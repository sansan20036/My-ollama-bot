# app/services/chat_service.py
import logging
import os
import re
import traceback
import json
from datetime import date, timedelta
import pandas as pd
from typing import Any, Optional, AsyncGenerator, List, Dict
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
# 新增：引入建構多模態訊息所需的套件
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.cache_service import SemanticCacheService
from app.services.table_analyzer_service import TableAnalyzerService
from app.utils.schedule_utils import clean_doctor_name
from app.utils.time_utils import augment_query_with_time_hints

logger = logging.getLogger(__name__)


@tool
def calculate_medical_fee(has_transfer: bool, drug_cost: int = -1) -> str:
    """
    計算醫院看診的總費用。當使用者詢問看診費用、藥費、部分負擔時，必須呼叫此工具。

    Args:
        has_transfer: 是否有經過診所轉診 (如果有轉診為 True，未經轉診直接來為 False)
        drug_cost: 藥品的總費用。🚨 如果使用者沒有提供藥費，或者你不知道，請務必填寫 -1。
    """
    # 🚨 只要 AI 填了 -1 (代表它不知道藥費)，就發動反問！
    if drug_cost == -1:
        return "【系統警告】：資料不足！請直接以自然語言回覆使用者：『請問您的藥費大約是多少元呢？我需要藥費才能為您計算總金額。』"

    # 防呆：確保一定是數字
    try:
        cost_val = int(drug_cost)
    except:
        return "【系統警告】：藥費格式錯誤！請反問使用者藥費是多少。"

    # 1. 醫院寫死的黃金準則：門診基本負擔
    base_fee = 280 if has_transfer else 420

    # 2. 醫院寫死的黃金準則：藥費部分負擔級距表
    if cost_val <= 100:
        drug_fee = 0
    elif cost_val <= 200:
        drug_fee = 40
    elif cost_val <= 300:
        drug_fee = 60
    elif cost_val <= 400:
        drug_fee = 80
    elif cost_val <= 500:
        drug_fee = 100
    elif cost_val <= 600:
        drug_fee = 120
    elif cost_val <= 700:
        drug_fee = 140
    elif cost_val <= 800:
        drug_fee = 160
    elif cost_val <= 900:
        drug_fee = 180
    elif cost_val <= 1000:
        drug_fee = 200
    else:
        # 超過 1000 元，上限就是 300 元
        drug_fee = 300

    total = base_fee + drug_fee

    # 回傳結果給 AI
    return f"【系統計算結果】門診基本負擔: {base_fee}元，藥費部分負擔: {drug_fee}元。總計應繳費用: {total}元。"


# 將工具打包成列表
tools = [calculate_medical_fee]


class ChatService:
    def __init__(self):
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        self.vector_store = VectorStoreService.get_instance()
        self.cache = SemanticCacheService.get_instance()
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        self.cached_df = None
        self.cached_file_path = ""
        self.cached_file_mtime = 0
        self._schedule_overrides_cache = None
        self._schedule_overrides_mtime = 0.0

        # 修改：將預設模型切換為設定檔中的模型
        target_model = settings.OLLAMA_MODEL

        logger.info(f"初始化全能文件聊天服務: {target_model}")

        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=target_model,
            temperature=0,  # Agent 運算設為 0，確保程式碼與數學精準
            keep_alive="1h",
            num_ctx=16384,
            num_predict=4096,
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {settings.OLLAMA_API_KEY}"
                }
            }
        )

    def _get_valid_files(self) -> list:
        if not os.path.exists(self.upload_dir): return []

        overrides_basename = os.path.basename(
            str(getattr(settings, "SCHEDULE_OVERRIDES_FILE", "") or "").strip()
        )

        # 新增防呆：過濾掉結尾是 _tables.csv 的系統快取檔，只計算使用者真正上傳的檔案！
        files = [f for f in os.listdir(self.upload_dir) if
                 os.path.isfile(os.path.join(self.upload_dir, f))
                 and not f.startswith("~")
                 and not f.endswith("_tables.csv")
                 and (not overrides_basename or f != overrides_basename)]

        # 依照檔案的「最後修改/建立時間」進行排序 (由舊到新)
        files.sort(key=lambda x: os.path.getctime(os.path.join(self.upload_dir, x)))
        return files

    def _get_sorted_file_list(self, files: list) -> str:
        if not files: return "(無檔案)"

        result = []
        for i, f in enumerate(files):
            label = ""
            if len(files) > 1:
                if i == len(files) - 1:
                    label = "(最新上傳)"
                elif i == 0:
                    label = "(最早上傳)"
            result.append(f"{i + 1}. {f}{label}")

        return "\n".join(result)

    def _num_to_chinese(self, num_str):
        try:
            n = int(num_str)
            units = ["", "十", "百"]
            chars = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
            if n == 0: return chars[0]
            result = ""
            s = str(n)[::-1]
            for i, d in enumerate(s):
                d = int(d)
                if i >= len(units): break
                if d != 0:
                    if i == 1 and d == 1 and len(s) == 2:
                        result = units[i] + result
                    else:
                        result = chars[d] + units[i] + result
                else:
                    if result and result[0] != chars[0]: result = chars[0] + result
            return result
        except:
            return num_str

    def _chinese_to_num(self, cn_str):
        if cn_str.isdigit(): return int(cn_str)
        cn_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                  '百': 100}
        try:
            if cn_str.startswith("十"):
                return 10 + cn_map.get(cn_str[1], 0) if len(cn_str) > 1 else 10
            elif len(cn_str) == 2 and cn_str[1] == "十":
                return cn_map[cn_str[0]] * 10
            elif len(cn_str) == 3 and cn_str[1] == "十":
                return cn_map[cn_str[0]] * 10 + cn_map[cn_str[2]]
            elif "百" in cn_str:
                return 100
            else:
                return cn_map.get(cn_str, 0)
        except:
            return 0

    def _get_schedule_overrides_path(self) -> str:
        raw = str(getattr(settings, "SCHEDULE_OVERRIDES_FILE", "") or "").strip()
        if not raw:
            return ""
        if os.path.isabs(raw):
            return raw
        return os.path.join(self.upload_dir, raw)

    @staticmethod
    def _normalize_date_str(date_text: str) -> Optional[str]:
        value = str(date_text or "").strip()
        if not value:
            return None
        value = value.replace(".", "-").replace("/", "-")
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", value)
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            return None

    def _load_schedule_overrides(self) -> Dict[str, Any]:
        default_payload = {"closed_dates": {}, "closed_slots": []}
        path = self._get_schedule_overrides_path()
        if not path or not os.path.exists(path):
            return default_payload

        try:
            mtime = os.path.getmtime(path)
            if self._schedule_overrides_cache is not None and self._schedule_overrides_mtime == mtime:
                return self._schedule_overrides_cache

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}

            closed_dates: Dict[str, str] = {}
            raw_closed_dates = raw.get("closed_dates", {})
            if isinstance(raw_closed_dates, list):
                for item in raw_closed_dates:
                    if isinstance(item, str):
                        d = self._normalize_date_str(item)
                        if d:
                            closed_dates[d] = ""
                    elif isinstance(item, dict):
                        d = self._normalize_date_str(item.get("date"))
                        if d:
                            closed_dates[d] = str(item.get("reason") or "").strip()
            elif isinstance(raw_closed_dates, dict):
                for k, v in raw_closed_dates.items():
                    d = self._normalize_date_str(k)
                    if not d:
                        continue
                    if isinstance(v, str):
                        closed_dates[d] = v.strip()
                    elif isinstance(v, dict):
                        closed_dates[d] = str(v.get("reason") or "").strip()
                    else:
                        closed_dates[d] = ""

            closed_slots: List[Dict[str, str]] = []
            for item in (raw.get("closed_slots", []) or []):
                if not isinstance(item, dict):
                    continue
                d = self._normalize_date_str(item.get("date"))
                if not d:
                    continue
                closed_slots.append(
                    {
                        "date": d,
                        "department": str(item.get("department") or "").strip(),
                        "period": str(item.get("period") or "").strip(),
                        "reason": str(item.get("reason") or "").strip(),
                    }
                )

            payload = {"closed_dates": closed_dates, "closed_slots": closed_slots}
            self._schedule_overrides_cache = payload
            self._schedule_overrides_mtime = mtime
            logger.info(
                "已載入日期級停診覆寫：closed_dates=%s closed_slots=%s file=%s",
                len(closed_dates),
                len(closed_slots),
                os.path.basename(path),
            )
            return payload
        except Exception as e:
            logger.warning("載入 schedule_overrides.json 失敗，略過覆寫：%s", e)
            return default_payload

    @staticmethod
    def _resolve_day_date_map(day_list: List[str], week_offset: int, time_meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
        day_to_idx = {"星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6}
        mapping: Dict[str, str] = {}
        if not day_list:
            return mapping

        today = date.today()
        if isinstance(week_offset, int) and week_offset > 0:
            start_of_this_week = today - timedelta(days=today.weekday())
            target_week_start = start_of_this_week + timedelta(weeks=week_offset)
            for day in day_list:
                idx = day_to_idx.get(day)
                if idx is None:
                    continue
                mapping[day] = (target_week_start + timedelta(days=idx)).isoformat()
            return mapping

        dates_by_day = dict((time_meta or {}).get("dates_by_day", {}) or {})
        for day in day_list:
            d = dates_by_day.get(day)
            if d:
                mapping[day] = d
        return mapping

    def _apply_date_level_overrides(
            self,
            day_list: List[str],
            day_date_map: Dict[str, str],
            grouped: Dict[str, Dict[str, List[str]]],
            grouped_without_surname: Dict[str, Dict[str, List[str]]],
            dept_keyword: str,
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        overrides = self._load_schedule_overrides()
        closed_dates = overrides.get("closed_dates", {}) or {}
        closed_slots = overrides.get("closed_slots", []) or []
        if not closed_dates and not closed_slots:
            return {"all_day": {}, "period": {}}

        all_day_notes: Dict[str, str] = {}
        period_notes: Dict[str, Dict[str, str]] = {}

        for day in day_list:
            target_date = day_date_map.get(day)
            if not target_date:
                continue

            # 全院整日停診
            if target_date in closed_dates:
                reason = str(closed_dates.get(target_date) or "").strip()
                grouped[day] = {}
                grouped_without_surname[day] = {}
                all_day_notes[day] = reason or "全院停診"
                continue

            # 日期級時段停診（可選擇限定科別）
            for slot in closed_slots:
                if slot.get("date") != target_date:
                    continue

                slot_dept = slot.get("department", "")
                if slot_dept:
                    # 若使用者查詢未指定科別，避免誤套用到所有科。
                    if not dept_keyword:
                        continue
                    if slot_dept not in dept_keyword and dept_keyword not in slot_dept:
                        continue

                slot_period = slot.get("period", "")
                reason = slot.get("reason", "") or "停診"
                if slot_period:
                    grouped.setdefault(day, {}).pop(slot_period, None)
                    grouped_without_surname.setdefault(day, {}).pop(slot_period, None)
                    period_notes.setdefault(day, {})[slot_period] = reason
                else:
                    grouped[day] = {}
                    grouped_without_surname[day] = {}
                    all_day_notes[day] = reason
                    break

        return {"all_day": all_day_notes, "period": period_notes}

    def _build_lexical_fallback_terms(self, query: str, ai_keywords: str = "") -> list:
        """
        建立關鍵字補撈用詞（給 vector_store.keyword_search_in_file）。
        目的：向量相似度太低時，仍可在同檔全文中補抓關鍵字片段。
        """
        stopwords = {
            "請問", "幫我", "內容", "哪些", "什麼", "有沒有", "可以", "是否", "資訊", "一下", "一下子",
            "門診", "醫師", "醫生", "今天", "明天", "後天", "下週", "下下週",
        }
        terms = []

        for token in re.findall(r"[\u4e00-\u9fff]{2,}", str(query or "")):
            if token in stopwords:
                continue
            if token not in terms:
                terms.append(token)

        for token in str(ai_keywords or "").split():
            t = token.strip()
            if len(t) < 2 or t in stopwords:
                continue
            if t not in terms:
                terms.append(t)

        return terms[:24]

    @staticmethod
    def _parse_slot_sentence(content: str) -> Dict[str, str]:
        """
        從 schedule_slot 句型中補抓 department/day/period/doctor。
        兼容：
        【門診時段查詢】科別：「骨科」；時段：「上午」；星期三醫師：「王大明」。
        """
        text = str(content or "").strip()
        if not text:
            return {}

        pattern = re.compile(
            r"科別：「(?P<dept>[^」]+)」；時段：「(?P<period>[^」]+)」；(?P<day>[^醫師；]+)醫師：「(?P<doctor>[^」]+)」"
        )
        m = pattern.search(text)
        if not m:
            return {}

        day_raw = (m.group("day") or "").strip()
        day = day_raw
        if "星期天" in day_raw or "禮拜天" in day_raw or "週日" in day_raw or "周日" in day_raw:
            day = "星期日"
        elif "星期一" in day_raw or "週一" in day_raw or "周一" in day_raw or "禮拜一" in day_raw:
            day = "星期一"
        elif "星期二" in day_raw or "週二" in day_raw or "周二" in day_raw or "禮拜二" in day_raw:
            day = "星期二"
        elif "星期三" in day_raw or "週三" in day_raw or "周三" in day_raw or "禮拜三" in day_raw:
            day = "星期三"
        elif "星期四" in day_raw or "週四" in day_raw or "周四" in day_raw or "禮拜四" in day_raw:
            day = "星期四"
        elif "星期五" in day_raw or "週五" in day_raw or "周五" in day_raw or "禮拜五" in day_raw:
            day = "星期五"
        elif "星期六" in day_raw or "週六" in day_raw or "周六" in day_raw or "禮拜六" in day_raw:
            day = "星期六"

        return {
            "department": (m.group("dept") or "").strip(),
            "period": (m.group("period") or "").strip(),
            "day": day,
            "doctor": clean_doctor_name((m.group("doctor") or "").strip()),
        }

    @staticmethod
    def _split_slot_doctors(raw: str) -> List[str]:
        """
        將 slot metadata 中可能混雜的醫師欄位拆成乾淨姓名清單。
        會剔除說明文字、英文殘留與非姓名 token。
        """
        text = str(raw or "").strip()
        if not text:
            return []

        stop_tokens = {
            "不指定", "休診", "停診", "無門診", "未註明", "特約醫師", "上午診下午診",
            "上班時段", "轉分機", "說明", "注意",
        }
        noisy_patterns = (
            "說明", "注意", "醫師專長", "限掛", "掛號", "預約", "請於", "電話",
            "OfficeHours", "MorningClinic", "AfternoonClinic", "EveryDay", "aday",
        )

        parts = re.split(r"[、,，;；/\\\s]+", text)
        doctors: List[str] = []
        for p in parts:
            token = clean_doctor_name(p)
            if not token:
                continue
            if token in stop_tokens:
                continue
            if any(k in token for k in noisy_patterns):
                continue
            # 僅保留中文姓名常見字元，避免英文與整句說明混入
            if not re.fullmatch(r"[\u4e00-\u9fa5·．]{2,6}", token):
                continue
            if token not in doctors:
                doctors.append(token)
        return doctors

    def _format_schedule_slots_answer(
            self,
            query: str,
            file_path: str,
            filename: str,
            week_offset: int = 0,
            time_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        優先用向量庫中的結構化 schedule_slot 回答，避免 CSV 欄位偏移造成無門診誤判。
        回傳空字串代表不適用或查無可用資料，外層可退回 pandas/RAG。
        """
        try:
            slot_data = self.vector_store.get_schedule_slot_documents(
                file_path=file_path,
                filename=filename,
                slot_types=["schedule_slot_local", "schedule_slot"],
                limit=8000,
            )
            metadatas = slot_data.get("metadatas", []) or []
            documents = slot_data.get("documents", []) or []
            logger.info("門診 slot 直答來源筆數: %s", len(metadatas))
            if not metadatas:
                return ""

            constraints = TableAnalyzerService._extract_constraints(query)
            dept_keyword = TableAnalyzerService._infer_department_keyword(query)
            special_strategy = TableAnalyzerService.get_special_department_strategy(query)
            include_days = constraints.get("include_days", set()) or set()
            exclude_days = constraints.get("exclude_days", set()) or set()
            include_periods = constraints.get("include_periods", set()) or set()
            exclude_periods = constraints.get("exclude_periods", set()) or set()
            surname = constraints.get("surname")
            tm = time_meta or {}
            tm_days = set(tm.get("days", []) or [])
            tm_periods = set(tm.get("periods", []) or [])
            # 僅在「沒有明確 include 條件」時才套用時間解析器結果，
            # 並先扣除 exclude，避免「除了星期一」被誤當成 include=星期一。
            if not include_days and tm_days:
                inferred_days = tm_days - exclude_days
                if inferred_days:
                    include_days = inferred_days
            if not include_periods and tm_periods:
                inferred_periods = tm_periods - exclude_periods
                if inferred_periods:
                    include_periods = inferred_periods

            slot_rows: List[Dict[str, Any]] = []
            for idx, meta in enumerate(metadatas):
                if not isinstance(meta, dict):
                    continue
                content = str(documents[idx] if idx < len(documents) else "")
                dept = str(meta.get("department") or meta.get("dept") or "").strip()
                day = str(meta.get("day") or "").strip()
                period = str(meta.get("period") or "").strip() or "未註明"
                doctor = clean_doctor_name(str(meta.get("doctor") or "").strip())
                table_title = str(meta.get("table_title") or "").strip()

                # fallback：舊資料常只有 page_content，無結構化 metadata
                if (not dept or not day or not doctor) and content:
                    parsed = self._parse_slot_sentence(content)
                    if parsed:
                        dept = dept or parsed.get("department", "")
                        day = day or parsed.get("day", "")
                        period = (period if period != "未註明" else "") or parsed.get("period", "未註明")
                        doctor = clean_doctor_name(doctor or parsed.get("doctor", ""))

                doctor_list = self._split_slot_doctors(doctor)
                if not dept or not day or not doctor_list:
                    continue
                if include_days and day not in include_days:
                    continue
                if day in exclude_days:
                    continue
                if include_periods and period not in include_periods:
                    continue
                if period in exclude_periods:
                    continue
                slot_rows.append(
                    {
                        "dept": dept,
                        "day": day,
                        "period": period,
                        "doctors": doctor_list,
                        "content": content,
                        "table_title": table_title,
                    }
                )

            def build_grouping(row_matcher):
                grouped_local: Dict[str, Dict[str, List[str]]] = {}
                grouped_no_surname_local: Dict[str, Dict[str, List[str]]] = {}
                matched_local = 0

                for row in slot_rows:
                    if not row_matcher(row):
                        continue
                    day = row["day"]
                    period = row["period"]
                    doctors = list(row["doctors"])

                    grouped_no_surname_local.setdefault(day, {})
                    grouped_no_surname_local[day].setdefault(period, [])
                    for dname in doctors:
                        if dname not in grouped_no_surname_local[day][period]:
                            grouped_no_surname_local[day][period].append(dname)

                    filtered_doctors = doctors
                    if surname:
                        filtered_doctors = [dname for dname in doctors if dname.startswith(surname) or surname in dname]
                        if not filtered_doctors:
                            continue

                    grouped_local.setdefault(day, {})
                    grouped_local[day].setdefault(period, [])
                    for dname in filtered_doctors:
                        if dname not in grouped_local[day][period]:
                            grouped_local[day][period].append(dname)
                    matched_local += len(filtered_doctors)

                return grouped_local, grouped_no_surname_local, matched_local

            def primary_matcher(row: Dict[str, Any]) -> bool:
                if special_strategy:
                    terms = list(special_strategy.get("primary_terms", []) or [])
                    hay_dept = str(row.get("dept", ""))
                    hay_content = str(row.get("content", ""))
                    hay_title = str(row.get("table_title", ""))
                    hay_doctors = "、".join(row.get("doctors", []) or [])
                    hay_raw = f"{hay_dept} {hay_content} {hay_title} {hay_doctors}"
                    hay_norm = re.sub(r"[^\w\u4e00-\u9fff]", "", hay_raw).lower()
                    for t in terms:
                        if not t:
                            continue
                        t_norm = re.sub(r"[^\w\u4e00-\u9fff]", "", str(t)).lower()
                        if not t_norm:
                            continue
                        if (
                            t in hay_dept
                            or t in hay_content
                            or t in hay_title
                            or t in hay_doctors
                            or t_norm in hay_norm
                        ):
                            return True
                    return False
                if dept_keyword:
                    return dept_keyword in str(row.get("dept", ""))
                return True

            def summarize_rows(rows: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
                items: List[Dict[str, Any]] = []
                for row in rows[:limit]:
                    doctors = list(row.get("doctors", []) or [])
                    items.append(
                        {
                            "day": row.get("day", ""),
                            "period": row.get("period", ""),
                            "dept": row.get("dept", ""),
                            "table_title": row.get("table_title", ""),
                            "doctor_preview": "、".join(doctors[:3]),
                            "doctor_count": len(doctors),
                        }
                    )
                return items

            primary_candidate_rows = [row for row in slot_rows if primary_matcher(row)]
            if special_strategy:
                logger.info(
                    "特診第一階段命中: label=%s terms=%s rows=%s sample=%s",
                    str((special_strategy or {}).get("label", "")),
                    list((special_strategy or {}).get("primary_terms", []) or []),
                    len(primary_candidate_rows),
                    summarize_rows(primary_candidate_rows, limit=5),
                )

            grouped, grouped_without_surname, matched_count = build_grouping(primary_matcher)
            used_fallback_department = ""
            if matched_count == 0 and special_strategy and special_strategy.get("fallback_department"):
                fallback_dept = str(special_strategy.get("fallback_department") or "").strip()
                if fallback_dept:
                    fallback_candidate_rows = [
                        row for row in slot_rows if fallback_dept in str(row.get("dept", ""))
                    ]
                    logger.info(
                        "特診第二階段降級: fallback_dept=%s rows=%s sample=%s",
                        fallback_dept,
                        len(fallback_candidate_rows),
                        summarize_rows(fallback_candidate_rows, limit=5),
                    )
                    grouped, grouped_without_surname, matched_count = build_grouping(
                        lambda row: fallback_dept in str(row.get("dept", ""))
                    )
                    if matched_count > 0:
                        used_fallback_department = fallback_dept

            logger.info(
                "門診 slot 直答過濾結果: matched=%s dept=%s special=%s fallback=%s include_days=%s exclude_days=%s include_periods=%s exclude_periods=%s surname=%s",
                matched_count,
                dept_keyword or "",
                str((special_strategy or {}).get("label", "")),
                used_fallback_department,
                sorted(list(include_days)) if include_days else [],
                sorted(list(exclude_days)) if exclude_days else [],
                sorted(list(include_periods)) if include_periods else [],
                sorted(list(exclude_periods)) if exclude_periods else [],
                surname or "",
            )

            # 決定要輸出的星期順序
            weekday_order = {"星期一": 1, "星期二": 2, "星期三": 3, "星期四": 4, "星期五": 5, "星期六": 6, "星期日": 7}
            if include_days:
                day_list = sorted(list(include_days), key=lambda d: weekday_order.get(d, 99))
            else:
                # 無 include 條件時，盡量保留 baseline 的日別（含姓氏過濾前）
                day_pool = set(grouped.keys()) | set(grouped_without_surname.keys())
                day_list = sorted(list(day_pool), key=lambda d: weekday_order.get(d, 99))

            if not day_list:
                return ""

            # 日期級停診覆寫：把「相對週模板」對應成實際日期後，覆寫停診/停時段。
            day_date_map = self._resolve_day_date_map(
                day_list=day_list,
                week_offset=week_offset,
                time_meta=time_meta,
            )
            override_notes = self._apply_date_level_overrides(
                day_list=day_list,
                day_date_map=day_date_map,
                grouped=grouped,
                grouped_without_surname=grouped_without_surname,
                dept_keyword=dept_keyword or "",
            )
            all_day_notes = override_notes.get("all_day", {}) or {}
            period_notes = override_notes.get("period", {}) or {}

            lines: List[str] = []
            if used_fallback_department:
                label = str((special_strategy or {}).get("label", "指定特診"))
                lines.append(f"※ 目前查無「{label}」專屬特診標記，以下提供「{used_fallback_department}」門診供參考。")
                lines.append("")
            if isinstance(week_offset, int) and week_offset > 0:
                lines.append(f"※ 此查詢為相對時間推算（{week_offset} 週後），依門診週模板（星期幾）顯示。")
                lines.append("")
            for day in day_list:
                date_label = day_date_map.get(day)
                if date_label:
                    lines.append(f"【{day}（{date_label}）】")
                else:
                    lines.append(f"【{day}】")

                if day in all_day_notes:
                    lines.append(f" - 全院停診（{all_day_notes[day]}）")
                    lines.append("")
                    continue

                periods = grouped.get(day, {})
                if not periods:
                    baseline_periods = grouped_without_surname.get(day, {})
                    closed_periods = period_notes.get(day, {})
                    if closed_periods and not baseline_periods:
                        ordered_closed = sorted(list(closed_periods.keys()), key=lambda p: {"上午": 1, "下午": 2, "夜間": 3}.get(p, 99))
                        for p in ordered_closed:
                            lines.append(f" - {p}：停診（{closed_periods.get(p) or '日期級停診'}）")
                    elif surname and baseline_periods:
                        lines.append(f" - 當天有門診，但沒有姓「{surname}」的醫師")
                        alt_ordered_periods = [p for p in ("上午", "下午", "夜間") if p in baseline_periods]
                        alt_ordered_periods += [p for p in baseline_periods.keys() if p not in alt_ordered_periods]
                        for p in alt_ordered_periods:
                            alt_names = baseline_periods.get(p, [])
                            if not alt_names:
                                continue
                            preview = alt_names[:8]
                            suffix = " 等" if len(alt_names) > 8 else ""
                            lines.append(f" - 可參考 {p}：{'、'.join(preview)}{suffix}")
                    else:
                        lines.append(" - 無門診")
                    lines.append("")
                    continue
                ordered_periods = [p for p in ("上午", "下午", "夜間") if p in periods]
                ordered_periods += [p for p in periods.keys() if p not in ordered_periods]
                for p in ordered_periods:
                    names = periods.get(p, [])
                    if names:
                        lines.append(f" - {p}：{'、'.join(names)}")
                # 顯示被日期級規則覆寫掉的時段
                day_closed_periods = period_notes.get(day, {})
                for p in sorted(day_closed_periods.keys(), key=lambda x: {"上午": 1, "下午": 2, "夜間": 3}.get(x, 99)):
                    if p not in periods:
                        lines.append(f" - {p}：停診（{day_closed_periods.get(p) or '日期級停診'}）")
                lines.append("")

            return "\n".join(lines).strip()
        except Exception as e:
            logger.warning(f"schedule_slot 直答失敗，改走 pandas: {e}")
            return ""

    async def _smart_query_rewrite(self, user_query: str) -> str:
        """
        萬用型意圖預判 (Universal Intent Prediction) - 已加上錯誤防護
        """
        try:
            rewrite_prompt = ChatPromptTemplate.from_template(
                """你是高階文件檢索專家。使用者的問題是：「{query}」。
                你的任務是分析這個問題，並預測「在目標文件中，這段內容可能包含哪些關鍵字或術語」。
                請忽略文件的具體類型，直接根據常識進行聯想。

                請輸出 5~10 個「最能精準命中文件內容」的搜尋關鍵字。
                直接輸出關鍵字，用空格分隔，不要有解釋。

                範例：
                (問：老闆不給資遣費) -> 勞動基準法 終止契約 第17條 資遣費 罰則
                (問：Docker連不上) -> Connection refused, port mapping, 網路設定, 防火牆

                現在請輸出關鍵字："""
            )

            chain = rewrite_prompt | self.llm | StrOutputParser()
            logger.info("AI 正在進行萬用關鍵字聯想...")
            refined_query = await chain.ainvoke({"query": user_query})
            clean_query = refined_query.replace("\n", " ").strip()
            logger.info("AI 聯想關鍵字: %s", clean_query)
            return clean_query
        except Exception as e:
            logger.error(f"關鍵字聯想失敗 (略過此步驟): {e}")
            return ""  # 聯想失敗時優雅退回，不讓程式崩潰

    async def process_query(
            self,
            query: str,
            history: list,
            images: list = None,
            model_name: str = None,
            session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            # 動態切換邏輯：有傳名字就用傳來的，沒有就用 config 預設的
            actual_model = model_name if model_name else settings.OLLAMA_MODEL
            logger.info(f"本次對話使用的模型為: {actual_model}")

            # 每次發問都重新設定一次 llm
            self.llm = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=actual_model,
                temperature=0,
                keep_alive="1h",
                num_ctx=16384,
                num_predict=4096,
                client_kwargs={
                    "headers": {
                        "Authorization": f"Bearer {settings.OLLAMA_API_KEY}"
                    }
                }
            )

            real_query = query
            retrieval_query, time_meta = augment_query_with_time_hints(real_query)
            logger.info(
                "時間解析: days=%s periods=%s week_offset=%s past=%s",
                time_meta.get("days", []),
                time_meta.get("periods", []),
                time_meta.get("week_offset", 0),
                time_meta.get("is_past_reference", False),
            )

            schedule_keywords = [
                "門診", "看診", "醫師", "醫生", "科", "掛號", "夜診", "上午", "下午",
                "週末", "周末", "星期", "禮拜", "週", "周"
            ]
            if time_meta.get("is_past_reference") and any(k in real_query for k in schedule_keywords):
                yield "抱歉，目前僅支援查詢目前與未來門診，暫不支援上個月、上週或更早的歷史門診。"
                return

            # 確保 images 絕對不是 None
            if images is None:
                images = []

            valid_files = self._get_valid_files()
            file_count = len(valid_files)
            file_list_str = self._get_sorted_file_list(valid_files)

            # 相容 dict 與 Pydantic Message 兩種型別，避免 Message 物件不可下標錯誤
            history_lines = []
            for msg in (history[-2:] if history else []):
                if isinstance(msg, dict):
                    role = str(msg.get("role", ""))
                    content = str(msg.get("content", ""))
                else:
                    role = str(getattr(msg, "role", ""))
                    content = str(getattr(msg, "content", ""))
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines) if history_lines else "(無歷史紀錄)"

            # 雙模式架構分流器(聊天模式與檔案問答模式)
            if file_count == 0:
                final_context = "使用者目前沒有提供任何文件。請直接以你豐富的常識與專業知識回答他的問題。"
                domain_rules = """
                    [GENERAL CONVERSATION MODE]
                    - You are a friendly, knowledgeable AI assistant.
                    - Since no documents are provided, answer the user's question directly based on your internal knowledge base.
                    - Be helpful, conversational, and precise.
                    - Do not mention that you are reading from a document.
                """
            else:
                target_file = os.path.join(self.upload_dir, valid_files[-1])  # 取最新上傳的檔案
                file_name_without_ext = os.path.splitext(target_file)[0]
                file_ext = target_file.lower().split('.')[-1]  # 取得副檔名
                df = None
                current_mtime = os.path.getmtime(target_file)

                # 尋找是否已有在上傳階段提煉好的 CSV 快取
                possible_csv = f"{file_name_without_ext}_tables.csv"

                if self.cached_file_path == target_file and self.cached_file_mtime == current_mtime:
                    logger.info("使用記憶體中的 DataFrame，跳過檔案解析")
                    df = self.cached_df
                else:
                    try:
                        # 優先檢查：上傳時是否已經提煉出表格了？
                        if os.path.exists(possible_csv):
                            logger.info("發現預處理的 PDF 表格快取！直接秒讀載入...")
                            df = pd.read_csv(possible_csv)
                        elif file_ext in ['xlsx', 'xls']:
                            logger.info("偵測到原生 Excel 檔案，直接載入...")
                            df = pd.read_excel(target_file)
                            df.columns = [re.split(r'[\s\n(]', str(col))[0] for col in df.columns]
                        elif file_ext == 'csv':
                            logger.info(" 偵測到原生 CSV 檔案，直接載入...")
                            df = pd.read_csv(target_file)
                            df.columns = [re.split(r'[\s\n(]', str(col))[0] for col in df.columns]
                    except Exception as e:
                        logger.error(f"讀取 DataFrame 時發生錯誤: {e}")

                    # 將結果存入快取
                    self.cached_df = df
                    self.cached_file_path = target_file
                    self.cached_file_mtime = current_mtime

                if df is not None and not df.empty:
                    # 企業級升級：LLM 語意路由器 (Semantic Router)

                    logger.info(" 啟動混合型語意路由器 (Hybrid Router)...")

                    # 1. 物理攔截：算錢這種大事，交給 Python 決定最穩，保證 100% 觸發率！
                    fee_keywords = ["錢", "費用", "藥費", "負擔", "計算", "多少"]
                    if any(kw in real_query for kw in fee_keywords):
                        logger.info(" 物理攔截：偵測到費用關鍵字，強制導向 CALCULATOR")
                        intent_result = "CALCULATOR"
                    else:
                        # 2. 其他問題再交給 AI 判斷 (使用最傳統、所有模型都支援的純文字模式)
                        router_prompt = ChatPromptTemplate.from_template(
                            "你是一個分類系統。請判斷以下問題屬於哪一類：\n"
                            "1. 如果是找特定科別的醫生、門診時間，請輸出 PANDAS\n"
                            "2. 如果是其他問題(如圖片內容、醫院規定、文件內容)，請輸出 RAG\n\n"
                            "問題：「{query}」\n"
                            "請嚴格只輸出 PANDAS 或 RAG 單字："
                        )
                        router_chain = router_prompt | self.llm | StrOutputParser()

                        try:
                            raw_result = await router_chain.ainvoke({"query": real_query})
                            # 用正則表達式把 PANDAS 或 RAG 抓出來，防止 AI 講廢話
                            if "PANDAS" in raw_result.upper():
                                intent_result = "PANDAS"
                            else:
                                intent_result = "RAG"
                        except Exception as e:
                            logger.error(f"路由失敗，降級為 RAG: {e}")
                            intent_result = "RAG"

                    logger.info(f" 🎯 最終路由判定: {intent_result}")
                    if "PANDAS" in intent_result and not TableAnalyzerService.looks_like_schedule_query(real_query):
                        logger.info("PANDAS 防呆：偵測為非門診查詢，改走 RAG")
                        intent_result = "RAG"

                    if "PANDAS" in intent_result:
                        logger.info(" 執行路線：啟動 [自建 Python 直譯引擎]")
                        try:
                            # 優先使用結構化 schedule_slot，避免 CSV 欄位偏移造成「無門診」誤判。
                            slot_answer = self._format_schedule_slots_answer(
                                query=retrieval_query,
                                file_path=target_file,
                                filename=valid_files[-1],
                                week_offset=int(time_meta.get("week_offset", 0) or 0),
                                time_meta=time_meta,
                            )
                            if slot_answer:
                                yield slot_answer
                                return

                            result_str = await TableAnalyzerService.query_and_format_schedule(
                                df=df,
                                query=real_query,
                                llm=self.llm
                            )

                            if "查無「" in result_str and "」的相關資料" in result_str:
                                yield "目前查無符合您條件的門診資料。"
                                return

                            # 直接回傳結構化結果，避免二次 LLM 改寫造成時間語意偏差或誤答。
                            yield result_str
                            return
                        except Exception as e:
                            logger.error(f"數據運算失敗，降級回傳統 RAG 模式: {e}")
                    elif "CALCULATOR" in intent_result:
                        logger.info(" 執行路線：啟動 [Agent 工具計算引擎]")

                        # 建立 Agent 專用 Prompt
                        agent_prompt = ChatPromptTemplate.from_messages([
                            ("system",
                             "你是一個專業的醫療費用計算客服。你的任務是計算「門診負擔」與「藥費負擔」。\n\n"
                             "【執行規則】：\n"
                             "1. 必須同時擁有「轉診狀態」與「藥費金額」才能進行計算。\n"
                             "2. 請先檢查 [使用者問題] 與 [歷史對話記憶]。\n"
                             "3. 🚨 如果藥費金額不明，請『絕對不要』亂編數字。請直接呼叫工具並在藥費填寫 -1。\n\n"
                             "【歷史對話記憶】：\n{history}"),
                            ("human", "{input}"),
                            ("placeholder", "{agent_scratchpad}"),
                        ])

                        try:
                            # 綁定工具並建立 Agent
                            agent = create_tool_calling_agent(self.llm, tools, agent_prompt)

                            # 👉 加入 handle_parsing_errors=True 允許 AI 自我糾錯
                            agent_executor = AgentExecutor(
                                agent=agent,
                                tools=tools,
                                verbose=True,
                                handle_parsing_errors=True
                            )

                            logger.info("🤖 Agent 思考與計算中...")
                            response = await agent_executor.ainvoke({
                                "input": real_query,
                                "history": history_text
                            })
                            final_answer = response.get("output", "抱歉，計算費用時發生錯誤。")

                            # 直接將 Agent 的完美回答回傳給前端
                            yield final_answer
                            return

                        except Exception as e:
                            logger.error(f"Agent 執行失敗，已攔截錯誤: {e}")
                            yield "【系統提示】計算機精靈剛才腦袋打結了，請再問我一次，或直接提供藥費給我喔！"
                            return

                    else:
                        logger.info(" 執行路線：跳過表格運算，進入 [RAG 文本檢索模式]")

            # 傳統 RAG 模式 (如果沒表格、或是意圖判定為閱讀理解，就會順暢地走到這裡)
            current_file_path = os.path.join(self.upload_dir, valid_files[-1]) if valid_files else ""
            file_filter = None
            sid = (session_id or "").strip()
            if sid:
                # metadata 寫入時使用 upload_session_id
                file_filter = {"upload_session_id": sid}
                logger.info("已套用同批次檔案範圍(upload_session_id=%s)", sid)

            # 1. 第一輪：通用檢索
            ai_keywords = await self._smart_query_rewrite(retrieval_query)
            search_query = f"{retrieval_query} {ai_keywords}"

            matches = re.findall(r'(?:第\s*\d+\s*[章節條頁]|(?<!\d)\d+\.\d+(?:\.\d+)?(?!\d))', real_query)

            if matches:
                for m in matches:
                    search_query += f" {m}"

            logger.info("執行檢索: %s (限定檔案: %s)", search_query, valid_files[-1] if valid_files else "無檔案")

            docs = self.vector_store.search(search_query, k=50, filter=file_filter)
            if sid and not docs:
                logger.warning(
                    "session 過濾後無結果，改用無 filter 重查。sid=%s query=%s",
                    sid,
                    search_query,
                )
                docs = self.vector_store.search(search_query, k=50, filter=None)

            if docs:
                logger.info("啟動 Reranker 精讀專家，重新評分中...")
                try:
                    # 載入 BAAI 多語系重排序模型 (第一次執行會自動下載模型檔)
                    reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3')

                    # 將「使用者的真實問題」與「這 50 筆資料」配對
                    sentence_pairs = [[real_query, doc.page_content] for doc in docs]

                    # 讓模型對每一對給出精準的關聯分數
                    scores = reranker_model.predict(sentence_pairs)

                    # 將分數寫入 doc 的 metadata 中，並依據分數由高到低重新排序
                    for doc, score in zip(docs, scores):
                        doc.metadata["rerank_score"] = float(score)

                    docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)

                    # 經過精讀後，我們只保留最精華、關聯度最高的前 10 筆給大語言模型！
                    # 徹底解決「迷失在中間」的問題！
                    docs = docs[:10]

                    logger.info("Reranker 篩選完畢！最高分: %.4f", docs[0].metadata['rerank_score'])
                except Exception as e:
                    logger.error(f"Reranker 執行失敗，退回原始檢索結果: {e}")

            # 向量相似度極低時，做一次同檔關鍵字補撈（lexical fallback）
            # 例如「防疫小叮嚀」這種段落，可能在 PDF 可見但向量沒命中。
            top_score = 0.0
            if docs:
                top_score = float(docs[0].metadata.get("rerank_score", 0.0) or 0.0)
            lexical_docs = []
            if valid_files and (not docs or top_score < 0.05):
                lexical_terms = self._build_lexical_fallback_terms(real_query, ai_keywords)
                if lexical_terms:
                    lexical_docs = self.vector_store.keyword_search_in_file(
                        filename=valid_files[-1],
                        keywords=lexical_terms,
                        session_id=sid if sid else None,
                        limit=15,
                    )
                    if lexical_docs:
                        logger.info(
                            "低相似度觸發關鍵字補撈：新增 %s 筆（top_score=%.4f）",
                            len(lexical_docs),
                            top_score,
                        )
                        docs.extend(lexical_docs)

            # 若是「逐頁搜尋/列出頁碼」題型，且已命中 lexical fallback，直接回覆頁碼，避免 LLM 漏答。
            page_intent = any(k in real_query for k in ["逐頁", "頁碼", "第幾頁", "哪一頁", "頁面"])
            if page_intent and lexical_docs:
                lexical_docs = self.vector_store.backfill_pages_for_documents(
                    filename=valid_files[-1],
                    docs=lexical_docs,
                    session_id=sid if sid else None,
                )
                pages: list[int] = []
                unknown_count = 0
                for d in lexical_docs:
                    page_val = d.metadata.get("page")
                    if page_val in (None, ""):
                        unknown_count += 1
                        continue
                    try:
                        p = int(page_val)
                        if p not in pages:
                            pages.append(p)
                    except Exception:
                        unknown_count += 1
                pages = sorted(pages)
                if pages:
                    page_text = "、".join([f"第 {p} 頁" for p in pages])
                    if unknown_count:
                        logger.info("頁碼查詢：仍有 %s 筆命中無頁碼（僅回傳可定位頁碼）", unknown_count)
                    yield f"已找到相關內容，頁碼如下：{page_text}"
                    return

            # ======== 👇 將「最新檔案狙擊模式」搬移到這裡 (繞過 Reranker 保送 VIP) ========
            if valid_files and any(kw in real_query for kw in ["最新", "這個", "這份", "這檔案"]):
                logger.info("偵測到代名詞，啟動「最新檔案狙擊模式」...")
                try:
                    latest_file_name = valid_files[-1]
                    latest_file_path = os.path.join(self.upload_dir, latest_file_name)

                    # 暴力破解：嘗試各種路徑格式，確保在 Windows 絕對抓得到資料！
                    latest_docs = self.vector_store.search(real_query, k=15, filter={"source": latest_file_path})
                    if not latest_docs:
                        latest_docs = self.vector_store.search(real_query, k=15,
                                                               filter={"source": latest_file_path.replace("\\", "/")})
                    if not latest_docs:
                        latest_docs = self.vector_store.search(real_query, k=15, filter={"source": latest_file_name})
                    if not latest_docs:
                        temp_docs = self.vector_store.search(real_query, k=100, filter=None)
                        latest_docs = [d for d in temp_docs if latest_file_name in str(d.metadata.get("source", ""))]
                        latest_docs = latest_docs[:15]

                    for d in latest_docs:
                        d.page_content = f"【使用者指定調閱：最新檔案內容】\n{d.page_content}"
                        docs.append(d)

                    logger.info("狙擊成功：已將最新檔案 (%s) 強制加入 %s 筆候選池！", latest_file_name, len(latest_docs))
                except Exception as e:
                    logger.warning("最新檔案狙擊發生錯誤: %s", e)
            # ================================================================

            # 新增：狙擊模式 (Sniper Mode)
            if matches:
                logger.info("偵測到明確條號/章節 %s，啟用狙擊模式", matches)
                existing_ids = set()
                for d in docs:
                    aid = d.metadata.get("article_id")
                    if aid: existing_ids.add(str(aid))

                for m in matches:
                    target_id = re.sub(r'[^\d.]', '', m)
                    if not target_id: continue

                    is_snipe_success = False

                    # 關鍵修改：如果是找「頁碼」，直接啟動硬核過濾器 (Metadata Filter)！
                    if "頁" in m:
                        logger.info("啟動硬核過濾：強制調閱第 %s 頁...", target_id)
                        try:
                            page_filter = {"page": int(target_id), "source": current_file_path}
                            sniper_docs = self.vector_store.search(real_query, k=5, filter=page_filter)

                            if sniper_docs:
                                for d in sniper_docs:
                                    d.page_content = f"【使用者指定調閱：第 {target_id} 頁】\n{d.page_content}"
                                    docs.insert(0, d)
                                existing_ids.add(target_id)
                                is_snipe_success = True
                                logger.info("狙擊成功：已將第 %s 頁內容強制拉至最前！", target_id)
                                continue  # 這頁找完了，跳到下一個 match
                        except Exception as e:
                            logger.warning("硬核過濾發生錯誤: %s", e)

                    # 以下保留給「非頁碼」的條號搜尋 (例如第 X 條)
                    if target_id in existing_ids: continue

                    sniper_query = f"第{target_id}條 第{target_id}章 第{target_id}節 {target_id}"
                    label_text = f"指定段落 {target_id}"
                    sniper_k = 1000
                    logger.info("啟動全域掃描尋找條號：目標 [%s]...", target_id)

                    sniper_docs = self.vector_store.search(sniper_query, k=sniper_k, filter=file_filter)

                    for d in sniper_docs:
                        fetched_id = str(d.metadata.get("article_id", ""))
                        if fetched_id == target_id:
                            d.page_content = f"【使用者指定調閱：{label_text}】\n{d.page_content}"
                            docs.insert(0, d)
                            existing_ids.add(target_id)
                            is_snipe_success = True
                            logger.info("狙擊成功：已將目標 [%s] 內容拉至最前！", target_id)
                            break

                    if not is_snipe_success:
                        logger.info("狙擊失敗：找不到包含 '%s' 的精確內容。", target_id)

            # 2. 第二輪：彈性補完
            existing_ids = set()
            has_structured_data = False

            for doc in docs:
                aid = doc.metadata.get("article_id")
                if aid:
                    existing_ids.add(str(aid))
                    has_structured_data = True

            if has_structured_data:
                logger.info("偵測到結構化資料，嘗試分析引用關係...")
                referenced_ids = set()
                for doc in docs:
                    content = doc.page_content
                    refs = re.findall(r'第\s*([0-9]+|[零一二三四五六七八九十百]+)\s*條', content)
                    for ref in refs:
                        if ref not in existing_ids and ref not in ["一", "二"]:
                            referenced_ids.add(ref)

                if referenced_ids:
                    target_refs = list(referenced_ids)[:5]
                    logger.info("發現引用，嘗試補完: %s", target_refs)

                    for ref_art in target_refs:
                        target_id = self._chinese_to_num(ref_art)
                        if target_id == 0: continue

                        fetch_query = f"第{ref_art}條"
                        supplementary_docs = self.vector_store.search(fetch_query, k=50, filter=file_filter)

                        for d in supplementary_docs:
                            fetched_id = str(d.metadata.get("article_id", ""))
                            if fetched_id == str(target_id) and fetched_id not in existing_ids:
                                d.page_content = f"【系統自動補完引用：第{ref_art}條】\n{d.page_content}"
                                docs.append(d)
                                existing_ids.add(fetched_id)
                                logger.info("成功補完 ID: %s", fetched_id)
                                break

            # 3. 排序與 Context
            def final_rank(doc):
                score = 0
                content = doc.page_content
                # 取得這塊資料的來源絕對路徑
                source_path = str(doc.metadata.get("source", ""))
                # 取得純檔名
                source_name = os.path.basename(source_path)
                # 取得最新檔案的檔名
                latest_file_name = os.path.basename(valid_files[-1]) if valid_files else ""

                # 1. 檔名精準命中霸王條款
                if source_name and (source_name in real_query or source_name.replace(".pdf", "") in real_query):
                    score += 5000

                # 2. 🚀 新增：「最新/這份/這個」代名詞霸王條款
                # 如果使用者問「最新檔案」，且這筆資料剛好來自最新檔案，給予絕對高分！
                if latest_file_name and source_name == latest_file_name:
                    if any(kw in real_query for kw in ["最新", "這個", "這份", "這檔案"]):
                        score += 5000

                if "【使用者指定調閱" in content: score += 2000
                if "【系統自動補完" in content: score += 50
                if doc.metadata.get("lexical_hit"): score += 1200
                if doc.metadata.get("type") == "file_summary": score += 1000
                if real_query in content: score += 100
                return score

            docs.sort(key=final_rank, reverse=True)

            final_context_list = []
            # 修改：把截斷限制從 [:10] 放大到 [:20] 或 [:25]
            # gemma3:27b 的胃口很大，多塞一點資料可以防止同時問兩份文件時被擠出去
            for doc in docs[:25]:
                source_raw = str(doc.metadata.get("source", "unknown"))
                source = os.path.basename(source_raw) if source_raw != "unknown" else "unknown"
                page = doc.metadata.get("page", "")
                article_id = doc.metadata.get("article_id", "")

                label = ""
                if article_id:
                    label = f" | 第 {article_id} 條"
                elif page:
                    label = f" | Page {page}"

                if doc.metadata.get("type") == "file_summary":
                    prefix = f"【全域摘要：{source}】"
                else:
                    prefix = f"【來源：{source}{label}】"

                final_context_list.append(f"{prefix}\n{doc.page_content}")

            schedule_like_query = TableAnalyzerService.looks_like_schedule_query(real_query)

            if file_count > 0:
                final_context = "\n\n".join(final_context_list) if final_context_list else "無具體內容。"
                if df is not None and schedule_like_query:
                    logger.info(" 啟動表格與文本融合，將表格標題列補給 RAG 引擎...")
                    table_info = f"表格欄位名稱：{list(df.columns)}\n前兩筆資料：{df.head(2).to_dict('records')}"
                    final_context += f"\n\n【系統強制補充：表格輔助資訊 (極可能包含預約電話與規定)】\n{table_info}"

            logger.info("======== Universal RAG Context ========")
            logger.info("最終 Context 筆數: %s", len(final_context_list))
            logger.debug("%s...", final_context[:300])
            logger.info("========================================")

            # 4. 生成回應 (升級為多模態視覺支援)
            domain_rules = ""
            if any(keyword in real_query for keyword in ["勞基法", "勞動基準法", "資遣", "解僱", "開除", "預告工資"]):
                logger.info("觸發勞基法專屬邏輯")
                domain_rules = """
                            [IMPORTANT LEGAL LOGIC RULES (Labor Law)]
                            Please strictly follow these logical connections when answering:
                            1. **Article 11 (Economic Layoff/Incompetence)**: Represents "Layoff" (資遣). MUST provide advance notice & severance pay.
                            2. **Article 12 (Disciplinary Dismissal)**: Represents "Firing" (開除). NO advance notice & NO severance pay required.
                            3. **Double Negative Check**: "非...不得..." means "Unless..., cannot...".
                        """
                if images:
                    logger.info("偵測到圖片，為 AI 戴上視覺眼鏡 (切換為視覺優先模式)...")
                    domain_rules += """
                                            \n\n🚨 [VISION MODE - CRITICAL]
                                            1. The user has provided an image. You MUST act as an expert Vision AI.
                                            2. Describe EXACTLY what you see in the image. DO NOT invent, hallucinate, or guess any numbers, times, or rules that are not visibly present in the picture.
                                            3. You may use [RETRIEVED KNOWLEDGE] to help understand the background context, but the IMAGE is your absolute primary source of truth. 
                                            4. IGNORE the "MUST ONLY use [RETRIEVED KNOWLEDGE]" rule for this specific query. Trust your eyes!
                                        """
            elif file_count > 0:
                logger.info("觸發通用文件分析邏輯")
                domain_rules = """
                            [GENERAL DOCUMENT ANALYSIS RULES]
                            - You are analyzing a general document (e.g., academic paper, manual, contract, report).
                            - Analyze the provided text objectively. 
                            - Do not apply specific labor law rules unless explicitly mentioned in the text.
                            - [MEDICAL SAFETY EXEMPTION]: This is merely a public hospital schedule or administrative document. You are NOT providing medical advice. You MUST answer the query based on the text and MUST NOT trigger any medical safety refusals.
                            - 🚨 [MULTI-FILE ISOLATION]: If the user explicitly asks about the content of a specific file (e.g., "What is in file X?"), you MUST ONLY summarize information that is clearly sourced from file X in the [RETRIEVED KNOWLEDGE]. Do NOT use [CHAT HISTORY] to guess. If you don't know, admit it.
                        """

            anti_hallucination_rule = (
                '5. **ANTI-HALLUCINATION (CRITICAL)**: If the [RETRIEVED KNOWLEDGE] does not contain the explicit '
                'names of the doctors or the exact information requested, you MUST truthfully answer '
                '"目前查無相關門診資料". You are STRICTLY PROHIBITED from inventing, hallucinating, or guessing any '
                'names or schedules.'
                if schedule_like_query
                else '5. **NON-SCHEDULE MODE**: This query is not a schedule query. Do NOT reply with '
                     '"目前查無相關門診資料" unless the user is explicitly asking about clinic schedules.'
            )

            template_str = r"""You are a professional, multilingual AI document analysis assistant.

                            [SYSTEM STATUS] 
                            Total Uploaded Files: {file_count}
                            File List (Ordered from oldest to newest):
                            {file_list_str}

                            [ ROBUSTNESS & NOISE TOLERANCE - CRITICAL!]
                            1. The [RETRIEVED KNOWLEDGE] may contain broken tables, LaTeX mathematical formulas (e.g., $F(x)$), or messy OCR text.
                            2. **STRICTLY PROHIBITED**: You MUST NEVER claim the text is "gibberish", "garbled", "corrupted", or "unreadable".
                            3. **YOUR DUTY**: Ignore formatting errors, raw formulas, and meaningless symbols. Focus ONLY on extracting the readable natural language sentences to answer the question.

                            [MATH & FORMATTING RULES - CRITICAL!]
                            When outputting mathematical formulas, equations, or variables, YOU MUST strictly use LaTeX formatting.
                            - For inline math and variables, wrap them in single dollar signs (e.g., $O(n^3)$, $A$, $\sigma_i$).
                            - For block equations, wrap them in double dollar signs on new lines (e.g., $$A w = b$$).
                            - DO NOT use raw unicode characters for complex math (like fractions or matrices). Always write them in standard LaTeX code.

                            {domain_rules}

                            [RETRIEVED KNOWLEDGE]
                            {context}

                            [CHAT HISTORY] {history}

                            [USER QUESTION] {question}

                            [MANDATORY LANGUAGE PROTOCOL]
                            1. **AUTO-DETECT**: Detect the language used in the [USER QUESTION].
                            2. **MATCH LANGUAGE**: Your entire response MUST be in the **SAME language** as the [USER QUESTION].
                            3. **TRANSLATION REQUIRED**: Read the context, understand it, and TRANSLATE & EXPLAIN it in the user's target language.

                            [CRITICAL READING RULES]
                            1. **NO SIMPLIFICATION**: When citation involves numbers, money, or days, DO NOT output a single number if the document lists a range or conditions.
                            2. **FULL LISTING**: Always list out all the tiered conditions found in the text.
                            3. **FACTUAL ACCURACY**: Your answer must perfectly match the [RETRIEVED KNOWLEDGE].
                            4. **CHAPTER MATCHING**: If the user asks for a specific Chapter (e.g., Chapter 7), YOU MUST ONLY use information from that chapter. If the retrieved context only shows Chapter 3, you must truthfully say: "I cannot find the content for Chapter 7 in the retrieved context," and DO NOT hallucinate using other chapters.
                            {anti_hallucination_rule}
                            """

            # 將變數塞入系統 Prompt 中
            system_content = template_str.format(
                file_count=str(file_count),
                file_list_str=file_list_str,
                domain_rules=domain_rules,
                context=final_context,
                history=history_text,
                question=real_query,
                anti_hallucination_rule=anti_hallucination_rule,
            )

            # 組合使用者的多模態訊息 (文字 + 圖片)
            human_content = [{"type": "text", "text": real_query}]

            if images:
                logger.info(f" 接收到 {len(images)} 張圖片，啟動視覺分析引擎...")
                for img_b64 in images:
                    human_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })

            # 將兩種訊息封裝為 LangChain 標準格式
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=human_content)
            ]

            # 繞過只能處理純文字的 prompt chain，直接丟給模型做 astream 串流輸出
            async for chunk in self.llm.astream(messages):
                text_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
                clean_chunk = text_chunk.replace("<br>", "\n").replace("<b>", "**").replace("</b>", "**")
                yield clean_chunk

        except Exception as e:
            # 關鍵防護：捕捉到任何錯誤，印出完整追蹤碼，並傳送友善的錯誤訊息給前端
            traceback.print_exc()
            logger.error(f"嚴重系統崩潰: {str(e)}")
            yield f"\n\n **系統遭遇錯誤**：無法完成處理。\n錯誤細節：`{str(e)}`"
