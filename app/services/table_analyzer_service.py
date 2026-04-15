import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.utils.schedule_utils import DEFAULT_PERIODS, WEEKDAY_ALIASES, extract_days, sort_days

logger = logging.getLogger(__name__)


DEPT_ALIAS_TO_KEYWORD = {
    "腸胃科": "胃腸",
    "胃腸科": "胃腸",
    "胃腸肝膽科": "胃腸",
    "消化內科": "胃腸",
    "身心科": "精神",
    "精神科": "精神",
    "精神部": "精神",
    "骨科": "骨科",
    "心臟科": "心臟",
    "心臟內科": "心臟",
    "神經內科": "神經",
    "兒童心智/青少年保健門診": "心智",
    "兒童心智青少年保健門診": "心智",
    # 糖尿病足採兩段式：先精準命中糖尿病足，再降級新陳代謝
    "糖尿病足照護特診": "糖尿病足",
    "糖尿病足照護特別門診": "糖尿病足",
    "糖尿病足特診": "糖尿病足",
    "糖尿病特診": "糖尿病足",
    "高齡整合門診": "高齡整合",
    "高齡醫學整合門診": "高齡整合",
    "高齡門診": "高齡整合",
}

PERIOD_ALIASES = {
    "上午": ("上午", "早上", "早診", "am", "AM", "morning", "Morning"),
    "下午": ("下午", "午診", "pm", "PM", "afternoon", "Afternoon"),
    "夜間": ("夜間", "夜診", "晚上", "晚間", "night", "Night", "evening", "Evening"),
}

NEGATIVE_TOKENS = ("不要", "不看", "排除", "避開", "不要看", "不想要", "除了", "除外", "絕對不要")


class TableAnalyzerService:
    @staticmethod
    def get_special_department_strategy(query: str) -> Optional[Dict[str, Any]]:
        text = str(query or "")
        if any(k in text for k in ("糖尿病足", "糖足")):
            return {
                "label": "糖尿病足特診",
                "primary_terms": [
                    "糖尿病足照護特診",
                    "糖尿病足照護",
                    "糖尿病足特診",
                    "糖尿病足",
                    "糖足",
                ],
                "fallback_department": "新陳代謝",
            }
        return None

    @staticmethod
    def looks_like_schedule_query(query: str) -> bool:
        text = str(query or "")
        schedule_hints = (
            "門診", "看診", "醫師", "醫生", "時段", "星期", "週", "周",
            "上午", "下午", "夜間", "掛號", "姓"
        )
        non_schedule_hints = (
            "防疫小叮嚀", "內容有哪些", "摘要", "注意事項", "規定", "政策", "停車", "接駁"
        )

        dept = TableAnalyzerService._infer_department_keyword(text)
        c = TableAnalyzerService._extract_constraints(text)
        has_constraints = bool(
            c["include_days"] or c["exclude_days"] or c["include_periods"] or
            c["exclude_periods"] or c["surname"]
        )
        has_schedule_hint = any(k in text for k in schedule_hints)
        has_non_schedule_hint = any(k in text for k in non_schedule_hints)

        if dept or has_constraints or has_schedule_hint:
            return True
        if has_non_schedule_hint:
            return False
        return False

    @staticmethod
    async def query_and_format_schedule(df: pd.DataFrame, query: str, llm: Any) -> str:
        """
        將自然語言查詢轉為 Pandas 一行程式碼、執行後再排版回傳。
        若查無資料，回傳固定格式訊息供上層判斷。
        """
        normalized_query = TableAnalyzerService._normalize_query_for_codegen(query)
        python_code = await TableAnalyzerService._generate_query_code(df=df, query=normalized_query, llm=llm)
        result = TableAnalyzerService._safe_eval_dataframe_code(df=df, python_code=python_code)

        # 第一層 fallback：若 LLM 產碼過度過濾，至少用科別關鍵字把整週資料撈出來。
        if TableAnalyzerService._is_empty_result(result):
            result = TableAnalyzerService._fallback_by_department(df=df, query=normalized_query)

        if TableAnalyzerService._is_empty_result(result):
            return (
                f"很抱歉，在目前的門診表快取中查無「{query}」的相關資料。\n"
                "建議您直接參考實體門診表或撥打諮詢專線確認。"
            )

        result_str = TableAnalyzerService._format_result(result=result, query=query)
        if len(result_str) > 30000:
            logger.warning("資料量過大，啟動防護截斷機制")
            result_str = result_str[:30000] + "\n... (資料過多，僅顯示部分) ..."
        return result_str

    @staticmethod
    def _normalize_query_for_codegen(query: str) -> str:
        text = str(query or "")
        for alias, keyword in DEPT_ALIAS_TO_KEYWORD.items():
            if alias in text:
                text = text.replace(alias, keyword)
        return text

    @staticmethod
    async def _generate_query_code(df: pd.DataFrame, query: str, llm: Any) -> str:
        code_prompt = (
            f"你是一個頂級的 Python 資料分析師。我有一個 pandas DataFrame 叫做 `df`。\n"
            f"這個表格的真實欄位有：{list(df.columns)}\n"
            f"前 3 筆資料範例如下：\n{df.head(3).to_dict('records')}\n\n"
            f"請寫出『一行』Python 程式碼來取得以下問題的答案：\n"
            f"問題：「{query}」\n\n"
            f"【嚴格規定】：\n"
            f"1. 請『只』輸出那行 Python 程式碼，絕對不要包含任何解釋。\n"
            f"2. 絕對不要使用 `print()`。\n"
            f"3. 請回傳過濾後的完整 DataFrame，句尾必須加上 `.to_dict('records')`。\n"
            f"4. 因為 PDF 欄位名稱不規則，請不要指定固定欄位名稱。\n"
            f"5. 使用全表模糊搜尋："
            f"`df[df.astype(str).apply(lambda x: x.str.contains('科別關鍵字', na=False, regex=False)).any(axis=1)].to_dict('records')`\n"
            f"6. 【時間與人名豁免】若問題有星期、時段、姓氏（如姓林），請不要把這些條件寫進程式碼，只過濾科別。\n"
            f"7. 【俗稱轉換】若問題提到腸胃科，請在 contains 只用「胃腸」關鍵字。\n"
            f"現在請輸出程式碼："
        )

        logger.info("AI 正在撰寫 Pandas 分析程式碼...")
        ai_code_response = await llm.ainvoke(code_prompt)
        raw_code_text = ai_code_response.content if hasattr(ai_code_response, "content") else str(ai_code_response)
        python_code = raw_code_text.replace("```python", "").replace("```", "").strip()
        return python_code

    @staticmethod
    def _safe_eval_dataframe_code(df: pd.DataFrame, python_code: str):
        safe_builtins = {
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
        }
        exec_env = {"df": df, "pd": pd, "__builtins__": safe_builtins}
        try:
            result = eval(python_code, exec_env)
            logger.info("Pandas 程式碼執行成功")
            return result
        except Exception as e:
            logger.error("Pandas 程式碼執行失敗: %s | code=%s", e, python_code)
            raise

    @staticmethod
    def _fallback_by_department(df: pd.DataFrame, query: str):
        dept_keyword = TableAnalyzerService._infer_department_keyword(query)
        if not dept_keyword:
            return []
        logger.info("啟用科別 fallback 過濾: %s", dept_keyword)
        mask = df.astype(str).apply(
            lambda row: row.str.contains(dept_keyword, na=False, regex=False)
        ).any(axis=1)
        return df.loc[mask].to_dict("records")

    @staticmethod
    def _infer_department_keyword(query: str) -> Optional[str]:
        text = str(query or "")
        for alias, keyword in DEPT_ALIAS_TO_KEYWORD.items():
            if alias in text:
                return keyword

        # 先移除常見時間/語氣詞，避免「星期六骨科」被誤抓成「星期六骨」
        normalized = (
            text.replace("禮拜", "星期")
            .replace("週", "星期")
            .replace("周", "星期")
        )
        normalized = re.sub(
            r"(今天|明天|後天|大後天|下星期|下下星期|下下下星期|這星期|本星期|星期[一二三四五六日天]|上午|下午|夜間|早上|晚上|哪天|有看診|看診|有哪些|醫師|醫生)",
            " ",
            normalized,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # 泛用後綴：xxx科 / xxx門診 / xxx特診 / xxx專診
        # 使用 finditer 取最有可能的候選，避免前綴殘留干擾。
        candidates = []
        for m in re.finditer(r"([\u4e00-\u9fa5A-Za-z/]{1,20})(科|門診|特診|專診)", normalized):
            head = (m.group(1) or "").strip()
            suffix = (m.group(2) or "").strip()
            if not head:
                continue
            full = f"{head}{suffix}"
            candidates.append((head, full))

        if candidates:
            # 優先用完整詞去 alias map（二次正規化）
            for head, full in reversed(candidates):
                for alias, keyword in DEPT_ALIAS_TO_KEYWORD.items():
                    if alias == full or alias == head:
                        return keyword
            # 退而求其次：回傳 head（例如「骨」不合理時會在下層過濾掉）
            # 優先最長、且靠後出現的候選。
            candidates.sort(key=lambda x: len(x[0]), reverse=True)
            return candidates[0][0]

        return None

    @staticmethod
    def _is_empty_result(result: Any) -> bool:
        if not result:
            return True
        if isinstance(result, list) and len(result) == 0:
            return True
        if len(str(result)) < 15:
            return True
        return False

    @staticmethod
    def _format_result(result: Any, query: str) -> str:
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return TableAnalyzerService._format_dict_rows(rows=result, query=query)
        return str(result)

    @staticmethod
    def _extract_constraints(query: str) -> Dict[str, Any]:
        text = str(query or "")
        text_norm = (
            text.replace("禮拜", "星期")
            .replace("週", "星期")
            .replace("周", "星期")
            .replace("星期天", "星期日")
            .replace("禮拜天", "星期日")
        )
        include_days: Set[str] = set()
        exclude_days: Set[str] = set()
        include_periods: Set[str] = set()
        exclude_periods: Set[str] = set()

        # 抓「排除片段」：除了/不要/排除...直到標點前
        negative_spans = []
        for m in re.finditer(r"(?:除了|除外|不要看|不要|不看|排除|避開|絕對不要)([^。！？；;，,\n]*)", text_norm):
            seg = (m.group(1) or "").strip()
            negative_spans.append(m.span())

            if "週末" in seg or "周末" in seg:
                exclude_days.update({"星期六", "星期日"})

            for canonical, aliases in WEEKDAY_ALIASES.items():
                if any(alias in seg for alias in aliases):
                    exclude_days.add(canonical)
            for canonical, aliases in PERIOD_ALIASES.items():
                if any(alias.lower() in seg.lower() for alias in aliases):
                    exclude_periods.add(canonical)

        # 移除排除片段後再抓 include，避免把「除了星期一」誤當 include
        text_for_include = text_norm
        for start, end in sorted(negative_spans, reverse=True):
            text_for_include = text_for_include[:start] + " " + text_for_include[end:]

        # 週末快捷語（include）
        if "週末" in text_for_include or "周末" in text_for_include:
            include_days.update({"星期六", "星期日"})

        # 顯式星期（先全抓，再扣掉排除）
        include_days.update(extract_days(text_for_include))

        for canonical, aliases in WEEKDAY_ALIASES.items():
            for alias in aliases:
                if any(re.search(rf"{neg}\s*{re.escape(alias)}", text_norm) for neg in NEGATIVE_TOKENS):
                    exclude_days.add(canonical)

        for canonical, aliases in PERIOD_ALIASES.items():
            for alias in aliases:
                if any(re.search(rf"{neg}\s*{re.escape(alias)}", text_norm, flags=re.IGNORECASE) for neg in NEGATIVE_TOKENS):
                    exclude_periods.add(canonical)
                elif re.search(re.escape(alias), text_for_include, flags=re.IGNORECASE):
                    include_periods.add(canonical)

        include_days -= exclude_days
        include_periods -= exclude_periods

        surname = None
        m = re.search(r"姓\s*[「『'\"`]?\s*([\u4e00-\u9fa5])", text_norm)
        if m:
            surname = m.group(1)

        return {
            "include_days": include_days,
            "exclude_days": exclude_days,
            "include_periods": include_periods,
            "exclude_periods": exclude_periods,
            "surname": surname,
        }

    @staticmethod
    def _split_doctors(raw_items: List[str]) -> List[str]:
        parts: List[str] = []
        stop_words = {"不指定", "休診", "停診", "未安排", "無門診", "未註明", "上午", "下午", "夜間"}
        for item in raw_items:
            tokens = re.split(r"[、,，;；\s]+", str(item))
            for token in tokens:
                name = token.strip()
                if not name or name in {"nan", "None", "-", "null"}:
                    continue
                # 去除括號備註、尾隨診間號、星期文字等雜訊
                name = re.sub(r"\([^)]*\)", "", name)
                name = re.sub(r"星期[一二三四五六日天]", "", name)
                name = re.sub(r"\d{2,4}$", "", name)
                # 只保留中文姓名常見字元
                name = re.sub(r"[^\u4e00-\u9fa5．·]", "", name).strip()
                if not name:
                    continue
                if name in stop_words:
                    continue
                if len(name) < 2:
                    continue
                parts.append(name)
        # 保留順序去重
        return list(dict.fromkeys(parts))

    @staticmethod
    def _detect_periods(text: str) -> List[str]:
        raw = str(text or "")
        if not raw:
            return []

        periods: List[str] = []
        for canonical, aliases in PERIOD_ALIASES.items():
            if any(alias.lower() in raw.lower() for alias in aliases):
                periods.append(canonical)
        if "全天" in raw or "全日" in raw:
            periods.extend(["上午", "下午"])

        # 去重保序
        deduped: List[str] = []
        for p in periods:
            if p not in deduped:
                deduped.append(p)
        return deduped

    @staticmethod
    def _format_dict_rows(rows: List[Dict[str, Any]], query: str) -> str:
        clean_df = pd.DataFrame(rows).astype(str).drop_duplicates()
        dept_keyword = TableAnalyzerService._infer_department_keyword(query)
        if dept_keyword:
            dept_mask = clean_df.astype(str).apply(
                lambda row: row.str.contains(dept_keyword, na=False, regex=False)
            ).any(axis=1)
            clean_df = clean_df.loc[dept_mask]
            if clean_df.empty:
                return "目前查無符合您條件的門診資料。"

        rename_map = {
            "未命名欄位_6": "星期一",
            "未命名欄位_7": "星期二",
            "未命名欄位_8": "星期三",
            "未命名欄位_9": "星期四",
            "未命名欄位_10": "星期五",
            "未命名欄位_11": "星期六",
            "未命名欄位_12": "星期日",
        }
        clean_df = clean_df.rename(columns=rename_map)

        for col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(
                lambda x: re.sub(r"(\d{4}|\))([\u4e00-\u9fa5])", r"\1、\2", str(x))
            )

        day_column_map: Dict[str, str] = {}
        for col in clean_df.columns:
            days = extract_days(str(col))
            if days:
                col_vals = clean_df[col].astype(str).str.strip()
                if col_vals.isin(["", "nan", "None", "-"]).all():
                    continue
                day_column_map.setdefault(days[0], col)

        days_of_week = sort_days(list(day_column_map.keys()))

        time_col = None
        for col in clean_df.columns:
            if clean_df[col].astype(str).str.contains("上午|下午|夜間", na=False).any():
                time_col = col
                break

        c = TableAnalyzerService._extract_constraints(query)
        include_days = c["include_days"]
        exclude_days = c["exclude_days"]
        include_periods = c["include_periods"]
        exclude_periods = c["exclude_periods"]
        surname = c["surname"]

        # 逐列解析，避免 time_col 判斷失敗導致整天都無門診
        day_period_names: Dict[str, Dict[str, List[str]]] = {}
        for _, row in clean_df.iterrows():
            row_text = " ".join([str(v) for v in row.values if str(v).strip()])
            periods = []
            if time_col:
                periods = TableAnalyzerService._detect_periods(str(row.get(time_col, "")))
            if not periods:
                periods = TableAnalyzerService._detect_periods(row_text)
            if not periods:
                periods = ["未註明"]

            # 套用時段 include/exclude 條件
            filtered_periods = []
            for p in periods:
                if include_periods and p not in include_periods:
                    continue
                if p in exclude_periods:
                    continue
                filtered_periods.append(p)
            if not filtered_periods:
                continue

            for day in days_of_week:
                if include_days and day not in include_days:
                    continue
                if day in exclude_days:
                    continue

                day_col = day_column_map.get(day)
                if not day_col:
                    continue

                names = TableAnalyzerService._split_doctors([str(row.get(day_col, ""))])
                if surname:
                    names = [n for n in names if n.startswith(surname) or surname in n]
                if not names:
                    continue

                day_period_names.setdefault(day, {})
                for p in filtered_periods:
                    existing = day_period_names[day].setdefault(p, [])
                    for n in names:
                        if n not in existing:
                            existing.append(n)

        structured_text = ""
        for day in days_of_week:
            if include_days and day not in include_days:
                continue
            if day in exclude_days:
                continue
            if day not in day_period_names or not day_period_names[day]:
                structured_text += f"【{day}】\n - 無門診\n\n"
                continue

            structured_text += f"【{day}】\n"
            # 優先顯示常見時段，其他時段放最後
            ordered_periods = [p for p in DEFAULT_PERIODS if p in day_period_names[day]]
            ordered_periods += [p for p in day_period_names[day].keys() if p not in ordered_periods]
            for p in ordered_periods:
                names = day_period_names[day].get(p, [])
                if names:
                    structured_text += f" - {p}：{'、'.join(names)}\n"
            structured_text += "\n"

        if structured_text.strip():
            return structured_text

        fallback_text = "【系統原始資料（表頭遺失，請依順序推斷）】\n"
        for _, row in clean_df.iterrows():
            time_val = row[time_col] if time_col else "未知時段"
            fallback_text += f"▶ 時段：{time_val}\n"
            for col in clean_df.columns:
                val = str(row[col]).strip()
                if val and val not in {"nan", "None", ""} and col != time_col:
                    fallback_text += f"  - {col}: {val}\n"
            fallback_text += "\n"
        return fallback_text
