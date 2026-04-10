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
    "骨科": "骨科",
    "心臟科": "心臟",
    "心臟內科": "心臟",
    "神經內科": "神經",
    "兒童心智/青少年保健門診": "心智",
    "兒童心智青少年保健門診": "心智",
}

PERIOD_ALIASES = {
    "上午": ("上午", "早上", "早診", "am", "AM", "morning", "Morning"),
    "下午": ("下午", "午診", "pm", "PM", "afternoon", "Afternoon"),
    "夜間": ("夜間", "夜診", "晚上", "晚間", "night", "Night", "evening", "Evening"),
}

NEGATIVE_TOKENS = ("不要", "不看", "排除", "避開", "不要看", "不想要")


class TableAnalyzerService:
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

        # 泛用後綴：xxx科 / xxx門診
        m = re.search(r"([\u4e00-\u9fa5]{1,12})(?:科|門診)", text)
        if m:
            return m.group(1)
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
        include_days: Set[str] = set()
        exclude_days: Set[str] = set()
        include_periods: Set[str] = set()
        exclude_periods: Set[str] = set()

        # 週末快捷語
        if "週末" in text or "周末" in text:
            include_days.update({"星期六", "星期日"})

        # 顯式星期（先全抓，再扣掉排除）
        include_days.update(extract_days(text))

        for canonical, aliases in WEEKDAY_ALIASES.items():
            for alias in aliases:
                if any(re.search(rf"{neg}\s*{re.escape(alias)}", text) for neg in NEGATIVE_TOKENS):
                    exclude_days.add(canonical)

        for canonical, aliases in PERIOD_ALIASES.items():
            for alias in aliases:
                if any(re.search(rf"{neg}\s*{re.escape(alias)}", text, flags=re.IGNORECASE) for neg in NEGATIVE_TOKENS):
                    exclude_periods.add(canonical)
                elif re.search(re.escape(alias), text, flags=re.IGNORECASE):
                    include_periods.add(canonical)

        include_days -= exclude_days
        include_periods -= exclude_periods

        surname = None
        m = re.search(r"姓\s*[「『'\"`]?\s*([\u4e00-\u9fa5])", text)
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
        for item in raw_items:
            tokens = re.split(r"[、,，;；\s]+", str(item))
            for token in tokens:
                name = token.strip()
                if not name or name in {"nan", "None", "-"}:
                    continue
                # 過濾誤入的星期字樣
                if re.search(r"星期[一二三四五六日天]", name):
                    continue
                parts.append(name)
        # 保留順序去重
        return list(dict.fromkeys(parts))

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

        structured_text = ""
        for day in days_of_week:
            if include_days and day not in include_days:
                continue
            if day in exclude_days:
                continue

            day_col = day_column_map.get(day)
            if not day_col:
                continue

            structured_text += f"【{day}】\n"
            day_has_data = False

            if time_col:
                for period in DEFAULT_PERIODS:
                    if include_periods and period not in include_periods:
                        continue
                    if period in exclude_periods:
                        continue

                    mask = clean_df[time_col].astype(str).str.contains(period, na=False)
                    doctors = clean_df.loc[mask, day_col].tolist()
                    names = TableAnalyzerService._split_doctors(doctors)

                    if surname:
                        names = [n for n in names if n.startswith(surname) or surname in n]

                    if names:
                        structured_text += f" - {period}：{'、'.join(names)}\n"
                        day_has_data = True

            if not day_has_data:
                structured_text += " - 無門診\n"
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
