import logging
import re
from typing import Any, Dict, List

import pandas as pd

from app.utils.schedule_utils import DEFAULT_PERIODS, extract_days, sort_days

logger = logging.getLogger(__name__)


class TableAnalyzerService:
    @staticmethod
    async def query_and_format_schedule(df: pd.DataFrame, query: str, llm: Any) -> str:
        """
        將自然語言查詢轉為 Pandas 一行程式碼、執行後再排版回傳。
        若查無資料，回傳固定格式訊息供上層判斷。
        """
        python_code = await TableAnalyzerService._generate_query_code(df=df, query=query, llm=llm)
        result = TableAnalyzerService._safe_eval_dataframe_code(df=df, python_code=python_code)

        if TableAnalyzerService._is_empty_result(result):
            return (
                f"很抱歉，在目前的門診表快取中查無「{query}」的相關資料。\n"
                "建議您直接參考實體門診表或撥打諮詢專線確認。"
            )

        result_str = TableAnalyzerService._format_result(result)
        if len(result_str) > 30000:
            logger.warning("資料量過大，啟動防護截斷機制")
            result_str = result_str[:30000] + "\n... (資料過多，僅顯示部分) ..."
        return result_str

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
            f"3. 請務必回傳過濾後的「完整 DataFrame」，並且務必在句尾加上 `.to_dict('records')`。\n"
            f"4. 因為 PDF 萃取的欄位名稱充滿不規則，請『絕對不要』指定欄位名稱來過濾！\n"
            f"5. 請直接套用全表模糊搜尋："
            f"`df[df.astype(str).apply(lambda x: x.str.contains('科別關鍵字', na=False)).any(axis=1)].to_dict('records')`\n"
            f"6. 【時間過濾豁免】如果使用者問「星期幾」或「上下午」，"
            f"請『絕對不要』將時間加入 `str.contains` 的條件！你只需要過濾『科別』即可。\n"
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
    def _is_empty_result(result: Any) -> bool:
        if not result:
            return True
        if isinstance(result, list) and len(result) == 0:
            return True
        if len(str(result)) < 15:
            return True
        return False

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return TableAnalyzerService._format_dict_rows(result)
        return str(result)

    @staticmethod
    def _format_dict_rows(rows: List[Dict[str, Any]]) -> str:
        clean_df = pd.DataFrame(rows).astype(str).drop_duplicates()
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
                day_column_map.setdefault(days[0], col)

        days_of_week = sort_days(list(day_column_map.keys()))
        time_col = None
        for col in clean_df.columns:
            if clean_df[col].astype(str).str.contains("上午|下午|夜間", na=False).any():
                time_col = col
                break

        structured_text = ""
        for day in days_of_week:
            day_col = day_column_map.get(day)
            if not day_col:
                continue

            structured_text += f"【{day}】\n"
            day_has_data = False

            if time_col:
                for period in DEFAULT_PERIODS:
                    mask = clean_df[time_col].astype(str).str.contains(period, na=False)
                    doctors = clean_df.loc[mask, day_col].tolist()
                    valid_docs = [str(d).strip() for d in doctors if str(d).strip() not in ["", "nan", "None"]]
                    if valid_docs:
                        structured_text += f" - {period}：{'、'.join(valid_docs)}\n"
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
                if val and val not in ["nan", "None", ""] and col != time_col:
                    fallback_text += f"  - {col}: {val}\n"
            fallback_text += "\n"
        return fallback_text

