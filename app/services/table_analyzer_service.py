import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.utils.schedule_utils import DEFAULT_PERIODS, WEEKDAY_ALIASES, extract_days, sort_days

logger = logging.getLogger(__name__)


DEPT_ALIAS_TO_KEYWORD = {
    "?貉?蝘?: "?",
    "?蝘?: "?",
    "??蝘?: "?",
    "瘨??抒?": "?",
    "頨怠?蝘?: "蝎曄?",
    "蝎曄?蝘?: "蝎曄?",
    "蝎曄???: "蝎曄?",
    "撉函?": "撉函?",
    "敹?蝘?: "敹?",
    "敹??抒?": "敹?",
    "蟡??抒?": "蟡?",
    "?咱敹/??撟港??仿?閮?: "敹",
    "?咱敹??撟港??仿?閮?: "敹",
    # 蝟倏?雲?∪畾萄?嚗?蝎暹??賭葉蝟倏?雲嚗????圈隞??
    "蝟倏?雲?扯風?寡那": "蝟倏?雲",
    "蝟倏?雲?扯風?孵?閮?: "蝟倏?雲",
    "蝟倏?雲?寡那": "蝟倏?雲",
    "蝟倏?閮?: "蝟倏?雲",
    "擃翩?游??閮?: "擃翩?游?",
    "擃翩?怠飛?游??閮?: "擃翩?游?",
    "擃翩?閮?: "擃翩?游?",
}

PERIOD_ALIASES = {
    "銝?": ("銝?", "?拐?", "?抵那", "am", "AM", "morning", "Morning"),
    "銝?": ("銝?", "?那", "pm", "PM", "afternoon", "Afternoon"),
    "憭?": ("憭?", "憭那", "??", "??", "night", "Night", "evening", "Evening"),
}

NEGATIVE_TOKENS = ("銝?", "銝?", "?", "?輸?", "銝???, "銝閬?, "?支?", "?文?", "蝯?銝?")


class TableAnalyzerService:
    @staticmethod
    def get_special_department_strategy(query: str) -> Optional[Dict[str, Any]]:
        text = str(query or "")
        if any(k in text for k in ("蝟倏?雲", "蝟雲")):
            return {
                "label": "蝟倏?雲?寡那",
                "primary_terms": [
                    "蝟倏?雲?扯風?寡那",
                    "蝟倏?雲?扯風",
                    "蝟倏?雲?寡那",
                    "蝟倏?雲",
                    "蝟雲",
                ],
                "fallback_department": "?圈隞??",
            }
        return None

    @staticmethod
    def looks_like_schedule_query(query: str) -> bool:
        text = str(query or "")
        schedule_hints = (
            "?閮?, "?那", "?怠葦", "?怎?", "?挾", "??", "??, "??,
            "銝?", "銝?", "憭?", "??", "憪?
        )
        non_schedule_hints = (
            "?脩撠?", "?批捆?鈭?, "??", "瘜冽?鈭?", "閬?", "?輻?", "??", "?仿?"
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
        撠?嗉?閮?亥岷頧 Pandas 銝銵?撘Ⅳ?銵??????喋?        ?交?∟?????箏??澆?閮靘?撅文?瑯?        """
        normalized_query = TableAnalyzerService._normalize_query_for_codegen(query)
        python_code = await TableAnalyzerService._generate_query_code(df=df, query=normalized_query, llm=llm)
        result = TableAnalyzerService._safe_eval_dataframe_code(df=df, python_code=python_code)

        # 蝚砌?撅?fallback嚗 LLM ?ＹⅣ?漲?蕪嚗撠蝘?摮??湧梯????箔???        if TableAnalyzerService._is_empty_result(result):
            result = TableAnalyzerService._fallback_by_department(df=df, query=normalized_query)

        if TableAnalyzerService._is_empty_result(result):
            return (
                f"敺甇??函???閮箄”敹怠?銝剜?～query}???賊?鞈??n"
                "撱箄降?函?亙??祕擃?閮箄”??垣閰Ｗ?蝺Ⅱ隤?
            )

        result_str = TableAnalyzerService._format_result(result=result, query=query)
        if len(result_str) > 30000:
            logger.warning("鞈???憭改????脰風?芣璈")
            result_str = result_str[:30000] + "\n... (鞈???嚗?憿舐內?典?) ..."
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
            f"雿銝??蝝? Python 鞈???撣怒?????pandas DataFrame ?怠? `df`?n"
            f"?”?潛??祕甈???{list(df.columns)}\n"
            f"??3 蝑???靘?銝?\n{df.head(3).to_dict('records')}\n\n"
            f"隢神?箝?銵ython 蝔?蝣潔???隞乩?????獢?\n"
            f"??嚗query}?n\n"
            f"??潸?摰?\n"
            f"1. 隢?撓?粹銵?Python 蝔?蝣潘?蝯?銝??隞颱?閫???n"
            f"2. 蝯?銝?雿輻 `print()`?n"
            f"3. 隢??喲?瞈曉?????DataFrame嚗撠曉???銝?`.to_dict('records')`?n"
            f"4. ? PDF 甈??迂銝???隢?閬?摰摰?雿?蝔晞n"
            f"5. 雿輻?刻”璅∠???嚗?
            f"`df[df.astype(str).apply(lambda x: x.str.contains('蝘?摮?, na=False, regex=False)).any(axis=1)].to_dict('records')`\n"
            f"6. ????鈭箏?鞊????????畾萸?瘞?憒???嚗?銝???璇辣撖恍脩?撘Ⅳ嚗?蕪蝘?n"
            f"7. ??蝔梯??????貉?蝘?隢 contains ?芰???詻??萄??n"
            f"?曉隢撓?箇?撘Ⅳ嚗?
        )

        logger.info("AI 甇??啣神 Pandas ??蝔?蝣?..")
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
            logger.info("Pandas 蝔?蝣澆銵???)
            return result
        except Exception as e:
            logger.error("Pandas 蝔?蝣澆銵仃?? %s | code=%s", e, python_code)
            raise

    @staticmethod
    def _fallback_by_department(df: pd.DataFrame, query: str):
        dept_keyword = TableAnalyzerService._infer_department_keyword(query)
        if not dept_keyword:
            return []
        logger.info("?蝘 fallback ?蕪: %s", dept_keyword)
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

        normalized = (
            text.replace("禮拜", "星期")
            .replace("週", "星期")
            .replace("周", "星期")
        )

        normalized = re.sub(
            r"(今天|明天|後天|大後天|下星期|下下星期|下下下星期|這星期|本星期|"
            r"星期[一二三四五六日天]|上午|下午|夜間|早上|晚上|哪天|有看診|看診|有哪些|"
            r"醫師|醫生|門診時間|門診時刻|門診時間表|門診表|支援|分院|醫院|榮民總醫院|榮總)",
            " ",
            normalized,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()

        for alias, keyword in DEPT_ALIAS_TO_KEYWORD.items():
            if alias in normalized:
                return keyword

        short_candidates = re.findall(r"([\u4e00-\u9fa5A-Za-z/]{1,10}(?:科|門診|特診|專診))", normalized)
        noise_terms = ("醫院", "分院", "時間", "支援", "掛號")
        for full in reversed(short_candidates):
            full = str(full or "").strip()
            if not full:
                continue
            if any(n in full for n in noise_terms):
                continue

            for alias, keyword in DEPT_ALIAS_TO_KEYWORD.items():
                if alias == full or alias in full or full in alias:
                    return keyword
            return full

        candidates = []
        for m in re.finditer(r"([\u4e00-\u9fa5A-Za-z/]{1,20})(科|門診|特診|專診)", normalized):
            head = (m.group(1) or "").strip()
            suffix = (m.group(2) or "").strip()
            if not head:
                continue
            full = f"{head}{suffix}"
            candidates.append((head, full))

        if candidates:
            for head, full in reversed(candidates):
                for alias, keyword in DEPT_ALIAS_TO_KEYWORD.items():
                    if alias == full or alias == head:
                        return keyword
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
            text.replace("蝳格?", "??")
            .replace("??, "??")
            .replace("??, "??")
            .replace("??憭?, "????)
            .replace("蝳格?憭?, "????)
        )
        include_days: Set[str] = set()
        exclude_days: Set[str] = set()
        include_periods: Set[str] = set()
        exclude_periods: Set[str] = set()

        # ???斤?畾萸??支?/銝?/?...?游璅???        negative_spans = []
        for m in re.finditer(r"(?:?支?|?文?|銝??銝?|銝?|?|?輸?|蝯?銝?)([^??嚗?;嚗?\n]*)", text_norm):
            seg = (m.group(1) or "").strip()
            negative_spans.append(m.span())

            if "?望" in seg or "?冽" in seg:
                exclude_days.update({"????, "????})

            for canonical, aliases in WEEKDAY_ALIASES.items():
                if any(alias in seg for alias in aliases):
                    exclude_days.add(canonical)
            for canonical, aliases in PERIOD_ALIASES.items():
                if any(alias.lower() in seg.lower() for alias in aliases):
                    exclude_periods.add(canonical)

        # 蝘駁??挾敺???include嚗???鈭????炊??include
        text_for_include = text_norm
        for start, end in sorted(negative_spans, reverse=True):
            text_for_include = text_for_include[:start] + " " + text_for_include[end:]

        # ?望敹急隤?include嚗?        if "?望" in text_for_include or "?冽" in text_for_include:
            include_days.update({"????, "????})

        # 憿臬???嚗??冽?嚗?????嚗?        include_days.update(extract_days(text_for_include))

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
        m = re.search(r"憪s*[??\"`]?\s*([\u4e00-\u9fa5])", text_norm)
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
        stop_words = {"銝?摰?, "隡那", "?那", "?芸???, "?⊿?閮?, "?芾酉??, "銝?", "銝?", "憭?"}
        for item in raw_items:
            tokens = re.split(r"[??嚗?嚗s]+", str(item))
            for token in tokens:
                name = token.strip()
                if not name or name in {"nan", "None", "-", "null"}:
                    continue
                # ?駁?祈??酉?偏?刻那??????摮???
                name = re.sub(r"\([^)]*\)", "", name)
                name = re.sub(r"??[銝鈭????剜憭夜", "", name)
                name = re.sub(r"\d{2,4}$", "", name)
                # ?芯??葉???虜閬???                name = re.sub(r"[^\u4e00-\u9fa5嚗愍", "", name).strip()
                if not name:
                    continue
                if name in stop_words:
                    continue
                if len(name) < 2:
                    continue
                parts.append(name)
        # 靽????駁?
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
        if "?典予" in raw or "?冽" in raw:
            periods.extend(["銝?", "銝?"])

        # ?駁?靽?
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
                return "?桀??亦蝚血??冽?隞嗥??閮箄???

        rename_map = {
            "?芸??雿6": "??銝",
            "?芸??雿7": "??鈭?,
            "?芸??雿8": "??銝?,
            "?芸??雿9": "????,
            "?芸??雿10": "??鈭?,
            "?芸??雿11": "????,
            "?芸??雿12": "????,
        }
        clean_df = clean_df.rename(columns=rename_map)

        for col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(
                lambda x: re.sub(r"(\d{4}|\))([\u4e00-\u9fa5])", r"\1?2", str(x))
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
            if clean_df[col].astype(str).str.contains("銝?|銝?|憭?", na=False).any():
                time_col = col
                break

        c = TableAnalyzerService._extract_constraints(query)
        include_days = c["include_days"]
        exclude_days = c["exclude_days"]
        include_periods = c["include_periods"]
        exclude_periods = c["exclude_periods"]
        surname = c["surname"]

        # ??閫??嚗??time_col ?斗憭望?撠?游予?賜?閮?        day_period_names: Dict[str, Dict[str, List[str]]] = {}
        for _, row in clean_df.iterrows():
            row_text = " ".join([str(v) for v in row.values if str(v).strip()])
            periods = []
            if time_col:
                periods = TableAnalyzerService._detect_periods(str(row.get(time_col, "")))
            if not periods:
                periods = TableAnalyzerService._detect_periods(row_text)
            if not periods:
                periods = ["?芾酉??]

            # 憟?挾 include/exclude 璇辣
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
                structured_text += f"?day}?n - ?⊿?閮暝n\n"
                continue

            structured_text += f"?day}?n"
            # ?芸?憿舐內撣貉??挾嚗隞?畾菜?敺?            ordered_periods = [p for p in DEFAULT_PERIODS if p in day_period_names[day]]
            ordered_periods += [p for p in day_period_names[day].keys() if p not in ordered_periods]
            for p in ordered_periods:
                names = day_period_names[day].get(p, [])
                if names:
                    structured_text += f" - {p}嚗'??.join(names)}\n"
            structured_text += "\n"

        if structured_text.strip():
            return structured_text

        fallback_text = "?頂蝯勗?憪???銵券?箏仃嚗?靘?摨?瘀??n"
        for _, row in clean_df.iterrows():
            time_val = row[time_col] if time_col else "?芰?挾"
            fallback_text += f"???挾嚗time_val}\n"
            for col in clean_df.columns:
                val = str(row[col]).strip()
                if val and val not in {"nan", "None", ""} and col != time_col:
                    fallback_text += f"  - {col}: {val}\n"
            fallback_text += "\n"
        return fallback_text
