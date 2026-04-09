import re
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from app.utils.schedule_utils import WEEKDAY_ALIASES, WEEKDAY_FULL, extract_days

EN_WEEKDAY_ALIASES = {
    "星期一": ("Mon", "Monday"),
    "星期二": ("Tue", "Tuesday"),
    "星期三": ("Wed", "Wednesday"),
    "星期四": ("Thu", "Thursday"),
    "星期五": ("Fri", "Friday"),
    "星期六": ("Sat", "Saturday"),
    "星期日": ("Sun", "Sunday"),
}

PERIOD_SYNONYMS = {
    "上午": ("上午", "早上", "早診", "AM", "am", "Morning", "morning"),
    "下午": ("下午", "午診", "PM", "pm", "Afternoon", "afternoon"),
    "夜間": ("夜間", "夜診", "晚上", "晚間", "Evening", "evening", "Night", "night"),
}

PAST_TIME_PATTERNS = [
    r"上個月", r"上月", r"上週", r"上周", r"上星期", r"上禮拜", r"上礼拜",
    r"昨天", r"昨日", r"前天", r"去年", r"前年", r"過去", r"过去", r"先前", r"之前"
]


def has_past_time_reference(query: str) -> bool:
    text = str(query or "")
    return any(re.search(pattern, text) for pattern in PAST_TIME_PATTERNS)


def _weekday_by_offset(reference_date: date, offset: int) -> str:
    target = reference_date + timedelta(days=offset)
    # date.weekday(): Monday=0 ... Sunday=6
    return WEEKDAY_FULL[target.weekday()]


def _extract_week_offset(text: str) -> int:
    # 下週=1, 下下週=2, 下下下週=3 ...
    matches = re.findall(r"((?:下)+)(?:個|个)?(?:週|周|星期|禮拜|礼拜)", text)
    if not matches:
        if re.search(r"下(?:個|个)?(?:週|周|星期|禮拜|礼拜)", text):
            return 1
        return 0
    return max(len(m) for m in matches)


def _extract_relative_days(text: str, reference_date: date) -> List[str]:
    days: List[str] = []
    has_future_specific = any(k in text for k in ("明天", "後天", "大後天"))

    if (not has_future_specific) and any(k in text for k in ("今天", "今日")):
        days.append(_weekday_by_offset(reference_date, 0))
    if "明天" in text:
        days.append(_weekday_by_offset(reference_date, 1))
    if "大後天" in text:
        days.append(_weekday_by_offset(reference_date, 3))
    elif "後天" in text:
        days.append(_weekday_by_offset(reference_date, 2))

    # 週末：固定映射星期六、星期日
    if any(k in text for k in ("週末", "周末", "假日")):
        days.extend(["星期六", "星期日"])

    # 平日：固定映射星期一到星期五
    if any(k in text for k in ("平日", "週間", "周間", "上班日")):
        days.extend(["星期一", "星期二", "星期三", "星期四", "星期五"])

    return list(dict.fromkeys(days))


def _extract_periods(text: str) -> List[str]:
    matched: List[str] = []
    for canonical, aliases in PERIOD_SYNONYMS.items():
        if any(alias in text for alias in aliases):
            matched.append(canonical)
    return list(dict.fromkeys(matched))


def augment_query_with_time_hints(query: str, reference_date: Optional[date] = None) -> Tuple[str, Dict]:
    """
    將相對時間詞補成可檢索的關鍵詞，回傳 (augmented_query, time_meta)
    """
    text = str(query or "").strip()
    today = reference_date or date.today()

    explicit_days = extract_days(text)
    relative_days = _extract_relative_days(text, today)
    days = list(dict.fromkeys(explicit_days + relative_days))
    periods = _extract_periods(text)
    week_offset = _extract_week_offset(text)
    past_reference = has_past_time_reference(text)

    extra_terms: List[str] = []
    for day in days:
        extra_terms.append(day)
        extra_terms.extend(WEEKDAY_ALIASES.get(day, ()))
        extra_terms.extend(EN_WEEKDAY_ALIASES.get(day, ()))
    for period in periods:
        extra_terms.append(period)
        extra_terms.extend(PERIOD_SYNONYMS.get(period, ()))

    # 去重並去空值
    compact_terms = list(dict.fromkeys([t for t in extra_terms if t]))
    augmented_query = f"{text} {' '.join(compact_terms)}".strip() if compact_terms else text

    meta = {
        "days": days,
        "periods": periods,
        "week_offset": week_offset,
        "is_past_reference": past_reference,
    }
    return augmented_query, meta
