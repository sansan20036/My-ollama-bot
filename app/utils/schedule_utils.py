from typing import Dict, List

WEEKDAY_BASE = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六"]
WEEKDAY_FULL = WEEKDAY_BASE + ["星期日"]
WEEKDAY_ORDER = {day: idx for idx, day in enumerate(WEEKDAY_FULL, start=1)}
WEEKDAY_ALIASES = {
    "星期一": ("星期一", "週一", "周一", "禮拜一"),
    "星期二": ("星期二", "週二", "周二", "禮拜二"),
    "星期三": ("星期三", "週三", "周三", "禮拜三"),
    "星期四": ("星期四", "週四", "周四", "禮拜四"),
    "星期五": ("星期五", "週五", "周五", "禮拜五"),
    "星期六": ("星期六", "週六", "周六", "禮拜六"),
    "星期日": ("星期日", "星期天", "週日", "周日", "禮拜日", "禮拜天"),
}
DEFAULT_PERIODS = ("上午", "下午", "夜間")


def all_days(include_sunday: bool = True) -> List[str]:
    return list(WEEKDAY_FULL if include_sunday else WEEKDAY_BASE)


def day_order() -> Dict[str, int]:
    return WEEKDAY_ORDER


def extract_days(text: str) -> List[str]:
    day_text = str(text or "")
    matched: List[str] = []
    for canonical, aliases in WEEKDAY_ALIASES.items():
        if any(alias in day_text for alias in aliases):
            matched.append(canonical)
    return matched


def sort_days(days: List[str]) -> List[str]:
    order = day_order()
    unique_days = list(dict.fromkeys([d for d in days if d]))
    return sorted(unique_days, key=lambda d: order.get(d, 99))

