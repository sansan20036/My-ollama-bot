import re
from typing import Dict, List

WEEKDAY_BASE = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六"]
WEEKDAY_FULL = WEEKDAY_BASE + ["星期日"]
WEEKDAY_ORDER = {day: idx for idx, day in enumerate(WEEKDAY_FULL, start=1)}
WEEKDAY_ALIASES = {
    "星期一": ("星期一", "週一", "周一", "禮拜一", "Mon", "Monday"),
    "星期二": ("星期二", "週二", "周二", "禮拜二", "Tue", "Tuesday"),
    "星期三": ("星期三", "週三", "周三", "禮拜三", "Wed", "Wednesday"),
    "星期四": ("星期四", "週四", "周四", "禮拜四", "Thu", "Thursday"),
    "星期五": ("星期五", "週五", "周五", "禮拜五", "Fri", "Friday"),
    "星期六": ("星期六", "週六", "周六", "禮拜六", "Sat", "Saturday"),
    "星期日": ("星期日", "星期天", "週日", "周日", "禮拜日", "禮拜天", "Sun", "Sunday"),
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
        found = False
        for alias in aliases:
            alias_str = str(alias)
            if not alias_str:
                continue
            if re.fullmatch(r"[A-Za-z]+", alias_str):
                if re.search(rf"\b{re.escape(alias_str)}\b", day_text, flags=re.IGNORECASE):
                    found = True
                    break
            else:
                if alias_str in day_text:
                    found = True
                    break
        if found:
            matched.append(canonical)
    return matched


def sort_days(days: List[str]) -> List[str]:
    order = day_order()
    unique_days = list(dict.fromkeys([d for d in days if d]))
    return sorted(unique_days, key=lambda d: order.get(d, 99))


def clean_doctor_name(raw: str) -> str:
    """
    清洗醫師姓名：
    - 去括號備註
    - 去尾端診間號（3~5 位數）
    - 去尾端日期註記（例：4、8、15、22、29）
    """
    text = str(raw or "").strip()
    if not text:
        return ""

    # 括號註記
    text = re.sub(r"[（(][^）)]*[）)]", "", text)

    # 先去診間號（通常 3~5 碼）
    text = re.sub(r"\d{3,5}$", "", text).strip()

    # 去尾端日期清單（單雙位數，允許、/,，,空白分隔）
    text = re.sub(r"(?:[、,，/\\\s]*\d{1,2}){1,10}$", "", text).strip()

    # 去尾端殘留分隔符
    text = re.sub(r"[、,，;；/\\\s]+$", "", text).strip()

    # 去中間空白
    text = text.replace(" ", "")

    # 避免純數字殘留
    if re.fullmatch(r"\d+", text):
        return ""

    return text

