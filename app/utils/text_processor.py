# app/utils/text_processor.py
import re
from difflib import SequenceMatcher
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入專門的語言偵測庫 (做錯誤處理以免沒裝套件時崩潰)
try:
    from langdetect import detect, LangDetectException
except ImportError:
    # Fallback: 如果沒裝套件，就給個假函數防止程式掛掉
    detect = lambda x: "unknown"
    LangDetectException = Exception


class TextProcessor:

    @staticmethod
    def text_to_arabic(text: str) -> str:
        """中文數字轉阿拉伯數字"""
        cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                  '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
        for cn, arab in cn_map.items():
            text = text.replace(cn, arab)
        return text

    @staticmethod
    def detect_precise_lang(text: str) -> str:
        """精準語言偵測"""
        # 1. Regex 過濾亞洲語言
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text): return "ja"
        if re.search(r'[\uAC00-\uD7AF]', text): return "ko"
        if re.search(r'[\u0400-\u04FF]', text): return "ru"
        if re.search(r'[\u4e00-\u9fff]', text): return "zh"

        # 2. LangDetect
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

    @staticmethod
    def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
        """計算相似度 (模糊去重用)"""
        clean_a = re.sub(r'\s+', '', a)
        clean_b = re.sub(r'\s+', '', b)
        return SequenceMatcher(None, clean_a, clean_b).ratio() > threshold

    @staticmethod
    def smart_merge(docs: List[Document]) -> str:
        """智慧縫合文件片段"""
        if not docs: return ""
        unique_docs = {}
        for d in docs:
            # 假設 metadata 有 chunk_id，如果沒有就用前20字當 id
            cid = d.metadata.get('chunk_id', d.page_content[:20])
            if cid not in unique_docs:
                unique_docs[cid] = d

        sorted_docs = list(unique_docs.values())
        # 嘗試排序 (防呆處理)
        sorted_docs.sort(
            key=lambda x: x.metadata.get('chunk_id', 0) if isinstance(x.metadata.get('chunk_id'), int) else 0)

        merged_text = sorted_docs[0].page_content
        for i in range(1, len(sorted_docs)):
            prev_text = merged_text
            curr_text = sorted_docs[i].page_content
            overlap_found = False
            check_len = min(len(prev_text), len(curr_text), 300)
            for k in range(check_len, 10, -1):
                suffix = prev_text[-k:]
                if curr_text.startswith(suffix):
                    merged_text += curr_text[k:]
                    overlap_found = True
                    break
            if not overlap_found:
                merged_text += "\n" + curr_text
        return merged_text

    @staticmethod
    async def analyze_user_intent(question: str, llm) -> Optional[str]:
        """
        利用 LLM 判斷使用者想查哪種語言 (回傳 ISO 代碼)
        """
        q_lower = question.lower()
        if 'english' in q_lower or '英文' in q_lower: return 'en'
        if 'chinese' in q_lower or '中文' in q_lower: return 'zh'

        system_prompt = (
            "你是一個意圖識別助手。請分析問題判斷使用者【想要查詢的目標語言】。\n"
            "規則：\n"
            "1. 回傳 ISO 639-1 代碼 (如 en, zh, ja, ko, fr...)\n"
            "2. 若未指定語言，回傳 'None'\n"
            "3. 嚴格只回傳代碼，不要有標點符號。"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        chain = prompt | llm | StrOutputParser()
        try:
            # 這裡我們需要非同步調用，因為整個體系都升級成 Async 了
            code = await chain.ainvoke({"input": question})
            code = code.strip().lower()
            if "none" in code: return None
            match = re.search(r'\b[a-z]{2}\b', code)
            return match.group(0) if match else None
        except:
            return None