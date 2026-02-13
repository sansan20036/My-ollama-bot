# app/services/chat_service.py
import logging
import os
import re
from typing import AsyncGenerator, List, Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.cache_service import SemanticCacheService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        self.vector_store = VectorStoreService.get_instance()
        self.cache = SemanticCacheService.get_instance()
        self.upload_dir = os.path.join(os.getcwd(), "uploads")

        # ✅ 建議使用 8b 模型以獲得最佳速度與通用性
        target_model = "gpt-oss:20b"

        logger.info(f"🔥 初始化全能文件聊天服務: {target_model}")

        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=target_model,
            temperature=0.1,
            keep_alive="1h",
            num_ctx=8192,
            num_predict=4096
        )

    def _get_valid_files(self) -> list:
        if not os.path.exists(self.upload_dir): return []
        return [f for f in os.listdir(self.upload_dir) if
                os.path.isfile(os.path.join(self.upload_dir, f)) and not f.startswith("~")]

    def _get_sorted_file_list(self, files: list) -> str:
        if not files: return "(無檔案)"
        return "\n".join([f"{i + 1}. {f}" for i, f in enumerate(files)])

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

    async def _smart_query_rewrite(self, user_query: str) -> str:
        """
        🔥 萬用型意圖預判 (Universal Intent Prediction)
        """
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
        print(f"🤔 AI 正在進行萬用關鍵字聯想...")
        refined_query = await chain.ainvoke({"query": user_query})
        clean_query = refined_query.replace("\n", " ").strip()
        print(f"✨ AI 聯想關鍵字: {clean_query}")
        return clean_query

    async def process_query(self, query: str, history: list) -> AsyncGenerator[str, None]:
        yield "🧠 **AI 正在分析文件內容...**\n\n"

        real_query = query
        valid_files = self._get_valid_files()
        file_count = len(valid_files)
        file_list_str = self._get_sorted_file_list(valid_files)

        if file_count == 0:
            yield "⚠️ 資料庫為空。請先上傳檔案。"
            return

        # =========================================================
        # 1. 第一輪：通用檢索
        # =========================================================
        ai_keywords = await self._smart_query_rewrite(real_query)
        search_query = f"{real_query} {ai_keywords}"

        matches = re.findall(r'第\s*(\d{1,3})\s*[條章節]', real_query)
        if matches:
            for m in matches:
                cn_num = self._num_to_chinese(m)
                search_query += f" 第{cn_num}條 第{cn_num}章"

        # 呼叫搜尋的地方
        print(f"🚀 執行檢索: {search_query}") # 這裡呼叫了上面的那個 search 方法
        docs = self.vector_store.search(search_query, k=15) # 回傳最相關的 15 筆文件 (docs)

        # =========================================================
        # 🔥 新增：狙擊模式 (Sniper Mode)
        # 如果使用者明確說了「第X條」，我們就強制去資料庫挖出來，不看運氣
        # =========================================================
        if matches:
            print(f"🎯 偵測到明確條號 {matches}，啟動狙擊模式...")
            # 建立目前已抓到的 ID 集合，避免重複
            existing_ids = set()
            for d in docs:
                aid = d.metadata.get("article_id")
                if aid: existing_ids.add(str(aid))

            for m in matches:
                target_id = str(int(m))  # 轉成字串 ID (如 "80")

                # 如果廣泛搜尋已經抓到了，就不用忙了
                if target_id in existing_ids:
                    print(f"✅ 第 {m} 條已在搜尋結果中，跳過強制調閱。")
                    continue

                # 如果沒抓到，發起精準搜尋 (使用 filter 如果支援，或廣搜後過濾)
                print(f"🔫 執行強制調閱：第 {m} 條...")
                sniper_query = f"第{m}條"

                # 利用我們剛在 VectorStore 實作的 filter 功能 (最穩)
                # 注意：這需要您的 vector_store.search 支援 filter 參數
                # 如果不支援，我們用 k=50 暴力搜
                sniper_docs = self.vector_store.search(sniper_query, k=50)

                found_target = False
                for d in sniper_docs:
                    fetched_id = str(d.metadata.get("article_id", ""))
                    if fetched_id == target_id:
                        d.page_content = f"【使用者指定調閱：第{m}條】\n{d.page_content}"
                        # 🔥 強制插入到最前面 (置頂)
                        docs.insert(0, d)
                        existing_ids.add(target_id)
                        print(f"✅ 狙擊成功：已強制載入 第 {m} 條")
                        found_target = True
                        break

                if not found_target:
                    print(f"⚠️ 狙擊失敗：資料庫中找不到 ID={target_id} 的條文")

        # =========================================================
        # 2. 第二輪：彈性補完 (Adaptive Auto-Completion)
        # =========================================================
        existing_ids = set()
        has_structured_data = False

        for doc in docs:
            aid = doc.metadata.get("article_id")
            if aid:
                existing_ids.add(str(aid))
                has_structured_data = True

        if has_structured_data:
            print("🕵️‍♂️ 偵測到結構化資料，嘗試分析引用關係...")
            referenced_ids = set()
            for doc in docs:
                content = doc.page_content
                refs = re.findall(r'第\s*([0-9]+|[零一二三四五六七八九十百]+)\s*條', content)
                for ref in refs:
                    if ref not in existing_ids and ref not in ["一", "二"]:
                        referenced_ids.add(ref)

            if referenced_ids:
                target_refs = list(referenced_ids)[:5]
                print(f"🔗 發現引用，嘗試補完: {target_refs}")
                yield f"🔗 **正在調閱相關章節 ({len(target_refs)} 筆)...**\n\n"

                for ref_art in target_refs:
                    target_id = self._chinese_to_num(ref_art)
                    if target_id == 0: continue

                    fetch_query = f"第{ref_art}條"
                    supplementary_docs = self.vector_store.search(fetch_query, k=50)

                    for d in supplementary_docs:
                        fetched_id = str(d.metadata.get("article_id", ""))
                        if fetched_id == str(target_id) and fetched_id not in existing_ids:
                            d.page_content = f"【系統自動補完引用：第{ref_art}條】\n{d.page_content}"
                            docs.append(d)
                            existing_ids.add(fetched_id)
                            print(f"✅ 成功補完 ID: {fetched_id}")
                            break

        # =========================================================
        # 3. 排序與 Context
        # =========================================================
        def final_rank(doc):
            score = 0
            content = doc.page_content
            if "【使用者指定調閱" in content: score += 2000  # 最高權重
            if "【系統自動補完" in content: score += 50
            if doc.metadata.get("type") == "file_summary": score += 1000
            if query in content: score += 100
            return score

        docs.sort(key=final_rank, reverse=True)

        final_context_list = []
        for doc in docs[:10]:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            article_id = doc.metadata.get("article_id", "")

            label = ""
            if article_id:
                label = f" | 第 {article_id} 條"
            elif page:
                label = f" | Page {page}"

            if doc.metadata.get("type") == "file_summary":
                prefix = f"🔥【全域摘要：{source}】"
            else:
                prefix = f"【來源：{source}{label}】"

            final_context_list.append(f"{prefix}\n{doc.page_content}")

        final_context = "\n\n".join(final_context_list) if final_context_list else "無具體內容。"

        # Debug View
        print("\n======== 🕵️‍♂️ Universal RAG Context ========")
        print(f"最終 Context 筆數: {len(final_context_list)}")
        print(final_context[:300] + "...")
        print("==========================================\n")

        # =========================================================
        # 4. 生成回應
        # =========================================================
        yield "⚡ **AI 正在生成解答...**\n\n"

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-2:]]) if history else "(無歷史紀錄)"

        template_str = """You are a professional, multilingual AI legal assistant.

                [SYSTEM STATUS] Uploaded Files: {file_count}

                [IMPORTANT LEGAL LOGIC RULES]
                Please strictly follow these logical connections when answering:
                1. **Article 11 (Economic Layoff/Incompetence)**:
                   - Represents "Layoff" (資遣).
                   - **MUST** provide advance notice (Article 16).
                   - **MUST** pay severance pay (Article 17).
                2. **Article 12 (Disciplinary Dismissal)**:
                   - Represents "Firing" (開除/懲戒性解僱).
                   - **NO** advance notice required.
                   - **NO** severance pay required.
                3. **Double Negative Check**:
                   - "非...不得..." means "Unless..., cannot...". It does NOT mean "No notice needed".

                [RETRIEVED KNOWLEDGE]
                The following content is retrieved from the database (mostly in Traditional Chinese).
                Use this knowledge to answer the user's question.
                {context}

                [CHAT HISTORY] {history}

                [USER QUESTION] {question}

                [⚠️ MANDATORY LANGUAGE PROTOCOL ⚠️]
                You must strictly follow these rules to determine the output language:

                1. **AUTO-DETECT**: Detect the language used in the [USER QUESTION].
                2. **MATCH LANGUAGE**: Your entire response MUST be in the **SAME language** as the [USER QUESTION].
                   - If user asks in **Japanese**, answer in **Japanese**.
                   - If user asks in **English**, answer in **English**.
                   - If user asks in **Chinese** (Simplified/Traditional), answer in **Traditional Chinese**.
                3. **TRANSLATION REQUIRED**: 
                   - The [RETRIEVED KNOWLEDGE] is in Chinese. 
                   - You must **READ** the Chinese context, **UNDERSTAND** it, and then **TRANSLATE & EXPLAIN** it in the user's target language.
                   - **DO NOT** output Traditional Chinese if the user asked in English or Japanese (unless it's for specific proper nouns).

                [RESPONSE FORMAT]
                - Be precise and helpful.
                - If the document mentions specific articles (e.g., 第12條), cite them in the target language (e.g., Article 12, 第12条).
                
                [CRITICAL READING RULES]
        1. **NO SIMPLIFICATION**: When citation involves numbers, money, or days, DO NOT output a single number if the document lists a range or conditions. (e.g., if text says "10 to 30 days", do not say "30 days").
        2. **FULL LISTING**: Always list out all the tiered conditions found in the text.
        3. **FACTUAL ACCURACY**: Your answer must perfectly match the [RETRIEVED KNOWLEDGE]. Do not use your own training data if it conflicts with the file.
                """

        prompt = ChatPromptTemplate.from_template(template_str)
        chain = (
                {
                    "context": lambda x: final_context,
                    "question": RunnablePassthrough(),
                    "history": lambda x: history_text,
                    "file_list_str": lambda x: file_list_str,
                    "file_count": lambda x: str(file_count)
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        try:
            async for chunk in chain.astream(real_query):
                clean_chunk = (chunk.replace("<br>", "\n").replace("<b>", "**").replace("</b>", "**"))
                yield clean_chunk
        except Exception as e:
            logger.error(f"Chat Error: {e}")
            yield f"\n\n⚠️ 發生錯誤: {str(e)}"