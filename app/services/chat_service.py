# app/services/chat_service.py
import logging
import os
import re
import pandas as pd
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
        self.cached_df = None
        self.cached_file_path = ""
        self.cached_file_mtime = 0

        target_model = "gpt-oss:20b"

        logger.info(f"🔥 初始化全能文件聊天服務: {target_model}")

        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=target_model,
            temperature=0,  # Agent 運算設為 0，確保程式碼與數學精準
            keep_alive="1h",
            num_ctx=16384,
            num_predict=4096
        )

    def _get_valid_files(self) -> list:
        if not os.path.exists(self.upload_dir): return []

        # 🔥 新增防呆：過濾掉結尾是 _tables.csv 的系統快取檔，只計算使用者真正上傳的檔案！
        files = [f for f in os.listdir(self.upload_dir) if
                 os.path.isfile(os.path.join(self.upload_dir, f))
                 and not f.startswith("~")
                 and not f.endswith("_tables.csv")]

        # 依照檔案的「最後修改/建立時間」進行排序 (由舊到新)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.upload_dir, x)))
        return files

    def _get_sorted_file_list(self, files: list) -> str:
        if not files: return "(無檔案)"

        result = []
        for i, f in enumerate(files):
            label = ""
            if len(files) > 1:
                if i == len(files) - 1:
                    label = " 🟢 (最新上傳)"
                elif i == 0:
                    label = " ⚪ (最早上傳)"
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
        real_query = query
        valid_files = self._get_valid_files()
        file_count = len(valid_files)
        file_list_str = self._get_sorted_file_list(valid_files)

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-2:]]) if history else "(無歷史紀錄)"

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

            # 🎯 終極秒殺邏輯：尋找是否已有在上傳階段提煉好的 CSV 快取
            possible_csv = f"{file_name_without_ext}_tables.csv"

            if self.cached_file_path == target_file and self.cached_file_mtime == current_mtime:
                logger.info("⚡ 使用記憶體中的 DataFrame，跳過檔案解析")
                df = self.cached_df
            else:
                try:
                    # 優先檢查：上傳時是否已經提煉出表格了？
                    if os.path.exists(possible_csv):
                        logger.info("⚡ 發現預處理的 PDF 表格快取！直接秒讀載入...")
                        df = pd.read_csv(possible_csv)
                    elif file_ext in ['xlsx', 'xls']:
                        logger.info("📊 偵測到原生 Excel 檔案，直接載入...")
                        df = pd.read_excel(target_file)
                        df.columns = [re.split(r'[\s\n(]', str(col))[0] for col in df.columns]
                    elif file_ext == 'csv':
                        logger.info("📊 偵測到原生 CSV 檔案，直接載入...")
                        df = pd.read_csv(target_file)
                        df.columns = [re.split(r'[\s\n(]', str(col))[0] for col in df.columns]
                except Exception as e:
                    logger.error(f"讀取 DataFrame 時發生錯誤: {e}")

                # 將結果存入快取
                self.cached_df = df
                self.cached_file_path = target_file
                self.cached_file_mtime = current_mtime

                if df is not None and not df.empty:
                    logger.info("✨ 啟動 [自建 Python 直譯引擎]")

                    python_code = ""
                    try:
                        # 第一步：讓 AI 根據使用者的問題，寫出「一行」Pandas 程式碼
                        code_prompt = (
                            f"你是一個頂級的 Python 資料分析師。我有一個 pandas DataFrame 叫做 `df`。\n"
                            f"這個表格的真實欄位有：{list(df.columns)}\n\n"
                            f"請寫出『一行』Python 程式碼來取得以下問題的答案：\n"
                            f"問題：「{real_query}」\n\n"

                            f"【⚠️ 嚴格規定】：\n"
                            f"1. 請『只』輸出那行 Python 程式碼，絕對不要包含任何解釋、不要使用 markdown 標記。\n"
                            f"2. 絕對不要使用 `print()`。\n"
                            f"3. 為了避免文字被截斷，如果答案是整行資料 (DataFrame)，請務必在句尾加上 `.to_dict('records')`。\n"
                            f"4. ⚠️ 【極度重要】如果要對某個欄位進行加總 (sum) 或數學運算，該欄位可能混雜中文字，請務必先使用 `pd.to_numeric(..., errors='coerce')` 轉換型別！例如：pd.to_numeric(df['時數'], errors='coerce').sum()\n"
                            f"5. 如果問最後一筆，請輸出：df.tail(1).to_dict('records')\n"
                            f"6. 如果問總筆數，請輸出：len(df)\n"
                            f"現在請輸出程式碼："
                        )

                        logger.info("🧠 AI 正在撰寫分析程式碼...")

                        ai_code_response = await self.llm.ainvoke(code_prompt)
                        # 確保拿到的是純字串
                        raw_code_text = ai_code_response.content if hasattr(ai_code_response, 'content') else str(
                            ai_code_response)
                        python_code = raw_code_text.replace("```python", "").replace("```", "").strip()

                        # 第二步：安全地在後端執行這行程式碼
                        safe_builtins = {
                            "len": len, "sum": sum, "min": min, "max": max,
                            "abs": abs, "round": round, "int": int, "float": float,
                            "str": str, "list": list, "dict": dict
                        }
                        exec_env = {"df": df, "pd": pd, "__builtins__": safe_builtins}
                        result = eval(python_code, exec_env)
                        logger.info(f"📈 程式碼執行結果: {result}")

                        # 第三步：把冰冷的結果，請 AI 翻譯成溫暖的中文回答
                        answer_prompt = (
                            f"使用者問的問題是：「{real_query}」\n"
                            f"我們透過 Python 計算出來的精確結果是：\n{result}\n\n"
                            f"請用專業、流暢的繁體中文，將這個結果完整地回答給使用者。"
                        )

                        logger.info("🗣️ AI 正在翻譯最終解答...")
                        # 將最終答案串流輸出給前端
                        async for chunk in self.llm.astream(answer_prompt):
                            text_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
                            clean_chunk = text_chunk.replace("<br>", "\n").replace("<b>", "**").replace("</b>", "**")
                            yield clean_chunk

                        return  # 執行完畢，提早結束，不進入 RAG

                    except Exception as e:
                        logger.error(f"⚠️ 數據運算失敗，降級回傳統 RAG 模式: {e}\n(嘗試執行的程式碼: {python_code})")
                        # 靜默降級，不干擾前端畫面

        # =========================================================
        # 📄 路線 B：傳統 RAG 模式 (如果沒表格，或 Agent 失敗)
        # =========================================================

        # 1. 第一輪：通用檢索
        ai_keywords = await self._smart_query_rewrite(real_query)
        search_query = f"{real_query} {ai_keywords}"

        matches = re.findall(r'(?:第\s*\d+\s*[章節條]|(?<!\d)\d+\.\d+(?:\.\d+)?(?!\d))', real_query)

        if matches:
            for m in matches:
                search_query += f" {m}"

        print(f"🚀 執行檢索: {search_query}")
        docs = self.vector_store.search(search_query, k=50)

        # 新增：狙擊模式 (Sniper Mode)
        if matches:
            print(f"🎯 偵測到明確條號/章節 {matches}，啟用狙擊模式")
            existing_ids = set()
            for d in docs:
                aid = d.metadata.get("article_id")
                if aid: existing_ids.add(str(aid))

            for m in matches:
                target_id = re.sub(r'[^\d.]', '', m)
                if not target_id: continue

                sniper_query = f"第{target_id}條 第{target_id}章 第{target_id}節 Section {target_id} Chapter {target_id} {target_id}"
                label_text = f"指定段落 {target_id}"
                sniper_k = 1000
                print(f"🔫 啟動全域掃描：尋找目標 [{target_id}]...")

                if target_id in existing_ids: continue

                sniper_docs = self.vector_store.search(sniper_query, k=sniper_k)
                is_snipe_success = False

                for d in sniper_docs:
                    fetched_id = str(d.metadata.get("article_id", ""))
                    is_match = False

                    if fetched_id:
                        # 防線 A：如果是結構化法規，只看 ID！絕對不看內文，避免誤判
                        if fetched_id == target_id or (target_id.isdigit() and fetched_id == str(int(target_id))):
                            is_match = True
                    else:
                        # 防線 B：如果是非結構化文件（如普通 PDF），才用 Regex 找內文
                        match_text = bool(re.search(rf'(?<![\d.]){re.escape(target_id)}(?![\d.])', d.page_content))
                        if match_text:
                            is_match = True

                    if is_match:
                        d.page_content = f"【使用者指定調閱：{label_text}】\n{d.page_content}"
                        docs.insert(0, d)
                        existing_ids.add(target_id)
                        is_snipe_success = True
                        print(f"✅ 狙擊成功：已將目標 [{target_id}] 內容拉至最前！(來源 ID: {fetched_id})")
                        break

                if not is_snipe_success:
                    print(f"⚠️ 狙擊失敗：在 {sniper_k} 筆切片中，完全找不到包含 '{target_id}' 的精確內容。")

        # 2. 第二輪：彈性補完
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

        # 3. 排序與 Context
        def final_rank(doc):
            score = 0
            content = doc.page_content
            if "【使用者指定調閱" in content: score += 2000
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

        if file_count > 0:
            final_context = "\n\n".join(final_context_list) if final_context_list else "無具體內容。"

        print("\n======== 🕵️‍♂️ Universal RAG Context ========")
        print(f"最終 Context 筆數: {len(final_context_list)}")
        print(final_context[:300] + "...")
        print("==========================================\n")

        # 4. 生成回應
        domain_rules = ""
        if any(keyword in real_query for keyword in ["勞基法", "勞動基準法", "資遣", "解僱", "開除", "預告工資"]):
            print("⚖️ 觸發勞基法專屬邏輯")
            domain_rules = """
                        [IMPORTANT LEGAL LOGIC RULES (Labor Law)]
                        Please strictly follow these logical connections when answering:
                        1. **Article 11 (Economic Layoff/Incompetence)**: Represents "Layoff" (資遣). MUST provide advance notice & severance pay.
                        2. **Article 12 (Disciplinary Dismissal)**: Represents "Firing" (開除). NO advance notice & NO severance pay required.
                        3. **Double Negative Check**: "非...不得..." means "Unless..., cannot...".
                    """
        elif file_count > 0:
            print("📄 觸發通用文件分析邏輯")
            domain_rules = """
                        [GENERAL DOCUMENT ANALYSIS RULES]
                        - You are analyzing a general document (e.g., academic paper, manual, contract, report).
                        - Analyze the provided text objectively. 
                        - Do not apply specific labor law rules unless explicitly mentioned in the text.
                    """

        template_str = """You are a professional, multilingual AI document analysis assistant.

                        [SYSTEM STATUS] 
                        Total Uploaded Files: {file_count}
                        File List (Ordered from oldest to newest):
                        {file_list_str}

                        [🛡️ ROBUSTNESS & NOISE TOLERANCE - CRITICAL!]
                        1. The [RETRIEVED KNOWLEDGE] may contain broken tables, LaTeX mathematical formulas (e.g., $F(x)$), or messy OCR text.
                        2. **STRICTLY PROHIBITED**: You MUST NEVER claim the text is "gibberish", "garbled", "corrupted", or "unreadable".
                        3. **YOUR DUTY**: Ignore formatting errors, raw formulas, and meaningless symbols. Focus ONLY on extracting the readable natural language sentences to answer the question.

                        [📐 MATH & FORMATTING RULES - CRITICAL!]
                        When outputting mathematical formulas, equations, or variables, YOU MUST strictly use LaTeX formatting.
                        - For inline math and variables, wrap them in single dollar signs (e.g., $O(n^3)$, $A$, $\sigma_i$).
                        - For block equations, wrap them in double dollar signs on new lines (e.g., $$A w = b$$).
                        - DO NOT use raw unicode characters for complex math (like fractions or matrices). Always write them in standard LaTeX code.

                        {domain_rules}

                        [RETRIEVED KNOWLEDGE]
                        {context}

                        [CHAT HISTORY] {history}

                        [USER QUESTION] {question}

                        [⚠️ MANDATORY LANGUAGE PROTOCOL ⚠️]
                        1. **AUTO-DETECT**: Detect the language used in the [USER QUESTION].
                        2. **MATCH LANGUAGE**: Your entire response MUST be in the **SAME language** as the [USER QUESTION].
                        3. **TRANSLATION REQUIRED**: Read the context, understand it, and TRANSLATE & EXPLAIN it in the user's target language.

                        [CRITICAL READING RULES]
                        1. **NO SIMPLIFICATION**: When citation involves numbers, money, or days, DO NOT output a single number if the document lists a range or conditions.
                        2. **FULL LISTING**: Always list out all the tiered conditions found in the text.
                        3. **FACTUAL ACCURACY**: Your answer must perfectly match the [RETRIEVED KNOWLEDGE].
                        4. **CHAPTER MATCHING**: If the user asks for a specific Chapter (e.g., Chapter 7), YOU MUST ONLY use information from that chapter. If the retrieved context only shows Chapter 3, you must truthfully say: "I cannot find the content for Chapter 7 in the retrieved context," and DO NOT hallucinate using other chapters.
                        """

        prompt = ChatPromptTemplate.from_template(template_str)
        chain = (
                {
                    "context": lambda x: final_context,
                    "question": RunnablePassthrough(),
                    "history": lambda x: history_text,
                    "file_count": lambda x: str(file_count),
                    "file_list_str": lambda x: file_list_str,
                    "domain_rules": lambda x: domain_rules
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