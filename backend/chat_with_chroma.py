import os
import re
from difflib import SequenceMatcher  # 【新增】用來計算相似度
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 引入專門的語言偵測庫
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("❌ 請先安裝 langdetect: pip install langdetect")
    exit()

# 1. 設定
DB_PATH = "./chroma_db"
OLLAMA_HOST = "http://git.tedpc.com.tw:11434/"

# 設定 Embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)

# 使用最強模型 gemma3:27b
llm = ChatOllama(model="gemma3:27b", base_url=OLLAMA_HOST, temperature=0)

# 載入現有的 Chroma DB
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


def text_to_arabic(text):
    cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
              '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
    for cn, arab in cn_map.items():
        text = text.replace(cn, arab)
    return text


# 精準語言偵測
def detect_precise_lang(text):
    # 1. Regex 過濾亞洲語言
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "ja"
    if re.search(r'[\uAC00-\uD7AF]', text):
        return "ko"
    if re.search(r'[\u0400-\u04FF]', text):
        return "ru"
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"

        # 2. LangDetect 過濾其他語系
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"


def smart_merge(docs):
    if not docs: return ""

    unique_docs = {}
    for d in docs:
        cid = d.metadata.get('chunk_id', d.page_content[:20])
        if cid not in unique_docs:
            unique_docs[cid] = d

    sorted_docs = list(unique_docs.values())
    sorted_docs.sort(key=lambda x: x.metadata.get('chunk_id', 0))

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


# 【新增】相似度計算函數
def is_similar(a, b, threshold=0.85):
    """計算兩個字串是否高度相似 (忽略空白)"""
    # 移除所有空白後再比對，準確度更高
    clean_a = re.sub(r'\s+', '', a)
    clean_b = re.sub(r'\s+', '', b)
    return SequenceMatcher(None, clean_a, clean_b).ratio() > threshold

# 【新增】利用 LLM 自動判斷使用者想要找什麼語言
def analyze_user_intent(question):
    """
    輸入: "Hindi第五句是什麼?"
    輸出: "hi"
    輸入: "英文的內文"
    輸出: "en"
    輸入: "第五句是什麼" (沒指定)
    輸出: "None"
    """
    system_prompt = (
        "你是一個意圖識別助手。使用者的問題通常是想查詢某種語言的文件內容。\n"
        "請分析使用者的問題，判斷他【想要查詢的目標語言】是什麼。\n\n"
        "規則：\n"
        "1. 請回傳該語言的【ISO 639-1 代碼】(例如：英文=en, 中文=zh, 日文=ja, 韓文=ko, 印地文=hi, 法文=fr, 德文=de, 西文=es, 俄文=ru, 葡萄牙文=pt, 越南文=vi...等)。\n"
        "2. 如果使用者【沒有指定語言】(例如只問 '第五點是什麼?')，請回傳 'None'。\n"
        "3. 嚴格只回傳代碼，不要有任何標點符號或解釋。\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"input": question})

    code = response.content.strip().lower()

    # 清理一下 AI 可能的多餘輸出 (雖然 Prompt 限制了，防呆一下)
    if "none" in code: return None
    # 抓取純字母代碼 (避免 AI 回傳 'code: en')
    match = re.search(r'\b[a-z]{2}\b', code)
    if match:
        return match.group(0)
    return None


def ask_ai(question):
    print(f"\n [Chroma版] 正在檢索「{question}」...")
    try:
        # --- 步驟 A: 檢索 ---
        results = vector_store.similarity_search(question, k=100)

        if not results:
            print(" 資料庫是空的！請先執行 ingest_chroma.py")
            return

        print(f" 檢索到 {len(results)} 個片段，準備進行處理...")

        # --- 步驟 C: 數字偵測 ---
        normalized_question = text_to_arabic(question)
        target_numbers = re.findall(r'\d+', normalized_question)

        final_context = ""
        mode = "full"

        q_lower = question.lower()
        target_lang_code = None

        # 語言判斷 (擴充版)
        if 'english' in q_lower or '英文' in q_lower:
            target_lang_code = 'en'
        elif 'chinese' in q_lower or '中文' in q_lower:
            target_lang_code = 'zh'
        elif 'korean' in q_lower or '韓文' in q_lower:
            target_lang_code = 'ko'
        elif 'japanese' in q_lower or '日文' in q_lower:
            target_lang_code = 'ja'
        elif 'hindi' in q_lower or '印地' in q_lower:
            target_lang_code = 'hi'
        elif 'french' in q_lower or '法文' in q_lower:
            target_lang_code = 'fr'
        elif 'german' in q_lower or '德文' in q_lower:
            target_lang_code = 'de'
        elif 'spanish' in q_lower or '西文' in q_lower:
            target_lang_code = 'es'
        elif 'russian' in q_lower or '俄文' in q_lower:
            target_lang_code = 'ru'

        if target_lang_code:
            print(f" 使用者目標語言代碼: {target_lang_code}")

        if target_numbers:
            # === 模式 1: 手術刀模式 (Specific) ===
            full_context = smart_merge(results)
            flat_context = full_context.replace('\n', '  ')

            target_num = target_numbers[0]
            next_num = str(int(target_num) + 1)
            print(f" 偵測到目標編號: {target_num}")

            if f"{target_num} " not in flat_context:
                print(f" 錯誤：文本中找不到編號 {target_num}。")
                return

            pattern = re.compile(rf"{target_num}\s+(.*?)\s+{next_num}", re.DOTALL)
            candidates = pattern.findall(flat_context)

            if not candidates:
                print(f" 找不到結束編號 {next_num}，改用「寬鬆模式」...")
                fallback_pattern = re.compile(rf"{target_num}\s+([^0-9]{{10,800}})", re.DOTALL)
                candidates = fallback_pattern.findall(flat_context)

            if candidates:
                print(f"---------- [DEBUG] Python 強力過濾前 ({len(candidates)} 個) ----------")

                # 1. 語言篩選
                lang_filtered_candidates = []
                for cand in candidates:
                    detected = detect_precise_lang(cand)

                    is_english_structure = False
                    if target_lang_code == 'en':
                        if re.search(r'\b(the|is|are|and|to|of|in|that|will|not|do|we|you)\b', cand.lower()):
                            is_english_structure = True
                        if "english" in cand.lower():
                            is_english_structure = True

                    should_keep = True
                    if target_lang_code:
                        if target_lang_code == 'en':
                            if not (detected == 'en' or is_english_structure): should_keep = False
                        elif target_lang_code == 'zh':
                            if detected not in ['zh', 'zh-cn', 'zh-tw']: should_keep = False
                        else:
                            if detected != target_lang_code: should_keep = False

                    if should_keep:
                        lang_filtered_candidates.append(cand)

                # 2. 【核心修改】模糊去重 (Fuzzy Deduplication)
                # 從長排到短，優先保留資訊量最多的
                lang_filtered_candidates.sort(key=len, reverse=True)
                unique_candidates = []

                for cand in lang_filtered_candidates:
                    is_duplicate = False
                    for existing in unique_candidates:
                        # 檢查 1: 子字串包含
                        if cand in existing:
                            is_duplicate = True
                            break
                        # 檢查 2: 高度相似 (解決 Hindi 亂碼差異問題)
                        if is_similar(cand, existing, threshold=0.85):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        unique_candidates.append(cand)
                        detected = detect_precise_lang(cand)
                        preview = cand[:30].replace('\n', ' ')
                        print(f"  [保留] {preview}... (偵測為: {detected})")

                print("-" * 60)
                final_context = "Candidates found:\n"

                output_list = unique_candidates if unique_candidates else candidates
                if not unique_candidates: print("過濾後沒有候選人，恢復原始列表")

                for cand in output_list:
                    final_context += f"- {cand}\n"

                mode = "specific"
            else:
                final_context = flat_context

        else:
            # === 模式 2: 全文模式 (Full Mode) ===
            print(" 偵測到全文請求，正在進行「縫合 -> 切行 -> 去重」...")

            full_document_text = smart_merge(results)
            lines = full_document_text.split('\n')

            filtered_lines = []
            seen_lines = []  # 改用 list 配合模糊比對

            for line in lines:
                clean_line = line.strip()
                if not clean_line: continue

                # 【全文模式模糊去重】
                is_duplicate = False
                for seen in seen_lines:
                    if clean_line in seen or seen in clean_line:  # 包含檢查
                        is_duplicate = True
                        break
                    if is_similar(clean_line, seen, threshold=0.9):  # 相似檢查
                        is_duplicate = True
                        break

                if is_duplicate: continue

                content = line
                detected = detect_precise_lang(content)
                should_keep = True

                if target_lang_code:
                    should_keep = False
                    if target_lang_code == 'en':
                        if detected == 'en' or "English" in content:
                            should_keep = True
                        elif re.search(r'\b(the|is|are|and|to|of|in|that|will|not|do|we|you|for)\b', content.lower()):
                            should_keep = True
                    elif target_lang_code == 'zh':
                        if detected in ['zh', 'zh-cn', 'zh-tw']: should_keep = True
                    else:
                        if detected == target_lang_code: should_keep = True

                if should_keep:
                    filtered_lines.append(content)
                    seen_lines.append(clean_line)

            if filtered_lines:
                print(f" 成功過濾！最終保留了 {len(filtered_lines)} 行內容 (已去除重複)。")
                final_context = "\n".join(filtered_lines)
            else:
                print(" 語言過濾後沒有內容。停止發送給 AI 以免幻覺。")
                print("=" * 48)
                print("系統回答：抱歉，在文件中找不到符合該語言的內容。")
                print("=" * 48)
                return

            mode = "full"

        # --- 步驟 D: 讓 AI 回答 ---
        print(f" AI ({llm.model}) 正在處理 ({mode} mode)...")

        # 【關鍵修改】Prompt 再次強調只輸出一種版本
        system_prompt_final = (
            "你是一個專業的內容整理助手。我會給你經過篩選的文字列表 (Context)。\n"
            "你的任務是根據使用者的問題，整理並列出 Context 中的內容。\n\n"
            "嚴格規則：\n"
            "1. 【提取與列出】：使用者想看特定語言的內容，請條列式列出。\n"
            "2. 【去重與合併】：如果 Context 中出現【內容極度相似】的多個版本（可能是因為亂碼或重複），請自動判斷並【只輸出最完整、最乾淨的那一個】，不要重複列出。\n"
            "3. 【去除雜訊】：刪除句尾不相關的語言標籤（如 'Hindi:', 'French:'）。\n"
            "4. 【只讀取 Context】：只輸出 Context 裡的內容。\n\n"
            "Context (已過濾):\n{context}"
        )

        prompt_final = ChatPromptTemplate.from_messages([
            ("system", system_prompt_final),
            ("human", "使用者指令: {input}"),
        ])

        chain_final = prompt_final | llm

        result = chain_final.invoke({
            "input": question,
            "context": final_context
        })

        print("\n" + "=" * 20 + " AI 回答 " + "=" * 20)
        print(result.content)
        print("=" * 48)

    except Exception as e:
        print(f" 執行發生錯誤: {e}")


if __name__ == "__main__":
    while True:
        user_input = input("\n[Chroma版] 請輸入問題 (quit 結束): ").strip()
        if not user_input: continue
        if user_input.lower() in ['quit', 'exit', '結束']: break

        ask_ai(user_input)