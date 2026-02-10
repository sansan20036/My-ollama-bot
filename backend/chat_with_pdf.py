import os
import re
from supabase import create_client
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 【新增】引入專門的語言偵測庫
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print(" 請先安裝 langdetect: pip install langdetect")
    exit()

# 1. 連線配置
SUPABASE_URL = "https://abuxyukbleiauunrroks.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFidXh5dWtibGVpYXV1bnJyb2tzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgyODM4NTUsImV4cCI6MjA4Mzg1OTg1NX0.w9g1xGbyHXGjCIj3wWl_0lkVojRzlkoQNTUEKZLRn8Q"
OLLAMA_HOST = "http://git.tedpc.com.tw:11434/"

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)

# 依舊使用強大的 gemma3:27b
llm = ChatOllama(model="gemma3:27b", base_url=OLLAMA_HOST, temperature=0)


def text_to_arabic(text):
    cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
              '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
    for cn, arab in cn_map.items():
        text = text.replace(cn, arab)
    return text


# 【核心升級】超級精準的語言偵測器
def detect_precise_lang(text):
    # 1. 先用 Regex 快速過濾亞洲語言 (Regex 對這些語言最準)
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "ja"  # 日文
    if re.search(r'[\uAC00-\uD7AF]', text):
        return "ko"  # 韓文
    if re.search(r'[\u0400-\u04FF]', text):
        return "ru"  # 俄文
    if re.search(r'[\u4e00-\u9fff]', text):  # 包含漢字
        return "zh"  # 中文 (可能是繁體或簡體)

    # 2. 如果是拉丁字母 (歐語系)，使用 langdetect 進行細分
    try:
        # langdetect 可以分辨 en(英), es(西), fr(法), de(德), it(義)...
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"


def smart_merge(docs):
    if not docs: return ""
    docs.sort(key=lambda x: x['id'])
    merged_text = docs[0]['content']

    for i in range(1, len(docs)):
        prev_text = merged_text
        curr_text = docs[i]['content']
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


def ask_ai(question):
    print(f"\n [Supabase版] 正在檢索「{question}」...")
    try:
        # --- 步驟 A: 檢索 ---
        query_vector = embeddings.embed_query(question)
        rpc_params = {
            "query_embedding": query_vector,
            "match_threshold": -1.0,
            "match_count": 100,
            "filter": {}
        }
        response = supabase_client.rpc("get_documents_nuclear", rpc_params).execute()

        # --- 步驟 B: 縫合 ---
        raw_data = response.data
        if raw_data:
            full_context = smart_merge(raw_data)
            flat_context = full_context.replace('\n', '  ')
        else:
            full_context = ""
            flat_context = ""

        print(f" 資料庫回傳並智慧去重 (長度 {len(flat_context)} 字元)")

        # --- 步驟 C: 數字偵測 ---
        normalized_question = text_to_arabic(question)
        target_numbers = re.findall(r'\d+', normalized_question)

        final_context = ""
        mode = "full"

        if target_numbers:
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
                # 偵測使用者想要什麼語言
                q_lower = question.lower()
                target_lang_code = None

                if 'english' in q_lower or '英文' in q_lower:
                    target_lang_code = 'en'
                    print(" 使用者目標語言: English (en)")
                elif 'chinese' in q_lower or '中文' in q_lower:
                    target_lang_code = 'zh'
                    print(" 使用者目標語言: Chinese (zh)")

                print(f"---------- [DEBUG] Python 強力過濾前 ({len(candidates)} 個) ----------")

                valid_candidates = []
                final_context = "Candidates found:\n"

                for cand in candidates:
                    # 使用新的強力偵測
                    detected = detect_precise_lang(cand)
                    preview = cand[:30].replace('\n', ' ')

                    # 【核心過濾邏輯】
                    # 如果我們明確知道使用者要英文 (en)，而這句被偵測為 es(西), fr(法), de(德)... 直接剔除！
                    if target_lang_code == 'en':
                        if detected != 'en':
                            print(f"  [剔除] {preview}... (偵測為: {detected}, 目標: en)")
                            continue

                    # 如果使用者要中文 (zh)，而這句被偵測為 ja(日), ko(韓)... 直接剔除！
                    if target_lang_code == 'zh':
                        if detected not in ['zh', 'zh-cn', 'zh-tw']:
                            print(f"  [剔除] {preview}... (偵測為: {detected}, 目標: zh)")
                            continue

                    print(f"  [保留] {preview}... (偵測為: {detected})")
                    final_context += f"- {cand}\n"
                    valid_candidates.append(cand)

                print("-" * 60)

                if not valid_candidates:
                    print("過濾後沒有候選人，恢復原始列表 (可能偵測失敗)")
                    for cand in candidates:
                        final_context += f"- {cand}\n"
                mode = "specific"
            else:
                print(" 寬鬆模式也抓不到，降級使用全文")
                final_context = flat_context
        else:
            print(" 偵測到全文請求")
            final_context = flat_context
            mode = "full"

        # --- 步驟 D: 讓 AI 回答 ---
        print(f" AI ({llm.model}) 正在處理 ({mode} mode)...")

        system_prompt_final = (
            "你是一個嚴格的內容提取助手。我會給你一個候選文字列表 (Context)。\n"
            "你的任務是根據使用者的問題，從列表中挑出【唯一正確的語言版本】。\n\n"
            "規則：\n"
            "1. 【英文鎖定】：如果使用者要英文，Context 裡現在應該只剩下英文候選句了。請直接輸出。\n"
            "2. 【完整複製】：選中正確的候選句後，請完整輸出該句內容，不要刪減，不要翻譯。\n"
            "3. 【只輸出內容】：不要解釋你的選擇理由，直接給出句子。\n\n"
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
        user_input = input("\n[Supabase版] 請輸入問題 (quit 結束): ").strip()
        if not user_input: continue
        if user_input.lower() in ['quit', 'exit', '結束']: break

        ask_ai(user_input)