import ollama
import re  # 🟢 引入正規表示式用於語言偵測
from pypdf import PdfReader
from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 設定連線資訊 (保持不變)
SUPABASE_URL = "https://abuxyukbleiauunrroks.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFidXh5dWtibGVpYXV1bnJyb2tzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgyODM4NTUsImV4cCI6MjA4Mzg1OTg1NX0.w9g1xGbyHXGjCIj3wWl_0lkVojRzlkoQNTUEKZLRn8Q"
OLLAMA_HOST = "http://git.tedpc.com.tw:11434/"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
ollama_client = ollama.Client(host=OLLAMA_HOST)

# 2. 切片邏輯
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=80,
    separators=["\n\n", "\n", " "]
)

import re


def detect_language(text):
    # 🟢 優先檢查是否含有日文字元 (平假名 \u3040-\u309f 或 片假名 \u30a0-\u30ff)
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
        return "Japanese"

    # 🟢 檢查是否含有韓文字元
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"

    # 🟢 最後才檢查漢字，確保不是日文後才判定為中文
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese (Simplified)"

    if "English:" in text:
        return "English"

    return "Other"

def process_pdf_to_supabase(file_path):
    print(f"正在執行「自動標籤化」處理: {file_path}")
    reader = PdfReader(file_path)

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text.strip(): continue

        chunks = text_splitter.split_text(page_text)
        print(f"第 {i + 1} 頁切分為 {len(chunks)} 個片段")

        for chunk in chunks:
            # 🟢 偵測當前片段的語言
            lang = detect_language(chunk)

            # 向量化
            response = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk)

            # 🟢 寫入資料庫：在 metadata 增加 language 欄位
            supabase.table("documents").insert({
                "content": chunk,
                "embedding": response['embedding'],
                "metadata": {
                    "page": i + 1,
                    "language": lang  # 給予語言標籤
                }
            }).execute()

    print("資料已成功分類入庫！")


if __name__ == "__main__":
    process_pdf_to_supabase(r"C:\Users\sansa\PycharmProjects\Ollama\.venv\sample-multilingual-text.pdf")