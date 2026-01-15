import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 設定
PDF_PATH = "sample-multilingual-text.pdf"  # <--- 請確認你的 PDF路徑
DB_PATH = "./chroma_db"  # 資料庫儲存位置
OLLAMA_HOST = "http://git.tedpc.com.tw:11434/"

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)


def ingest():
    print(f"開始讀取 PDF: {PDF_PATH}")
    if not os.path.exists(PDF_PATH):
        print(" 找不到 PDF 檔案！請檢查路徑。")
        return

    # 2. 讀取
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f" 讀取到 {len(pages)} 頁")

    # 3. 切割
    # 這裡我們故意切小一點，並保持重疊，模擬之前的環境
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(pages)

    # 【關鍵】手動加入 ID/Index，確保之後可以排序
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i  # 加入流水號
        chunk.metadata['source'] = PDF_PATH

    print(f"✂ 切割成 {len(chunks)} 個片段")

    # 4. 存入 ChromaDB (本地)
    print(" 正在存入 ChromaDB (這可能需要一點時間)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH  # 指定儲存資料夾
    )

    print(f" 成功！資料已儲存至 {DB_PATH}")


if __name__ == "__main__":
    ingest()