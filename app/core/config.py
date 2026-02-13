# app/core/config.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 專案基本設定
    PROJECT_NAME: str = "Syspower RAG Bot"
    VERSION: str = "1.0.0"

    # Ollama 連線設定
    OLLAMA_BASE_URL: str = "http://git.tedpc.com.tw:11434"
    OLLAMA_MODEL: str = "gpt-oss:20b"

    # 向量與 Embedding 設定
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # 自動計算路徑 (這就是為什麼我們要有這個檔案)
    # BASE_DIR 會自動抓到 app 資料夾的上一層 (也就是專案根目錄)
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 資料庫與上傳路徑
    CHROMA_DB_DIR: str = os.path.join(BASE_DIR, "data", "chroma_db")
    CACHE_DB_DIR: str = os.path.join(BASE_DIR, "data", "chroma_cache")
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "data", "uploads")

    # 確保資料夾存在
    def ensure_dirs(self):
        os.makedirs(self.CHROMA_DB_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DB_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

    class Config:
        env_file = ".env"


settings = Settings()
settings.ensure_dirs()  # 啟動時自動建立資料夾