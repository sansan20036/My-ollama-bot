# app/core/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
os.environ["ANONYMIZED_TELEMETRY"] = "False"

class Settings(BaseSettings):
    # 專案基本設定
    PROJECT_NAME: str = "Syspower RAG Bot"
    VERSION: str = "1.0.0"

    # Ollama 連線設定
    OLLAMA_BASE_URL: str = "http://git.tedpc.com.tw:11434"
    OLLAMA_MODEL: str = "gemma3:27b"
    OLLAMA_API_KEY: str = ""
    LLAMA_CLOUD_API_KEY: Optional[str] = None

    # 向量與 Embedding 設定
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"

    # PDF 解析策略
    # auto: 大檔優先走本地快速解析，小檔走 LlamaParse
    # local_fast: 一律走本地快速解析
    # llamaparse: 一律走 LlamaParse
    PDF_PARSE_MODE: str = "auto"
    PDF_FAST_MODE_SIZE_MB: int = 6
    PURGE_ON_STARTUP: bool = False

    # 🚀 自動計算專案根目錄路徑
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 🚀 資料庫與上傳路徑 (拿掉 data 層，直接與 service 端的 uploads 同步)
    CHROMA_DB_DIR: str = os.path.join(BASE_DIR, "chroma_db")
    CACHE_DB_DIR: str = os.path.join(BASE_DIR, "chroma_cache")
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")

    # 🚀 Pydantic v2 的標準寫法，取代舊版的 class Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # 忽略 .env 中多餘的變數，防止報錯
    )

    def ensure_dirs(self):
        """確保系統啟動時，必要的資料夾都存在"""
        os.makedirs(self.CHROMA_DB_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DB_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)


# 初始化並建立資料夾
settings = Settings()
settings.ensure_dirs()
