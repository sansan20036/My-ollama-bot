# app/services/vector_store.py
import logging
import shutil
import os
from langchain_chroma import Chroma  # 建議使用新版 import，若報錯改回 langchain_community.vectorstores
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    _instance = None

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self._init_db()

    def _init_db(self):
        """初始化連線"""
        self.db = Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_documents(self, docs):
        if docs:
            self.db.add_documents(docs)
            logger.info(f"📥 存入 {len(docs)} 筆向量資料")

    def search(self, query: str, k: int = 4):
        return self.db.similarity_search(query, k=k)

    def list_sources(self):
        """
        🔥 新增：列出目前資料庫中所有不重複的檔案名稱
        這是為了解決 AI 不知道「目前有幾個檔案」的問題
        """
        try:
            # 只抓取 metadata，不抓 embedding 向量，速度快
            data = self.db.get(include=['metadatas'])
            metadatas = data.get("metadatas", [])

            sources = set()
            if metadatas:
                for m in metadatas:
                    # 確保 metadata 存在且有 source 欄位
                    if m and "source" in m:
                        sources.add(m["source"])

            # 回傳排序後的檔案清單
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []

    def delete_file(self, filename: str):
        """🔥 新增：刪除指定檔案的所有向量資料"""
        try:
            # 1. 先找出該檔案對應的所有 ID
            # ChromaDB 支援透過 where 條件查詢 metadata
            data = self.db.get(where={"source": filename})
            ids = data.get("ids", [])

            if ids:
                # 2. 根據 ID 刪除
                self.db.delete(ids)
                logger.info(f"🗑️ 已刪除檔案 '{filename}'，共移除 {len(ids)} 筆向量片段")
                return True
            else:
                logger.warning(f"⚠️ 找不到檔案 '{filename}' 的資料")
                return False
        except Exception as e:
            logger.error(f"刪除檔案失敗: {e}")
            raise e

    def get_file_content(self, filename: str) -> str:
        """🔥 新增：讀取指定檔案的完整內容 (將切片縫合)"""
        try:
            # 透過 metadata 找出所有片段
            data = self.db.get(where={"source": filename})
            documents = data.get("documents", [])

            if not documents:
                return "無內容或是圖片檔案 (未儲存純文字)。"

            # 簡單縫合 (如果你的切片有重疊，這裡會看到重複文字，這是正常的 RAG 現象)
            # 如果要完美還原，通常會在儲存時保留一份原始檔，但在這裡我們直接用向量庫還原
            return "\n\n...[分段分隔]...\n\n".join(documents)
        except Exception as e:
            logger.error(f"讀取檔案內容失敗: {e}")
            return f"讀取錯誤: {str(e)}"

    def reset(self):
        """🔥 清空資料庫"""
        try:
            # 1. 嘗試從 Chroma 刪除所有資料
            ids = self.db.get()['ids']
            if ids:
                self.db.delete(ids)
                logger.info(f"🗑️ 已從 Chroma 刪除 {len(ids)} 筆資料")

            # 2. 為了保險，重新初始化物件
            self.db = None
            self._init_db()
            logger.info("✅ 資料庫重置完成")

        except Exception as e:
            logger.error(f"Reset failed: {e}")