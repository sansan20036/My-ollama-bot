import logging
import os
import shutil
from typing import List, Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import settings
from app.utils.smart_parser import SmartFileParser  # 確保這支程式已存在

logger = logging.getLogger(__name__)


class VectorStoreService:
    _instance = None

    def __init__(self):
        # 初始化 Embedding 模型
        # 使用 HuggingFace 模型將文字轉為向量
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self._init_db()

    def _init_db(self):
        """初始化 ChromaDB 連線"""
        self.db = Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )

    @classmethod
    def get_instance(cls):
        """Singleton 模式，確保全域只有一個實例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_documents(self, docs: List[Document]):
        """
        將文件存入向量資料庫 (同步方法)
        """
        if docs:
            try:
                # ChromaDB 的 add_documents 會自動處理 ID 和向量化
                self.db.add_documents(docs)
                logger.info(f"📥 成功存入 {len(docs)} 筆向量資料片段")
            except Exception as e:
                logger.error(f"❌ 存入向量資料庫失敗: {e}")
                raise e

    async def process_file(self, file_path: str):
        """
        🔥 核心流程：讀取 -> 智慧解析 -> 儲存
        """
        try:
            # 避免循環引用，在函式內 import
            from app.services.file_service import FileLoaderFactory

            filename = os.path.basename(file_path)

            # 1. 讀取原始文字 (使用 FileLoaderFactory)
            loader = FileLoaderFactory.get_loader(file_path, filename)
            text_content = loader.extract_text()

            if not text_content:
                logger.warning(f"⚠️ 檔案 {filename} 無內容或無法讀取，跳過處理")
                return

            # 2. 🔥 啟動 SmartFileParser 進行結構化解析
            logger.info(f"🧠 啟動 SmartFileParser 解析檔案: {filename}")
            parser = SmartFileParser()

            # 這會回傳一系列帶有豐富 metadata (如 article_id, type) 的 Document 物件
            docs = parser.parse(text_content, filename)

            # 3. 存入資料庫
            if docs:
                self.add_documents(docs)
                logger.info(f"✅ 檔案 '{filename}' 處理完成，共存入 {len(docs)} 筆結構化資料")
            else:
                logger.warning(f"⚠️ 檔案 '{filename}' 解析後無有效資料片段")

        except Exception as e:
            logger.error(f"❌ 處理檔案失敗 {file_path}: {e}")
            raise e
    # 定義搜尋動作的地方
    def search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None):
        """
        執行向量相似度搜尋
        Args:
            query: 搜尋關鍵字
            k: 回傳筆數
            filter: Metadata 過濾條件 (例如 {"article_id": "12"})
        """
        if filter:
            # 如果有指定 filter，使用帶過濾的搜尋
            return self.db.similarity_search(query, k=k, filter=filter)
        else:
            # 一般搜尋
            return self.db.similarity_search(query, k=k)

    def list_sources(self):
        """
        列出目前資料庫中所有不重複的檔案名稱
        """
        try:
            # 只抓取 metadata，不抓 embedding 向量，速度快
            data = self.db.get(include=['metadatas'])
            metadatas = data.get("metadatas", [])

            sources = set()
            if metadatas:
                for m in metadatas:
                    if m and "source" in m:
                        sources.add(m["source"])

            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []

    def delete_file(self, filename: str):
        """刪除指定檔案的所有向量資料"""
        try:
            # 1. 找出該檔案對應的所有 ID
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
        """讀取指定檔案的完整內容 (將切片縫合，用於前端預覽)"""
        try:
            # 根據 source 抓取所有片段
            data = self.db.get(where={"source": filename})
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            if not documents:
                return "無內容或是圖片檔案 (未儲存純文字)。"

            # 嘗試根據 'chunk_index' 或 'article_id' 排序，讓縫合後的文字順序正確
            combined = zip(documents, metadatas)

            # 優先嘗試用 metadata 裡的 chunk_id 排序，如果沒有就保持原樣
            try:
                sorted_combined = sorted(combined, key=lambda x: x[1].get('chunk_id', 0) if x[1] else 0)
                sorted_docs = [doc for doc, meta in sorted_combined]
            except:
                sorted_docs = documents

            return "\n\n-------------------\n\n".join(sorted_docs)
        except Exception as e:
            logger.error(f"讀取檔案內容失敗: {e}")
            return f"讀取錯誤: {str(e)}"

    def reset(self):
        """🔥 強制清空資料庫 (Purge System)"""
        try:
            # 1. 嘗試從 Chroma 刪除所有資料
            all_ids = self.db.get()['ids']
            if all_ids:
                # Chroma 限制一次刪除數量，分批刪除較安全
                batch_size = 5000
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    self.db.delete(batch_ids)
                logger.info(f"🗑️ 已從 Chroma 刪除 {len(all_ids)} 筆資料")

            # 2. 重新初始化
            self.db = None
            self._init_db()
            logger.info("✅ 資料庫重置完成")

        except Exception as e:
            logger.error(f"Reset failed: {e}")