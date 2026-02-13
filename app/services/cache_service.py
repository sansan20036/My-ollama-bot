# app/services/cache_service.py
import time
import logging
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.core.config import settings

logger = logging.getLogger(__name__)


class SemanticCacheService:
    _instance = None

    def __init__(self):
        logger.info(f"🔄 正在初始化語意快取: {settings.CACHE_DB_DIR}")
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

        # 注意：這裡用不同的 persist_directory 和 collection_name
        self.db = Chroma(
            collection_name="semantic_cache",
            persist_directory=settings.CACHE_DB_DIR,
            embedding_function=self.embeddings
        )
        self.threshold = 0.35  # 相似度門檻

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def check_cache(self, query: str):
        """檢查是否有快取"""
        start_time = time.time()
        results = self.db.similarity_search_with_score(query, k=1)

        if results:
            doc, score = results[0]
            if score < self.threshold:
                elapsed = time.time() - start_time
                logger.info(f"⚡ [Cache Hit] 命中快取! (距離: {score:.4f} | 耗時: {elapsed:.4f}s)")
                return doc.page_content

        return None

    def update_cache(self, query: str, answer: str):
        """寫入快取"""
        if not answer or len(answer) < 5: return

        self.db.add_documents([
            Document(page_content=answer, metadata={"question": query})
        ])
        logger.info(f"💾 [Cache Update] 已將問答寫入快取")