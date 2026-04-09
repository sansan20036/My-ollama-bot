# app/services/cache_service.py
import time
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.core.config import settings

logger = logging.getLogger(__name__)


class SemanticCacheService:
    _instance = None

    def __init__(self):
        # 1. 建立連線模式的 Embedding
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {settings.OLLAMA_API_KEY}"
                }
            }
        )
        # 2. 補上門檻值：距離越小越接近，0.35 是兼顧精準與速度的平衡點
        self.threshold = 0.35

        self._init_db()

    def _init_db(self):
        """初始化快取資料庫"""
        self.db = Chroma(
            persist_directory=settings.CACHE_DB_DIR,
            embedding_function=self.embeddings
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def check_cache(self, query: str):
        """檢查是否有語意相近的快取答案"""
        start_time = time.time()

        # 🚀 這裡的 query (新問題) 將會去跟資料庫裡的 page_content (舊問題) 算距離！
        results = self.db.similarity_search_with_score(query, k=1)

        if results:
            doc, score = results[0]
            # 只有當距離小於門檻時，才認定是同一個問題
            if score < self.threshold:
                elapsed = time.time() - start_time
                logger.info(f"[Cache Hit] 命中快取! (距離: {score:.4f} | 耗時: {elapsed:.4f}s)")

                # 🚀 關鍵修正：命中舊問題後，把藏在 metadata 裡的「答案」拿出來還給使用者！
                return doc.metadata.get("answer")

        return None

    def update_cache(self, query: str, answer: str):
        """將新的問答對寫入向量快取"""
        if not answer or len(answer) < 5:
            return

        # 🚀 關鍵修正：把「問題」當作向量主體 (page_content)！把「答案」當作包裹塞進 metadata！
        self.db.add_documents([
            Document(page_content=query, metadata={"answer": answer})
        ])
        logger.info(f" [Cache Update] 已將問答寫入快取")