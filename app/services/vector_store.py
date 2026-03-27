# app/services/vector_store.py
import logging
import os
import shutil
import re
from typing import List, Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from app.core.config import settings
from app.utils.smart_parser import SmartFileParser

logger = logging.getLogger(__name__)


class VectorStoreService:
    _instance = None

    def __init__(self):
        # 初始化 Embedding 模型
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,  # 會讀取 config 中的 "nomic-embed-text:latest"
            base_url=settings.OLLAMA_BASE_URL,  # 公司伺服器網址
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {settings.OLLAMA_API_KEY}"
                }
            }
        )
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
        """將文件存入向量資料庫 (同步方法)"""
        if docs:
            try:
                self.db.add_documents(docs)
                logger.info(f"成功存入 {len(docs)} 筆向量資料片段")
            except Exception as e:
                logger.error(f"存入向量資料庫失敗: {e}")
                raise e

    async def process_file(self, file_path: str):
        """核心流程：支援 PDF 逐頁讀取 (Lazy Loading) 防 OOM 引擎，其他檔案維持智能解析"""
        try:
            filename = os.path.basename(file_path)
            file_ext = filename.lower().split('.')[-1]

            # 升級改造：針對 PDF 啟用「一頁一頁讀取」的 Lazy Load 模式
            if file_ext == 'pdf':
                from langchain_community.document_loaders import PyMuPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                logger.info(f"啟動 PDF 逐頁解析引擎: {filename}")

                loader = PyMuPDFLoader(file_path)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                page_count = 0
                global_chunk_count = 0  # 關鍵新增：全域片段計數器，用來維持文章順序！

                for page_doc in loader.lazy_load():
                    page_count += 1
                    logger.info(f"正在處理 {filename} 的第 {page_count} 頁...")

                    # --- 文字清洗機 ---
                    raw_text = page_doc.page_content
                    clean_text = re.sub(r'-\n\s*', '', raw_text)  # 修復跨行斷字
                    clean_text = re.sub(r'(?:\b[a-zA-Z]{1,2}\b\s+){4,}', ' [數學公式/圖表] ', clean_text)  # 碎字殺手
                    clean_text = re.sub(r'\s+', ' ', clean_text)  # 壓縮空白
                    page_doc.page_content = clean_text

                    # --- 強化 Metadata ---
                    page_doc.metadata["source"] = filename
                    page_doc.metadata["page"] = page_doc.metadata.get("page", page_count)

                    # --- 切塊並打上順序 ID ---
                    chunks = text_splitter.split_documents([page_doc])
                    for chunk in chunks:
                        chunk.metadata["chunk_id"] = global_chunk_count
                        global_chunk_count += 1

                    # 存入資料庫
                    if chunks:
                        self.add_documents(chunks)

                logger.info(f"PDF '{filename}' 處理完成，共安全解析 {page_count} 頁")

            # 其他檔案 (如 txt, csv, docx) 維持 SmartParser 邏輯
            else:
                from app.services.file_service import FileLoaderFactory

                loader = FileLoaderFactory.get_loader(file_path, filename)
                text_content = loader.extract_text()

                if not text_content:
                    logger.warning(f"檔案 {filename} 無內容或無法讀取，跳過處理")
                    return

                logger.info(f"啟動 SmartFileParser 解析檔案: {filename}")
                parser = SmartFileParser()
                docs = parser.parse(text_content, filename)

                if docs:
                    self.add_documents(docs)
                    logger.info(f" 檔案 '{filename}' 處理完成，共存入 {len(docs)} 筆結構化資料")
                else:
                    logger.warning(f"檔案 '{filename}' 解析後無有效資料片段")

        except Exception as e:
            logger.error(f" 處理檔案失敗 {file_path}: {e}")
            raise e

    def search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None):
        """執行向量相似度搜尋 (已修復重複邏輯)"""
        if filter:
            return self.db.similarity_search(query, k=k, filter=filter)
        return self.db.similarity_search(query, k=k)

    def list_sources(self):
        """列出目前資料庫中所有不重複的檔案名稱"""
        try:
            data = self.db.get(include=['metadatas'])
            metadatas = data.get("metadatas", [])
            sources = {m["source"] for m in metadatas if m and "source" in m}
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []

    def delete_file(self, filename: str):
        """刪除指定檔案的所有向量資料"""
        try:
            data = self.db.get(where={"source": filename})
            ids = data.get("ids", [])
            if ids:
                self.db.delete(ids)
                logger.info(f"已刪除檔案 '{filename}'，共移除 {len(ids)} 筆向量片段")
                return True
            else:
                logger.warning(f"找不到檔案 '{filename}' 的資料")
                return False
        except Exception as e:
            logger.error(f"刪除檔案失敗: {e}")
            raise e

    def get_file_content(self, filename: str) -> str:
        """讀取指定檔案的完整內容 (將切片縫合，用於前端預覽)"""
        try:
            data = self.db.get(where={"source": filename})
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            if not documents:
                return "無內容或是圖片檔案 (未儲存純文字)。"

            # 根據 chunk_id 排序，確保縫合後的文字順序 100% 正確
            combined = sorted(zip(documents, metadatas), key=lambda x: x[1].get('chunk_id', 0) if x[1] else 0)
            sorted_docs = [doc for doc, meta in combined]

            return "\n\n-------------------\n\n".join(sorted_docs)

        except Exception as e:
            logger.error(f"讀取檔案內容失敗: {e}")
            return f"讀取錯誤: {str(e)}"

    def reset(self):
        """強制清空資料庫 (Purge System)"""
        try:
            all_ids = self.db.get()['ids']
            if all_ids:
                batch_size = 5000
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    self.db.delete(batch_ids)
                logger.info(f"️已從 Chroma 刪除 {len(all_ids)} 筆資料")

            self.db = None
            self._init_db()
            logger.info("資料庫重置完成")

        except Exception as e:
            logger.error(f"Reset failed: {e}")