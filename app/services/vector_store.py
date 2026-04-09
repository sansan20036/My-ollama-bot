# app/services/vector_store.py
import logging
import os
import shutil
import re
from typing import List, Optional, Dict, Any, Set

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from app.core.config import settings
from app.utils.smart_parser import SmartFileParser

logger = logging.getLogger(__name__)


class VectorStoreService:
    _instance = None

    def __init__(self):
        # 初始化 Embedding 模型 (使用新版的 kwargs 寫法)
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {settings.OLLAMA_API_KEY}"
                }
            }
        )
        self.db = None
        self._init_db()

    def _init_db(self):
        """初始化 ChromaDB 連線"""
        # 確保目錄存在
        os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
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

    # 🚀 將開發期清理邏輯獨立成一個方法，避免意外觸發
    def purge_development_data(self):
        """【開發期神兵利器】徹底清除舊資料庫與上傳檔案。僅建議在系統啟動時呼叫一次。"""
        logger.warning(f"🧹 [開發模式] 啟動物理刪除程序...")

        # 1. 清除 ChromaDB 目錄
        if os.path.exists(settings.CHROMA_DB_DIR):
            try:
                # 斷開目前的連線 (重要！)
                self.db = None
                shutil.rmtree(settings.CHROMA_DB_DIR, ignore_errors=True)
                logger.info("💥 舊資料庫已徹底超渡完畢！")
            except Exception as e:
                logger.error(f"刪除舊資料庫失敗，檔案可能被佔用: {e}")

        # 2. 清除 Uploads 目錄 (🚀 使用 settings.UPLOAD_DIR)
        if os.path.exists(settings.UPLOAD_DIR):
            for filename in os.listdir(settings.UPLOAD_DIR):
                file_path = os.path.join(settings.UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"無法刪除檔案 {file_path}: {e}")
            logger.info("💥 舊上傳檔案已清空！")

        # 重新初始化連線
        self._init_db()

    def add_documents(self, docs: List[Document]):
        """將文件存入向量資料庫 (同步方法)"""
        if docs:
            try:
                self.db.add_documents(docs)
                logger.info(f"成功存入 {len(docs)} 筆向量資料片段")
            except Exception as e:
                logger.error(f"存入向量資料庫失敗: {e}")
                raise e

    async def process_file(self, file_path: str, session_id: Optional[str] = None):
        """核心流程：處理非 PDF 的其他檔案"""
        try:
            filename = os.path.basename(file_path)
            file_ext = filename.lower().split('.')[-1]

            if file_ext == 'pdf':
                logger.warning(f"⚠️ 警告：{filename} 誤入舊版處理區塊，將略過處理。請確保 PDF 交由 LlamaParse 處理。")
                return
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
                    # 寫入上傳批次識別，供多檔案同批關聯檢索。
                    sid = (session_id or "").strip()
                    for doc in docs:
                        if not isinstance(doc.metadata, dict):
                            doc.metadata = {}
                        doc.metadata.setdefault("source", file_path)
                        doc.metadata.setdefault("filename", filename)
                        if sid:
                            doc.metadata["upload_session_id"] = sid

                    self.add_documents(docs)
                    logger.info(f" 檔案 '{filename}' 處理完成，共存入 {len(docs)} 筆結構化資料")
                else:
                    logger.warning(f"檔案 '{filename}' 解析後無有效資料片段")

        except Exception as e:
            logger.error(f" 處理檔案失敗 {file_path}: {e}")
            raise e

    def search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None):
        """執行向量相似度搜尋"""
        if not self.db:
            self._init_db()
        if filter:
            return self.db.similarity_search(query, k=k, filter=filter)
        return self.db.similarity_search(query, k=k)

    def _safe_get(
        self,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """包裝 Chroma get，避免外部重複 try/except。"""
        if not self.db:
            self._init_db()
        kwargs: Dict[str, Any] = {}
        if where:
            kwargs["where"] = where
        if include:
            kwargs["include"] = include
        if isinstance(limit, int) and limit > 0:
            kwargs["limit"] = limit
        try:
            return self.db.get(**kwargs)
        except TypeError:
            # 某些 Chroma 版本不支援 limit 參數，降級重試。
            kwargs.pop("limit", None)
            return self.db.get(**kwargs)

    def get_file_documents(
        self,
        filename: str,
        include_documents: bool = True,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        以 filename 精準取得向量資料，優先使用 where 過濾，避免全庫掃描。
        """
        safe_name = str(filename or "").strip()
        include = ["metadatas"]
        if include_documents:
            include = ["documents", "metadatas"]

        empty_docs = [] if include_documents else None
        if not safe_name:
            return {"ids": [], "metadatas": [], "documents": empty_docs}

        source_path = os.path.join(settings.UPLOAD_DIR, safe_name)
        source_path_norm = source_path.replace("\\", "/")
        where_candidates = [
            {"filename": safe_name},
            {"source": source_path},
            {"source": source_path_norm},
        ]

        collected: Dict[str, Dict[str, Any]] = {}
        for where in where_candidates:
            try:
                data = self._safe_get(where=where, include=include, limit=limit)
            except Exception:
                continue

            ids = data.get("ids", []) or []
            metadatas = data.get("metadatas", []) or []
            documents = data.get("documents", []) or []
            for idx, doc_id in enumerate(ids):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                doc = documents[idx] if include_documents and idx < len(documents) else None
                collected[str(doc_id)] = {"meta": meta, "doc": doc}

            if isinstance(limit, int) and limit > 0 and len(collected) >= limit:
                break

        # fallback：極端情況下 where 無法命中時，才退回全庫過濾。
        if not collected:
            try:
                data = self._safe_get(include=include)
                ids = data.get("ids", []) or []
                metadatas = data.get("metadatas", []) or []
                documents = data.get("documents", []) or []
                for idx, doc_id in enumerate(ids):
                    meta = metadatas[idx] if idx < len(metadatas) else {}
                    if not isinstance(meta, dict):
                        continue
                    source_val = str(meta.get("source", "")).strip()
                    filename_val = str(meta.get("filename", "")).strip()
                    source_name = os.path.basename(source_val.replace("\\", "/"))
                    if safe_name not in {filename_val, source_name}:
                        continue
                    doc = documents[idx] if include_documents and idx < len(documents) else None
                    collected[str(doc_id)] = {"meta": meta, "doc": doc}
                    if isinstance(limit, int) and limit > 0 and len(collected) >= limit:
                        break
            except Exception as e:
                logger.warning(f"get_file_documents fallback 失敗({safe_name}): {e}")

        ids_out = list(collected.keys())
        metas_out = [collected[i]["meta"] for i in ids_out]
        docs_out = [collected[i]["doc"] for i in ids_out] if include_documents else None
        return {
            "ids": ids_out,
            "metadatas": metas_out,
            "documents": docs_out,
        }

    def get_schedule_slot_documents(
        self,
        file_path: str = "",
        filename: str = "",
        slot_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        取得結構化門診 slot 文件，優先走 metadata where 過濾。
        """
        slot_types = slot_types or ["schedule_slot_local", "schedule_slot"]
        target_filename = str(filename or "").strip()
        target_path = str(file_path or "").strip()
        target_path_norm = target_path.replace("\\", "/") if target_path else ""
        if not target_filename and target_path:
            target_filename = os.path.basename(target_path_norm)

        collected: Dict[str, Dict[str, Any]] = {}
        for slot_type in slot_types:
            where_candidates = []
            if target_filename:
                where_candidates.append({"type": slot_type, "filename": target_filename})
            if target_path:
                where_candidates.append({"type": slot_type, "source": target_path})
            if target_path_norm and target_path_norm != target_path:
                where_candidates.append({"type": slot_type, "source": target_path_norm})
            if not where_candidates:
                where_candidates.append({"type": slot_type})

            for where in where_candidates:
                try:
                    data = self._safe_get(where=where, include=["documents", "metadatas"], limit=limit)
                except Exception:
                    continue

                ids = data.get("ids", []) or []
                metadatas = data.get("metadatas", []) or []
                documents = data.get("documents", []) or []
                for idx, doc_id in enumerate(ids):
                    meta = metadatas[idx] if idx < len(metadatas) else {}
                    doc = documents[idx] if idx < len(documents) else None
                    if target_filename:
                        source_val = str((meta or {}).get("source", "")).strip()
                        source_name = os.path.basename(source_val.replace("\\", "/"))
                        filename_val = str((meta or {}).get("filename", "")).strip()
                        if target_filename not in {source_name, filename_val}:
                            continue
                    collected[str(doc_id)] = {"meta": meta, "doc": doc}
                    if isinstance(limit, int) and limit > 0 and len(collected) >= limit:
                        break
                if isinstance(limit, int) and limit > 0 and len(collected) >= limit:
                    break
            if isinstance(limit, int) and limit > 0 and len(collected) >= limit:
                break

        ids_out = list(collected.keys())
        return {
            "ids": ids_out,
            "metadatas": [collected[i]["meta"] for i in ids_out],
            "documents": [collected[i]["doc"] for i in ids_out],
        }

    def list_schedule_departments(self) -> List[str]:
        """
        快速列出門診資料中的科別，供查詢意圖對齊使用。
        """
        deps: Set[str] = set()
        for slot_type in ["schedule_slot_local", "schedule_slot"]:
            try:
                data = self._safe_get(where={"type": slot_type}, include=["metadatas"])
            except Exception:
                continue
            for meta in data.get("metadatas", []) or []:
                if not isinstance(meta, dict):
                    continue
                dep = str(meta.get("department") or meta.get("dept") or meta.get("科別") or "").strip()
                if dep:
                    deps.add(dep)
        return sorted(deps)

    def _get_uploaded_files_set(self) -> Set[str]:
        """取得 uploads 目錄中目前存在的真實檔案名稱集合。"""
        uploaded_files: Set[str] = set()
        if not os.path.exists(settings.UPLOAD_DIR):
            return uploaded_files

        for filename in os.listdir(settings.UPLOAD_DIR):
            file_path = os.path.join(settings.UPLOAD_DIR, filename)
            if not os.path.isfile(file_path):
                continue
            if filename.startswith("~") or filename.endswith("_tables.csv"):
                continue
            uploaded_files.add(filename)
        return uploaded_files

    def cleanup_orphan_documents(self) -> Dict[str, Any]:
        """
        清理向量庫中沒有對應 uploads 實體檔案的殘留切片。
        這會「實際刪除」資料庫中的孤兒向量，而不只是隱藏。
        """
        report: Dict[str, Any] = {
            "total_vectors": 0,
            "deleted_vectors": 0,
            "deleted_sources": [],
            "uploaded_files_count": 0,
        }

        try:
            if not self.db:
                self._init_db()

            data = self.db.get(include=['metadatas'])
            ids = data.get("ids", []) or []
            metadatas = data.get("metadatas", []) or []

            report["total_vectors"] = len(ids)

            if not ids:
                return report

            uploaded_files = self._get_uploaded_files_set()
            report["uploaded_files_count"] = len(uploaded_files)

            ids_to_delete = []
            orphan_sources = set()

            for idx, doc_id in enumerate(ids):
                meta = metadatas[idx] if idx < len(metadatas) else None
                candidate_names = set()

                if isinstance(meta, dict):
                    filename_val = str(meta.get("filename", "")).strip()
                    source_val = str(meta.get("source", "")).strip()
                    if filename_val:
                        candidate_names.add(os.path.basename(filename_val))
                    if source_val:
                        candidate_names.add(os.path.basename(source_val))

                if not candidate_names:
                    ids_to_delete.append(doc_id)
                    orphan_sources.add("(missing_metadata)")
                    continue

                if not any(name in uploaded_files for name in candidate_names):
                    ids_to_delete.append(doc_id)
                    orphan_sources.update(candidate_names)

            if ids_to_delete:
                self.db.delete(ids_to_delete)
                report["deleted_vectors"] = len(ids_to_delete)
                report["deleted_sources"] = sorted(orphan_sources)
                logger.info(
                    f"已清理 orphan 向量: {len(ids_to_delete)} 筆，來源檔 {len(orphan_sources)} 個"
                )

            return report
        except Exception as e:
            logger.error(f"清理 orphan 向量失敗: {e}")
            report["error"] = str(e)
            return report

    def list_sources(self, require_uploaded_file: bool = True):
        """列出目前資料庫中所有不重複的檔案名稱。

        預設只回傳「向量庫有內容，且 uploads 目錄仍存在的實體檔案」，
        避免殘留 metadata 讓前端 Knowledge Base 計數失真。
        """
        try:
            if not self.db: return []
            data = self.db.get(include=['metadatas'])
            metadatas = data.get("metadatas", [])
            sources = set()
            for m in metadatas:
                if m:
                    name = m.get("filename") or os.path.basename(m.get("source", ""))
                    if name:
                        sources.add(name)

            if require_uploaded_file:
                uploaded_files = self._get_uploaded_files_set()
                sources = {name for name in sources if name in uploaded_files}

            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []

    def list_sources_by_session(self, session_id: str, require_uploaded_file: bool = True):
        """
        列出指定 upload_session_id 的檔案清單。
        """
        sid = str(session_id or "").strip()
        if not sid:
            return self.list_sources(require_uploaded_file=require_uploaded_file)

        try:
            if not self.db:
                return []

            data = self._safe_get(where={"upload_session_id": sid}, include=['metadatas'])
            metadatas = data.get("metadatas", [])
            sources = set()

            for m in metadatas:
                if not isinstance(m, dict):
                    continue
                meta_sid = str(m.get("upload_session_id", "")).strip()
                if meta_sid != sid:
                    continue
                name = m.get("filename") or os.path.basename(str(m.get("source", "")))
                if name:
                    sources.add(name)

            if require_uploaded_file:
                uploaded_files = self._get_uploaded_files_set()
                sources = {name for name in sources if name in uploaded_files}

            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error listing session sources: {e}")
            return []

    def delete_file(self, filename: str):
        """刪除指定檔案的所有向量資料"""
        try:
            if not self.db: return False
            data = self.get_file_documents(filename, include_documents=False)
            ids_to_delete = data.get("ids", []) or []

            if ids_to_delete:
                self.db.delete(ids_to_delete)
                logger.info(f"已刪除檔案 '{filename}'，共移除 {len(ids_to_delete)} 筆向量片段")
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
            if not self.db: return "資料庫未連線。"
            data = self.get_file_documents(filename, include_documents=True)
            documents = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []

            if not documents:
                return "無內容或是圖片檔案 (未儲存純文字)。"

            combined = sorted(zip(documents, metadatas), key=lambda x: x[1].get('chunk_id', 0) if x[1] else 0)
            sorted_docs = [doc for doc, meta in combined]

            return "\n\n-------------------\n\n".join(sorted_docs)

        except Exception as e:
            logger.error(f"讀取檔案內容失敗: {e}")
            return f"讀取錯誤: {str(e)}"

    def reset(self):
        """強制清空資料庫 (Purge System) - 物理核爆版"""
        try:
            logger.info("準備執行物理重置，銷毀舊的 Collection 維度...")
            # 1. 斷開目前的連線，釋放檔案鎖定
            self.db = None

            # 2. 物理刪除整個資料庫資料夾 (連同 384 維的記憶一起炸掉)
            if os.path.exists(settings.CHROMA_DB_DIR):
                shutil.rmtree(settings.CHROMA_DB_DIR, ignore_errors=True)

            # 3. 重新建立全新的連線與 768 維的新 Collection
            self._init_db()
            logger.info("資料庫重置完成 (已徹底重建 Collection)！")

        except Exception as e:
            logger.error(f"Reset failed: {e}")
