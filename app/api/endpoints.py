# app/api/endpoints.py
import shutil
import os
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import httpx

# 引入服務
from app.services.chat_service import ChatService
from app.services.vector_store import VectorStoreService
from app.services.file_service import FileLoaderFactory
from app.core.config import settings

# 引入 LangChain 切割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

router = APIRouter()
logger = logging.getLogger(__name__)

# 初始化服務
chat_service = ChatService()

# 🔥 設定檔案儲存目錄 (請確保此資料夾存在)
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 定義 Request Schema
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    model_name: str = "llama3.2"
    history: List[Message] = []
    images: List[str] = []


# ==========================================
# 1. 聊天與模型相關 API
# ==========================================

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """對話 API (包含歷史紀錄改寫)"""
    try:
        history_data = [m.model_dump() for m in request.history]
        return StreamingResponse(
            chat_service.process_query(request.query, history_data),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_models():
    """從 Ollama 伺服器動態抓取模型列表"""
    try:
        base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        target_url = f"{base_url}/api/tags"

        async with httpx.AsyncClient() as client:
            response = await client.get(target_url, timeout=5.0)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ollama 回應錯誤: {response.status_code}")
                return {"models": []}
    except Exception as e:
        logger.error(f"無法連線到 Ollama: {str(e)}")
        return {"models": [{"name": "gpt-oss:20b", "details": {"parameter_size": "20B"}}]}


# ==========================================
# 2. 檔案管理 API (CRUD & View)
# ==========================================

@router.get("/files")
async def list_files():
    """取得目前資料庫中的檔案列表"""
    try:
        vs = VectorStoreService.get_instance()
        files = vs.list_sources()
        return {"files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{filename}/view")
async def view_file(filename: str):
    """🔥 讓瀏覽器直接預覽檔案 (PDF, 圖片等)"""
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    # 簡單的 MIME type 判斷
    media_type = "application/octet-stream"
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        media_type = "application/pdf"
    elif lower_name.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif lower_name.endswith(".png"):
        media_type = "image/png"
    elif lower_name.endswith(".txt"):
        media_type = "text/plain"

    return FileResponse(file_path, media_type=media_type, filename=filename, content_disposition_type="inline")


@router.get("/files/{filename}/content")
async def view_file_content(filename: str):
    """檢視檔案內容 (純文字模式，保留給舊功能或 Debug 用)"""
    try:
        vs = VectorStoreService.get_instance()
        content = vs.get_file_content(filename)
        return {"filename": filename, "content": content}
    except Exception as e:
        logger.error(f"View content error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """刪除指定檔案 (同時刪除向量資料庫與實體檔案)"""
    try:
        # 1. 刪除向量資料庫
        vs = VectorStoreService.get_instance()
        success = vs.delete_file(filename)

        # 2. 刪除實體檔案 (如果存在)
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"實體檔案 {filename} 已刪除")

        if success:
            return {"status": "success", "message": f"File {filename} deleted"}
        else:
            raise HTTPException(status_code=404, detail="File not found in database")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Delete file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_database():
    """全域重置資料庫 (清空所有檔案)"""
    try:
        # 1. 重置 ChromaDB
        vs = VectorStoreService.get_instance()
        vs.reset()

        # 2. 清空 uploads 資料夾 (保留資料夾本身)
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")

        return {"message": "✅ 系統記憶與檔案已完全重置"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """檔案上傳與處理"""
    try:
        vs = VectorStoreService.get_instance()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        processed_files = []
        error_files = []

        for file in files:
            # 🔥 修改：直接存到 uploads 資料夾
            file_path = os.path.join(UPLOAD_DIR, file.filename)

            try:
                # 1. 永久儲存檔案 (為了預覽與時間戳記)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # 2. 解析檔案
                loader = FileLoaderFactory.get_loader(file_path, file.filename)
                raw_text = loader.extract_text()

                if not raw_text:
                    logger.warning(f"檔案 {file.filename} 無文字內容，跳過")
                    error_files.append(file.filename)
                    continue

                # 3. 切割與向量化
                chunks = text_splitter.split_text(raw_text)
                docs = [Document(page_content=c, metadata={"source": file.filename}) for c in chunks]

                if docs:
                    vs.add_documents(docs)
                    processed_files.append(file.filename)

            except Exception as e:
                logger.error(f"處理失敗 {file.filename}: {e}")
                error_files.append(file.filename)

        return {
            "status": "success",
            "processed": processed_files,
            "errors": error_files
        }

    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))