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
from app.core.config import settings

#  移除舊的 LangChain 切割器引用 (因為已經封裝進 process_file 了)
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

router = APIRouter()
logger = logging.getLogger(__name__)

# 初始化服務
chat_service = ChatService()

# 🔥 設定檔案儲存目錄
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 定義 Request Schema
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    model_name: str = "gpt-oss:20b"
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
        # 回傳預設值防止前端壞掉
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
    elif lower_name.endswith((".py", ".js", ".html", ".css", ".json", ".md")):
        media_type = "text/plain"  # 程式碼也當文字看

    return FileResponse(file_path, media_type=media_type, filename=filename, content_disposition_type="inline")


@router.get("/files/{filename}/content")
async def view_file_content(filename: str):
    """檢視檔案內容 (純文字模式，從 VectorStore 拼湊回來)"""
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
            # 即使資料庫沒找到，只要檔案刪了也算成功
            if os.path.exists(file_path) == False:
                return {"status": "success", "message": f"File {filename} deleted (was not in DB)"}
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
    """
    🔥 核心修改：檔案上傳與處理
    改用 vector_store.process_file 來觸發 Smart Parsing
    """
    try:
        vs = VectorStoreService.get_instance()
        processed_files = []
        error_files = []

        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)

            try:
                # 1. 永久儲存檔案
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # 2. 🔥 呼叫 process_file (這裡會去跑 SmartFileParser)
                # 這是最關鍵的一步！
                await vs.process_file(file_path)

                processed_files.append(file.filename)

            except Exception as e:
                logger.error(f"處理失敗 {file.filename}: {e}")
                error_files.append(file.filename)
                # 如果處理失敗，順便把殘留檔案刪掉
                if os.path.exists(file_path):
                    os.remove(file_path)

        return {
            "status": "success",
            "processed": processed_files,
            "errors": error_files,
            "message": f"成功處理 {len(processed_files)} 個檔案，結構化解析完成。"
        }

    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))