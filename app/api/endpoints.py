# app/api/endpoints.py
import logging
import os
import re
import shutil
from typing import List, Optional
from uuid import uuid4

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings
from app.models.schemas import ChatRequest, ScheduleDebugRequest, ScheduleSmokeRequest
from app.services.chat_service import ChatService
from app.services.file_service import FileService
from app.services.table_analyzer_service import TableAnalyzerService
from app.services.vector_store import VectorStoreService

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = settings.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_chat_service() -> ChatService:
    # Fallback only: normally we use app.state singleton set in main lifespan.
    return ChatService()


def get_file_service() -> FileService:
    # Fallback only: normally we use app.state singleton set in main lifespan.
    return FileService()


def get_chat_service_from_app(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        service = get_chat_service()
        request.app.state.chat_service = service
    return service


def get_file_service_from_app(request: Request) -> FileService:
    service = getattr(request.app.state, "file_service", None)
    if service is None:
        service = get_file_service()
        request.app.state.file_service = service
    return service


def _load_dataframe_from_file(chat_service: ChatService, target_file: str) -> Optional[pd.DataFrame]:
    """Load structured table DataFrame from cache/csv/xlsx/csv file."""
    if not os.path.exists(target_file):
        return None

    current_mtime = os.path.getmtime(target_file)
    if (
        chat_service.cached_file_path == target_file
        and chat_service.cached_file_mtime == current_mtime
        and chat_service.cached_df is not None
    ):
        return chat_service.cached_df

    file_name_without_ext = os.path.splitext(target_file)[0]
    file_ext = target_file.lower().split(".")[-1]
    possible_csv = f"{file_name_without_ext}_tables.csv"
    df = None

    if os.path.exists(possible_csv):
        df = pd.read_csv(possible_csv)
    elif file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(target_file)
        df.columns = [re.split(r"[\s\n(]", str(col))[0] for col in df.columns]
    elif file_ext == "csv":
        df = pd.read_csv(target_file)
        df.columns = [re.split(r"[\s\n(]", str(col))[0] for col in df.columns]

    chat_service.cached_df = df
    chat_service.cached_file_path = target_file
    chat_service.cached_file_mtime = current_mtime
    return df


def _pick_target_filename(valid_files: List[str], requested_filename: Optional[str]) -> str:
    if not valid_files:
        raise HTTPException(status_code=400, detail="No uploaded files found.")
    if requested_filename:
        safe_name = os.path.basename(requested_filename.strip())
        if safe_name not in valid_files:
            raise HTTPException(status_code=404, detail=f"File not found in uploads: {safe_name}")
        return safe_name
    return valid_files[-1]


async def _run_schedule_debug(chat_service: ChatService, query: str, requested_filename: Optional[str]) -> dict:
    valid_files = chat_service._get_valid_files()
    target_name = _pick_target_filename(valid_files, requested_filename)
    target_file = os.path.join(UPLOAD_DIR, target_name)

    df = _load_dataframe_from_file(chat_service, target_file)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail=f"Target file has no structured table data: {target_name}")

    constraints = TableAnalyzerService._extract_constraints(query)
    normalized_query = TableAnalyzerService._normalize_query_for_codegen(query)
    inferred_dept = TableAnalyzerService._infer_department_keyword(query)
    fallback_rows = TableAnalyzerService._fallback_by_department(df, normalized_query)
    result_text = await TableAnalyzerService.query_and_format_schedule(df=df, query=query, llm=chat_service.llm)

    return {
        "query": query,
        "target_file": target_name,
        "normalized_query": normalized_query,
        "inferred_department_keyword": inferred_dept,
        "constraints": {
            "include_days": sorted(list(constraints.get("include_days", set()))),
            "exclude_days": sorted(list(constraints.get("exclude_days", set()))),
            "include_periods": sorted(list(constraints.get("include_periods", set()))),
            "exclude_periods": sorted(list(constraints.get("exclude_periods", set()))),
            "surname": constraints.get("surname"),
        },
        "dataframe_shape": [int(df.shape[0]), int(df.shape[1])],
        "fallback_row_count": len(fallback_rows),
        "final_result": result_text,
    }


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service_from_app),
):
    """Streaming chat endpoint."""
    try:
        return StreamingResponse(
            chat_service.process_query(
                query=request.query,
                history=request.history,
                images=request.images,
                model_name=request.model_name,
                session_id=request.session_id,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_models():
    """Fetch model list from Ollama."""
    try:
        base_url = getattr(settings, "OLLAMA_BASE_URL", "http://git.tedpc.com.tw:11434").rstrip("/")
        target_url = f"{base_url}/api/tags"

        headers = {"Content-Type": "application/json"}
        api_key = (getattr(settings, "OLLAMA_API_KEY", "") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.get(target_url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                return response.json()

            logger.error(f"Ollama response error: {response.status_code} - {response.text}")
            return {"models": []}
    except Exception as e:
        logger.error(f"Failed to query Ollama models: {e}")
        return {"models": []}


@router.get("/files")
async def list_files(session_id: Optional[str] = None):
    """List indexed files (optionally scoped by upload session_id)."""
    try:
        vs = VectorStoreService.get_instance()
        vs.cleanup_orphan_documents()
        sid = (session_id or "").strip()
        if sid:
            files = vs.list_sources_by_session(sid, require_uploaded_file=True)
        else:
            files = vs.list_sources()
        return {"files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{filename}/view")
async def view_file(filename: str):
    """Preview file in browser."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    media_type = "application/octet-stream"
    lower_name = safe_filename.lower()

    if lower_name.endswith(".pdf"):
        media_type = "application/pdf"
    elif lower_name.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif lower_name.endswith(".png"):
        media_type = "image/png"
    elif lower_name.endswith(".webp"):
        media_type = "image/webp"
    elif lower_name.endswith(".bmp"):
        media_type = "image/bmp"
    elif lower_name.endswith(".gif"):
        media_type = "image/gif"
    elif lower_name.endswith((".tif", ".tiff")):
        media_type = "image/tiff"
    elif lower_name.endswith(".txt"):
        media_type = "text/plain"
    elif lower_name.endswith((".py", ".js", ".html", ".css", ".json", ".md")):
        media_type = "text/plain"

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=safe_filename,
        content_disposition_type="inline",
    )


@router.get("/files/{filename}/content")
async def view_file_content(filename: str):
    """Fetch reconstructed text content from vector store."""
    try:
        safe_filename = os.path.basename(filename)
        vs = VectorStoreService.get_instance()
        content = vs.get_file_content(safe_filename)
        return {"filename": safe_filename, "content": content}
    except Exception as e:
        logger.error(f"View content error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete file from vector DB and uploads folder."""
    try:
        safe_filename = os.path.basename(filename)

        vs = VectorStoreService.get_instance()
        success = vs.delete_file(safe_filename)

        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        csv_path = file_path.rsplit(".", 1)[0] + "_tables.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)

        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Physical file deleted: {safe_filename}")

        if success:
            return {"status": "success", "message": f"File {safe_filename} deleted"}
        if not os.path.exists(file_path):
            return {"status": "success", "message": f"File {safe_filename} deleted (was not in DB)"}
        raise HTTPException(status_code=404, detail="File not found in database")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_database(chat_service: ChatService = Depends(get_chat_service_from_app)):
    """Global reset: clear vector DB and uploads."""
    try:
        vs = VectorStoreService.get_instance()
        vs.reset()

        # Clear chat-side caches if available.
        if hasattr(chat_service, "cached_df"):
            chat_service.cached_df = None
        if hasattr(chat_service, "cached_file_path"):
            chat_service.cached_file_path = ""
        if hasattr(chat_service, "cached_file_mtime"):
            chat_service.cached_file_mtime = 0

        if os.path.exists(UPLOAD_DIR):
            for name in os.listdir(UPLOAD_DIR):
                path = os.path.join(UPLOAD_DIR, name)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    logger.error(f"Failed to delete {path}: {e}")

        return {"message": "System memory, cache, and uploaded files have been reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    file_service: FileService = Depends(get_file_service_from_app),
):
    """Upload files and process into vector store."""
    try:
        processed_files = []
        error_files = []
        effective_session_id = (session_id or "").strip() or uuid4().hex

        for file in files:
            try:
                await file_service.upload_file(file, session_id=effective_session_id)
                processed_files.append(file.filename)
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                error_files.append(file.filename)

        return {
            "status": "success",
            "processed": processed_files,
            "errors": error_files,
            "session_id": effective_session_id,
            "message": f"Processed {len(processed_files)} file(s).",
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debug/schedule-eval")
async def debug_schedule_eval(
    payload: ScheduleDebugRequest,
    chat_service: ChatService = Depends(get_chat_service_from_app),
):
    """Run one schedule query in debug mode and return intermediate filters/constraints."""
    try:
        q = (payload.query or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="query is required")
        result = await _run_schedule_debug(chat_service, q, payload.filename)
        return {"status": "ok", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug schedule eval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debug/schedule-smoke")
async def debug_schedule_smoke(
    payload: ScheduleSmokeRequest,
    chat_service: ChatService = Depends(get_chat_service_from_app),
):
    """
    Run default smoke tests for schedule filtering:
    1) 下下週週末 + 姓氏 + 科別
    2) 科別 + 排除星期 + 排除時段
    """
    try:
        queries = payload.queries or [
            "幫我查下下週週末姓林的骨科醫生",
            "我要看腸胃科，但不要星期一也不要上午，下週有哪些選擇",
        ]
        results = []
        for q in queries:
            results.append(await _run_schedule_debug(chat_service, q, payload.filename))
        return {"status": "ok", "count": len(results), "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug schedule smoke error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

