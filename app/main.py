# app/main.py
import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.services.vector_store import VectorStoreService
from app.services.chat_service import ChatService
from app.services.file_service import FileService
# 🚀 引入集中管理的設定檔
from app.core.config import settings

# 設定 Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("系統啟動中... 檢查環境設定")

    try:
        # 1. 🚀 確保上傳資料夾存在 (改用 settings 統一管理的路徑)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # 2. 初始化向量資料庫
        vs = VectorStoreService.get_instance()

        # 3. 啟動時自動清理 orphan 向量，避免殘留資料干擾回答與前端計數
        cleanup_report = vs.cleanup_orphan_documents()
        if cleanup_report.get("deleted_vectors", 0) > 0:
            logger.info(
                f"啟動時已清理 orphan 向量 {cleanup_report['deleted_vectors']} 筆"
            )

        # ==========================================
        # 💡 啟動是否核爆資料由設定檔控制，預設為 False（保留歷史檔案）
        # ==========================================
        if getattr(settings, "PURGE_ON_STARTUP", False):
            logger.warning("設定啟動清空資料：PURGE_ON_STARTUP=True，執行核爆清理")
            vs.purge_development_data()
        else:
            logger.info("確認上傳資料夾準備完畢，目前為【持久化模式】(歷史資料將被保留)")

        # 4. 初始化全域單例 Service，避免每個 request 重複建構重物件
        app.state.chat_service = ChatService()
        app.state.file_service = FileService()

        logger.info("向量資料庫載入完成，準備就緒！")

    except Exception as e:
        logger.error(f"系統啟動時發生錯誤: {e}")

    yield  # 這裡代表伺服器正在運作中...

    # 關閉時執行
    logger.info("系統關閉")


# 建立 App (名稱改用 settings 統一管理)
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

# 後端開機紀錄這次伺服器啟動的時間
BOOT_TIME = str(time.time())


@app.get("/api/status")
async def get_status():
    """讓前端用來檢查後端是否重新啟動過"""
    return {"boot_time": BOOT_TIME}


# 設定 CORS (允許前端連線)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 載入路由
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    # 🚀 建議使用字串 "app.main:app" 啟動，這樣 reload 熱重載機制才會最穩定
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
