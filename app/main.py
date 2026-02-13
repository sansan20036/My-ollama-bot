# app/main.py
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.services.vector_store import VectorStoreService

# 設定 Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定義上傳資料夾路徑 (必須與 endpoints.py 一致)
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================================
    # 🟢 啟動時執行：自動同步檢查 (Auto-Sync)
    # ==========================================
    logger.info("🚀 系統啟動中... 正在執行檔案一致性檢查...")

    try:
        # 1. 初始化資料庫服務
        vs = VectorStoreService.get_instance()

        # 2. 取得資料庫內所有檔案清單 (帳本)
        db_files = vs.list_sources()
        logger.info(f"📋 資料庫紀錄檔案數: {len(db_files)}")

        removed_count = 0

        # 3. 逐一檢查硬碟裡有沒有這些檔案 (盤點)
        for filename in db_files:
            file_path = os.path.join(UPLOAD_DIR, filename)

            # 如果硬碟裡找不到這個檔案...
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ 發現幽靈檔案 (無實體): {filename} -> 正在自動移除...")

                # ...就從資料庫中刪除它
                vs.delete_file(filename)
                removed_count += 1

        if removed_count > 0:
            logger.info(f"✅ 自動修復完成：已清除 {removed_count} 個無效的檔案紀錄。")
        else:
            logger.info("✅ 系統健康：資料庫與硬碟檔案完全同步。")

    except Exception as e:
        logger.error(f"❌ 啟動檢查失敗: {e}")

    yield  # 這裡代表伺服器正在運作中...

    # 🔴 關閉時執行
    logger.info("🛑 系統關閉")


# 建立 App
app = FastAPI(title="Ollama RAG API", lifespan=lifespan)

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

    uvicorn.run(app, host="0.0.0.0", port=8000)