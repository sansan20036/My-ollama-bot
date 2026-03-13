# app/main.py
import os
import shutil
import logging
import time  # 新增這個模組，用來產生開機時間戳記
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

    # 啟動時執行：強制清空所有資料
    logger.info("系統啟動中... 準備執行強制環境初始化 (PURGE)")

    try:
        # 1. 無情清空硬碟裡的上傳資料夾 (刪除所有舊的 PDF 等檔案)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)  # 把整個資料夾連同裡面的檔案拔掉
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # 重新建立一個乾淨的空資料夾
        logger.info("[大掃除] 已清空本機上傳檔案資料夾")

        # 2. 清空 ChromaDB 向量資料庫
        vs = VectorStoreService.get_instance()

        # 假設VectorStoreService 有 reset() 函數可以清空資料庫
        if hasattr(vs, 'reset'):
            vs.reset()
            logger.info("[大掃除] 已重置向量資料庫 (ChromaDB)")
        else:
            # 備用方案：如果沒有 reset 函數，就逐一刪除清單內的檔案
            db_files = vs.list_sources()
            for filename in db_files:
                vs.delete_file(filename)
            logger.info(f" [大掃除] 已逐一刪除 {len(db_files)} 筆資料庫紀錄")

    except Exception as e:
        logger.error(f" [大掃除] 清理舊資料時發生錯誤: {e}")
    logger.info(" 系統啟動完成，目前為 100% 乾淨的全新狀態！")
    yield  # 這裡代表伺服器正在運作中...

    #關閉時執行
    logger.info(" 系統關閉")


# 建立 App
app = FastAPI(title="Ollama RAG API", lifespan=lifespan)

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

    uvicorn.run(app, host="0.0.0.0", port=8000)