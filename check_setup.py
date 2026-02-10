# check_setup.py
import sys
import os

# 1. 確保 Python 找得到 app 資料夾
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚀 開始系統自我檢查...\n")

try:
    # --- 測試 1: 讀取設定檔 ---
    print("1️⃣ [Config] 正在讀取設定檔...")
    from app.core.config import settings
    print(f"   ✅ 專案名稱: {settings.PROJECT_NAME}")
    print(f"   ✅ 資料庫路徑: {settings.CHROMA_DB_DIR}")
    print(f"   ✅ 快取路徑: {settings.CACHE_DB_DIR}")

    # --- 測試 2: 初始化向量資料庫服務 ---
    print("\n2️⃣ [VectorStore] 正在測試向量資料庫連線 (這可能會花幾秒鐘載入模型)...")
    from app.services.vector_store import VectorStoreService
    vector_store = VectorStoreService.get_instance()
    print("   ✅ VectorStoreService 初始化成功！")

    # --- 測試 3: 初始化快取服務 ---
    print("\n3️⃣ [SemanticCache] 正在測試語意快取...")
    from app.services.cache_service import SemanticCacheService
    cache = SemanticCacheService.get_instance()
    print("   ✅ SemanticCacheService 初始化成功！")

    print("\n🎉 恭喜！你的基礎架構重構非常完美，所有模組都能正常載入！")

except ImportError as e:
    print(f"\n❌ [Import Error] 模組匯入失敗：{e}")
    print("   👉 請檢查：")
    print("   1. 資料夾內是否有 __init__.py")
    print("   2. 檔名是否正確 (例如沒有空格)")
    print("   3. 類別名稱是否正確")
except Exception as e:
    print(f"\n❌ [System Error] 系統錯誤：{e}")
    import traceback
    traceback.print_exc()