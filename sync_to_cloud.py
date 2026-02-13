import chromadb
from chromadb.config import Settings

# 1. 配置設定
LOCAL_PATH = "./chroma_db"          # 你本地資料夾
COLLECTION_NAME = "medical_sample_demo"
REMOTE_HOST = "git.tedpc.com.tw"      # 例如: git.tedpc.com.tw
REMOTE_PORT = 8000                 # Chroma 預設通常是 8000

print(f" 正在讀取本地資料庫: {LOCAL_PATH}...")

# 2. 連接本地端 (PersistentClient)
local_client = chromadb.PersistentClient(path=LOCAL_PATH)
try:
    local_col = local_client.get_collection(name=COLLECTION_NAME)
    # 抓出所有資料 (包含向量、文檔、元數據)
    local_data = local_col.get(include=['embeddings', 'documents', 'metadatas'])
    total_items = len(local_data['ids'])
    print(f" 成功提取 {total_items} 筆資料。")
except Exception as e:
    print(f" 讀取本地 Collection 失敗: {e}")
    exit()

# 3. 連接雲端端 (HttpClient)
print(f" 正在連接雲端 ChromaDB ({REMOTE_HOST}:{REMOTE_PORT})...")
try:
    remote_client = chromadb.HttpClient(host=REMOTE_HOST, port=REMOTE_PORT)
    remote_col = remote_client.get_or_create_collection(name=COLLECTION_NAME)

    # 4. 執行同步 (Upsert: 若存在則更新，不存在則新增)
    if total_items > 0:
        print(f"開始同步到雲端...")
        remote_col.upsert(
            ids=local_data['ids'],
            embeddings=local_data['embeddings'],
            metadatas=local_data['metadatas'],
            documents=local_data['documents']
        )
        print(f"同步成功！雲端現在擁有 {remote_col.count()} 筆資料。")
    else:
        print("本地資料庫是空的，無需同步。")

except Exception as e:
    print(f"雲端同步失敗: {e}")
    print("\n小提醒：請確保雲端主機的 ChromaDB 服務已開啟，且防火牆允許 8000 Port 通訊。")