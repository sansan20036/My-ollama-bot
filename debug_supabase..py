import os
from supabase import create_client
from langchain_ollama import OllamaEmbeddings

# 1. 設定 (請確保這些跟你原本的一樣)
SUPABASE_URL = "https://abuxyukbleiauunrroks.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFidXh5dWtibGVpYXV1bnJyb2tzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgyODM4NTUsImV4cCI6MjA4Mzg1OTg1NX0.w9g1xGbyHXGjCIj3wWl_0lkVojRzlkoQNTUEKZLRn8Q"
OLLAMA_HOST = "http://git.tedpc.com.tw:11434/"

# 2. 建立連線
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)


def test_nuclear_function():
    print("🚀 開始進行直連測試 (Bypassing LangChain)...")

    # 3. 產生一個測試向量
    print("正在產生向量...")
    query_vector = embeddings.embed_query("English")

    # 4. 直接呼叫 Supabase RPC (不透過 LangChain)
    # 我們直接呼叫那個「核能版」函數
    rpc_params = {
        "query_embedding": query_vector,
        "match_threshold": -1.0,  # 負數，確保不過濾
        "match_count": 100,  # 這裡雖然傳 100，但 SQL 裡我們已經鎖死了，只是形式上傳一下
        "filter": {}
    }

    print(f"正在呼叫 RPC 函數: get_documents_nuclear ...")

    try:
        response = supabase.rpc("get_documents_nuclear", rpc_params).execute()

        # 5. 分析結果
        data = response.data
        count = len(data)
        print(f"\n✅ RPC 呼叫成功！")
        print(f"🔥 資料庫回傳筆數: {count} 筆")
        print("-" * 30)

        if count == 0:
            print("⚠️ 警告：回傳 0 筆。請檢查資料庫是否真的有資料？")
        else:
            # 檢查裡面有沒有英文
            found_english = False
            for i, item in enumerate(data):
                content = item.get('content', '')
                preview = content.replace('\n', ' ')[:50]
                print(f"[{i + 1}] {preview}...")

                # 簡單檢查一下有沒有常見英文單字
                if "rich" in content.lower() or "wear" in content.lower():
                    found_english = True
                    print(f"   >>> 🎉 找到疑似英文片段！: {content[:100]}")

            print("-" * 30)
            if count > 10:
                print("🎉 恭喜！我們突破 10 筆的魔咒了！LangChain 是兇手。")
            elif count == 10:
                print("💀 仍然是 10 筆... 看來我要去吃鍵盤了 (或者資料庫裡真的只有 10 筆資料)。")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")


if __name__ == "__main__":
    test_nuclear_function()