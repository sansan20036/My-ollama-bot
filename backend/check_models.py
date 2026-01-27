from langchain_ollama import ChatOllama

OLLAMA_HOST = "http://git.tedpc.com.tw:11434/"

# 這是 Ollama 的 API 接口，用來列出所有模型
import requests

try:
    response = requests.get(f"{OLLAMA_HOST}/api/tags")
    if response.status_code == 200:
        models = response.json().get('models', [])
        print("====== 伺服器上可用的模型 ======")
        for m in models:
            print(f"📦 {m['name']}")
            # 顯示詳細資訊 (如大小)
            size_gb = m.get('size', 0) / (1024**3)
            print(f"   - 大小: {size_gb:.2f} GB")
        print("================================")
    else:
        print(f"❌ 無法連線，狀態碼: {response.status_code}")
except Exception as e:
    print(f"❌ 發生錯誤: {e}")