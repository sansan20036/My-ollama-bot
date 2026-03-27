# app/services/llm_service.py
from ollama import Client
import json
from pathlib import Path
from typing import List, Dict, Optional


class OllamaChat:
    def __init__(
            self,
            model: str,
            host: str = "http://git.tedpc.com.tw:11434/",
            api_key: Optional[str] = None,  # 新增這個參數接收 API Key
            system_prompt: Optional[str] = None,
            history_path: Optional[str] = None,
            options: Optional[dict] = None,
    ):
        # 建立帶有身分驗證的 headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # 實例化 Client 時把 headers 傳進去
        self.client = Client(host=host, headers=headers)
        self.model = model
        self.history: List[Dict[str, str]] = []
        self.history_path = Path(history_path) if history_path else None
        self.options = options or {}

        # 嘗試檢查模型是否存在
        try:
            self.client.show(model=model)
        except Exception:
            print(f" 警告：伺服器找不到模型 '{model}'，或無權限存取。")
            print(f"嘗試下載模型 {model}...")
            try:
                self.client.pull(model=model)
                print("下載完成！")
            except Exception as e:
                print(f"下載失敗（伺服器可能無外網權限）: {e}")
                print("請手動修改 model 參數為伺服器現有的模型。")

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

        if self.history_path and self.history_path.exists():
            self.load_history()

    def save_history(self):
        if not self.history_path:
            return
        self.history_path.write_text(json.dumps(self.history, ensure_ascii=False, indent=2))

    def load_history(self):
        if not self.history_path or not self.history_path.exists():
            return
        self.history = json.loads(self.history_path.read_text())

    def clear_history(self, keep_system: bool = True):
        if keep_system:
            sys_msgs = [m for m in self.history if m.get("role") == "system"]
            self.history = sys_msgs
        else:
            self.history = []
        self.save_history()

    def ask(self, user_message: str, stream: bool = True) -> str:
        self.history.append({"role": "user", "content": user_message})
        try:
            if stream:
                resp_stream = self.client.chat(
                    model=self.model,
                    messages=self.history,
                    options=self.options,
                    stream=True,
                )
                full = []
                for part in resp_stream:
                    chunk = part.get("message", {}).get("content", "")
                    if chunk:
                        print(chunk, end="", flush=True)
                        full.append(chunk)
                print()
                answer = "".join(full)
            else:
                resp = self.client.chat(
                    model=self.model,
                    messages=self.history,
                    options=self.options,
                    stream=False,
                )
                answer = resp["message"]["content"]
                print(answer)

            self.history.append({"role": "assistant", "content": answer})
            self.save_history()
            return answer
        except Exception as e:
            error_msg = f"\n 對話出錯: {e}"
            print(error_msg)
            return error_msg


if __name__ == "__main__":
    from app.core.config import settings

    TARGET_HOST = settings.OLLAMA_BASE_URL
    EXPECTED_MODEL = settings.OLLAMA_MODEL
    API_KEY = settings.OLLAMA_API_KEY  #  從 config 抓取 API KEY

    print(" 正在掃描伺服器現有的模型...")

    check_headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    check_client = Client(host=TARGET_HOST, headers=check_headers)

    try:
        # 👇 抓取伺服器模型清單
        raw_models = check_client.list()

        # 👇 相容不同版本的 ollama API 回傳格式 (字典或物件)
        if isinstance(raw_models, dict):
            available_models = [m.get('model', m.get('name', '')) for m in raw_models.get('models', [])]
        else:
            available_models = [getattr(m, 'model', getattr(m, 'name', '')) for m in raw_models.models]

        if available_models:
            print(" 目前伺服器可用的模型如下：")
            for name in available_models:
                if name:  # 濾掉空值
                    print(f"   - {name}")

            if EXPECTED_MODEL in available_models:
                default_model = EXPECTED_MODEL
            else:
                default_model = available_models[0]
                print(f" 警告：伺服器沒有 {EXPECTED_MODEL}，自動降級為 {default_model}")
        else:
            print(" 伺服器中沒有任何模型。")
            default_model = EXPECTED_MODEL

    except Exception as e:
        print(f" 無法連線至伺服器或解析失敗: {e}")
        default_model = EXPECTED_MODEL

    print("-" * 30)

    # 初始化我們封裝好的機器人時，把 api_key 傳進去
    bot = OllamaChat(
        model=default_model,
        host=TARGET_HOST,
        api_key=API_KEY,
        system_prompt="你是專業的中文助手，回答請精簡。",
        history_path="chat_history.json",
        options={"temperature": 0.3}
    )

    print(f"使用模型: [{bot.model}] 開始聊天！")
    print("輸入 /reset 清空；/quit 結束。")

    while True:
        msg = input("\n我：").strip()
        if not msg: continue
        if msg == "/quit":
            break
        elif msg == "/reset":
            bot.clear_history(keep_system=True)
            print("→ 已清空紀錄")
            continue

        bot.ask(msg, stream=True)