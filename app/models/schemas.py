# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

# 定義單條對話紀錄的格式 (完美對應前端狀態)
class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = []  # 接住歷史紀錄裡的圖片預覽
    sources: Optional[List[str]] = [] # 接住 AI 的參考來源
    isTyping: Optional[bool] = False  # 接住前端的打字狀態

# 定義前端發送過來的請求格式
class ChatRequest(BaseModel):
    query: str
    model_name: str = "gemma3:27b"
    history: List[Message] = []
    images: List[str] = []  # 接住本次對話要傳給大模型的 Base64 圖片陣列