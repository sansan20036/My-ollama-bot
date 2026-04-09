# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

# 定義單條對話紀錄的格式 (完美對應前端狀態)
class Message(BaseModel):
    role: str
    content: str
    # 🚀 優化 1：使用 Field(default_factory=list) 安全地產生空陣列
    images: Optional[List[str]] = Field(default_factory=list)  # 接住歷史紀錄裡的圖片預覽
    sources: Optional[List[str]] = Field(default_factory=list) # 接住 AI 的參考來源
    isTyping: Optional[bool] = False                           # 接住前端的打字狀態

# 定義前端發送過來的請求格式
class ChatRequest(BaseModel):
    query: str
    # 🚀 優化 2：把原本的 Dict[str, Any] 換成上面定義好的 Message！
    # 這樣只要前端傳來的歷史紀錄少了一個欄位或型別不對，FastAPI 就會自動幫你擋下來報錯！
    history: List[Message] = Field(default_factory=list)
    images: Optional[List[str]] = Field(default_factory=list)
    model_name: Optional[str] = None  # 允許前端傳送模型名稱
    session_id: Optional[str] = None  # 同批次上傳關聯識別
