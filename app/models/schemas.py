# app/models/schemas.py
from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = Field(default_factory=list)
    sources: Optional[List[str]] = Field(default_factory=list)
    isTyping: Optional[bool] = False


class ChatRequest(BaseModel):
    query: str
    history: List[Message] = Field(default_factory=list)
    images: Optional[List[str]] = Field(default_factory=list)
    model_name: Optional[str] = None
    session_id: Optional[str] = None


class ScheduleDebugRequest(BaseModel):
    query: str
    filename: Optional[str] = None


class ScheduleSmokeRequest(BaseModel):
    queries: Optional[List[str]] = None
    filename: Optional[str] = None
