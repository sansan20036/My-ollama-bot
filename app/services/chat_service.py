import logging
import os
import time
from typing import AsyncGenerator
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.cache_service import SemanticCacheService
from app.utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        # 強制設定不使用代理
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        self.vector_store = VectorStoreService.get_instance()
        self.cache = SemanticCacheService.get_instance()
        self.upload_dir = os.path.join(os.getcwd(), "uploads")

        # 🔥🔥🔥 回歸原點：使用您原本運作正常的 gpt-oss:20b 🔥🔥🔥
        target_model = "gpt-oss:20b"

        logger.info(f"正在初始化聊天模型: {target_model}")

        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=target_model,
            temperature=0.1,  # 保持低溫，讓回答穩定
            keep_alive="1h",

            # ✅ 只保留這兩個必要的 RAG 參數，其他花俏的設定通通移除
            num_ctx=8192,  # 確保能讀取長篇 PDF
            num_predict=4096  # 確保表格能畫完，不會斷在中間
        )

    def _get_valid_files(self) -> list:
        """過濾掉隱藏檔與暫存檔"""
        if not os.path.exists(self.upload_dir):
            return []

        return [
            f for f in os.listdir(self.upload_dir)
            if os.path.isfile(os.path.join(self.upload_dir, f))
               and not f.startswith("~$")
               and f.lower() != "thumbs.db"
               and not f.endswith(".tmp")
        ]

    def _get_sorted_file_list(self, files: list) -> str:
        """生成檔案清單字串"""
        if not files:
            return "(目前資料庫為空)"

        try:
            file_info_list = []
            for f in files:
                file_path = os.path.join(self.upload_dir, f)
                mod_time = os.path.getmtime(file_path)
                file_info_list.append((f, mod_time))

            file_info_list.sort(key=lambda x: x[1], reverse=True)

            top_n = 10
            recent_files = file_info_list[:top_n]

            formatted_list = []
            for index, (fname, timestamp) in enumerate(recent_files):
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)) if timestamp > 0 else "Unknown"

                icon = "📄"
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    icon = "🖼️"

                if index == 0:
                    formatted_list.append(f"- {icon} {fname} (✨ NEWEST / 最新上傳) [Time: {time_str}]")
                else:
                    formatted_list.append(f"- {icon} {fname} [Time: {time_str}]")

            if len(file_info_list) > top_n:
                formatted_list.append(f"... (以及其他 {len(file_info_list) - top_n} 個較舊的檔案)")

            return "\n".join(formatted_list)
        except Exception as e:
            logger.error(f"排序失敗: {e}")
            return "(無法取得檔案列表)"

    async def process_query(self, query: str, history: list) -> AsyncGenerator[str, None]:
        real_query = query

        valid_files = self._get_valid_files()
        has_files = len(valid_files) > 0

        if not has_files:
            yield "⚠️ 目前資料庫是空的 (0 個檔案)。\n\n請先上傳文件。"
            return

        file_count = len(valid_files)
        file_list_str = self._get_sorted_file_list(valid_files)

        # RAG 檢索
        docs = self.vector_store.search(real_query, k=4)
        full_text = TextProcessor.smart_merge(docs)
        final_context = full_text if full_text else "沒有檢索到具體內容。"

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-2:]]) if history else "(無歷史紀錄)"

        template_str = """You are a helpful AI assistant.

        【Files】:
        {file_list_str}

        【Context】:
        {context}

        【History】:
        {history}

        【User Question】: {question}

        Instructions:
        1. Answer based on the Context.
        2. If the user asks for content, summarize it clearly.
        3. ✅ **Use Markdown Tables** for structured data.
        4. ❌ **Do NOT use HTML tags** (like <br>). Use standard Markdown newlines.
        5. Answer in Traditional Chinese (繁體中文).

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template_str)

        chain = (
                {
                    "context": lambda x: final_context,
                    "question": RunnablePassthrough(),
                    "history": lambda x: history_text,
                    "file_list_str": lambda x: file_list_str,
                    "file_count": lambda x: str(file_count)
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        try:
            async for chunk in chain.astream(real_query):
                # GPT-OSS 通常格式比較標準，我們只做最基本的防呆，不做過度清洗
                clean_chunk = (chunk
                               .replace("<br>", "\n")
                               .replace("<br/>", "\n")
                               .replace("<b>", "**")
                               .replace("</b>", "**"))
                yield clean_chunk

        except Exception as e:
            logger.error(f"Chat Error: {e}")
            yield f"\n\n⚠️ 發生錯誤: {str(e)}\n請檢查遠端伺服器連線。"