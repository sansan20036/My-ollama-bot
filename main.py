import shutil
import os
import logging
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

# 引用你的檔案讀取工廠
from file_factory import FileLoaderFactory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("正在載入 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

VECTOR_DB_PATH = "./chroma_db_api"


def get_vector_db():
    return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)


vector_db = get_vector_db()
# 原本可能是 500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

CHAT_HISTORY = []


class ChatRequest(BaseModel):
    query: str
    model_name: str = "llama3.2"


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/reset")
async def reset_history():
    global CHAT_HISTORY, vector_db
    try:
        CHAT_HISTORY = []
        existing_data = vector_db.get()
        existing_ids = existing_data['ids']
        if existing_ids:
            vector_db.delete(existing_ids)
            logger.info(f"已清空資料庫，共刪除 {len(existing_ids)} 筆資料")
        return {"message": "系統已完全重置"}
    except Exception as e:
        logger.error(f"Reset Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global CHAT_HISTORY, vector_db
    CHAT_HISTORY = []

    try:
        existing_ids = vector_db.get()['ids']
        if existing_ids:
            vector_db.delete(existing_ids)

        total_chunks = 0
        processed_files = []
        error_files = []

        for file in files:
            temp_filename = f"temp_{file.filename}"
            try:
                with open(temp_filename, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                logger.info(f"正在處理: {file.filename}")
                loader = FileLoaderFactory.get_loader(temp_filename, file.filename)
                raw_text = loader.extract_text()

                if not raw_text or len(raw_text.strip()) == 0:
                    logger.warning(f"{file.filename} 內容為空，跳過")
                    error_files.append(file.filename)
                    continue

                chunks_text = text_splitter.split_text(raw_text)
                documents = [
                    Document(page_content=chunk, metadata={"source": file.filename})
                    for chunk in chunks_text
                ]

                if documents:
                    vector_db.add_documents(documents)
                    total_chunks += len(documents)
                    processed_files.append(file.filename)

            except Exception as e:
                logger.error(f"處理 {file.filename} 失敗: {str(e)}")
                error_files.append(f"{file.filename} ({str(e)})")

            finally:
                try:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                except Exception as cleanup_error:
                    logger.warning(f"暫存檔刪除失敗 (忽略): {str(cleanup_error)}")

        return {
            "status": "success",
            "processed_files": processed_files,
            "message": f"成功讀取 {len(processed_files)} 個檔案"
        }

    except Exception as e:
        logger.error(f"批次上傳錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 🟢 修正重點：加入「對話改寫機制」以支援多輪對話
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global CHAT_HISTORY
    try:
        llm = ChatOllama(
            model=request.model_name,
            temperature=0.1,
            base_url="http://localhost:11434",
            request_timeout=180.0
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 10})

        # 將歷史紀錄轉成字串
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in CHAT_HISTORY])


        # 如果有歷史對話，先請 AI 把使用者的問題改寫成「包含完整上下文的問句」
        real_query = request.query

        if CHAT_HISTORY:
            rephrase_prompt = ChatPromptTemplate.from_template("""
            Given the following conversation history and a follow-up question, 
            rephrase the follow-up question to be a standalone question that includes the necessary context.
            Do NOT answer the question, just rephrase it.

            Chat History:
            {history}

            Follow Up Input: {question}

            Standalone Question:""")

            rephrase_chain = rephrase_prompt | llm | StrOutputParser()

            # 取得改寫後的搜尋關鍵字
            real_query = rephrase_chain.invoke({"history": history_str, "question": request.query})
            logger.info(f"原始問題: {request.query} -> 改寫後搜尋: {real_query}")



        docs = retriever.invoke(real_query)

        # 準備 Sources (回傳給前端顯示用)
        sources = list(set([doc.metadata.get("source", "未知") for doc in docs]))
        sources_json = json.dumps(sources, ensure_ascii=True)


        # 這裡的 Prompt 可以維持原樣，或是強調參考 Context
        user_query_lower = request.query.lower()
        force_english = "english" in user_query_lower or "英文" in user_query_lower

        if force_english:
            template = """You are a helpful AI assistant.
            Answer the user's question based on the Context below.
            If the answer is not in the context, say "I don't have that information."

            Context:
            {context}

            History:
            {history}

            User Question: {question}

            Answer:"""
        else:
            template = """你是一個專業助理。
            請根據下方的「已知資訊」與「歷史對話」來回答問題。

            重要：
            1. 如果這個問題是承接上一句的(例如「那缺貨的是哪個？」)，請務必結合歷史對話來理解。
            2. 如果資料庫真的沒有相關資訊，請直說。

            已知資訊:
            {context}

            歷史對話:
            {history}

            使用者問題: {question}

            回答:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
                {
                    "context": lambda x: docs,  # 直接使用剛剛檢索到的 docs
                    "question": RunnablePassthrough(),
                    "history": lambda x: history_str
                }
                | prompt
                | llm
                | StrOutputParser()
        )

        async def generate_response():
            full_response = ""
            # 這裡我們傳入原始問題 request.query 給 LLM 生成回答，因為 Context 已經抓對了
            async for chunk in rag_chain.astream(request.query):
                full_response += chunk
                yield chunk

            # 存入歷史紀錄
            CHAT_HISTORY.append({"role": "User", "content": request.query})
            CHAT_HISTORY.append({"role": "AI", "content": full_response})

        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"X-Sources": sources_json}
        )

    except Exception as e:
        logger.error(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))