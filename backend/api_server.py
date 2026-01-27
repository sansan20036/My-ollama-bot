import shutil
import os
import logging
import json
import gc
import pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import random
import requests

# 確保 file_factory 存在
from file_factory import FileLoaderFactory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# 🟢 Import 區塊
from langchain_community.retrievers import BM25Retriever

# 🟢 手動定義 EnsembleRetriever (Polyfill)
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class EnsembleRetriever(BaseRetriever):
    """
    自定義的混合檢索器 (Polyfill)
    """
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        all_docs = []
        seen_contents = set()
        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query)
                for doc in docs:
                    if doc.page_content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)
            except Exception as e:
                print(f"⚠️ 檢索器執行錯誤: {e}")
                continue
        return all_docs[:10]


# 引入 Agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 設定 Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 配置區 =================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

DEFAULT_MODEL = "gpt-oss:20b"

# 初始化變數
vector_db = None
CHAT_HISTORY = []
GLOBAL_FILE_CONTENT = ""
GLOBAL_DFS = []
GLOBAL_DOCS = []


def init_vector_db():
    global vector_db, GLOBAL_DOCS

    # 🟢 1. 設定資料庫儲存路徑
    persist_dir = "./chroma_db"

    logger.info(f"🔄 正在初始化資料庫，路徑: {persist_dir}")

    # 🟢 2. 初始化 Chroma (持久化)
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # 🟢 3. 重建 BM25 索引
    try:
        existing_data = vector_db.get()
        if existing_data and len(existing_data['ids']) > 0:
            print(f"📦 偵測到歷史存檔，正在重建關鍵字索引 (共 {len(existing_data['ids'])} 筆)...")

            GLOBAL_DOCS = []
            for i in range(len(existing_data['ids'])):
                doc = Document(
                    page_content=existing_data['documents'][i],
                    metadata=existing_data['metadatas'][i] if existing_data['metadatas'] else {}
                )
                GLOBAL_DOCS.append(doc)

            print(f"✅ 成功恢復記憶！目前擁有 {len(GLOBAL_DOCS)} 筆知識片段。")
        else:
            print("✨ 全新開始：資料庫目前是空的。")

    except Exception as e:
        print(f"⚠️ 重建索引時發生小插曲: {e}")

    logger.info(f"✅ 資料庫初始化完成")


init_vector_db()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)


class ChatRequest(BaseModel):
    query: str
    model_name: str = DEFAULT_MODEL


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


# ================= 重置邏輯 (Deep Clean) =================
@app.post("/reset")
async def reset_history():
    global CHAT_HISTORY, vector_db, GLOBAL_FILE_CONTENT, GLOBAL_DFS, GLOBAL_DOCS
    logger.info("🧹 執行系統重置 (Deep Clean)...")

    try:
        CHAT_HISTORY = []
        GLOBAL_FILE_CONTENT = ""
        GLOBAL_DFS = []
        GLOBAL_DOCS = []

        persist_dir = "./chroma_db"

        if vector_db:
            try:
                ids = vector_db.get()['ids']
                if ids:
                    vector_db.delete(ids)
            except Exception as e:
                logger.warning(f"邏輯刪除失敗: {e}")

            vector_db = None
            del vector_db
            gc.collect()

        if os.path.exists(persist_dir):
            try:
                await asyncio.sleep(0.5)
                shutil.rmtree(persist_dir, ignore_errors=True)
                logger.info("🗑️ 已刪除實體資料庫檔案")
            except Exception as e:
                logger.error(f"❌ 物理刪除失敗: {e}")

        init_vector_db()
        return {"message": "系統已完全重置 (記憶已清除)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= 獲取模型清單 API =================
@app.get("/models")
async def get_models():
    try:
        ollama_api_url = "http://git.tedpc.com.tw:11434/api/tags"
        response = requests.get(ollama_api_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        return {"models": models}
    except Exception as e:
        logger.error(f"無法獲取模型清單: {e}")
        return {"models": ["gpt-oss:20b", "llama3.1:latest"]}


# ================= 上傳邏輯 =================
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    #await reset_history()
    global vector_db, GLOBAL_FILE_CONTENT, GLOBAL_DFS, GLOBAL_DOCS

    try:
        processed_files = []
        full_text_list = []
        temp_dfs = []

        for file in files:
            temp_filename = f"temp_{file.filename}"
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            if file.filename.endswith(('.xlsx', '.xls')):
                try:
                    df = pd.read_excel(temp_filename, engine='openpyxl')
                    temp_dfs.append(df)
                except Exception:
                    pass
            elif file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(temp_filename)
                    temp_dfs.append(df)
                except Exception:
                    pass

            try:
                loader = FileLoaderFactory.get_loader(temp_filename, file.filename)
                raw_text = loader.extract_text()
                if raw_text:
                    full_text_list.append(f"【檔案: {file.filename}】\n{raw_text}\n")

                    chunks = text_splitter.split_text(raw_text)
                    docs = [Document(page_content=c, metadata={"source": file.filename}) for c in chunks]
                    if docs:
                        vector_db.add_documents(docs)
                        GLOBAL_DOCS.extend(docs)
                        processed_files.append(file.filename)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        GLOBAL_DFS = temp_dfs
        combined_text = "\n".join(full_text_list)
        if len(combined_text) < 10000:
            GLOBAL_FILE_CONTENT = combined_text
        else:
            GLOBAL_FILE_CONTENT = ""

        mode = "RAG_MODE"
        if len(GLOBAL_DFS) > 0:
            mode = "PANDAS_AGENT"
        elif GLOBAL_FILE_CONTENT:
            mode = "GOD_MODE"

        return {"status": "success", "processed_files": processed_files, "mode": mode}

    except Exception as e:
        logger.error(f"上傳失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================= 聊天邏輯 (乾淨版 - 無思考過程) =================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global CHAT_HISTORY, GLOBAL_FILE_CONTENT, GLOBAL_DFS, GLOBAL_DOCS

    logger.info(f"🗣️ 收到問題: {request.query}")

    try:
        model_to_use = request.model_name if request.model_name else DEFAULT_MODEL

        llm = ChatOllama(
            model=model_to_use,
            temperature=0.1,
            base_url="http://git.tedpc.com.tw:11434",
            request_timeout=500.0,
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

        real_query = request.query
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in CHAT_HISTORY[-4:]])

        # 🟢 準備回傳的生成器
        async def generate_response():
            final_query = real_query

            # 策略 1: Pandas Agent
            if len(GLOBAL_DFS) > 0:
                try:
                    agent = create_pandas_dataframe_agent(
                        llm, GLOBAL_DFS, verbose=True, allow_dangerous_code=True,
                        agent_type="zero-shot-react-description", handle_parsing_errors=True
                    )
                    result = agent.invoke(f"請根據資料表回答。問題: {final_query}")
                    response_text = result['output']

                    # 模擬打字機 (僅針對 Agent)
                    full_text = f"🤖 (數據分析): {response_text}"
                    for char in full_text:
                        yield char
                        await asyncio.sleep(random.uniform(0.01, 0.05))

                    CHAT_HISTORY.append({"role": "User", "content": request.query})
                    CHAT_HISTORY.append({"role": "AI", "content": response_text})
                    return
                except Exception:
                    pass

            # 策略 2: RAG 混合檢索
            final_context = ""
            mode = "chitchat"

            if len(GLOBAL_DOCS) > 0 and vector_db:
                print(f"🚀 啟動混合檢索 (Hybrid Search): {real_query}")
                try:
                    chroma_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
                    bm25_retriever = BM25Retriever.from_documents(GLOBAL_DOCS)
                    bm25_retriever.k = 5

                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[chroma_retriever, bm25_retriever],
                        weights=[0.5, 0.5]
                    )
                    initial_docs = ensemble_retriever.invoke(final_query)

                    if initial_docs:
                        mode = "rag"
                        context_pieces = []
                        for doc in initial_docs:
                            source_name = doc.metadata.get("source", "未知來源")
                            piece = f"【來源：{source_name}】\n{doc.page_content}"
                            context_pieces.append(piece)
                        final_context = "\n\n".join(context_pieces)
                        print(f"✅ 混合檢索命中！參考了 {len(initial_docs)} 個片段。")
                except Exception as e:
                    print(f"Error: {e}")
                    mode = "chitchat"

            # 3. 生成回答
            if mode == "rag":
                template = """系統指令:
                            你是一位專業助理。請根據下方的 [參考資料] 與 [歷史對話] 回答 [目前問題]。
                            ⚠️ 回答規則：
                            1. **直接回答問題**，語氣自然、有禮貌。
                            2. 答案必須來自參考資料，不可瞎掰。
                            3. 如果是表格數據，請整理成 Markdown 表格。
                            4. 如果需要計算，請列出算式。
                            5. **非常重要：** 在回答的每一點後面，請用括號標註來源檔名，例如：(出處: 勞基法.pdf)。
                            6. **格式排版**：
                               - 條列式重點請用 Markdown 清單。
                               - 數據請整理成 Markdown 表格。
                               - 數學公式請用 LaTeX 格式 (例如 $E=mc^2$)。

                            [參考資料]: {context}
                            [歷史對話]: {history}
                            [目前問題]: {question}
                            回答:"""
            else:
                template = """系統指令:
                你是一位友善且博學的 AI 助理。根據 [歷史對話] 回答 [目前問題]。
                [歷史對話]: {history}
                [目前問題]: {question}
                回答:"""

            prompt = ChatPromptTemplate.from_template(template)
            rag_chain = (
                    {"context": lambda x: final_context, "question": RunnablePassthrough(),
                     "history": lambda x: history_str}
                    | prompt | llm | StrOutputParser()
            )

            full_response = ""
            async for chunk in rag_chain.astream(final_query):
                full_response += chunk
                yield chunk
                await asyncio.sleep(0)  # 讓出資源，但不刻意延遲

            CHAT_HISTORY.append({"role": "User", "content": request.query})
            CHAT_HISTORY.append({"role": "AI", "content": full_response})

        return StreamingResponse(generate_response(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)