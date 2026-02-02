import shutil
import os
import logging
import json
import gc
import pandas as pd
from typing import List, Optional  # 🟢 補上 Optional
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
# 🟢 新增：用於建構多模態訊息 (處理圖片的關鍵)
from langchain_core.messages import HumanMessage

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

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:

        all_docs = []
        seen_contents = set()

        for retriever in self.retrievers:
            docs = retriever.invoke(query)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    # 在 metadata 加入「標註狀態」，防止重複引用
                    doc.metadata["already_cited"] = False
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)
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


def get_optimized_history(history, max_turns=10):
    """
    博士論文亮點：動態滑動窗口與資訊壓縮
    1. 限制對話輪數防止 Token 溢位
    2. 截斷過長的內容（如先前的表格數據）保留語意核心
    """
    recent_history = history[-max_turns:]
    history_lines = []
    for msg in recent_history:
        role = "使用者" if msg['role'] == "User" else "助理"
        # 🟢 關鍵優化：如果內容超過 200 字（通常是 Agent 產生的報表），則進行摘要截斷
        content = (msg['content'][:150] + " [後續內容已省略...]") if len(msg['content']) > 200 else msg['content']
        history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines)


# ================= 配置區 =================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

DEFAULT_MODEL = "gpt-oss:20b"

# 初始化變數
vector_db = None
CHAT_HISTORY = []
GLOBAL_FILE_CONTENT = ""
GLOBAL_DFS = {}
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


class ChatRequest(BaseModel):
    query: str
    model_name: str = DEFAULT_MODEL
    # 🟢 新增：接收圖片 Base64 列表 (Optional)
    # 沒有這行，後端就會忽略前端傳來的圖片！
    images: Optional[List[str]] = None


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
        GLOBAL_DFS = {}
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
            gc.collect()  # 🟢 強制回收垃圾，確保檔案指標被釋放
            await asyncio.sleep(1)  # 🟢 給 Windows 一秒鐘反應時間

        if os.path.exists(persist_dir):
            try:
                await asyncio.sleep(1)
                shutil.rmtree(persist_dir, ignore_errors=True)
                logger.info("🗑️ 已刪除實體資料庫檔案")
            except Exception as e:
                logger.error(f"❌ 物理刪除失敗: {e}")

        init_vector_db()
        return {"message": "系統已完全重置 (記憶已清除)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= 新增：軟重置 (新對話，保留檔案) =================
@app.post("/new_chat")
async def start_new_chat():
    global CHAT_HISTORY
    logger.info("🧹 執行新對話")

    # 只清空對話紀錄
    CHAT_HISTORY = []

    # 回傳目前的檔案數量，讓前端知道檔案還活著
    current_file_count = len(GLOBAL_DFS) + len(set([d.metadata.get("source") for d in GLOBAL_DOCS]))

    return {
        "message": "對話紀錄已清除 (檔案記憶已保留)",
        "status": "success",
        "kept_files": current_file_count
    }

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
    # 🟢 註解掉重置，允許累加檔案
    # await reset_history()

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
                    # 🟢 使用 with 確保 excel_file 在讀取完後自動關閉釋放檔案鎖
                    with pd.ExcelFile(temp_filename) as excel_file:
                        for sheet_name in excel_file.sheet_names:
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)
                            if not df.empty:
                                display_name = f"{file.filename} ({sheet_name})"
                                temp_dfs.append((display_name, df))
                except Exception as e:
                    logger.error(f"讀取 Excel 失敗: {e}")

            elif file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(temp_filename)
                    if not df.empty:
                        temp_dfs.append((file.filename, df))  # 🟢 存成 tuple
                except Exception as e:
                    logger.error(f"讀取 CSV 失敗: {e}")

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

        # 🟢 修復：使用 extend 累積資料表，而不是覆蓋
        if temp_dfs:
            for file_name, df in temp_dfs:
                GLOBAL_DFS[file_name] = df  # 這是字典賦值，正確

        combined_text = "\n".join(full_text_list)

        # 🟢 修復：累積純文字內容
        if len(GLOBAL_FILE_CONTENT) + len(combined_text) < 10000:
            GLOBAL_FILE_CONTENT += "\n" + combined_text
        else:
            pass

        has_dfs = len(GLOBAL_DFS) > 0

        mode = "RAG_MODE"
        if has_dfs:
            mode = "PANDAS_AGENT"
        elif GLOBAL_FILE_CONTENT:
            mode = "GOD_MODE"

        return {
            "status": "success",
            "processed_files": processed_files,
            "current_inventory": list(GLOBAL_DFS.keys()),
            "mode": mode
        }

    except Exception as e:
        logger.error(f"上傳失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================= 聊天邏輯 (完整版：多模態 + 混合檢索) =================
# ================= 聊天邏輯 (修正版：視覺 + 檔案記憶共存) =================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global CHAT_HISTORY, GLOBAL_FILE_CONTENT, GLOBAL_DFS, GLOBAL_DOCS

    # 1. 準備基礎資訊
    raw_sources = [doc.metadata.get("source", "") for doc in GLOBAL_DOCS] + list(GLOBAL_DFS.keys())
    clean_sources = []
    for s in raw_sources:
        if not s: continue
        name = s.replace("temp_", "")
        name = name.split(" (")[0] if " (" in name else name
        clean_sources.append(name)
    unique_files_list = list(set(clean_sources))
    total_file_count = len(unique_files_list)

    current_images = len(request.images) if request.images else 0
    logger.info(f"🗣️ 收到問題: {request.query} (圖片: {current_images}, 檔案庫存: {total_file_count})")

    try:
        model_to_use = request.model_name if request.model_name else DEFAULT_MODEL

        llm = ChatOllama(
            model=model_to_use,
            temperature=0.2,
            base_url="http://git.tedpc.com.tw:11434",
            request_timeout=900.0,
            num_thread=8,
            num_predict=2048,
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        )

        history_str = get_optimized_history(CHAT_HISTORY, max_turns=5)

        async def generate_response():
            # 🟢 [修正關鍵 1]：先把檔案摘要 (Context) 準備好，不管有沒有圖都要用
            context_summary_list = []

            # 1. 檔案清單
            if unique_files_list:
                context_summary_list.append(f"已載入檔案清單：{', '.join(unique_files_list)}")
            else:
                context_summary_list.append("目前無載入檔案")

            # 2. Excel/CSV 結構摘要
            if len(GLOBAL_DFS) > 0:
                context_summary_list.append("\n【表格檔案摘要資訊】(AI請注意此區塊)：")
                for display_name, df in GLOBAL_DFS.items():
                    clean_name = display_name.replace("temp_", "")
                    columns = ", ".join(list(df.columns))
                    if len(df) <= 20:
                        preview_data = df.to_markdown(index=False)
                    else:
                        preview_data = df.head(3).to_markdown(index=False) + "\n(只顯示前 3 筆...)"

                    summary = (
                        f"- 檔案 `{clean_name}`：\n"
                        f"  - 類型：結構化數據表\n"
                        f"  - 資料筆數：{len(df)} 筆\n"
                        f"  - 包含欄位：[{columns}]\n"
                        f"  - 資料預覽：\n{preview_data}\n"
                    )
                    context_summary_list.append(summary)

            # 3. RAG 檢索 (如果是混合模式，也要嘗試撈一點文字資料)
            #    (這裡做一個輕量級檢索，確保如果問文字檔也能回答)
            rag_context = ""
            if len(GLOBAL_DOCS) > 0 and vector_db:
                try:
                    # 簡單用 BM25 撈前 3 筆相關的，作為背景知識
                    bm25_retriever = BM25Retriever.from_documents(GLOBAL_DOCS)
                    bm25_retriever.k = 3
                    docs = bm25_retriever.invoke(request.query)
                    if docs:
                        rag_context = "\n【相關文字檔案片段】:\n" + "\n".join([d.page_content[:200] for d in docs])
                        context_summary_list.append(rag_context)
                except Exception:
                    pass

            # 組合最終的 System Context
            files_context = "\n".join(context_summary_list)

            # 🟢 [修正關鍵 2]: 視覺模式 (Vision Mode) 現在也包含 files_context 了
            if request.images and len(request.images) > 0:
                # 組合 Prompt：包含 使用者問題 + 檔案背景知識 + 歷史對話
                # 🟢 修正：強制 AI 先描述圖片，再關聯檔案，防止瞎掰
                system_instruction = """系統指令：你是一個具備視覺能力的數據分析師。

                使用者正在展示一張圖片並詢問問題。

                ⚠️ 你的思考步驟 (必須嚴格遵守)：
                1. **視覺檢測**：首先，請客觀、誠實地描述你「真正」在圖片中看到了什麼。如果圖片中沒有顯示卡，請直接說出來，不要瞎掰。
                2. **關聯性分析**：接著，將你看到的內容與 [背景知識] 進行比對。
                3. **回答問題**：根據比對結果回答使用者的問題。
                """

                # 2. 安全拼接 (Safe Concatenation)
                mixed_prompt = (
                        system_instruction +
                        "\n\n【背景知識/檔案內容】：\n" + files_context +
                        "\n\n【歷史對話】：\n" + history_str +
                        "\n\n【使用者目前問題】：\n" + request.query +
                        "\n\n請綜合圖片內容與上述背景知識回答。"
                )

                content_parts = [{"type": "text", "text": mixed_prompt}]
                for img_base64 in request.images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_base64}"
                    })

                message = HumanMessage(content=content_parts)
                async for chunk in llm.astream([message]):
                    yield chunk.content

                CHAT_HISTORY.append({"role": "User", "content": f"[圖片] {request.query}"})
                # 注意：這裡依然 Return，因為視覺模型不適合跑 Pandas Agent，直接回答即可
                return

            # --- 以下維持原本的邏輯 (快速通道、Agent、RAG) ---

            # 檔案清單快速通道
            trigger_keywords = ["哪些", "哪幾份", "分別是", "檔案清單", "列出", "有什麼"]
            exclusion_keywords = ["內容", "欄位", "數據", "資料", "關於", "描述", "翻譯", "計算", "總結"]
            is_listing_query = any(k in request.query for k in trigger_keywords)
            has_exclusion = any(k in request.query for k in exclusion_keywords)

            if is_listing_query and not has_exclusion:
                fast_response = f"目前系統中已載入的 {len(unique_files_list)} 份檔案如下：\n\n"
                for i, f_name in enumerate(unique_files_list):
                    fast_response += f"{i + 1}. {f_name}\n"
                for char in fast_response:
                    yield char
                    await asyncio.sleep(0.005)
                CHAT_HISTORY.append({"role": "User", "content": request.query})
                CHAT_HISTORY.append({"role": "AI", "content": fast_response})
                return

            # Pandas Agent (數據分析)
            is_calc_query = any(k in request.query for k in ["算", "平均", "總和", "加總", "幾份", "檔案", "資料"])
            if len(GLOBAL_DFS) > 0 and is_calc_query:
                try:
                    df_list = list(GLOBAL_DFS.values())
                    file_names = list(GLOBAL_DFS.keys())
                    inventory_str = "\n".join([f"df{i + 1}: {name}" for i, name in enumerate(file_names)])

                    # 臨時建立 temp=0 的 Agent 專用 LLM
                    agent_llm = ChatOllama(
                        model=model_to_use,
                        temperature=0.0,
                        base_url="http://git.tedpc.com.tw:11434",
                        request_timeout=900.0,
                        num_thread=8,
                        num_predict=2048,
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
                    )

                    custom_prefix = f"""你現在是一個具備嚴格邏輯的數據分析官。
                                    ⚠️ 系統環境資訊：
                                    - 你的 Python 環境中有 {len(df_list)} 個資料表：{inventory_str}。
                                    - 總檔案清單：[{files_context}]。
                                    你的思考規則 (極重要)：
                                    1. 必須使用 ReAct 框架：Thought -> Action -> Observation -> Final Answer。
                                    2. **禁止**使用 JSON 格式回應，請使用純文字格式。
                                    3. Action 必須是 `python_repl_ast`。
                                    4. 當你得出結論後，**必須**以 'Final Answer: ' 開頭輸出結果。
                                    5. 禁止直接回覆 "Answer:" 或 "答案是"，這會導致系統解析失敗。
                                    6. 如果計算出現錯誤，請重新檢查 DataFrame 的欄位名稱。
                                    """
                    agent = create_pandas_dataframe_agent(
                        agent_llm,
                        df_list,
                        verbose=True,
                        allow_dangerous_code=True,
                        agent_type="zero-shot-react-description",
                        prefix=custom_prefix
                    )
                    result = await asyncio.to_thread(agent.invoke, {"input": request.query},
                                                     {"handle_parsing_errors": True})
                    response_text = result['output']
                    full_text = f"🤖 (數據分析): {response_text}"
                    for char in full_text:
                        yield char
                        await asyncio.sleep(0)
                    CHAT_HISTORY.append({"role": "User", "content": request.query})
                    CHAT_HISTORY.append({"role": "AI", "content": response_text})
                    return
                except Exception as e:
                    logger.warning(f"Agent 執行失敗: {e}")
                    total_rows = sum(len(df) for df in list(GLOBAL_DFS.values()))
                    if is_calc_query and total_rows > 20:
                        error_msg = f"⚠️ 抱歉，計算引擎暫時無法處理 (錯誤代碼: OutputParserException)。\n建議您：\n1. 嘗試更明確的指令\n2. 檢查檔案內容是否乾淨"
                        yield error_msg
                        CHAT_HISTORY.append({"role": "User", "content": request.query})
                        CHAT_HISTORY.append({"role": "AI", "content": error_msg})
                        return
                    logger.warning(f"切換至 RAG 模式嘗試回答...")

            # RAG 混合檢索 (文字模式)
            mode = "rag"
            # 這裡不需要再重新檢索了，上面已經產生 files_context，且會包含 RAG 的文字片段
            # 我們只需要加強檢索，如果上面的簡易檢索不夠
            if len(GLOBAL_DOCS) > 0 and vector_db:
                try:
                    chroma_retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # 加強檢索深度
                    bm25_retriever = BM25Retriever.from_documents(GLOBAL_DOCS)
                    bm25_retriever.k = 5
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[chroma_retriever, bm25_retriever],
                        weights=[0.7, 0.3]
                    )
                    initial_docs = ensemble_retriever.invoke(request.query)
                    if initial_docs:
                        # 重新組合更詳細的 Context
                        rag_details = "\n\n".join(
                            [f"【來源：{d.metadata.get('source', '未知')}】\n{d.page_content}" for d in initial_docs])
                        files_context += f"\n\n=== 詳細檢索內容 ===\n{rag_details}"
                except Exception as e:
                    logger.error(f"深度檢索失敗: {e}")

            template = f"""系統指令:
            你是一位專業助理。目前系統中已載入的檔案清單與內容摘要如下：
            [{files_context}]

            請優先參考上述 [參考資料] 回答問題。

            ⚠️ 回答規則：
            1. **優先參考歷史對話**：若 [目前問題] 提到「剛才」、「那個公式」，請從 [歷史對話] 找尋上下文。
            2. **直接回答問題**，語氣自然。
            3. **任務區分**：
                - **事實查詢**：引用參考資料。
                - **翻譯/總結**：請使用你的語言能力生成。
            4. 若檔案中包含多個題目，請仔細辨別。

            [歷史對話]: {{history}}
            [目前問題]: {{question}}
            回答:"""

            prompt = ChatPromptTemplate.from_template(template)
            rag_chain = (
                    {"question": RunnablePassthrough(), "history": lambda x: history_str}
                    | prompt | llm | StrOutputParser()
            )

            full_response = ""
            async for chunk in rag_chain.astream(request.query):
                full_response += chunk
                yield chunk

            CHAT_HISTORY.append({"role": "User", "content": request.query})
            CHAT_HISTORY.append({"role": "AI", "content": full_response})

        return StreamingResponse(generate_response(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    print("🚀 正在啟動後端 API 伺服器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
