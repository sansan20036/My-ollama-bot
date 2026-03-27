# app/services/chat_service.py
import logging
import os
import re
import traceback
import pandas as pd
from typing import Any, Optional, AsyncGenerator, List, Dict
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
# 新增：引入建構多模態訊息所需的套件
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.cache_service import SemanticCacheService

logger = logging.getLogger(__name__)


@tool
def calculate_medical_fee(has_transfer: bool, drug_cost: int = -1) -> str:
    """
    計算醫院看診的總費用。當使用者詢問看診費用、藥費、部分負擔時，必須呼叫此工具。

    Args:
        has_transfer: 是否有經過診所轉診 (如果有轉診為 True，未經轉診直接來為 False)
        drug_cost: 藥品的總費用。🚨 如果使用者沒有提供藥費，或者你不知道，請務必填寫 -1。
    """
    # 🚨 只要 AI 填了 -1 (代表它不知道藥費)，就發動反問！
    if drug_cost == -1:
        return "【系統警告】：資料不足！請直接以自然語言回覆使用者：『請問您的藥費大約是多少元呢？我需要藥費才能為您計算總金額。』"

    # 防呆：確保一定是數字
    try:
        cost_val = int(drug_cost)
    except:
        return "【系統警告】：藥費格式錯誤！請反問使用者藥費是多少。"

    # 1. 醫院寫死的黃金準則：門診基本負擔
    base_fee = 280 if has_transfer else 420

    # 2. 醫院寫死的黃金準則：藥費部分負擔級距表
    if cost_val <= 100:
        drug_fee = 0
    elif cost_val <= 200:
        drug_fee = 40
    elif cost_val <= 300:
        drug_fee = 60
    elif cost_val <= 400:
        drug_fee = 80
    elif cost_val <= 500:
        drug_fee = 100
    elif cost_val <= 600:
        drug_fee = 120
    elif cost_val <= 700:
        drug_fee = 140
    elif cost_val <= 800:
        drug_fee = 160
    elif cost_val <= 900:
        drug_fee = 180
    elif cost_val <= 1000:
        drug_fee = 200
    else:
        # 超過 1000 元，上限就是 300 元
        drug_fee = 300

    total = base_fee + drug_fee

    # 回傳結果給 AI
    return f"【系統計算結果】門診基本負擔: {base_fee}元，藥費部分負擔: {drug_fee}元。總計應繳費用: {total}元。"


# 將工具打包成列表
tools = [calculate_medical_fee]


class ChatService:
    def __init__(self):
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        self.vector_store = VectorStoreService.get_instance()
        self.cache = SemanticCacheService.get_instance()
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        self.cached_df = None
        self.cached_file_path = ""
        self.cached_file_mtime = 0

        # 修改：將預設模型切換為設定檔中的模型
        target_model = settings.OLLAMA_MODEL

        logger.info(f"初始化全能文件聊天服務: {target_model}")

        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=target_model,
            temperature=0,  # Agent 運算設為 0，確保程式碼與數學精準
            keep_alive="1h",
            num_ctx=16384,
            num_predict=4096,
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {settings.OLLAMA_API_KEY}"
                }
            }
        )

    def _get_valid_files(self) -> list:
        if not os.path.exists(self.upload_dir): return []

        # 新增防呆：過濾掉結尾是 _tables.csv 的系統快取檔，只計算使用者真正上傳的檔案！
        files = [f for f in os.listdir(self.upload_dir) if
                 os.path.isfile(os.path.join(self.upload_dir, f))
                 and not f.startswith("~")
                 and not f.endswith("_tables.csv")]

        # 依照檔案的「最後修改/建立時間」進行排序 (由舊到新)
        files.sort(key=lambda x: os.path.getctime(os.path.join(self.upload_dir, x)))
        return files

    def _get_sorted_file_list(self, files: list) -> str:
        if not files: return "(無檔案)"

        result = []
        for i, f in enumerate(files):
            label = ""
            if len(files) > 1:
                if i == len(files) - 1:
                    label = "(最新上傳)"
                elif i == 0:
                    label = "(最早上傳)"
            result.append(f"{i + 1}. {f}{label}")

        return "\n".join(result)

    def _num_to_chinese(self, num_str):
        try:
            n = int(num_str)
            units = ["", "十", "百"]
            chars = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
            if n == 0: return chars[0]
            result = ""
            s = str(n)[::-1]
            for i, d in enumerate(s):
                d = int(d)
                if i >= len(units): break
                if d != 0:
                    if i == 1 and d == 1 and len(s) == 2:
                        result = units[i] + result
                    else:
                        result = chars[d] + units[i] + result
                else:
                    if result and result[0] != chars[0]: result = chars[0] + result
            return result
        except:
            return num_str

    def _chinese_to_num(self, cn_str):
        if cn_str.isdigit(): return int(cn_str)
        cn_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                  '百': 100}
        try:
            if cn_str.startswith("十"):
                return 10 + cn_map.get(cn_str[1], 0) if len(cn_str) > 1 else 10
            elif len(cn_str) == 2 and cn_str[1] == "十":
                return cn_map[cn_str[0]] * 10
            elif len(cn_str) == 3 and cn_str[1] == "十":
                return cn_map[cn_str[0]] * 10 + cn_map[cn_str[2]]
            elif "百" in cn_str:
                return 100
            else:
                return cn_map.get(cn_str, 0)
        except:
            return 0

    async def _smart_query_rewrite(self, user_query: str) -> str:
        """
        萬用型意圖預判 (Universal Intent Prediction) - 已加上錯誤防護
        """
        try:
            rewrite_prompt = ChatPromptTemplate.from_template(
                """你是高階文件檢索專家。使用者的問題是：「{query}」。
                你的任務是分析這個問題，並預測「在目標文件中，這段內容可能包含哪些關鍵字或術語」。
                請忽略文件的具體類型，直接根據常識進行聯想。

                請輸出 5~10 個「最能精準命中文件內容」的搜尋關鍵字。
                直接輸出關鍵字，用空格分隔，不要有解釋。

                範例：
                (問：老闆不給資遣費) -> 勞動基準法 終止契約 第17條 資遣費 罰則
                (問：Docker連不上) -> Connection refused, port mapping, 網路設定, 防火牆

                現在請輸出關鍵字："""
            )

            chain = rewrite_prompt | self.llm | StrOutputParser()
            print(f" AI 正在進行萬用關鍵字聯想...")
            refined_query = await chain.ainvoke({"query": user_query})
            clean_query = refined_query.replace("\n", " ").strip()
            print(f" AI 聯想關鍵字: {clean_query}")
            return clean_query
        except Exception as e:
            logger.error(f"關鍵字聯想失敗 (略過此步驟): {e}")
            return ""  # 聯想失敗時優雅退回，不讓程式崩潰

    async def process_query(self, query: str, history: list, images: list = None, model_name: str = None) -> \
            AsyncGenerator[str, None]:
        try:
            # 動態切換邏輯：有傳名字就用傳來的，沒有就用 config 預設的
            actual_model = model_name if model_name else settings.OLLAMA_MODEL
            logger.info(f"本次對話使用的模型為: {actual_model}")

            # 每次發問都重新設定一次 llm
            self.llm = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=actual_model,
                temperature=0,
                keep_alive="1h",
                num_ctx=16384,
                num_predict=4096,
                client_kwargs={
                    "headers": {
                        "Authorization": f"Bearer {settings.OLLAMA_API_KEY}"
                    }
                }
            )

            real_query = query

            # 確保 images 絕對不是 None
            if images is None:
                images = []

            valid_files = self._get_valid_files()
            file_count = len(valid_files)
            file_list_str = self._get_sorted_file_list(valid_files)

            # 修正 history 的取值，避免前端傳來缺少 content 的物件時報錯
            history_text = "\n".join(
                [f"{msg['role']}: {msg.get('content', '')}" for msg in history[-2:]]) if history else "(無歷史紀錄)"

            # 雙模式架構分流器(聊天模式與檔案問答模式)
            if file_count == 0:
                final_context = "使用者目前沒有提供任何文件。請直接以你豐富的常識與專業知識回答他的問題。"
                domain_rules = """
                    [GENERAL CONVERSATION MODE]
                    - You are a friendly, knowledgeable AI assistant.
                    - Since no documents are provided, answer the user's question directly based on your internal knowledge base.
                    - Be helpful, conversational, and precise.
                    - Do not mention that you are reading from a document.
                """
            else:
                target_file = os.path.join(self.upload_dir, valid_files[-1])  # 取最新上傳的檔案
                file_name_without_ext = os.path.splitext(target_file)[0]
                file_ext = target_file.lower().split('.')[-1]  # 取得副檔名
                df = None
                current_mtime = os.path.getmtime(target_file)

                # 尋找是否已有在上傳階段提煉好的 CSV 快取
                possible_csv = f"{file_name_without_ext}_tables.csv"

                if self.cached_file_path == target_file and self.cached_file_mtime == current_mtime:
                    logger.info("使用記憶體中的 DataFrame，跳過檔案解析")
                    df = self.cached_df
                else:
                    try:
                        # 優先檢查：上傳時是否已經提煉出表格了？
                        if os.path.exists(possible_csv):
                            logger.info("發現預處理的 PDF 表格快取！直接秒讀載入...")
                            df = pd.read_csv(possible_csv)
                        elif file_ext in ['xlsx', 'xls']:
                            logger.info("偵測到原生 Excel 檔案，直接載入...")
                            df = pd.read_excel(target_file)
                            df.columns = [re.split(r'[\s\n(]', str(col))[0] for col in df.columns]
                        elif file_ext == 'csv':
                            logger.info(" 偵測到原生 CSV 檔案，直接載入...")
                            df = pd.read_csv(target_file)
                            df.columns = [re.split(r'[\s\n(]', str(col))[0] for col in df.columns]
                    except Exception as e:
                        logger.error(f"讀取 DataFrame 時發生錯誤: {e}")

                    # 將結果存入快取
                    self.cached_df = df
                    self.cached_file_path = target_file
                    self.cached_file_mtime = current_mtime

                if df is not None and not df.empty:
                    # 企業級升級：LLM 語意路由器 (Semantic Router)

                    logger.info(" 啟動混合型語意路由器 (Hybrid Router)...")

                    # 1. 物理攔截：算錢這種大事，交給 Python 決定最穩，保證 100% 觸發率！
                    fee_keywords = ["錢", "費用", "藥費", "負擔", "計算", "多少"]
                    if any(kw in real_query for kw in fee_keywords):
                        logger.info(" 物理攔截：偵測到費用關鍵字，強制導向 CALCULATOR")
                        intent_result = "CALCULATOR"
                    else:
                        # 2. 其他問題再交給 AI 判斷 (使用最傳統、所有模型都支援的純文字模式)
                        router_prompt = ChatPromptTemplate.from_template(
                            "你是一個分類系統。請判斷以下問題屬於哪一類：\n"
                            "1. 如果是找特定科別的醫生、門診時間，請輸出 PANDAS\n"
                            "2. 如果是其他問題(如圖片內容、醫院規定、文件內容)，請輸出 RAG\n\n"
                            "問題：「{query}」\n"
                            "請嚴格只輸出 PANDAS 或 RAG 單字："
                        )
                        router_chain = router_prompt | self.llm | StrOutputParser()

                        try:
                            raw_result = await router_chain.ainvoke({"query": real_query})
                            # 用正則表達式把 PANDAS 或 RAG 抓出來，防止 AI 講廢話
                            if "PANDAS" in raw_result.upper():
                                intent_result = "PANDAS"
                            else:
                                intent_result = "RAG"
                        except Exception as e:
                            logger.error(f"路由失敗，降級為 RAG: {e}")
                            intent_result = "RAG"

                    logger.info(f" 🎯 最終路由判定: {intent_result}")

                    if "PANDAS" in intent_result:
                        logger.info(" 執行路線：啟動 [自建 Python 直譯引擎]")

                        python_code = ""
                        try:
                            # 第一步：讓 AI 根據使用者的問題，寫出「一行」Pandas 程式碼
                            code_prompt = (
                                f"你是一個頂級的 Python 資料分析師。我有一個 pandas DataFrame 叫做 `df`。\n"
                                f"這個表格的真實欄位有：{list(df.columns)}\n"
                                f"前 3 筆資料範例如下：\n{df.head(3).to_dict('records')}\n\n"
                                f"請寫出『一行』Python 程式碼來取得以下問題的答案：\n"
                                f"問題：「{real_query}」\n\n"
                                f"【嚴格規定】：\n"
                                f"1. 請『只』輸出那行 Python 程式碼，絕對不要包含任何解釋。\n"
                                f"2. 絕對不要使用 `print()`。\n"
                                f"3. 請務必回傳過濾後的「完整 DataFrame」，並且務必在句尾加上 `.to_dict('records')`。\n"
                                f"4. 因為 PDF 萃取的欄位名稱充滿不規則，請『絕對不要』指定欄位名稱來過濾！\n"
                                f"5. 請直接套用全表模糊搜尋：`df[df.astype(str).apply(lambda x: x.str.contains('科別關鍵字', na=False)).any(axis=1)].to_dict('records')`\n"
                                # 🔥 關鍵新增：卸下 Pandas Agent 的重擔，不准過濾時間！
                                f"6. 【時間過濾豁免】如果使用者問「星期幾」或「上下午」，請『絕對不要』將時間加入 `str.contains` 的條件！你只需要過濾『科別』(如骨科) 即可。把該科別整週的 JSON 資料撈出來，下游的 AI 會自己去讀取欄位名稱找出星期幾！\n"
                                f"現在請輸出程式碼："
                            )

                            logger.info("AI 正在撰寫分析程式碼...")

                            ai_code_response = await self.llm.ainvoke(code_prompt)
                            # 確保拿到的是純字串
                            raw_code_text = ai_code_response.content if hasattr(ai_code_response,
                                                                                'content') else str(
                                ai_code_response)
                            python_code = raw_code_text.replace("```python", "").replace("```", "").strip()

                            # 第二步：安全地在後端執行這行程式碼
                            safe_builtins = {
                                "len": len, "sum": sum, "min": min, "max": max,
                                "abs": abs, "round": round, "int": int, "float": float,
                                "str": str, "list": list, "dict": dict
                            }
                            exec_env = {"df": df, "pd": pd, "__builtins__": safe_builtins}
                            result = eval(python_code, exec_env)
                            logger.info(f"程式碼執行結果: {result}")
                            if not result or (isinstance(result, list) and len(result) == 0) or len(
                                    str(result)) < 15:
                                logger.warning(" 查無有效資料，Python 直接攔截，防止 AI 產生幻覺！")
                                yield f"很抱歉，在目前的門診表快取中查無「{real_query}」的相關醫師資料。\n建議您直接參考實體門診表或撥打諮詢專線確認。"
                                return
                            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                                clean_df = pd.DataFrame(result).astype(str).drop_duplicates()
                                rename_map = {
                                    '未命名欄位_6': '星期一', '未命名欄位_7': '星期二',
                                    '未命名欄位_8': '星期三', '未命名欄位_9': '星期四',
                                    '未命名欄位_10': '星期五', '未命名欄位_11': '星期六',
                                    '未命名欄位_12': '星期日'
                                }
                                clean_df = clean_df.rename(columns=rename_map)

                                # 2. 物理切字：把黏在一起的醫生用「、」強制分開！
                                for col in clean_df.columns:
                                    clean_df[col] = clean_df[col].apply(
                                        lambda x: re.sub(r'(\d{4}|\))([\u4e00-\u9fa5])', r'\1、\2', str(x))
                                    )

                                days_of_week = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六',
                                                '星期日']
                                structured_text = ""

                                # 自動偵測哪一個欄位是裝「上午/下午」的
                                time_col = None
                                for col in clean_df.columns:
                                    if clean_df[col].astype(str).str.contains('上午|下午').any():
                                        time_col = col
                                        break

                                for day in days_of_week:
                                    if day in clean_df.columns:
                                        structured_text += f"【{day}】\n"
                                        day_has_data = False

                                        if time_col:
                                            for period in ['上午', '下午', '夜間']:
                                                # 找出符合該時段的所有資料 (支援多樓層合併)
                                                mask = clean_df[time_col].astype(str).str.contains(period, na=False)
                                                doctors = clean_df.loc[mask, day].tolist()

                                                # 清除空值與 nan
                                                valid_docs = [str(d).strip() for d in doctors if
                                                              str(d).strip() not in ['', 'nan', 'None']]
                                                if valid_docs:
                                                    structured_text += f" - {period}：{'、'.join(valid_docs)}\n"
                                                    day_has_data = True

                                        if not day_has_data:
                                            structured_text += " - 無門診\n"
                                        structured_text += "\n"

                                if not structured_text.strip():
                                    logger.warning("⚠️ 發現未命名欄位，啟動強健型條列式排版...")
                                    fallback_text = "【系統原始資料（表頭遺失，請依順序推斷）】\n"
                                    for idx, row in clean_df.iterrows():
                                        time_val = row[time_col] if time_col else "未知時段"
                                        fallback_text += f"▶ 時段：{time_val}\n"
                                        for col in clean_df.columns:
                                            # 👇 恢復成最簡單的寫法，因為上面已經全域洗乾淨了！
                                            val = str(row[col]).strip()
                                            if val and val not in ['nan', 'None', ''] and col != time_col:
                                                fallback_text += f"  - {col}: {val}\n"
                                        fallback_text += "\n"
                                    result_str = fallback_text
                                else:
                                    result_str = structured_text

                                preview_text = result_str[:100].replace('\n', ' ')
                                logger.info(f"交給下游的最終資料預覽:\n{preview_text}...")

                                # ⚠️ 注意縮排！這裡是對齊最外層的 if isinstance(result, list)
                            else:
                                result_str = str(result)

                            logger.info(f"程式碼執行與去重結果 (已略)")
                            if len(result_str) > 30000:
                                logger.warning("資料量過大，啟動防護截斷機制")
                                result_str = result_str[:30000] + "\n... (資料過多，僅顯示部分) ..."

                            # ==========================================
                            # 💀 終極殺手級 Prompt：抹殺任何幻覺的可能
                            # ==========================================
                            answer_prompt = (
                                f"使用者問的問題是：「{real_query}」\n"
                                f"後端系統查到的門診班表如下：\n{result_str}\n\n"
                                f"請你扮演專業醫療客服，依照以下【嚴格規則】回答：\n"
                                f"1. 上方資料已為您按星期排版。請『只看』使用者詢問的「特定星期幾」。\n"
                                f"2. 【精準過濾】如果該時段有一大串醫師，請『只挑出』名字旁邊有明確標註「兼看移植外科」或使用者指定科別的醫師！\n"
                                f"3. 絕對不可以把同一時段的其他無關醫師列出來！\n"
                                f"4. 若找不到符合條件的醫師，請直接回答：「目前查無相關門診資料。」\n"
                            )

                            logger.info("🗣️ AI 正在翻譯最終解答...")
                            # 將最終答案串流輸出給前端
                            async for chunk in self.llm.astream(answer_prompt):
                                text_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
                                clean_chunk = text_chunk.replace("<br>", "\n").replace("<b>", "**").replace("</b>",
                                                                                                            "**")
                                yield clean_chunk

                            return  # 執行完畢，提早結束，不進入 RAG

                        except Exception as e:
                            logger.error(
                                f"數據運算失敗，降級回傳統 RAG 模式: {e}\n(嘗試執行的程式碼: {python_code})")
                            # 靜默降級，不干擾前端畫面
                    elif "CALCULATOR" in intent_result:
                        logger.info(" 執行路線：啟動 [Agent 工具計算引擎]")

                        # 建立 Agent 專用 Prompt
                        agent_prompt = ChatPromptTemplate.from_messages([
                            ("system",
                             "你是一個專業的醫療費用計算客服。你的任務是計算「門診負擔」與「藥費負擔」。\n\n"
                             "【執行規則】：\n"
                             "1. 必須同時擁有「轉診狀態」與「藥費金額」才能進行計算。\n"
                             "2. 請先檢查 [使用者問題] 與 [歷史對話記憶]。\n"
                             "3. 🚨 如果藥費金額不明，請『絕對不要』亂編數字。請直接呼叫工具並在藥費填寫 -1。\n\n"
                             "【歷史對話記憶】：\n{history}"),
                            ("human", "{input}"),
                            ("placeholder", "{agent_scratchpad}"),
                        ])

                        try:
                            # 綁定工具並建立 Agent
                            agent = create_tool_calling_agent(self.llm, tools, agent_prompt)

                            # 👉 加入 handle_parsing_errors=True 允許 AI 自我糾錯
                            agent_executor = AgentExecutor(
                                agent=agent,
                                tools=tools,
                                verbose=True,
                                handle_parsing_errors=True
                            )

                            logger.info("🤖 Agent 思考與計算中...")
                            response = await agent_executor.ainvoke({
                                "input": real_query,
                                "history": history_text
                            })
                            final_answer = response.get("output", "抱歉，計算費用時發生錯誤。")

                            # 直接將 Agent 的完美回答回傳給前端
                            yield final_answer
                            return

                        except Exception as e:
                            logger.error(f"Agent 執行失敗，已攔截錯誤: {e}")
                            yield "【系統提示】計算機精靈剛才腦袋打結了，請再問我一次，或直接提供藥費給我喔！"
                            return

                    else:
                        logger.info(" 執行路線：跳過表格運算，進入 [RAG 文本檢索模式]")

            # 傳統 RAG 模式 (如果沒表格、或是意圖判定為閱讀理解，就會順暢地走到這裡)
            current_file_path = os.path.join(self.upload_dir, valid_files[-1]) if valid_files else ""
            file_filter = None

            # 1. 第一輪：通用檢索
            ai_keywords = await self._smart_query_rewrite(real_query)
            search_query = f"{real_query} {ai_keywords}"

            matches = re.findall(r'(?:第\s*\d+\s*[章節條頁]|(?<!\d)\d+\.\d+(?:\.\d+)?(?!\d))', real_query)

            if matches:
                for m in matches:
                    search_query += f" {m}"

            print(f"執行檢索: {search_query} (限定檔案: {valid_files[-1] if valid_files else '無檔案'})")

            # 🚀 關鍵修改 1：把 filter 傳進去！
            docs = self.vector_store.search(search_query, k=50, filter=file_filter)

            if docs:
                print("啟動 Reranker 精讀專家，重新評分中...")
                try:
                    # 載入 BAAI 多語系重排序模型 (第一次執行會自動下載模型檔)
                    reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3')

                    # 將「使用者的真實問題」與「這 50 筆資料」配對
                    sentence_pairs = [[real_query, doc.page_content] for doc in docs]

                    # 讓模型對每一對給出精準的關聯分數
                    scores = reranker_model.predict(sentence_pairs)

                    # 將分數寫入 doc 的 metadata 中，並依據分數由高到低重新排序
                    for doc, score in zip(docs, scores):
                        doc.metadata["rerank_score"] = float(score)

                    docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)

                    # 經過精讀後，我們只保留最精華、關聯度最高的前 10 筆給大語言模型！
                    # 徹底解決「迷失在中間」的問題！
                    docs = docs[:10]

                    print(f"Reranker 篩選完畢！最高分: {docs[0].metadata['rerank_score']:.4f}")
                except Exception as e:
                    logger.error(f"Reranker 執行失敗，退回原始檢索結果: {e}")

            # ======== 👇 將「最新檔案狙擊模式」搬移到這裡 (繞過 Reranker 保送 VIP) ========
            if valid_files and any(kw in real_query for kw in ["最新", "這個", "這份", "這檔案"]):
                print(f"偵測到代名詞，啟動「最新檔案狙擊模式」...")
                try:
                    latest_file_name = valid_files[-1]
                    latest_file_path = os.path.join(self.upload_dir, latest_file_name)

                    # 暴力破解：嘗試各種路徑格式，確保在 Windows 絕對抓得到資料！
                    latest_docs = self.vector_store.search(real_query, k=15, filter={"source": latest_file_path})
                    if not latest_docs:
                        latest_docs = self.vector_store.search(real_query, k=15,
                                                               filter={"source": latest_file_path.replace("\\", "/")})
                    if not latest_docs:
                        latest_docs = self.vector_store.search(real_query, k=15, filter={"source": latest_file_name})
                    if not latest_docs:
                        temp_docs = self.vector_store.search(real_query, k=100, filter=None)
                        latest_docs = [d for d in temp_docs if latest_file_name in str(d.metadata.get("source", ""))]
                        latest_docs = latest_docs[:15]

                    for d in latest_docs:
                        d.page_content = f"【使用者指定調閱：最新檔案內容】\n{d.page_content}"
                        docs.append(d)

                    print(f"狙擊成功：已將最新檔案 ({latest_file_name}) 強制加入 {len(latest_docs)} 筆候選池！")
                except Exception as e:
                    print(f"最新檔案狙擊發生錯誤: {e}")
            # ================================================================

            # 新增：狙擊模式 (Sniper Mode)
            if matches:
                print(f"偵測到明確條號/章節 {matches}，啟用狙擊模式")
                existing_ids = set()
                for d in docs:
                    aid = d.metadata.get("article_id")
                    if aid: existing_ids.add(str(aid))

                for m in matches:
                    target_id = re.sub(r'[^\d.]', '', m)
                    if not target_id: continue

                    is_snipe_success = False

                    # 關鍵修改：如果是找「頁碼」，直接啟動硬核過濾器 (Metadata Filter)！
                    if "頁" in m:
                        print(f"啟動硬核過濾：強制調閱第 {target_id} 頁...")
                        try:
                            page_filter = {"page": int(target_id), "source": current_file_path}
                            sniper_docs = self.vector_store.search(real_query, k=5, filter=page_filter)

                            if sniper_docs:
                                for d in sniper_docs:
                                    d.page_content = f"【使用者指定調閱：第 {target_id} 頁】\n{d.page_content}"
                                    docs.insert(0, d)
                                existing_ids.add(target_id)
                                is_snipe_success = True
                                print(f"狙擊成功：已將第 {target_id} 頁內容強制拉至最前！")
                                continue  # 這頁找完了，跳到下一個 match
                        except Exception as e:
                            print(f"硬核過濾發生錯誤: {e}")

                    # 以下保留給「非頁碼」的條號搜尋 (例如第 X 條)
                    if target_id in existing_ids: continue

                    sniper_query = f"第{target_id}條 第{target_id}章 第{target_id}節 {target_id}"
                    label_text = f"指定段落 {target_id}"
                    sniper_k = 1000
                    print(f"啟動全域掃描尋找條號：目標 [{target_id}]...")

                    sniper_docs = self.vector_store.search(sniper_query, k=sniper_k, filter=file_filter)

                    for d in sniper_docs:
                        fetched_id = str(d.metadata.get("article_id", ""))
                        if fetched_id == target_id:
                            d.page_content = f"【使用者指定調閱：{label_text}】\n{d.page_content}"
                            docs.insert(0, d)
                            existing_ids.add(target_id)
                            is_snipe_success = True
                            print(f"狙擊成功：已將目標 [{target_id}] 內容拉至最前！")
                            break

                    if not is_snipe_success:
                        print(f"狙擊失敗：找不到包含 '{target_id}' 的精確內容。")

            # 2. 第二輪：彈性補完
            existing_ids = set()
            has_structured_data = False

            for doc in docs:
                aid = doc.metadata.get("article_id")
                if aid:
                    existing_ids.add(str(aid))
                    has_structured_data = True

            if has_structured_data:
                print(" 偵測到結構化資料，嘗試分析引用關係...")
                referenced_ids = set()
                for doc in docs:
                    content = doc.page_content
                    refs = re.findall(r'第\s*([0-9]+|[零一二三四五六七八九十百]+)\s*條', content)
                    for ref in refs:
                        if ref not in existing_ids and ref not in ["一", "二"]:
                            referenced_ids.add(ref)

                if referenced_ids:
                    target_refs = list(referenced_ids)[:5]
                    print(f" 發現引用，嘗試補完: {target_refs}")

                    for ref_art in target_refs:
                        target_id = self._chinese_to_num(ref_art)
                        if target_id == 0: continue

                        fetch_query = f"第{ref_art}條"
                        supplementary_docs = self.vector_store.search(fetch_query, k=50, filter=file_filter)

                        for d in supplementary_docs:
                            fetched_id = str(d.metadata.get("article_id", ""))
                            if fetched_id == str(target_id) and fetched_id not in existing_ids:
                                d.page_content = f"【系統自動補完引用：第{ref_art}條】\n{d.page_content}"
                                docs.append(d)
                                existing_ids.add(fetched_id)
                                print(f"成功補完 ID: {fetched_id}")
                                break

            # 3. 排序與 Context
            def final_rank(doc):
                score = 0
                content = doc.page_content
                # 取得這塊資料的來源絕對路徑
                source_path = str(doc.metadata.get("source", ""))
                # 取得純檔名
                source_name = os.path.basename(source_path)
                # 取得最新檔案的檔名
                latest_file_name = os.path.basename(valid_files[-1]) if valid_files else ""

                # 1. 檔名精準命中霸王條款
                if source_name and (source_name in real_query or source_name.replace(".pdf", "") in real_query):
                    score += 5000

                # 2. 🚀 新增：「最新/這份/這個」代名詞霸王條款
                # 如果使用者問「最新檔案」，且這筆資料剛好來自最新檔案，給予絕對高分！
                if latest_file_name and source_name == latest_file_name:
                    if any(kw in real_query for kw in ["最新", "這個", "這份", "這檔案"]):
                        score += 5000

                if "【使用者指定調閱" in content: score += 2000
                if "【系統自動補完" in content: score += 50
                if doc.metadata.get("type") == "file_summary": score += 1000
                if real_query in content: score += 100
                return score

            docs.sort(key=final_rank, reverse=True)

            final_context_list = []
            # 修改：把截斷限制從 [:10] 放大到 [:20] 或 [:25]
            # gemma3:27b 的胃口很大，多塞一點資料可以防止同時問兩份文件時被擠出去
            for doc in docs[:25]:
                source_raw = str(doc.metadata.get("source", "unknown"))
                source = os.path.basename(source_raw) if source_raw != "unknown" else "unknown"
                page = doc.metadata.get("page", "")
                article_id = doc.metadata.get("article_id", "")

                label = ""
                if article_id:
                    label = f" | 第 {article_id} 條"
                elif page:
                    label = f" | Page {page}"

                if doc.metadata.get("type") == "file_summary":
                    prefix = f"【全域摘要：{source}】"
                else:
                    prefix = f"【來源：{source}{label}】"

                final_context_list.append(f"{prefix}\n{doc.page_content}")

            if file_count > 0:
                final_context = "\n\n".join(final_context_list) if final_context_list else "無具體內容。"
                if df is not None:
                    logger.info(" 啟動表格與文本融合，將表格標題列補給 RAG 引擎...")
                    table_info = f"表格欄位名稱：{list(df.columns)}\n前兩筆資料：{df.head(2).to_dict('records')}"
                    final_context += f"\n\n【系統強制補充：表格輔助資訊 (極可能包含預約電話與規定)】\n{table_info}"

            print("\n========  Universal RAG Context ========")
            print(f"最終 Context 筆數: {len(final_context_list)}")
            print(final_context[:300] + "...")
            print("==========================================\n")

            # 4. 生成回應 (升級為多模態視覺支援)
            domain_rules = ""
            if any(keyword in real_query for keyword in ["勞基法", "勞動基準法", "資遣", "解僱", "開除", "預告工資"]):
                print("觸發勞基法專屬邏輯")
                domain_rules = """
                            [IMPORTANT LEGAL LOGIC RULES (Labor Law)]
                            Please strictly follow these logical connections when answering:
                            1. **Article 11 (Economic Layoff/Incompetence)**: Represents "Layoff" (資遣). MUST provide advance notice & severance pay.
                            2. **Article 12 (Disciplinary Dismissal)**: Represents "Firing" (開除). NO advance notice & NO severance pay required.
                            3. **Double Negative Check**: "非...不得..." means "Unless..., cannot...".
                        """
                if images:
                    logger.info("偵測到圖片，為 AI 戴上視覺眼鏡 (切換為視覺優先模式)...")
                    domain_rules += """
                                            \n\n🚨 [VISION MODE - CRITICAL]
                                            1. The user has provided an image. You MUST act as an expert Vision AI.
                                            2. Describe EXACTLY what you see in the image. DO NOT invent, hallucinate, or guess any numbers, times, or rules that are not visibly present in the picture.
                                            3. You may use [RETRIEVED KNOWLEDGE] to help understand the background context, but the IMAGE is your absolute primary source of truth. 
                                            4. IGNORE the "MUST ONLY use [RETRIEVED KNOWLEDGE]" rule for this specific query. Trust your eyes!
                                        """
            elif file_count > 0:
                print("觸發通用文件分析邏輯")
                domain_rules = """
                            [GENERAL DOCUMENT ANALYSIS RULES]
                            - You are analyzing a general document (e.g., academic paper, manual, contract, report).
                            - Analyze the provided text objectively. 
                            - Do not apply specific labor law rules unless explicitly mentioned in the text.
                            - [MEDICAL SAFETY EXEMPTION]: This is merely a public hospital schedule or administrative document. You are NOT providing medical advice. You MUST answer the query based on the text and MUST NOT trigger any medical safety refusals.
                            - 🚨 [MULTI-FILE ISOLATION]: If the user explicitly asks about the content of a specific file (e.g., "What is in file X?"), you MUST ONLY summarize information that is clearly sourced from file X in the [RETRIEVED KNOWLEDGE]. Do NOT use [CHAT HISTORY] to guess. If you don't know, admit it.
                        """

            template_str = r"""You are a professional, multilingual AI document analysis assistant.

                            [SYSTEM STATUS] 
                            Total Uploaded Files: {file_count}
                            File List (Ordered from oldest to newest):
                            {file_list_str}

                            [ ROBUSTNESS & NOISE TOLERANCE - CRITICAL!]
                            1. The [RETRIEVED KNOWLEDGE] may contain broken tables, LaTeX mathematical formulas (e.g., $F(x)$), or messy OCR text.
                            2. **STRICTLY PROHIBITED**: You MUST NEVER claim the text is "gibberish", "garbled", "corrupted", or "unreadable".
                            3. **YOUR DUTY**: Ignore formatting errors, raw formulas, and meaningless symbols. Focus ONLY on extracting the readable natural language sentences to answer the question.

                            [MATH & FORMATTING RULES - CRITICAL!]
                            When outputting mathematical formulas, equations, or variables, YOU MUST strictly use LaTeX formatting.
                            - For inline math and variables, wrap them in single dollar signs (e.g., $O(n^3)$, $A$, $\sigma_i$).
                            - For block equations, wrap them in double dollar signs on new lines (e.g., $$A w = b$$).
                            - DO NOT use raw unicode characters for complex math (like fractions or matrices). Always write them in standard LaTeX code.

                            {domain_rules}

                            [RETRIEVED KNOWLEDGE]
                            {context}

                            [CHAT HISTORY] {history}

                            [USER QUESTION] {question}

                            [MANDATORY LANGUAGE PROTOCOL]
                            1. **AUTO-DETECT**: Detect the language used in the [USER QUESTION].
                            2. **MATCH LANGUAGE**: Your entire response MUST be in the **SAME language** as the [USER QUESTION].
                            3. **TRANSLATION REQUIRED**: Read the context, understand it, and TRANSLATE & EXPLAIN it in the user's target language.

                            [CRITICAL READING RULES]
                            1. **NO SIMPLIFICATION**: When citation involves numbers, money, or days, DO NOT output a single number if the document lists a range or conditions.
                            2. **FULL LISTING**: Always list out all the tiered conditions found in the text.
                            3. **FACTUAL ACCURACY**: Your answer must perfectly match the [RETRIEVED KNOWLEDGE].
                            4. **CHAPTER MATCHING**: If the user asks for a specific Chapter (e.g., Chapter 7), YOU MUST ONLY use information from that chapter. If the retrieved context only shows Chapter 3, you must truthfully say: "I cannot find the content for Chapter 7 in the retrieved context," and DO NOT hallucinate using other chapters.
                            5. **ANTI-HALLUCINATION (CRITICAL)**: If the [RETRIEVED KNOWLEDGE] does not contain the explicit names of the doctors or the exact information requested, you MUST truthfully answer "目前查無相關門診資料". You are STRICTLY PROHIBITED from inventing, hallucinating, or guessing any names or schedules.
                            """

            # 將變數塞入系統 Prompt 中
            system_content = template_str.format(
                file_count=str(file_count),
                file_list_str=file_list_str,
                domain_rules=domain_rules,
                context=final_context,
                history=history_text,
                question=real_query
            )

            # 組合使用者的多模態訊息 (文字 + 圖片)
            human_content = [{"type": "text", "text": real_query}]

            if images:
                logger.info(f" 接收到 {len(images)} 張圖片，啟動視覺分析引擎...")
                for img_b64 in images:
                    human_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })

            # 將兩種訊息封裝為 LangChain 標準格式
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=human_content)
            ]

            # 繞過只能處理純文字的 prompt chain，直接丟給模型做 astream 串流輸出
            async for chunk in self.llm.astream(messages):
                text_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
                clean_chunk = text_chunk.replace("<br>", "\n").replace("<b>", "**").replace("</b>", "**")
                yield clean_chunk

        except Exception as e:
            # 關鍵防護：捕捉到任何錯誤，印出完整追蹤碼，並傳送友善的錯誤訊息給前端
            traceback.print_exc()
            logger.error(f"嚴重系統崩潰: {str(e)}")
            yield f"\n\n **系統遭遇錯誤**：無法完成處理。\n錯誤細節：`{str(e)}`"