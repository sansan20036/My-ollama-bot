# app/services/file_service.py
import os
import shutil
import logging
import pandas as pd
import requests
import base64
import io
import fitz
import numpy as np
import cv2
import pdfplumber  # 新增：用於精準提取 PDF 表格的套件
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
from abc import ABC, abstractmethod
from fastapi import UploadFile
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
import re  #  新增：處理欄位名稱的正規表達式

from app.core.config import settings
from app.services.vector_store import VectorStoreService

# 設定 Log
logger = logging.getLogger(__name__)


# 新增：背景表格提煉引擎 (ETL Processor)
def extract_and_save_tables(pdf_path: str):
    """在上傳階段，直接將 PDF 內的表格提煉並存成同名的 _tables.csv"""
    csv_path = pdf_path.rsplit('.', 1)[0] + "_tables.csv"
    all_data = []

    try:
        logger.info(f"⚙️ [背景任務] 正在掃描 {os.path.basename(pdf_path)} 是否包含表格...")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 🛡️ 第一道粗篩：連 4 條線都沒有的純文字，直接跳過
                if len(page.lines) < 4 and len(page.rects) < 1:
                    continue

                # 🛡️ 第二道精篩：利用 find_tables() 尋找真實網格，排除裝飾線
                if not page.find_tables():
                    continue

                # ⚔️ 確定有表格結構，才啟動文字萃取
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        # 將 None 換成空字串，並去除多餘換行
                        cleaned_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                        if any(cleaned_row):
                            all_data.append(cleaned_row)

        # 如果有抓到表格資料，就把它存成 CSV 備用
        if len(all_data) > 1:
            df = pd.DataFrame(all_data[1:], columns=all_data[0])
            df = df[df.iloc[:, 0].astype(str).str.strip() != ""]
            df.columns = [re.split(r'[\s\n(]', str(col))[0] for col in df.columns]

            # 存成 CSV (使用 utf-8-sig 確保 Excel 打開不會亂碼)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"✅ [背景任務] 提煉成功！已產生表格快取: {os.path.basename(csv_path)}")
            return True
        else:
            logger.info("ℹ️ [背景任務] 此 PDF 無表格，標記為純文字檔。")
            return False

    except Exception as e:
        logger.error(f"⚠️ [背景任務] 表格提煉失敗: {e}")
        return False


# PART 1: 檔案讀取邏輯 (Loader)

class FileLoader(ABC):
    def __init__(self, file_path, original_filename="未知檔案"):
        self.file_path = file_path
        self.original_filename = original_filename

    @abstractmethod
    def extract_text(self) -> str:
        pass


class PDFFileLoader(FileLoader):
    def extract_text(self) -> str:
        text_content = []
        try:
            doc = fitz.open(self.file_path)
            # 初始化 OCR 引擎 (第一次跑會自動載入模型)
            ocr = RapidOCR()

            logger.info(f"🔍 開始解析 PDF (啟用 OCR 雙重防護): {self.original_filename}")

            for i, page in enumerate(doc):
                # 1. 先嘗試正規文字提取
                page_text = page.get_text().strip()

                # 2. 判斷是否為亂碼 (如果文字長度大於 20，但包含大量怪符號，或是像空空如也)
                # 設定一個簡單的機制：如果抽出來的字不到 50 個字，或是包含特殊亂碼特徵，就強制啟用 OCR
                if len(page_text) < 50 or "cid:" in page_text or "MNOP" in page_text:
                    logger.warning(f"⚠️ 第 {i + 1} 頁偵測到亂碼或空文字，啟動 OCR 視覺提取...")

                    # 將 PDF 該頁渲染成圖片 (dpi=150 足以辨識文字且不會太慢)
                    pix = page.get_pixmap(dpi=150)

                    # 將 PyMuPDF 的圖片轉換為 OpenCV 格式給 OCR 看
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:  # 如果有 Alpha 通道，轉為 RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                    elif pix.n == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    # 執行 OCR 辨識
                    result, _ = ocr(img_array)

                    if result:
                        # result 包含 [框座標, 文字, 信心度]，我們只要文字
                        page_text = "\n".join([line[1] for line in result])
                    else:
                        page_text = ""

                if page_text and page_text.strip():
                    text_content.append(page_text)
                else:
                    text_content.append(f"[第 {i + 1} 頁：無文字內容]")

            doc.close()

            if not text_content:
                return f"【檔案: {self.original_filename}】(無法提取文字)\n"

            return f"【檔案: {self.original_filename}】\n" + "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"PDF 解析失敗: {e}")
            return f"【檔案: {self.original_filename}】(PDF 解析錯誤: {e})\n"


class DocxFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            loader = Docx2txtLoader(self.file_path)
            docs = loader.load()
            return f"【檔案: {self.original_filename}】\n" + "\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error(f"DOCX 解析失敗: {e}")
            return f"【檔案: {self.original_filename}】(DOCX 解析錯誤: {e})\n"


class TextFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            loader = TextLoader(self.file_path, encoding='utf-8')
            return f"【檔案: {self.original_filename}】\n" + "\n".join([d.page_content for d in loader.load()])
        except:
            try:
                loader = TextLoader(self.file_path, encoding='big5')
                return f"【檔案: {self.original_filename}】\n" + "\n".join([d.page_content for d in loader.load()])
            except Exception as e:
                logger.error(f"TXT 解析失敗: {e}")
                return f"【檔案: {self.original_filename}】(TXT 解析錯誤: {e})\n"


class CSVFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            df = pd.read_csv(self.file_path, nrows=1000)
            return f"【檔案: {self.original_filename}】(前1000筆)\n" + df.to_csv(index=False)
        except Exception as e:
            logger.error(f"CSV 解析失敗: {e}")
            return f"【檔案: {self.original_filename}】(CSV 解析錯誤: {e})\n"


class ExcelFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            dfs = pd.read_excel(self.file_path, sheet_name=None, nrows=1000)
            content = []
            for sheet, df in dfs.items():
                content.append(f"\n【工作表: {sheet}】(前1000筆)\n" + df.to_csv(index=False))
            return f"【檔案: {self.original_filename}】\n" + "".join(content)
        except Exception as e:
            logger.error(f"Excel 解析失敗: {e}")
            return f"【檔案: {self.original_filename}】(Excel 解析錯誤: {e})\n"


class ImageFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            model_name = "minicpm-v"

            logger.info(f"🖼️ 開始處理圖片: {self.original_filename}，呼叫模型: {model_name}...")

            with open(self.file_path, "rb") as f:
                img_bytes = f.read()

            # 驗證圖片有效性
            Image.open(io.BytesIO(img_bytes)).verify()

            # 準備 Payload
            response = requests.post(
                "http://git.tedpc.com.tw:11434/api/tags",
                json={
                    "model": model_name,
                    # 強化 Prompt：明確要求它讀出圖片裡的文字
                    "prompt": "請詳細描述這張圖片。如果圖片中有「文字」，請務必將文字內容完整抄寫出來。請用繁體中文回答。",
                    "images": [base64.b64encode(img_bytes).decode('utf-8')],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 低溫模式，減少幻覺
                        "num_predict": 1024  # 給它足夠的長度寫字
                    }
                },
                # 延長超時：大模型看圖比較慢，給它 300 秒
                timeout=300
            )

            if response.status_code == 200:
                result = response.json().get('response', '')
                logger.info(f"✅ 圖片分析完成: {self.original_filename}")
                return f"【圖片: {self.original_filename}】\nAI ({model_name}) 視覺分析與文字提取結果：\n{result}\n"

            return f"【圖片: {self.original_filename}】(分析失敗 Status: {response.status_code})\n"

        except requests.exceptions.Timeout:
            return f"【圖片: {self.original_filename}】(錯誤：模型回應逾時，請檢查伺服器效能)\n"
        except Exception as e:
            logger.error(f"圖片解析失敗: {e}")
            return f"【圖片: {self.original_filename}】(錯誤: {e})\n"


class FileLoaderFactory:
    @staticmethod
    def get_loader(file_path: str, original_filename: str) -> FileLoader:
        ext = os.path.splitext(original_filename)[1].lower()
        if ext == ".pdf": return PDFFileLoader(file_path, original_filename)
        if ext in [".docx", ".doc"]: return DocxFileLoader(file_path, original_filename)
        if ext in [".csv"]: return CSVFileLoader(file_path, original_filename)
        if ext in [".xlsx", ".xls"]: return ExcelFileLoader(file_path, original_filename)
        if ext in [".jpg", ".png", ".jpeg"]: return ImageFileLoader(file_path, original_filename)
        return TextFileLoader(file_path, original_filename)


# PART 2: 檔案上傳服務 (Service)

class FileService:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.vector_store = VectorStoreService.get_instance()

    async def upload_file(self, file: UploadFile):
        """上傳檔案 (累加模式，保留舊檔)"""
        try:

            # 舊檔案會被保留
            file_path = os.path.join(self.upload_dir, file.filename)

            # 1. 儲存實體檔案
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info(f"✅ 檔案已儲存: {file.filename}")

            # 檔案儲存後，若為 PDF，立刻在背景提煉表格
            if file_path.lower().endswith(".pdf"):
                extract_and_save_tables(file_path)

            # 2. 呼叫向量資料庫進行處理 (RAG 純文字向量化)
            await self.vector_store.process_file(file_path)

            return {
                "filename": file.filename,
                "status": "uploaded",
                "message": f"成功上傳: {file.filename}"
            }

        except Exception as e:
            logger.error(f"上傳失敗: {str(e)}")
            raise e

    def get_files(self):
        """列出所有檔案"""
        if not os.path.exists(self.upload_dir):
            return []
        return [f for f in os.listdir(self.upload_dir) if os.path.isfile(os.path.join(self.upload_dir, f))]

    def clear_all_files(self):
        """只有按 PURGE 按鈕時才清空"""
        try:
            if os.path.exists(self.upload_dir):
                shutil.rmtree(self.upload_dir)
                os.makedirs(self.upload_dir)
                self.vector_store.reset()
                logger.info("🗑️ 系統已清空")
            return True
        except Exception as e:
            logger.error(f"清空失敗: {e}")
            return False