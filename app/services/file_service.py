import os
import shutil
import logging
import pandas as pd
import requests
import base64
import io
import pdfplumber
from PIL import Image
from abc import ABC, abstractmethod
from fastapi import UploadFile
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from app.core.config import settings
from app.services.vector_store import VectorStoreService

# 設定 Log
logger = logging.getLogger(__name__)


# ==========================================
# PART 1: 檔案讀取邏輯 (Loader)
# ==========================================

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
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                    else:
                        text_content.append(f"[第 {i + 1} 頁：無文字內容]")
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
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    # 🔥 強化 Prompt：明確要求它讀出圖片裡的文字
                    "prompt": "請詳細描述這張圖片。如果圖片中有「文字」，請務必將文字內容完整抄寫出來。請用繁體中文回答。",
                    "images": [base64.b64encode(img_bytes).decode('utf-8')],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 低溫模式，減少幻覺
                        "num_predict": 1024  # 給它足夠的長度寫字
                    }
                },
                # 🔥 延長超時：大模型看圖比較慢，給它 300 秒
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


# ==========================================
# PART 2: 檔案上傳服務 (Service) - 🔥 修正重點在這裡
# ==========================================

class FileService:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.vector_store = VectorStoreService.get_instance()

    async def upload_file(self, file: UploadFile):
        """上傳檔案 (累加模式，保留舊檔)"""
        try:
            # ✅ 這裡已經移除了 shutil.rmtree，所以舊檔案會被保留！

            file_path = os.path.join(self.upload_dir, file.filename)

            # 1. 儲存實體檔案
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info(f"✅ 檔案已儲存: {file.filename}")

            # 2. 呼叫向量資料庫進行處理
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
        """🔥 只有按 PURGE 按鈕時才清空"""
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