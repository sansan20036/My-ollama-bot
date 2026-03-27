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
import pdfplumber
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
from abc import ABC, abstractmethod
from fastapi import UploadFile
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
import re

from app.core.config import settings
from app.services.vector_store import VectorStoreService

# 設定 Log
logger = logging.getLogger(__name__)


# ==========================================
# 背景表格提煉引擎 (ETL Processor)
# ==========================================
def extract_and_save_tables(pdf_path: str):
    """在上傳階段，直接將 PDF 內的表格提煉並存成同名的 _tables.csv (寬容模式 + ETL正規化)"""
    csv_path = pdf_path.rsplit('.', 1)[0] + "_tables.csv"
    all_data = []

    try:
        logger.info(f"[背景任務] 啟動 PDF 表格提煉引擎: {os.path.basename(pdf_path)}")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    clean_table = []
                    for row in table:
                        cleaned_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                        if any(cleaned_row):
                            clean_table.append(cleaned_row)
                    if len(clean_table) > 1:
                        all_data.append(clean_table)

        if all_data:
            dfs = []
            for table_data in all_data:
                headers = table_data[0]
                headers = [str(h) if h else f"未命名欄位_{i}" for i, h in enumerate(headers)]
                df = pd.DataFrame(table_data[1:], columns=headers)
                dfs.append(df)

            # 🚀 垂直拼接所有表格
            final_df = pd.concat(dfs, ignore_index=True)

            try:
                logger.info(" [背景任務] 啟動 ETL 預處理：正規化表格與清洗雜訊...")

                # 🔥 0. 真・企業級表頭提升 (Header Promotion)
                header_row_idx = -1
                for idx, row in final_df.head(20).iterrows():
                    row_str = "".join(row.astype(str)).upper()
                    if "星期一" in row_str or "MON" in row_str or "TUE" in row_str:
                        header_row_idx = idx
                        break

                if header_row_idx != -1:
                    logger.info(f" [背景任務] 在第 {header_row_idx} 列找到真實表頭，執行提升...")

                    chopped_info = ""
                    if header_row_idx > 0:
                        # 將表頭上方的所有廢棄列，全部轉成字串並接起來
                        chopped_rows = final_df.iloc[:header_row_idx].fillna("").astype(str)
                        chopped_info = " ".join([" ".join(row) for row in chopped_rows.values]).replace('nan',
                                                                                                        '').strip()
                        logger.info(f" [背景任務] 成功搶救表頭上方資訊：{chopped_info[:30]}...")

                    new_headers = final_df.iloc[header_row_idx].astype(str).tolist()
                    clean_headers = []
                    for i, h in enumerate(new_headers):
                        h = h.strip().replace('\n', '')
                        if h == 'nan' or h == '':
                            clean_headers.append(f"分類標籤_{i}")
                        else:
                            clean_headers.append(h)

                    final_df.columns = clean_headers
                    # 斷頭台落下：砍掉上方的資料
                    final_df = final_df.iloc[header_row_idx + 1:].reset_index(drop=True)

                    # 🚑 將搶救出來的備註，作為一個新欄位「醫院預約重要備註」塞回資料表！
                    if chopped_info:
                        final_df['醫院預約重要備註'] = chopped_info

                # 1. 切分「左右並排」的表格
                if len(final_df.columns) >= 10:
                    logger.info(" [背景任務] 偵測到疑似左右並排表格，執行垂直切分與堆疊...")
                    mid_idx = len(final_df.columns) // 2
                    left_df = final_df.iloc[:, :mid_idx].copy()
                    right_df = final_df.iloc[:, mid_idx:mid_idx * 2].copy()
                    right_df.columns = left_df.columns.tolist()
                    final_df = pd.concat([left_df, right_df], ignore_index=True)

                # 2. 清除雜訊資料
                noise_keywords = ['講座', '衛教', '常規疫苗', '整合', '說明會', '特別門診']
                pattern = '|'.join(noise_keywords)
                final_df = final_df[
                    ~final_df.astype(str).apply(lambda x: x.str.contains(pattern, na=False)).any(axis=1)]

                # 3. 終極除垢與格式化：消滅隱形空白與 NaN 填補
                final_df.replace(r'\s+', '', regex=True, inplace=True)
                final_df.replace('', np.nan, inplace=True)
                final_df.replace('None', np.nan, inplace=True)
                final_df.dropna(how='all', inplace=True)

                # 4. 精準向下填補 (Forward Fill)：前 3 欄
                final_df.iloc[:, :3] = final_df.iloc[:, :3].ffill()
                final_df.fillna("", inplace=True)

            except Exception as etl_e:
                logger.error(f" [背景任務] ETL 清洗過程發生錯誤: {etl_e}")

            # 存成 CSV
            final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f" [背景任務] 表格提煉成功！已產生快取: {os.path.basename(csv_path)}")
            return True
        else:
            logger.info(" [背景任務] 此 PDF 中未偵測到實體格線表格。")
            return False

    except Exception as e:
        logger.error(f" [背景任務] 表格提煉失敗: {e}")
        return False


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
            doc = fitz.open(self.file_path)
            ocr = RapidOCR()
            logger.info(f" 開始解析 PDF (啟用 OCR 雙重防護): {self.original_filename}")

            for i, page in enumerate(doc):
                page_text = page.get_text().strip()
                if len(page_text) < 50 or "cid:" in page_text or "MNOP" in page_text:
                    logger.warning(f" 第 {i + 1} 頁偵測到亂碼或空文字，啟動 OCR 視覺提取...")
                    pix = page.get_pixmap(dpi=150)
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                    elif pix.n == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    result, _ = ocr(img_array)
                    if result:
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
            model_name = "llama3.2-vision:11b"
            logger.info(f"開始處理圖片: {self.original_filename}...")

            with open(self.file_path, "rb") as f:
                img_bytes = f.read()

            ocr_text = "(無明顯文字)"
            try:
                ocr = RapidOCR()
                img_array = cv2.imread(self.file_path)
                result, _ = ocr(img_array)
                if result:
                    ocr_text = "\n".join([line[1] for line in result])
                    logger.info(f" 圖片 OCR 預先辨識成功: {ocr_text}")
            except Exception as e:
                logger.warning(f"圖片 OCR 輔助失敗: {e}")

            target_url = getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip('/') + "/api/chat"
            enhanced_prompt = f"""You are an expert image analysis AI.
            The image contains the following text extracted by an OCR tool:
            [OCR TEXT START]
            {ocr_text}
            [OCR TEXT END]

            Please describe the humor, context, and visual elements of this image in Traditional Chinese (繁體中文). Use the OCR text to help you understand the meme."""

            response = requests.post(
                target_url,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": enhanced_prompt,
                                  "images": [base64.b64encode(img_bytes).decode('utf-8')]}],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024}
                },
                timeout=300
            )

            if response.status_code == 200:
                result = response.json().get('message', {}).get('content', '')
                logger.info(f"圖片分析完成: {self.original_filename}")
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
# PART 2: 檔案上傳服務 (Service)
# ==========================================
class FileService:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.vector_store = VectorStoreService.get_instance()

    async def upload_file(self, file: UploadFile):
        try:
            file_path = os.path.join(self.upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info(f"檔案已儲存: {file.filename}")

            if file_path.lower().endswith(".pdf"):
                extract_and_save_tables(file_path)

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
        if not os.path.exists(self.upload_dir):
            return []
        return [f for f in os.listdir(self.upload_dir) if os.path.isfile(os.path.join(self.upload_dir, f))]

    def clear_all_files(self):
        try:
            if os.path.exists(self.upload_dir):
                shutil.rmtree(self.upload_dir)
                os.makedirs(self.upload_dir)
                self.vector_store.reset()
                logger.info("系統已清空")
            return True
        except Exception as e:
            logger.error(f"清空失敗: {e}")
            return False