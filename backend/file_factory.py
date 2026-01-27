import os
import logging
import pandas as pd
import requests
import base64
import io
from PIL import Image  # 用來檢查圖片格式
from abc import ABC, abstractmethod
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# 設定 Log
logger = logging.getLogger(__name__)


class FileLoader(ABC):
    def __init__(self, file_path, original_filename="未知檔案"):
        self.file_path = file_path
        self.original_filename = original_filename

    @abstractmethod
    def extract_text(self) -> str:
        try:
            logger.info(f"開始處理圖片: {self.original_filename}，正在呼叫多模態模型 (llava-phi3)...")

            with open(self.file_path, "rb") as image_file:
                try:
                    img = Image.open(io.BytesIO(image_file.read()))
                    img.verify()
                    image_file.seek(0)
                except Exception:
                    return f"【檔案名稱: {self.original_filename}】\n(無效的圖片檔案)\n"

                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            url = "http://localhost:11434/api/generate"
            payload = {
                # 🟢 修改 1: 換成視力更好的模型
                "model": "llava-phi3",

                # 🟢 修改 2: Prompt 優化，強調逐字抄寫
                "prompt": """
                You are a text transcription machine.
                Your ONLY job is to read the text in the image and output it.

                Rules:
                1. Transcribe any text you see EXACTLY.
                2. If the text is Chinese or Japanese, output the Chinese/Japanese characters directly.
                3. DO NOT describe the colors, background, or characters.
                4. If there is no text, reply "No text found".

                Output format:
                Text: [The text you read]
                """,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.0  # 零容忍，禁止瞎掰
                }
            }

            # 記得 timeout 還是要留長一點
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            description = response.json().get("response", "").strip()

            # 把結果印在終端機給你看，讓你確定它到底讀到了什麼
            logger.info(f"======== 模型讀到的內容 ========\n{description}\n===============================")

            if not description:
                return f"【檔案名稱: {self.original_filename}】\n(AI 無法辨識此圖片內容)\n"

            header = f"【檔案名稱: {self.original_filename}】\n這是一張圖片檔案，以下是圖片上的文字內容：\n"
            return header + description + "\n"

        except Exception as e:
            logger.error(f"圖片解析失敗: {e}")
            return f"【檔案名稱: {self.original_filename}】\n(圖片解析失敗: {str(e)})\n"


# ... [PDF, Docx, Text, CSV, Excel Loader 保持不變，為了節省篇幅省略] ...
# ... [請保留你原本的這些 Loader Class] ...

class PDFFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            loader = PyPDFLoader(self.file_path)
            pages = loader.load_and_split()
            content = "\n".join([page.page_content for page in pages])
            return f"【檔案名稱: {self.original_filename}】\n{content}"
        except Exception as e:
            logger.error(f"PDF 解析失敗: {e}")
            return ""


class DocxFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            loader = Docx2txtLoader(self.file_path)
            documents = loader.load()
            content = "\n".join([doc.page_content for doc in documents])
            return f"【檔案名稱: {self.original_filename}】\n{content}"
        except Exception as e:
            logger.error(f"DOCX 解析失敗: {e}")
            return ""


class TextFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            loader = TextLoader(self.file_path, encoding='utf-8')
            documents = loader.load()
            content = "\n".join([doc.page_content for doc in documents])
            return f"【檔案名稱: {self.original_filename}】\n{content}"
        except Exception as e:
            logger.error(f"TXT 解析失敗: {e}")
            return ""


class CSVFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            df = pd.read_csv(self.file_path)
            df = df.astype(str)
            df = df.fillna("無")
            try:
                markdown_table = df.to_markdown(index=False)
            except ImportError:
                markdown_table = df.to_csv(index=False)
            header = f"【檔案名稱: {self.original_filename}】\n這是一份 CSV 數據表，內容如下：\n"
            return header + markdown_table
        except Exception as e:
            logger.error(f"CSV 解析失敗: {e}")
            return ""


class ExcelFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            all_text = []
            dfs = pd.read_excel(self.file_path, sheet_name=None)
            for sheet_name, df in dfs.items():
                df = df.astype(str)
                df = df.fillna("無")
                try:
                    markdown_table = df.to_markdown(index=False)
                except ImportError:
                    markdown_table = df.to_csv(index=False)
                sheet_content = f"\n\n【檔案名稱: {self.original_filename} | 工作表: {sheet_name}】\n這是一份表格數據，內容如下：\n{markdown_table}\n"
                all_text.append(sheet_content)
            return "\n".join(all_text)
        except Exception as e:
            logger.error(f"Excel 解析失敗: {e}")
            return ""


# 🟢 新增：圖片讀取器 (使用 Ollama llava 模型)
# 🟢 修改 file_factory.py 中的 ImageFileLoader

class ImageFileLoader(FileLoader):
    def extract_text(self) -> str:
        try:
            logger.info(f"開始處理圖片: {self.original_filename}，正在呼叫多模態模型 (minicpm-v)...")

            with open(self.file_path, "rb") as image_file:
                try:
                    img = Image.open(io.BytesIO(image_file.read()))
                    img.verify()
                    image_file.seek(0)
                except Exception:
                    return f"【檔案名稱: {self.original_filename}】\n(無效的圖片檔案)\n"

                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "minicpm-v",

                # 🔥 關鍵修改：混合模式 (Hybrid Prompt)
                # 讓 AI 先描述畫面 (防呆)，再嘗試讀字。
                # 這樣就算字讀錯，RAG 還是知道這張圖在幹嘛。
                "prompt": """
                Please analyze this image comprehensively.

                Step 1: Describe the visual content (What is happening? Who are the characters? What is the mood?).
                Step 2: If there is text, transcribe it EXACTLY. If the text is stylized or unclear, interpret its meaning.

                Output Format:
                [Visual Description]: ...
                [Detected Text]: ...
                """,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.2  # 稍微調高一點點，讓它在描述畫面時自然一點
                }
            }

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            description = response.json().get("response", "").strip()

            logger.info(f"======== 模型讀到的內容 ========\n{description}\n===============================")

            if not description:
                return f"【檔案名稱: {self.original_filename}】\n(AI 無法辨識此圖片內容)\n"

            # 這裡加上 Markdown 格式，讓之後 RAG 檢索時更清楚
            header = f"【檔案名稱: {self.original_filename}】\n這是一張圖片，AI 分析結果如下：\n"
            return header + description + "\n"

        except Exception as e:
            logger.error(f"圖片解析失敗: {e}")
            return f"【檔案名稱: {self.original_filename}】\n(圖片解析失敗: {str(e)})\n"


class FileLoaderFactory:
    @staticmethod
    def get_loader(file_path: str, original_filename: str) -> FileLoader:
        ext = os.path.splitext(original_filename)[1].lower()

        if ext == ".pdf":
            return PDFFileLoader(file_path, original_filename)
        elif ext in [".docx", ".doc"]:
            return DocxFileLoader(file_path, original_filename)
        elif ext in [".txt", ".md"]:
            return TextFileLoader(file_path, original_filename)
        elif ext in [".xlsx", ".xls"]:
            return ExcelFileLoader(file_path, original_filename)
        elif ext == ".csv":
            return CSVFileLoader(file_path, original_filename)
        # 🟢 新增：支援常見圖片格式
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            return ImageFileLoader(file_path, original_filename)
        else:
            raise ValueError(f"不支援的檔案格式: {ext}")