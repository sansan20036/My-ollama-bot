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
import nest_asyncio
from langchain_core.documents import Document
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
from abc import ABC, abstractmethod
from fastapi import UploadFile
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
import re
from typing import Optional
from dotenv import load_dotenv
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.utils.smart_parser import SmartFileParser

# 設定 Log
logger = logging.getLogger(__name__)
nest_asyncio.apply()
load_dotenv()


EMPTY_CELL_MARKERS = {"", "-", "—", "－", "–", "N/A", "NA", "nan", "None", "null"}

DAY_ALIASES = {
    "星期一": ["星期一", "週一", "周一", "禮拜一", "礼拜一", "Mon", "Monday", "一"],
    "星期二": ["星期二", "週二", "周二", "禮拜二", "礼拜二", "Tue", "Tuesday", "二"],
    "星期三": ["星期三", "週三", "周三", "禮拜三", "礼拜三", "Wed", "Wednesday", "三"],
    "星期四": ["星期四", "週四", "周四", "禮拜四", "礼拜四", "Thu", "Thursday", "四"],
    "星期五": ["星期五", "週五", "周五", "禮拜五", "礼拜五", "Fri", "Friday", "五"],
    "星期六": ["星期六", "週六", "周六", "禮拜六", "礼拜六", "Sat", "Saturday", "六"],
    "星期日": ["星期日", "星期天", "週日", "週天", "周日", "周天", "禮拜日", "禮拜天", "Sun", "Sunday", "日"],
}


def _sanitize_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _split_md_row(line: str) -> list[str]:
    """切割 Markdown 表格列，支援有無外框的 | ... | 格式。"""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]

    escaped_pipe_token = "__ESCAPED_PIPE__"
    s = s.replace(r"\|", escaped_pipe_token)
    cells = [_sanitize_cell_text(cell).replace(escaped_pipe_token, "|") for cell in s.split("|")]
    return cells


def _is_separator(line: str) -> bool:
    """判斷是否為 Markdown 表格分隔線，支援有無外框的格式。"""
    s = line.strip()
    if "|" not in s:
        return False

    parts = _split_md_row(s)
    if len(parts) < 2:
        return False

    for part in parts:
        p = part.replace(":", "").replace(" ", "")
        if "-" not in p:
            return False
        if any(ch != "-" for ch in p):
            return False
    return True


def _is_table_row(line: str, min_cols: int = 2) -> bool:
    s = line.strip()
    if not s or "|" not in s or s.startswith("```"):
        return False
    return len(_split_md_row(s)) >= min_cols


def _normalize_day_header(header: str) -> str:
    raw = _sanitize_cell_text(header)
    if not raw:
        return ""

    lowered = raw.lower()
    compact = lowered.replace(" ", "")

    for day, aliases in DAY_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.lower().replace(" ", "")
            if not alias_lower:
                continue
            # 單字元別名（如「一」「二」）僅在欄位完全相等時才命中，避免誤判。
            if len(alias_lower) == 1:
                if compact == alias_lower:
                    return day
                continue
            if alias_lower in compact:
                return day
    return ""


def _normalize_period(value: str) -> str:
    text = _sanitize_cell_text(value).lower().replace(" ", "")
    if not text:
        return ""

    periods = []
    if any(k in text for k in ["上午", "早上", "早診", "morning", "am"]):
        periods.append("上午")
    if any(k in text for k in ["下午", "午診", "afternoon", "pm"]):
        periods.append("下午")
    if any(k in text for k in ["夜間", "晚間", "晚上", "夜診", "evening", "night"]):
        periods.append("夜間")
    if any(k in text for k in ["全天", "全日", "整天", "all-day", "allday"]):
        periods.append("全天")

    deduped = []
    for p in periods:
        if p not in deduped:
            deduped.append(p)
    return "、".join(deduped)


def _extract_row_period(headers: list[str], cells: list[str]) -> str:
    time_col_keywords = ["時段", "診次", "診別", "班別", "時間", "session", "clinic"]
    candidates = []

    for h, c in zip(headers, cells):
        header_text = _sanitize_cell_text(h).lower()
        if any(k in header_text for k in time_col_keywords):
            candidates.append(c)

    # 有些表格時段放在最前面第一欄，保留前三欄做備援判斷
    candidates.extend(cells[:3])

    for c in candidates:
        normalized = _normalize_period(c)
        if normalized:
            return normalized
    return ""


def _extract_row_department(current_section: str, headers: list[str], cells: list[str]) -> str:
    dept_keywords = ["科別", "科", "門診別", "專科", "部門", "department"]
    for h, c in zip(headers, cells):
        header_text = _sanitize_cell_text(h).lower()
        if any(k in header_text for k in dept_keywords):
            val = _sanitize_cell_text(c)
            if val and val not in EMPTY_CELL_MARKERS:
                return val
    return current_section


def convert_md_table_to_sentences(md_text: str) -> str:
    """
    【萬能表格降維引擎】
    不再猜測哪裡是星期、哪裡是醫生。直接將每一列轉換為完整的 Key-Value 語意！
    """
    lines = md_text.split('\n')
    current_section = "全院共通"
    new_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            new_lines.append(lines[i])
            i += 1
            continue

        clean_title = line.replace("*", "").replace("#", "").strip()
        is_header = False

        if line.startswith("#"):
            is_header = True
        elif line.startswith("**") and line.endswith("**") and len(clean_title) <= 30:
            is_header = True
        elif 2 <= len(clean_title) <= 30 and any(clean_title.endswith(s) for s in ["科", "門診", "中心", "部", "特診", "外科", "內科"]):
            is_header = True

        if is_header and clean_title:
            current_section = clean_title
            new_lines.append(lines[i])
            i += 1
            continue

        # 支援兩種 Markdown 表格：
        # 1. | col1 | col2 |
        # 2. col1 | col2
        if i + 1 < len(lines) and _is_table_row(lines[i], min_cols=2) and _is_separator(lines[i + 1]):
            table_lines = [lines[i].strip(), lines[i + 1].strip()]
            i += 2

            header_cols = max(len(_split_md_row(table_lines[0])), 2)
            while i < len(lines):
                row = lines[i].strip()
                if not _is_table_row(row, min_cols=max(2, header_cols - 1)):
                    break
                table_lines.append(row)
                i += 1

            table_raw_text = "".join(table_lines)
            if "張三" in table_raw_text or "範例" in table_raw_text or "範例" in current_section:
                logger.info(" 🗑️ 攔截到假資料教學範例，已銷毀。")
                continue

            try:
                headers = _split_md_row(table_lines[0])
                if len(headers) < 2:
                    new_lines.extend(table_lines)
                    continue

                headers = [
                    _sanitize_cell_text(h) if _sanitize_cell_text(h) else f"欄位{idx + 1}"
                    for idx, h in enumerate(headers)
                ]

                new_lines.append(f"【原始表格欄位】{'｜'.join(headers)}")
                sentences_generated = 0
                last_seen = [""] * min(3, len(headers))

                for t_line in table_lines[2:]:
                    cells = _split_md_row(t_line)
                    if len(cells) < len(headers):
                        cells += [""] * (len(headers) - len(cells))
                    elif len(cells) > len(headers):
                        cells = cells[:len(headers)]

                    for col_idx in range(min(3, len(headers))):
                        val = _sanitize_cell_text(cells[col_idx])
                        if val in EMPTY_CELL_MARKERS:
                            cells[col_idx] = last_seen[col_idx]
                        else:
                            cells[col_idx] = val
                            last_seen[col_idx] = val

                    row_parts = []
                    for h, v in zip(headers, cells):
                        val = _sanitize_cell_text(v)
                        if val and val not in EMPTY_CELL_MARKERS:
                            row_parts.append(f"{h}: {val}")

                    if not row_parts:
                        continue

                    sentence = f"【門診班表】科別：「{current_section}」｜ " + "，".join(row_parts) + "。"
                    new_lines.append(sentence)
                    sentences_generated += 1

                    row_period = _extract_row_period(headers, cells) or "未註明"
                    row_dept = _extract_row_department(current_section, headers, cells)

                    for header, cell_val in zip(headers, cells):
                        day = _normalize_day_header(header)
                        doctor = _sanitize_cell_text(cell_val)
                        if not day:
                            continue
                        if not doctor or doctor in EMPTY_CELL_MARKERS:
                            continue

                        aliases = " ".join(DAY_ALIASES.get(day, [])[:4])
                        new_lines.append(
                            f"【門診時段查詢】科別：「{row_dept}」；時段：「{row_period}」；{day}醫師：「{doctor}」。"
                        )
                        new_lines.append(
                            f"【門診檢索詞】{row_dept} {day} {aliases} {row_period} 醫師 {doctor}"
                        )

                logger.info(f" 🪄 成功將【{current_section}】的表格轉換為 {sentences_generated} 筆萬能語意！")
            except Exception as e:
                logger.warning(f"表格降維轉換失敗: {e}")
                new_lines.extend(table_lines)
            continue

        new_lines.append(lines[i])
        i += 1

    return "\n".join(new_lines) + "\n\n"


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
        if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"]:
            return ImageFileLoader(file_path, original_filename)
        return TextFileLoader(file_path, original_filename)


# ==========================================
# PART 2: 檔案上傳服務 (Service)
# ==========================================
class FileService:
    def __init__(self):
        # 🚀 修正 1：全面套用設定檔裡的 UPLOAD_DIR
        self.upload_dir = settings.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
        self.vector_store = VectorStoreService.get_instance()

    def _attach_upload_session_metadata(
        self,
        docs: list[Document],
        file_path: str,
        filename: str,
        session_id: Optional[str] = None,
    ) -> list[Document]:
        """為本次上傳寫入統一 metadata，支援同批次檔案關聯。"""
        sid = (session_id or "").strip()
        for idx, doc in enumerate(docs or []):
            if not isinstance(doc.metadata, dict):
                doc.metadata = {}
            doc.metadata.setdefault("source", file_path)
            doc.metadata.setdefault("filename", filename)
            doc.metadata.setdefault("chunk_id", idx)
            if sid:
                doc.metadata["upload_session_id"] = sid
        return docs

    async def upload_file(self, file: UploadFile, session_id: Optional[str] = None):
        try:
            file_path = os.path.join(self.upload_dir, file.filename)
            self.vector_store.delete_file(file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info(f"檔案已儲存: {file.filename}")

            ext = os.path.splitext(file.filename)[1].lower()

            if ext in [".csv", ".xlsx", ".xls"]:
                schedule_docs = self._parse_schedule_spreadsheet(file_path, file.filename)
                if schedule_docs:
                    schedule_docs = self._attach_upload_session_metadata(
                        schedule_docs, file_path, file.filename, session_id
                    )
                    self.vector_store.add_documents(schedule_docs)
                    logger.info(f"已成功將 {len(schedule_docs)} 筆結構化門診資料存入 ChromaDB")
                else:
                    # 非標準門診格式時，退回通用檔案流程，保留可用性。
                    logger.info("未辨識為標準門診表格，改走通用檔案解析流程")
                    await self.vector_store.process_file(file_path, session_id=session_id)

            elif file_path.lower().endswith(".pdf"):
                parse_mode = self._resolve_pdf_parse_mode(file_path)
                logger.info(f"PDF 解析模式: {parse_mode}")

                if parse_mode == "local_fast":
                    docs_to_store = self._parse_pdf_locally_fast(file_path)
                else:
                    docs_to_store = await self._parse_pdf_with_llamaparse(file_path)
                    # LlamaParse 失敗時自動降級，避免上傳長時間卡住後仍無資料。
                    if not docs_to_store:
                        logger.warning("LlamaParse 無結果，啟動本地快速解析降級流程...")
                        docs_to_store = self._parse_pdf_locally_fast(file_path)

                if docs_to_store:
                    docs_to_store = self._attach_upload_session_metadata(
                        docs_to_store, file_path, file.filename, session_id
                    )
                    self.vector_store.add_documents(docs_to_store)
                    logger.info(f"已成功將 {len(docs_to_store)} 塊 Markdown (已降維) 存入 ChromaDB")
            elif ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"]:
                logger.info("🖼️ 偵測到圖片檔案，啟動圖片解析通道...")
                await self.vector_store.process_file(file_path, session_id=session_id)
            else:
                await self.vector_store.process_file(file_path, session_id=session_id)

            return {
                "filename": file.filename,
                "status": "uploaded",
                "message": f"成功上傳: {file.filename}"
            }

        except Exception as e:
            logger.error(f"上傳失敗: {str(e)}")
            raise e

    def _is_likely_medical_schedule_pdf(self, file_path: str) -> bool:
        """
        判斷 PDF 是否屬於「醫療門診班表」。
        只有命中時才啟用門診專屬解析，避免一般文件被硬轉成骨科/內科語意。
        """
        filename = os.path.basename(file_path).lower()
        filename_hints = [
            "sched", "schedule", "clinic", "outpatient", "reg_schedular",
            "門診", "掛號", "醫師", "科別"
        ]
        name_hit = any(h in filename for h in filename_hints)

        text_hit_score = 0
        has_day_signal = False
        content = ""
        try:
            with fitz.open(file_path) as doc:
                for page_idx in range(min(len(doc), 4)):
                    page = doc[page_idx]
                    content += "\n" + (page.get_text() or "")
        except Exception as e:
            logger.warning(f"門診文件判斷時讀取 PDF 失敗，改以檔名判斷: {e}")
            return name_hit

        compact = re.sub(r"\s+", "", content)
        keyword_groups = [
            ["門診", "醫師", "科別", "診間", "看診", "掛號", "時段"],
            ["clinic", "doctor", "department", "outpatient", "session", "weekday"],
        ]
        for group in keyword_groups:
            for kw in group:
                if kw.lower() in compact.lower():
                    text_hit_score += 1

        if re.search(r"(星期[一二三四五六日天]|週[一二三四五六日天]|周[一二三四五六日天]|禮拜[一二三四五六日天])", compact):
            has_day_signal = True
        if re.search(r"\b(mon|tue|wed|thu|fri|sat|sun)(day)?\b", compact, flags=re.IGNORECASE):
            has_day_signal = True

        # 命名強訊號或文字高命中 + 週期訊號才視為門診班表
        if name_hit:
            return True
        return text_hit_score >= 3 and has_day_signal

    def _split_markdown_to_documents(self, markdown_text: str, file_path: str) -> list[Document]:
        """
        將 Markdown 以通用策略切塊，適用於一般文件與門診文件。
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        final_splits = text_splitter.split_documents(md_header_splits)

        filename = os.path.basename(file_path)
        for idx, doc in enumerate(final_splits):
            doc.metadata["source"] = file_path
            doc.metadata["filename"] = filename
            doc.metadata.setdefault("chunk_id", idx)
            doc.metadata.setdefault("type", "general_document")
        return final_splits

    def _extract_pdf_plain_text_documents(self, file_path: str, filename: str) -> list[Document]:
        """
        從 PDF 每一頁保留原文文字，避免只索引表格語意導致費用/政策資訊遺失。
        """
        docs: list[Document] = []
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=120,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            )

            with fitz.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf, start=1):
                    page_text = (page.get_text("text") or "").strip()
                    page_text = re.sub(r"\n{3,}", "\n\n", page_text)
                    if len(re.sub(r"\s+", "", page_text)) < 40:
                        continue

                    chunks = splitter.create_documents([page_text])
                    for chunk_idx, chunk in enumerate(chunks):
                        content = (chunk.page_content or "").strip()
                        if len(re.sub(r"\s+", "", content)) < 30:
                            continue
                        docs.append(
                            Document(
                                page_content=f"[PDF原文 Page {page_idx}]\n{content}",
                                metadata={
                                    "source": file_path,
                                    "filename": filename,
                                    "page": page_idx,
                                    "type": "general_text_local",
                                    "page_chunk_id": chunk_idx,
                                },
                            )
                        )
            logger.info(f"📄 PDF 純文字萃取完成，共 {len(docs)} 筆")
        except Exception as e:
            logger.warning(f"PDF 純文字萃取失敗，略過純文字索引: {e}")
        return docs

    def _dedupe_documents(self, docs: list[Document]) -> list[Document]:
        """
        以 type + page + content 去重，降低多路解析造成的重複 chunk。
        """
        unique_docs: list[Document] = []
        seen: set[tuple[str, str, str]] = set()

        for doc in docs or []:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            content = re.sub(r"\s+", " ", (doc.page_content or "")).strip()
            if not content:
                continue

            dtype = str(metadata.get("type", ""))
            page = str(metadata.get("page", ""))
            key = (dtype, page, content)
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)

        return unique_docs

    def _build_llamaparse_parser(self, is_schedule_pdf: bool) -> LlamaParse:
        """
        建立 LlamaParse 解析器：
        - 醫療門診文件：啟用門診專屬 prompt
        - 一般文件：啟用中性 prompt，避免過度擬合成門診資料
        """
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if is_schedule_pdf:
            system_prompt = """
            This is a high-precision medical clinic schedule.
            1. Keep strict Markdown table grid alignment.
            2. Preserve all weekday and period columns.
            3. Keep department names and doctor names associated correctly.
            4. If a merged cell spans days, repeat values to each day cell when possible.
            """
            return LlamaParse(
                api_key=api_key,
                result_type="markdown",
                verbose=True,
                language="ch_tra",
                system_prompt=system_prompt,
            )

        generic_prompt = """
        Parse this document faithfully into Markdown.
        1. Preserve original semantics and section structure.
        2. Do not hallucinate domain-specific fields that are not present.
        3. Keep tables as neutral tables without converting to medical schedule terms.
        """
        return LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=True,
            system_prompt=generic_prompt,
        )

    def _resolve_pdf_parse_mode(self, file_path: str) -> str:
        """
        決定 PDF 解析策略：
        - local_fast: 低延遲、免等待雲端
        - llamaparse: 高品質版面還原
        - auto: 大檔走 local_fast，小檔走 llamaparse
        """
        mode = str(getattr(settings, "PDF_PARSE_MODE", "auto")).strip().lower()
        if mode not in {"auto", "local_fast", "llamaparse"}:
            mode = "auto"

        if mode == "local_fast":
            return "local_fast"
        if mode == "llamaparse":
            return "llamaparse"

        # auto mode
        api_key = os.getenv("LLAMA_CLOUD_API_KEY") or getattr(settings, "LLAMA_CLOUD_API_KEY", None)
        if not api_key:
            return "local_fast"

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        threshold_mb = int(getattr(settings, "PDF_FAST_MODE_SIZE_MB", 6))
        if size_mb >= threshold_mb:
            return "local_fast"
        return "llamaparse"

    def _split_doctor_names(self, raw_text: str) -> list[str]:
        """將表格儲存格中的醫師欄位盡可能拆成單一姓名。"""
        text = _sanitize_cell_text(raw_text)
        if not text or text in EMPTY_CELL_MARKERS:
            return []

        # 常見無效值
        if text in {"不指定", "休診", "停診", "無"}:
            return [text]

        # 常見格式：姓名後面接診間號，例如「王小明2311」
        direct_name_hits = re.findall(r"[\u4e00-\u9fff]{2,4}(?=\d{3,5})", text)
        if direct_name_hits:
            dedup = []
            seen = set()
            for n in direct_name_hits:
                n = n.strip()
                if not n or n in seen:
                    continue
                seen.add(n)
                dedup.append(n)
            if dedup:
                return dedup

        normalized = (
            text.replace("/", "、")
            .replace(";", "、")
            .replace(",", "、")
            .replace("；", "、")
            .replace("\n", "、")
        )

        parts = [p.strip() for p in normalized.split("、") if p.strip()]
        results = []
        seen = set()
        for part in parts:
            # 去掉尾端診間號數字與日期註記
            cleaned = re.sub(r"\(.*?\)", "", part).strip()
            cleaned = re.sub(r"\d{3,5}$", "", cleaned).strip()
            cleaned = cleaned.replace(" ", "")
            if not cleaned:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            results.append(cleaned)
        return results

    def _find_column_by_aliases(self, columns: list[str], aliases: list[str]) -> str:
        """在欄位中尋找最匹配的欄位名稱（忽略空白與大小寫）。"""
        normalized = {}
        for c in columns:
            key = _sanitize_cell_text(c).lower().replace(" ", "")
            normalized[key] = c

        for alias in aliases:
            target = alias.lower().replace(" ", "")
            if target in normalized:
                return normalized[target]

        for alias in aliases:
            target = alias.lower().replace(" ", "")
            for key, original in normalized.items():
                if target and target in key:
                    return original
        return ""

    def _normalize_day_value(self, raw_day: str) -> str:
        """把星期值正規化為 星期一~星期日。"""
        day = _normalize_day_header(raw_day or "")
        return day

    def _normalize_period_value(self, raw_period: str) -> str:
        """把時段值正規化為 上午/下午/夜間/全天。"""
        period = _normalize_period(raw_period or "")
        return period or "未註明"

    def _build_schedule_slot_sentence(self, dept: str, period: str, day: str, doctor: str) -> str:
        return f"【門診時段查詢】科別：「{dept}」；時段：「{period}」；{day}醫師：「{doctor}」。"

    def _build_schedule_docs_from_flat_df(
        self,
        df: pd.DataFrame,
        filename: str,
        file_path: str,
        sheet_name: str = "",
    ) -> list[Document]:
        """標準扁平格式：科別、星期、時段、醫師。"""
        docs: list[Document] = []
        seen = set()

        columns = [str(c) for c in df.columns]
        dept_col = self._find_column_by_aliases(columns, ["科別", "門診別", "科", "department", "dept"])
        day_col = self._find_column_by_aliases(columns, ["星期", "星期幾", "週", "周", "day", "weekday"])
        period_col = self._find_column_by_aliases(columns, ["時段", "時間", "診別", "session", "period", "time"])
        doctor_col = self._find_column_by_aliases(columns, ["醫師", "醫生", "醫師名單", "doctor", "doctors"])

        if not all([dept_col, day_col, period_col, doctor_col]):
            return []

        for _, row in df.iterrows():
            dept = _sanitize_cell_text(row.get(dept_col, ""))
            day = self._normalize_day_value(row.get(day_col, ""))
            period = self._normalize_period_value(row.get(period_col, ""))
            doctors_raw = _sanitize_cell_text(row.get(doctor_col, ""))
            if not dept or not day or not doctors_raw or doctors_raw in EMPTY_CELL_MARKERS:
                continue

            for doctor in self._split_doctor_names(doctors_raw):
                sentence = self._build_schedule_slot_sentence(dept, period, day, doctor)
                if sentence in seen:
                    continue
                seen.add(sentence)
                meta = {
                    "source": file_path,
                    "filename": filename,
                    "type": "schedule_slot",
                }
                if sheet_name:
                    meta["sheet"] = sheet_name
                docs.append(Document(page_content=sentence, metadata=meta))
        return docs

    def _build_schedule_docs_from_wide_df(
        self,
        df: pd.DataFrame,
        filename: str,
        file_path: str,
        sheet_name: str = "",
    ) -> list[Document]:
        """寬表格式：科別/時段 + 星期一~星期六/日 欄位。"""
        docs: list[Document] = []
        seen = set()
        columns = [str(c) for c in df.columns]

        dept_col = self._find_column_by_aliases(columns, ["科別", "門診別", "科", "department", "dept"])
        period_col = self._find_column_by_aliases(columns, ["時段", "時間", "診別", "session", "period", "time"])
        if not dept_col:
            return []

        day_cols = []
        for col in columns:
            day = _normalize_day_header(col)
            if day:
                day_cols.append((col, day))
        if not day_cols:
            return []

        for _, row in df.iterrows():
            dept = _sanitize_cell_text(row.get(dept_col, ""))
            if not dept:
                continue
            period = self._normalize_period_value(row.get(period_col, "")) if period_col else "未註明"

            for col_name, day in day_cols:
                doctors_raw = _sanitize_cell_text(row.get(col_name, ""))
                if not doctors_raw or doctors_raw in EMPTY_CELL_MARKERS:
                    continue
                for doctor in self._split_doctor_names(doctors_raw):
                    sentence = self._build_schedule_slot_sentence(dept, period, day, doctor)
                    if sentence in seen:
                        continue
                    seen.add(sentence)
                    meta = {
                        "source": file_path,
                        "filename": filename,
                        "type": "schedule_slot",
                    }
                    if sheet_name:
                        meta["sheet"] = sheet_name
                    docs.append(Document(page_content=sentence, metadata=meta))
        return docs

    def _parse_schedule_spreadsheet(self, file_path: str, filename: str) -> list[Document]:
        """
        CSV/XLSX 精準模式：用 Pandas 直接讀取，不經 LLM/OCR，避免表格位移污染。
        支援：
        1) 扁平格式（科別/星期/時段/醫師）
        2) 寬表格式（科別/時段 + 星期欄）
        """
        docs: list[Document] = []
        ext = os.path.splitext(filename)[1].lower()
        logger.info(f"偵測到表格檔案 {filename}，啟動 [Pandas 精準直讀模式]...")

        try:
            if ext == ".csv":
                df = pd.read_csv(file_path, dtype=str, keep_default_na=False).fillna("")
                docs.extend(self._build_schedule_docs_from_flat_df(df, filename, file_path))
                if not docs:
                    docs.extend(self._build_schedule_docs_from_wide_df(df, filename, file_path))
            else:
                all_sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
                for sheet_name, sheet_df in all_sheets.items():
                    if sheet_df is None or sheet_df.empty:
                        continue
                    sheet_df = sheet_df.fillna("")
                    flat_docs = self._build_schedule_docs_from_flat_df(
                        sheet_df, filename, file_path, str(sheet_name)
                    )
                    if flat_docs:
                        docs.extend(flat_docs)
                        continue
                    docs.extend(
                        self._build_schedule_docs_from_wide_df(
                            sheet_df, filename, file_path, str(sheet_name)
                        )
                    )
        except Exception as e:
            logger.error(f"解析表格檔案失敗: {e}")
            return []

        logger.info(f"✅ 表格精準入庫準備完成：{len(docs)} 筆 schedule_slot")
        return docs

    def _extract_schedule_docs_from_pdf_tables(self, file_path: str, filename: str) -> list[Document]:
        """
        從 PDF 實體表格抽出結構化門診句子，提供給直答引擎使用。
        """
        docs: list[Document] = []
        sentence_seen = set()

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables() or []
                    for table in tables:
                        rows = []
                        for row in table:
                            if not row:
                                continue
                            cleaned = [_sanitize_cell_text(c) if c else "" for c in row]
                            if any(cell for cell in cleaned):
                                rows.append(cleaned)

                        if len(rows) < 2:
                            continue

                        header_idx = -1
                        for idx, row in enumerate(rows[:10]):
                            day_count = sum(1 for c in row if _normalize_day_header(c))
                            if day_count >= 2:
                                header_idx = idx
                                if any(("時間" in c) or ("時段" in c) for c in row):
                                    break
                        if header_idx < 0:
                            continue

                        headers = rows[header_idx]
                        width = len(headers)
                        day_columns = {}
                        for col_idx, h in enumerate(headers):
                            day_name = _normalize_day_header(h)
                            if day_name:
                                day_columns[col_idx] = day_name
                        if not day_columns:
                            continue

                        # 有些 PDF 表格會把「星期欄」與「科別/時間欄」拆成兩列表頭：
                        # 例如第 N 列是星期一~六，第 N-1 列才有科別/時間。
                        # 因此這裡要跨列蒐集欄位線索，避免漏抓下午門診。
                        header_candidates = [headers]
                        if header_idx - 1 >= 0:
                            header_candidates.append(rows[header_idx - 1])
                        if header_idx - 2 >= 0:
                            header_candidates.append(rows[header_idx - 2])

                        def _collect_cols_by_keywords(candidates: list[list[str]], keywords: list[str]) -> list[int]:
                            cols = []
                            seen = set()
                            for cand in candidates:
                                for i, h in enumerate(cand):
                                    ht = _sanitize_cell_text(h)
                                    if any(k in ht for k in keywords):
                                        if i not in seen:
                                            seen.add(i)
                                            cols.append(i)
                            return cols

                        dept_cols = _collect_cols_by_keywords(
                            header_candidates, ["科別", "門診別", "專科", "科"]
                        )
                        time_cols = _collect_cols_by_keywords(
                            header_candidates, ["時間", "時段", "診次", "診別", "班別"]
                        )

                        # 仍抓不到欄位時，對星期欄之前的 prefix 區做內容推斷。
                        # 這對「表頭拆裂/掃描壓扁」的 PDF 很重要。
                        prefix_limit = min(day_columns.keys()) if day_columns else 0
                        sample_rows = rows[header_idx + 1: min(len(rows), header_idx + 40)]
                        if prefix_limit > 0 and sample_rows:
                            if not dept_cols:
                                inferred = []
                                for col_idx in range(prefix_limit):
                                    score = 0
                                    for r in sample_rows:
                                        if col_idx >= len(r):
                                            continue
                                        val = _sanitize_cell_text(r[col_idx])
                                        if not val or val in EMPTY_CELL_MARKERS:
                                            continue
                                        if "科" in val or "門診" in val:
                                            score += 1
                                    if score > 0:
                                        inferred.append((col_idx, score))
                                inferred.sort(key=lambda x: x[1], reverse=True)
                                dept_cols = [idx for idx, _ in inferred[:3]]

                            if not time_cols:
                                inferred = []
                                for col_idx in range(prefix_limit):
                                    score = 0
                                    for r in sample_rows:
                                        if col_idx >= len(r):
                                            continue
                                        if _normalize_period(r[col_idx]):
                                            score += 1
                                    if score > 0:
                                        inferred.append((col_idx, score))
                                inferred.sort(key=lambda x: x[1], reverse=True)
                                time_cols = [idx for idx, _ in inferred[:2]]

                        last_dept = ""
                        last_period = ""

                        for row in rows[header_idx + 1:]:
                            if len(row) < width:
                                row = row + [""] * (width - len(row))
                            elif len(row) > width:
                                row = row[:width]

                            dept = ""
                            for col_idx in dept_cols:
                                if col_idx >= len(row):
                                    continue
                                value = _sanitize_cell_text(row[col_idx])
                                if not value or value in EMPTY_CELL_MARKERS:
                                    continue
                                # 優先選擇「骨科、內科...」這種正式科別
                                if value.endswith("科"):
                                    dept = value
                                    break
                                if not dept:
                                    dept = value
                            if dept:
                                last_dept = dept
                            else:
                                dept = last_dept
                            if not dept:
                                continue

                            period = ""
                            for col_idx in time_cols:
                                if col_idx >= len(row):
                                    continue
                                period = _normalize_period(row[col_idx])
                                if period:
                                    break
                            if not period:
                                period = _normalize_period(" ".join(row[:min(6, len(row))]))
                            if period:
                                last_period = period
                            else:
                                period = last_period or "未註明"

                            for col_idx, day in day_columns.items():
                                if col_idx >= len(row):
                                    continue
                                doctor_cell = _sanitize_cell_text(row[col_idx])
                                if not doctor_cell or doctor_cell in EMPTY_CELL_MARKERS:
                                    continue
                                doctor_names = self._split_doctor_names(doctor_cell)
                                for doctor in doctor_names:
                                    sentence = (
                                        f"【門診時段查詢】科別：「{dept}」；時段：「{period}」；{day}醫師：「{doctor}」。"
                                    )
                                    if sentence in sentence_seen:
                                        continue
                                    sentence_seen.add(sentence)
                                    docs.append(
                                        Document(
                                            page_content=sentence,
                                            metadata={
                                                "source": file_path,
                                                "filename": filename,
                                                "page": page_idx,
                                                "type": "schedule_slot_local",
                                            },
                                        )
                                    )

            logger.info(f"⚡ 從 PDF 表格萃取到 {len(docs)} 筆結構化門診資料")
            logger.info(
                f"⚡ 從 PDF 表格萃取完成，共 {len(docs)} 筆結構化門診資料"
            )
            return docs
        except Exception as e:
            logger.warning(f"⚡ 本地表格結構化萃取失敗（將僅使用全文切塊）: {e}")
            return []

    def _parse_pdf_locally_fast(self, file_path: str) -> list[Document]:
        """
        本地快速 PDF 解析：不依賴雲端，優先追求上傳速度與可用性。
        """
        try:
            filename = os.path.basename(file_path)
            logger.info("⚡ 啟動本地快速 PDF 解析...")
            is_schedule_pdf = self._is_likely_medical_schedule_pdf(file_path)
            logger.info(f"⚡ 本地路由判斷: {'醫療門診文件' if is_schedule_pdf else '一般文件'}")

            loader = PDFFileLoader(file_path, filename)
            text_content = loader.extract_text()
            if not text_content or not text_content.strip():
                logger.warning("本地 PDF 解析無文字內容")
                return []

            parser = SmartFileParser()
            docs = parser.parse(text_content, filename)
            plain_text_docs = self._extract_pdf_plain_text_documents(file_path, filename)
            schedule_docs = []
            if is_schedule_pdf:
                schedule_docs = self._extract_schedule_docs_from_pdf_tables(file_path, filename)
                # 只有門診文件才注入結構化 slot，避免污染一般知識文件。

            if is_schedule_pdf:
                docs = schedule_docs + plain_text_docs + docs
            else:
                docs = plain_text_docs + docs

            docs = self._dedupe_documents(docs)

            for idx, doc in enumerate(docs):
                doc.metadata["source"] = file_path
                doc.metadata["filename"] = filename
                doc.metadata.setdefault("chunk_id", idx)

            logger.info(
                f"⚡ 本地快速解析完成，共 {len(docs)} 筆（結構化門診 {len(schedule_docs)} 筆）"
            )
            return docs
        except Exception as e:
            logger.error(f"⚡ 本地快速解析失敗: {e}")
            return []

    async def _parse_pdf_with_llamaparse(self, file_path: str) -> list[Document]:
        """
        使用 LlamaParse 將 PDF 轉換為 Markdown，並由路由器決定解析策略。
        """
        try:
            is_schedule_pdf = self._is_likely_medical_schedule_pdf(file_path)
            logger.info(
                f"📄 LlamaParse 路由判斷: {'醫療門診文件(啟用門診專屬解析)' if is_schedule_pdf else '一般文件(啟用通用解析)'}"
            )
            logger.info("📄 啟動 LlamaParse 視覺排版解析...")
            parser = self._build_llamaparse_parser(is_schedule_pdf)

            parsed_documents = parser.load_data(file_path)
            full_markdown = "\n\n".join([doc.text for doc in parsed_documents])

            if is_schedule_pdf:
                # 只有門診文件才做「表格降維」，避免一般文件被錯誤轉成骨科/內科語意。
                full_markdown = convert_md_table_to_sentences(full_markdown)

            final_splits = self._split_markdown_to_documents(full_markdown, file_path)
            filename = os.path.basename(file_path)
            plain_text_docs = self._extract_pdf_plain_text_documents(file_path, filename)
            schedule_docs = []

            if is_schedule_pdf:
                # 補上本地表格結構化 slot，加強後續精準問答。
                schedule_docs = self._extract_schedule_docs_from_pdf_tables(file_path, filename)

            logger.info(
                f"📄 LlamaParse 解析完成: {'門診文件' if is_schedule_pdf else '一般文件'}，共 {len(final_splits)} 筆"
            )
            if is_schedule_pdf:
                final_splits = schedule_docs + plain_text_docs + final_splits
            else:
                final_splits = plain_text_docs + final_splits
            final_splits = self._dedupe_documents(final_splits)
            logger.info(
                f"📄 LlamaParse 解析完成，共 {len(final_splits)} 筆（結構化門診 {len(schedule_docs)} 筆，純文字 {len(plain_text_docs)} 筆）"
            )
            return final_splits

        except Exception as e:
            logger.error(f"❌ LlamaParse 解析失敗: {e}")
            return []
