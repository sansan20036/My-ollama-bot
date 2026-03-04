# app/utils/smart_parser.py
import re
import unicodedata
from langchain_core.documents import Document

# 🔥 萬能引用區塊
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
    except ImportError:
        class Language:
            PYTHON = "python";
            JS = "js";
            TS = "ts";
            JAVA = "java";
            CPP = "cpp";
            GO = "go";
            HTML = "html"


        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError("❌ 嚴重錯誤：找不到 langchain 文字切割模組。")


class SmartFileParser:
    """
    智慧檔案解析器 (Smart Routing Parser) V2.0
    修正：解決法律條文引用造成的過度切割問題
    """

    def parse(self, raw_text: str, filename: str) -> list[Document]:
        text = self._clean_text(raw_text)
        documents = []

        if self._is_law_file(text):
            documents = self._parse_law(text, filename)
            doc_type = "法律規章"
        elif self._is_code_file(filename):
            documents = self._parse_code(text, filename)
            doc_type = "程式原始碼"
        else:
            documents = self._parse_general(text, filename)
            doc_type = "一般文件"

        # 生成摘要卡
        total_chars = len(text)
        total_chunks = len(documents)

        summary_content = (
            f"【檔案全域摘要】\n"
            f"檔案名稱：{filename}\n"
            f"文件類型：{doc_type}\n"
            f"總字數：約 {total_chars} 字\n"
            f"資料切片數：{total_chunks} 個段落\n"
        )

        if doc_type == "法律規章":
            article_count = len([d for d in documents if d.metadata.get("type") == "law_article"])
            summary_content += f"法規條文數：共 {article_count} 條\n"

        summary_doc = Document(
            page_content=summary_content,
            metadata={
                "source": filename,
                "type": "file_summary",
                "total_chars": total_chars,
                "doc_type": doc_type
            }
        )
        documents.insert(0, summary_doc)
        return documents

    def _clean_text(self, text: str) -> str:
        # 強制轉半形，但保留換行符號！
        return unicodedata.normalize('NFKC', text)

    def _is_law_file(self, text: str) -> bool:
        return text.count("第") > 5 and text.count("條") > 5 and ("法" in text or "條例" in text)

    def _is_code_file(self, filename: str) -> bool:
        if '.' not in filename: return False
        ext = filename.split('.')[-1].lower()
        return ext in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'html', 'css', 'sql', 'go', 'rs']

    def _parse_law(self, text: str, filename: str) -> list[Document]:
        """
        修正版：嚴格切割: 只抓取位於「行首」或「換行後」的第X條，避免抓到內文引用的條號
        """
        # Regex 解釋：
        # (?:\n|^)  -> 非捕獲群組：必須是字串開頭 (^) 或是換行符號 (\n)
        # \s* -> 允許前面有一些空白
        # (第...條) -> 捕獲群組：這是我們要抓的標題
        pattern = r'(?:\n|^)\s*(第\s*[0-9\u4e00-\u9fa5\d\-－]+\s*條)'

        # 使用 split 切割
        parts = re.split(pattern, text)
        docs = []

        # 處理前言
        if parts[0].strip():
            docs.append(Document(page_content=f"【前言】\n{parts[0].strip()}",
                                 metadata={"source": filename, "type": "law_preamble"}))

        # 因為 regex 加了行首判斷，split 出來的 list 結構會稍微變動
        # parts[0] = 前言
        # parts[1] = 標題1 (e.g. 第1條)
        # parts[2] = 內容1
        # parts[3] = 標題2 ...

        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts): break
            header = parts[i].strip()
            content = parts[i + 1].strip()

            # 如果內容太短且後面緊接著另一個標題，可能是誤切，但在嚴格模式下機率降低
            # 提取條號 ID
            try:
                article_id = header.replace("第", "").replace("條", "").strip()
            except:
                article_id = "unknown"

            # 組合完整條文
            full_text = f"{header}\n{content}"

            docs.append(Document(
                page_content=full_text,
                metadata={"source": filename, "type": "law_article", "article_id": article_id}
            ))
        return docs

    def _parse_code(self, text: str, filename: str) -> list[Document]:
        # (程式碼解析邏輯保持不變)
        ext = filename.split('.')[-1].lower()
        lang_map = {
            'py': Language.PYTHON, 'js': Language.JS, 'ts': Language.TS,
            'java': Language.JAVA, 'cpp': Language.CPP, 'c': Language.C,
            'html': Language.HTML, 'go': Language.GO
        }
        language = lang_map.get(ext, Language.PYTHON)
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=800, chunk_overlap=200
            )
            raw_docs = splitter.create_documents([text])
            for d in raw_docs:
                d.metadata.update({"source": filename, "type": "code_snippet"})
            return raw_docs
        except Exception:
            return self._parse_general(text, filename)

    def _parse_general(self, text: str, filename: str) -> list[Document]:
        # (通用解析邏輯保持不變)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        chunks = splitter.split_text(text)
        return [Document(page_content=c, metadata={"source": filename, "type": "general_text"}) for c in chunks]