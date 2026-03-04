import os
import sys
import re
import unicodedata

# 確保 python 能找到 app 模組
sys.path.append(os.getcwd())

from app.services.file_service import FileLoaderFactory


def debug_pdf_content():
    upload_dir = os.path.join(os.getcwd(), "uploads")

    # 1. 找出勞動基準法 PDF
    target_files = [f for f in os.listdir(upload_dir) if "勞動" in f and f.endswith(".pdf")]

    if not target_files:
        print(f"❌ 在 {upload_dir} 找不到包含 '勞動' 的 PDF 檔案！")
        return

    filename = target_files[0]
    file_path = os.path.join(upload_dir, filename)
    print(f"🕵️‍♂️ 正在分析檔案：{filename}")

    try:
        # 2. 讀取文字
        loader = FileLoaderFactory.get_loader(file_path, filename)
        raw_text = loader.extract_text()

        # 正規化 (轉半形)
        text = unicodedata.normalize('NFKC', raw_text)

        print(f"✅ 成功讀取文字，總長度：{len(text)} 字")

        # 3. 測試一：直接找 "79"
        print("\n🔍 --- 測試 1: 搜尋數字 '79' ---")
        indices_79 = [m.start() for m in re.finditer(r"79", text)]

        if not indices_79:
            print("❌ 整份文件裡完全找不到 '79' 這個數字！(可能是圖片或中文數字)")
        else:
            print(f"✅ 找到 {len(indices_79)} 處包含 '79'：")
            for idx in indices_79:
                # 印出前後 30 個字
                start = max(0, idx - 30)
                end = min(len(text), idx + 30)
                snippet = text[start:end].replace('\n', '【換行】')
                print(f"   📍 位置 {idx}: ...{snippet}...")

        # 4. 測試二：搜尋中文 "七十九"
        print("\n🔍 --- 測試 2: 搜尋中文 '七十九' ---")
        indices_zh = [m.start() for m in re.finditer(r"七十九", text)]
        if indices_zh:
            print(f"✅ 找到 {len(indices_zh)} 處包含 '七十九'：")
            for idx in indices_zh:
                start = max(0, idx - 30)
                end = min(len(text), idx + 30)
                snippet = text[start:end].replace('\n', '【換行】')
                print(f"   📍 位置 {idx}: ...{snippet}...")
        else:
            print("❌ 找不到 '七十九'。")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")


if __name__ == "__main__":
    debug_pdf_content()