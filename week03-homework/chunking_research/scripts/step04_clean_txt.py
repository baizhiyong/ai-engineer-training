
import re
import os
from pathlib import Path

# 获取当前脚本所在目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 从scripts目录向上找到chunking_research目录，然后定位data/raw和data/clean
# scripts -> chunking_research -> data -> raw/clean
chunking_research_dir = os.path.dirname(current_script_dir)
data_raw_dir = os.path.join(chunking_research_dir, "data", "raw")
data_clean_dir = os.path.join(chunking_research_dir, "data", "clean")

RAW_DIR = Path(data_raw_dir)
CLEAN_DIR = Path(data_clean_dir)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# 1. 通用正则：页眉页脚常见特征（可再补）
HEADER_FOOTER_RE = re.compile(
    r"(^|\n)\s*(\d+[\s\-–—]+\d+|\d+\s*\/\s*\d+|Page\s+\d+|\d+\s*of\s*\d+)(?=\n|$)",  # 1/5、Page 2、2 of 5
    flags=re.IGNORECASE | re.MULTILINE
)

def clean_one(text: str) -> str:
    """单文本清洗逻辑"""
    # 1. 去页眉页脚
    text = HEADER_FOOTER_RE.sub("", text)

    # 2. 合并多余空行（>=2 行 → 1 行）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. 去掉首尾空白
    text = text.strip()

    # 4. 去掉网址、邮件（可选）
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    return text

def main():
    for txt_file in RAW_DIR.glob("*.txt"):
        out_file = CLEAN_DIR / txt_file.name
        raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")

        cleaned = clean_one(raw_text)
        chars = len(cleaned)

        # 长度过滤和截取
        if chars < 1200:
            print(f"⚠️  {txt_file.name}  字数 {chars} 过少，已跳过")
        else:
            # 如果超过3000字符，截取前3000个字符
            if chars > 3000:
                cleaned = cleaned[:3000]
                chars = 3000
                print(f"✂️  {txt_file.name}  字数过多，已截取前 {chars} 字")
            else:
                print(f"✅ {txt_file.name}  →  {chars} 字")
            
            out_file.write_text(cleaned, encoding="utf-8")

if __name__ == "__main__":
    main()