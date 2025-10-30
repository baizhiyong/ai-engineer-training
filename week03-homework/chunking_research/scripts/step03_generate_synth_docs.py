#!/usr/bin/env python3
# generate_synth_docs.py
import os
import json
from pathlib import Path
import dashscope
from tqdm import tqdm

# 1. 基本配置
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
assert dashscope.api_key, "请先 export DASHSCOPE_API_KEY=你的KEY"

# 获取当前脚本所在目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 从scripts目录向上找到chunking_research目录，然后定位data/raw
# scripts -> chunking_research -> data -> raw
chunking_research_dir = os.path.dirname(current_script_dir)
data_raw_dir = os.path.join(chunking_research_dir, "data", "raw")

OUTPUT_DIR = Path(data_raw_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 提示模板：要求 1000 字以上，中文，禁止分点条列
PROMPTS = {
    "产品说明书": (
        "请你以‘智能降噪耳机 Pro-X’为题，写一篇面向消费者的完整产品说明书，"
        "内容需涵盖：产品亮点、技术规格、使用教程、注意事项、售后政策。全文连贯、"
        "不得使用条列或编号，,文档需包含复杂句式和段落结构,字数不少于 1000 字。"
    ),
    "小说节选": (
        "请创作一篇科幻题材的小说节选，故事背景设定在 2077 年的火星城市，"
        "主人公是一名记忆修复师。要求情节完整、人物形象饱满，不得使用条列，"
        "语言生动，文档需包含复杂句式和段落结构,字数不少于 1000 字。"
    ),
    "公司年报": (
        "请为‘XXXX科技有限公司’撰写 2025 年度财报致股东信，"
        "内容包括：年度营收与利润、核心研发进展、市场战略、ESG 实践、未来展望。全文连贯、"
        "不得使用条列，文档需包含复杂句式和段落结构,字数不少于 1000 字。"
    ),
}

# 3. 调用 qwen-plus 生成
def generate(text: str) -> str:
    """单次生成，流式输出拼合，返回完整文本"""
    full = ""
    resp = dashscope.Generation.call(
        model="qwen-plus",
        messages=[{"role": "user", "content": text}],  # type: ignore
        result_format="message",
        stream=True,
        top_p=0.95,
        max_tokens=2500,  # 约 2500 中文字
    )
    for chunk in resp:
        if chunk.status_code == 200:
            content = chunk.output.choices[0].message.content
            if isinstance(content, str):
                full += content
        else:
            raise RuntimeError(chunk.message)
    return full.strip()

# 4. 批量写入
def main():
    for name, prompt in tqdm(PROMPTS.items(), desc="生成中"):
        out_path = OUTPUT_DIR / f"synth_{name}.txt"
        if out_path.exists():
            tqdm.write(f"{out_path} 已存在，跳过")
            continue
        text = generate(prompt)
        # 简单校验字数（中文字符）
        if len(text) < 1000:
            tqdm.write(f"{name} 字数不足，重试一次")
            text = generate(prompt + "（请确保超过 1000 字）")
        out_path.write_text(text, encoding="utf-8")
        tqdm.write(f"✅ 已写入 {out_path}  字数：{len(text)}")

if __name__ == "__main__":
    main()