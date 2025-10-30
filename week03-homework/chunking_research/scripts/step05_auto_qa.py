#!/usr/bin/env python3
"""
自动问答对生成器 - 基于文本内容生成高质量问答数据集

功能说明：
========
本工具从清理后的文本文件中自动生成问答对，用于构建QA数据集。采用动态路径计算和流式API调用，具有以下特点：

1. 动态路径计算：
   - 自动基于脚本位置计算data/clean目录路径（输入）
   - 自动基于脚本位置计算outputs目录路径（输出）
   - 无论在哪个目录运行都能正确找到目标文件夹
   - 路径结构：scripts -> chunking_research -> data/clean & outputs

2. 智能问答生成：
   - 使用DashScope API (qwen-plus模型) 生成高质量问答对
   - 每个文本生成8个问题和对应答案
   - 问题覆盖：核心观点、关键技术、数据案例、潜在影响、结构逻辑
   - 答案要求：简短精准（≤50字），可直接定位到原文

3. 流式API调用：
   - 使用流式调用避免类型检查问题
   - 支持长文本处理（自动截断到8000字符）
   - 包含错误重试机制和异常处理

4. 数据格式和输出：
   - 输出格式：JSON列表，每个元素包含question、answer、doc字段
   - doc字段记录来源文件名，方便数据溯源
   - 自动创建输出目录，确保文件正确保存

使用方式：
========
# 基本使用（处理data/clean目录下的所有txt文件）
python auto_qa.py

# 环境变量设置
export DASHSCOPE_API_KEY="your_api_key_here"

依赖要求：
========
pip install dashscope tqdm

输入文件：
========
- 位置：data/clean/*.txt
- 格式：UTF-8编码的纯文本文件
- 要求：经过清理的高质量文本内容

输出文件：
========
- 位置：outputs/qa.json
- 格式：JSON数组，每个问答对包含：
  {
    "question": "问题内容",
    "answer": "答案内容",
    "doc": "源文件名"
  }

注意事项：
========
- 需要有效的DASHSCOPE_API_KEY环境变量
- API调用有频率限制，大量文件处理时请注意
- 生成质量依赖于输入文本的质量和结构
- 支持递归重试机制，但可能导致无限循环（需要手动中断）

作者: AI Assistant
日期: 2025-10-29
版本: 1.0.0
"""

import os, json, dashscope
from pathlib import Path
from tqdm import tqdm

def _get_default_data_dir() -> str:
    """获取默认的data/clean目录路径
    
    动态计算相对于当前脚本的data/clean目录路径，
    确保无论在哪里运行都能正确找到
    
    Returns:
        data/clean目录的绝对路径
    """
    # 获取当前脚本所在目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从scripts目录向上找到chunking_research目录，然后定位data/clean
    # scripts -> chunking_research -> data -> clean
    chunking_research_dir = os.path.dirname(current_script_dir)
    data_clean_dir = os.path.join(chunking_research_dir, "data", "clean")
    
    return data_clean_dir

def _get_default_output_dir() -> str:
    """获取默认的outputs目录路径
    
    动态计算相对于当前脚本的outputs目录路径，
    确保无论在哪里运行都能正确找到
    
    Returns:
        outputs目录的绝对路径
    """
    # 获取当前脚本所在目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从scripts目录向上找到chunking_research目录，然后定位outputs
    # scripts -> chunking_research -> outputs
    chunking_research_dir = os.path.dirname(current_script_dir)
    outputs_dir = os.path.join(chunking_research_dir, "outputs")
    
    return outputs_dir

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
DATA_DIR  = Path(_get_default_data_dir())
QA_FILE   = Path(_get_default_output_dir()) / "qa.json"

PROMPT = """
你是一名专业分析师。请仔细阅读以下文本，并生成 1-2 个高质量的问题及其对应答案。

要求：
1. 问题覆盖全文核心观点、关键技术、数据案例、潜在影响、结构逻辑。
2. 答案尽量简短（≤50字），可直接定位到原文。
3. 必须严格按照JSON数组格式输出，不要添加任何解释文字。
4. 每个问答对格式：{{"question": "问题内容", "answer": "答案内容"}}

示例输出格式：
[
  {{"question": "这是问题1", "answer": "这是答案1"}},
  {{"question": "这是问题2", "answer": "这是答案2"}}
]

请基于以下文本生成问答对，只输出JSON数组，不要其他内容：

{text}
"""

def build_qa_one(text: str, retry_count: int = 0) -> list:
    full_prompt = PROMPT.format(text=text[:8000])          # 截断避免超长
    
    try:
        # 使用非流式调用
        resp = dashscope.Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": full_prompt}],  # type: ignore
            result_format="message",
            temperature=0.3,
            max_tokens=2000,
            stream=False,  # 非流式调用
        )
        
        # 统一处理响应内容
        content = ""
        
        # 检查响应类型并提取内容
        if isinstance(resp, str):
            # 如果直接返回字符串
            content = resp
        elif hasattr(resp, 'output') and hasattr(resp.output, 'choices'):  # type: ignore
            # 标准响应对象格式
            try:
                content = resp.output.choices[0].message.content  # type: ignore
            except (AttributeError, IndexError) as e:
                print(f"响应格式异常: {e}, 响应类型: {type(resp)}")
                raise ValueError(f"无法从响应中提取内容: {e}")
        elif hasattr(resp, '__iter__'):
            # 迭代器格式（流式响应的情况）
            for chunk in resp:  # type: ignore
                if hasattr(chunk, 'status_code') and chunk.status_code == 200:  # type: ignore
                    chunk_content = chunk.output.choices[0].message.content  # type: ignore
                    if isinstance(chunk_content, str):
                        content = chunk_content  # 直接赋值最后一个完整响应
                elif hasattr(chunk, 'output'):
                    # 有些情况下可能没有status_code
                    try:
                        chunk_content = chunk.output.choices[0].message.content  # type: ignore
                        if isinstance(chunk_content, str):
                            content = chunk_content
                    except Exception:
                        continue
        else:
            # 其他未知格式
            print(f"未知响应格式: {type(resp)}")
            print(f"响应内容: {str(resp)[:200]}...")
            raise ValueError(f"未知的响应格式: {type(resp)}")
        
        # 确保内容是字符串类型且非空
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"响应内容为空或类型错误: {type(content)}, 内容: {content}")
        
        # 清理内容，去除可能的前缀和后缀干扰
        content = content.strip()
        
        # 处理流式响应可能导致的重复内容问题
        # 如果发现内容重复模式，尝试提取最后一个完整的JSON
        if content.count('[{') > 1:
            # 查找最后一个看起来像JSON开始的位置
            last_json_start = content.rfind('[{"question"')
            if last_json_start > 0:
                content = content[last_json_start:]
                print(f"检测到重复内容，提取最后部分: {content[:100]}...")
        
        # 尝试找到JSON数组的开始和结束
        if content.startswith('[') and content.endswith(']'):
            json_content = content
        else:
            # 如果不是标准JSON格式，尝试提取JSON部分
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = content[start_idx:end_idx+1]
            else:
                # 如果找不到完整的JSON，尝试查找部分JSON
                question_start = content.find('{"question"')
                if question_start != -1:
                    # 构造一个最小的JSON数组
                    partial_content = content[question_start:]
                    # 尝试找到第一个完整的对象
                    brace_count = 0
                    end_pos = -1
                    for i, char in enumerate(partial_content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if end_pos > 0:
                        json_content = '[' + partial_content[:end_pos] + ']'
                        print(f"构造部分JSON: {json_content}")
                    else:
                        raise ValueError(f"无法在响应中找到有效的JSON格式")
                else:
                    raise ValueError(f"无法在响应中找到有效的JSON数组格式")
        
        # 解析 JSON
        qa = json.loads(json_content)
        if not isinstance(qa, list) or len(qa) < 1:
            raise ValueError("生成的问答对数量不足或格式错误")
        
        return qa
        
    except json.JSONDecodeError as e:
        print(f"JSON解析失败 (尝试 {retry_count + 1}/3): {e}")
        print(f"问题内容前200字符: {content[:200] if 'content' in locals() else 'N/A'}")
        # 限制重试次数避免无限递归
        if retry_count < 2:
            print("重试中...")
            return build_qa_one(text, retry_count + 1)
        else:
            print("重试次数超限，跳过此文本")
            return []
    except Exception as e:
        print(f"生成问答对失败 (尝试 {retry_count + 1}/3): {e}")
        if retry_count < 2:
            print("重试中...")
            return build_qa_one(text, retry_count + 1)
        else:
            print("重试次数超限，跳过此文本")
            return []

def main():
    # 确保输出目录存在
    os.makedirs(QA_FILE.parent, exist_ok=True)
    
    all_qa = []
    for txt_file in tqdm(list(DATA_DIR.glob("*.txt")), desc="生成问答对"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        qa_list = build_qa_one(text)
        for qa in qa_list:
            qa["doc"] = txt_file.stem   # 记录来源，方便溯源
        all_qa.extend(qa_list)

    QA_FILE.write_text(json.dumps(all_qa, ensure_ascii=False, indent=2))
    print(f"完成！共 {len(all_qa)} 条问答对 → {QA_FILE}")

if __name__ == "__main__":
    main()