#!/usr/bin/env python3
"""
分块实验运行脚本

支持三种执行模式：
1. sequential - 顺序执行（默认，稳定可靠）
2. parallel - 并行执行（速度快，但可能遇到兼容性问题）
3. auto - 自动模式（先测试再选择最佳方式）

使用方法：
python run_exps.py                           # 默认顺序模式（推荐）
python run_exps.py --mode sequential         # 显式指定顺序模式
python run_exps.py --mode parallel --jobs 4  # 并行模式，4个线程
python run_exps.py --mode auto --jobs 2      # 自动模式，最多2个线程
"""
import os, json, yaml, itertools
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Tuple

# 添加NLTK数据初始化
try:
    import nltk
    # 尝试下载必要的NLTK数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("下载NLTK数据...")
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
except ImportError:
    print("NLTK未安装，跳过NLTK数据初始化")

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SentenceWindowNodeParser,
    MarkdownNodeParser,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# ---------- 1. 全局配置 ----------
Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True,
    temperature=0.3,
)
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)

def _get_project_dirs() -> Tuple[str, str]:
    """获取项目目录路径
    
    动态计算相对于当前脚本的目录路径，
    确保无论在哪里运行都能正确找到
    
    Returns:
        tuple: (data_clean_dir, outputs_dir)
    """
    # 获取当前脚本所在目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从scripts目录向上找到chunking_research目录
    # scripts -> chunking_research -> data -> clean
    chunking_research_dir = os.path.dirname(current_script_dir)
    data_clean_dir = os.path.join(chunking_research_dir, "data", "clean")
    outputs_dir = os.path.join(chunking_research_dir, "outputs")
    
    return data_clean_dir, outputs_dir

data_dir_str, output_dir_str = _get_project_dirs()
DATA_DIR: Path = Path(data_dir_str)
OUTPUT_DIR: Path = Path(output_dir_str)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- 2. 载入实验配置 ----------
def _get_config_path():
    """获取实验配置文件路径"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    chunking_research_dir = os.path.dirname(current_script_dir)
    return os.path.join(chunking_research_dir, "experiments.yaml")

cfg = yaml.safe_load(open(_get_config_path()))

# 检查数据目录是否存在
if not DATA_DIR.exists():
    print(f"错误: 数据目录不存在: {DATA_DIR}")
    print("请确保 data/clean 目录存在并包含文档文件")
    exit(1)

print(f"从 {DATA_DIR} 加载文档...")
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"已加载 {len(documents)} 个文档")

# 从 qa.json 文件中提取问题
def _get_qa_file_path():
    """获取 qa.json 文件路径"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    chunking_research_dir = os.path.dirname(current_script_dir)
    return os.path.join(chunking_research_dir, "outputs", "qa.json")

def _load_questions_from_qa_json():
    """从 qa.json 文件中加载问题列表"""
    qa_file_path = _get_qa_file_path()
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        # 提取所有问题，去重
        questions = list(set(item['question'] for item in qa_data if 'question' in item))
        return questions
    except FileNotFoundError:
        print(f"警告：qa.json 文件不存在 ({qa_file_path})，使用默认问题")
        return [
            "这篇文章的核心观点是什么？",
            "文中提到了哪些关键技术？",
            "作者给出了哪些数据或案例？",
            "这些技术可能带来哪些影响？",
            "文章的结构是怎样的？"
        ]
    except Exception as e:
        print(f"警告：加载 qa.json 文件失败 ({e})，使用默认问题")
        return [
            "这篇文章的核心观点是什么？",
            "文中提到了哪些关键技术？",
            "作者给出了哪些数据或案例？",
            "这些技术可能带来哪些影响？",
            "文章的结构是怎样的？"
        ]

QUESTIONS = _load_questions_from_qa_json()

# ---------- 3. splitter 工厂 ----------
def get_splitter(name: str, params: dict):
    if name == "SentenceSplitter":
        return SentenceSplitter(**params)
    if name == "TokenTextSplitter":
        return TokenTextSplitter(**params)
    if name == "SentenceWindowNodeParser":
        # window_size 是类方法参数
        return SentenceWindowNodeParser.from_defaults(
            window_size=params["window_size"],
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
    if name == "MarkdownNodeParser":
        return MarkdownNodeParser()
    raise ValueError(f"unknown splitter {name}")

# ---------- 4. 单组实验 ----------
def run_one(exp_id: str, splitter, docs, qs):
    try:
        print(f"开始实验: {exp_id}")
        nodes = splitter.get_nodes_from_documents(docs)
        print(f"生成了 {len(nodes)} 个节点")
        
        index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
        print(f"创建了向量索引")

        # SentenceWindow 需要特殊后处理
        post = None
        if isinstance(splitter, SentenceWindowNodeParser):
            post = [MetadataReplacementPostProcessor(target_metadata_key="window")]

        engine = index.as_query_engine(similarity_top_k=5, node_postprocessors=post or [])

        results = []
        for i, q in enumerate(qs):
            print(f"处理问题 {i+1}/{len(qs)}: {q[:50]}...")
            resp = engine.query(q)
            ctx = [n.node.get_content() for n in resp.source_nodes]
            results.append({"question": q, "answer": str(resp), "contexts": ctx})
            
        out_file = OUTPUT_DIR / f"{exp_id}.json"
        out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"完成实验: {exp_id}, 结果保存到: {out_file}")
        return exp_id
        
    except Exception as e:
        print(f"实验 {exp_id} 失败: {e}")
        import traceback
        traceback.print_exc()
        raise

# ---------- 5. 执行方式封装 ----------
def run_experiments_parallel(tasks, documents, questions, n_jobs=4):
    """并行执行实验
    
    Args:
        tasks: 任务列表，每个元素为 (exp_id, splitter)
        documents: 文档列表
        questions: 问题列表
        n_jobs: 并行任务数
        
    Returns:
        list: 成功完成的实验ID列表
    """
    # 过滤掉已存在结果文件的任务
    filtered_tasks = []
    skipped_count = 0
    
    for exp_id, sp in tasks:
        out_file = OUTPUT_DIR / f"{exp_id}.json"
        if out_file.exists():
            print(f"⏭️  跳过已存在的实验: {exp_id}")
            skipped_count += 1
        else:
            filtered_tasks.append((exp_id, sp))
    
    if skipped_count > 0:
        print(f"跳过了 {skipped_count} 个已完成的实验")
    
    if not filtered_tasks:
        print("所有实验都已完成！")
        return []
    
    print(f"使用并行模式执行 {len(filtered_tasks)} 个任务，并行度: {min(n_jobs, len(filtered_tasks))}")
    
    results = Parallel(n_jobs=min(n_jobs, len(filtered_tasks)), backend="threading")(
        delayed(run_one)(exp_id, sp, documents, questions) for exp_id, sp in tqdm(filtered_tasks, desc="并行执行")
    )
    return list(results)

def run_experiments_sequential(tasks, documents, questions):
    """顺序执行实验
    
    Args:
        tasks: 任务列表，每个元素为 (exp_id, splitter)
        documents: 文档列表
        questions: 问题列表
        
    Returns:
        list: 成功完成的实验ID列表
    """
    # 统计和过滤任务
    total_tasks = len(tasks)
    skipped_count = 0
    completed_count = 0
    failed_count = 0
    
    print(f"使用顺序模式执行 {total_tasks} 个任务")
    
    results = []
    for exp_id, sp in tqdm(tasks, desc="顺序执行"):
        # 检查输出文件是否已存在
        out_file = OUTPUT_DIR / f"{exp_id}.json"
        if out_file.exists():
            print(f"⏭️  跳过已存在的实验: {exp_id}")
            skipped_count += 1
            results.append(exp_id)  # 将已存在的实验也计入结果
            continue
            
        try:
            result = run_one(exp_id, sp, documents, questions)
            results.append(result)
            completed_count += 1
            print(f"✓ 完成: {exp_id}")
        except Exception as task_error:
            print(f"✗ 任务 {exp_id} 失败: {task_error}")
            failed_count += 1
            continue
    
    # 输出统计信息
    print(f"\n📊 执行统计:")
    print(f"   总任务数: {total_tasks}")
    print(f"   跳过已完成: {skipped_count}")
    print(f"   新完成: {completed_count}")
    print(f"   失败: {failed_count}")
    print(f"   成功率: {len(results)}/{total_tasks} ({len(results)/total_tasks*100:.1f}%)")
    
    return results

def run_experiments_auto(tasks, documents, questions, prefer_parallel=True, n_jobs=4):
    """自动选择执行方式
    
    Args:
        tasks: 任务列表
        documents: 文档列表
        questions: 问题列表
        prefer_parallel: 是否优先使用并行模式
        n_jobs: 并行任务数
        
    Returns:
        list: 成功完成的实验ID列表
    """
    print(f"共 {len(tasks)} 组实验，开始运行...")
    
    # 先检查已完成的任务
    remaining_tasks = []
    completed_existing = []
    
    for exp_id, sp in tasks:
        out_file = OUTPUT_DIR / f"{exp_id}.json"
        if out_file.exists():
            print(f"⏭️  跳过已存在的实验: {exp_id}")
            completed_existing.append(exp_id)
        else:
            remaining_tasks.append((exp_id, sp))
    
    if completed_existing:
        print(f"发现 {len(completed_existing)} 个已完成的实验")
    
    if not remaining_tasks:
        print("所有实验都已完成！")
        return completed_existing
    
    # 先运行一个测试任务
    print(f"测试运行第一个任务: {remaining_tasks[0][0]}")
    try:
        test_result = run_one(remaining_tasks[0][0], remaining_tasks[0][1], documents, questions)
        print(f"✓ 测试成功: {test_result}")
        # 移除已测试的任务
        final_remaining_tasks = remaining_tasks[1:]
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print("将使用顺序模式执行所有任务")
        return completed_existing + run_experiments_sequential(remaining_tasks, documents, questions)
    
    if not final_remaining_tasks:
        print("测试任务是最后一个任务，已全部完成")
        return completed_existing + [test_result]
    
    # 根据偏好选择执行方式
    if prefer_parallel:
        try:
            print("尝试并行执行剩余任务...")
            parallel_results = run_experiments_parallel(final_remaining_tasks, documents, questions, n_jobs)
            results = completed_existing + [test_result] + parallel_results
            print(f"✓ 并行执行完成，总共完成 {len(results)} 个任务")
            return results
        except Exception as e:
            print(f"✗ 并行执行失败: {e}")
            print("回退到顺序执行...")
            sequential_results = run_experiments_sequential(final_remaining_tasks, documents, questions)
            results = completed_existing + [test_result] + sequential_results
            print(f"✓ 顺序执行完成，总共完成 {len(results)} 个任务")
            return results
    else:
        print("使用顺序执行...")
        sequential_results = run_experiments_sequential(final_remaining_tasks, documents, questions)
        results = completed_existing + [test_result] + sequential_results
        print(f"✓ 顺序执行完成，总共完成 {len(results)} 个任务")
        return results

# ---------- 6. 构造参数笛卡尔积 ----------
tasks = []
for item in cfg["splitters"]:
    cls_name = item["cls"]
    # params 是列表，需要遍历每个参数组合
    for param_group in item["params"]:
        if not param_group:  # 空字典的情况
            param_dict = {}
            splitter = get_splitter(cls_name, param_dict)
            exp_id = f"{item['name']}"
            tasks.append((exp_id, splitter))
        else:
            # 把 dict-of-list 展开成 list-of-dict
            keys, values = zip(*param_group.items())
            for v in itertools.product(*values):
                param_dict = dict(zip(keys, v))
                splitter = get_splitter(cls_name, param_dict)
                # 实验编号：Splitter_cls_param1_val1_param2_val2
                exp_id = f"{item['name']}_" + "_".join(f"{k}_{v}" for k, v in param_dict.items())
                tasks.append((exp_id, splitter))

# ---------- 7. 执行实验 ----------
if __name__ == "__main__":
    import sys
    
    # 可以通过命令行参数选择执行模式
    # python run_exps.py                        # 默认顺序模式
    # python run_exps.py --mode parallel --jobs 4
    # python run_exps.py --mode sequential
    # python run_exps.py --mode auto
    
    mode = "sequential"  # 默认顺序模式，更稳定可靠
    n_jobs = 4          # 默认并行数（仅在并行/自动模式下使用）
    
    # 简单的命令行参数解析
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--mode" and i + 1 < len(sys.argv):
                mode = sys.argv[i + 1]
            elif arg == "--jobs" and i + 1 < len(sys.argv):
                try:
                    n_jobs = int(sys.argv[i + 1])
                except ValueError:
                    print(f"警告: 无效的并行数 '{sys.argv[i + 1]}'，使用默认值 {n_jobs}")
    
    print(f"执行模式: {mode}" + (f", 并行数: {n_jobs}" if mode in ["parallel", "auto"] else ""))
    
    # 根据模式执行
    if mode == "parallel":
        results = run_experiments_parallel(tasks, documents, QUESTIONS, n_jobs)
        print(f"\n🎉 并行执行完成，共完成 {len(results)} 个任务，结果见 outputs/*.json")
    elif mode == "sequential":
        results = run_experiments_sequential(tasks, documents, QUESTIONS)
        print(f"\n🎉 顺序执行完成，结果见 outputs/*.json")
    else:  # auto mode
        results = run_experiments_auto(tasks, documents, QUESTIONS, prefer_parallel=True, n_jobs=n_jobs)
        print(f"\n🎉 自动执行完成，结果见 outputs/*.json")