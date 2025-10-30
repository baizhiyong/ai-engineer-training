#!/usr/bin/env python3
"""
RAG系统评估指标计算工具

功能说明：
========
本工具用于计算RAG（检索增强生成）系统的多维度评估指标，包括检索效果和生成质量两大类指标。

评估指标详解：
============

检索效果指标：
-----------
1. Hit@1：
   - 含义：检索的Top-1文档中是否包含答案
   - 计算方法：检索到的第1个文档与标准答案的Rouge-L >= 0.5即算命中
   - 取值范围：[0, 1]，越高越好
   - 反映：检索系统的精确度

2. Hit@3：
   - 含义：检索的Top-3文档中是否包含答案
   - 计算方法：检索到的前3个文档中任一个与标准答案的Rouge-L >= 0.5即算命中
   - 取值范围：[0, 1]，越高越好
   - 反映：检索系统的召回能力（相对宽松）

3. Hit@5：
   - 含义：检索的Top-5文档中是否包含答案
   - 计算方法：检索到的前5个文档中任一个与标准答案的Rouge-L >= 0.5即算命中
   - 取值范围：[0, 1]，越高越好
   - 反映：检索系统的召回能力（更宽松）

4. MRR (Mean Reciprocal Rank)：
   - 含义：平均倒数排名，衡量相关文档在检索结果中的排序质量
   - 计算方法：1/首次命中位置的平均值，未命中为0
   - 取值范围：[0, 1]，越高越好
   - 反映：检索结果的排序质量，越早命中分数越高

5. Redundancy：
   - 含义：检索文档间的冗余度
   - 计算方法：1 - (去重token数 / 总token数)
   - 取值范围：[0, 1]，越低越好
   - 反映：检索结果的多样性，冗余度低说明文档差异化程度高

生成质量指标：
-----------
6. BLEU：
   - 含义：基于n-gram重叠的生成文本质量评估
   - 计算方法：生成答案与标准答案的n-gram精确度
   - 取值范围：[0, 1]，越高越好
   - 反映：生成文本与参考答案的词汇层面相似度

7. BERTScore_F1：
   - 含义：基于BERT嵌入的语义相似度评估
   - 计算方法：生成答案与标准答案在BERT语义空间的F1分数
   - 取值范围：[0, 1]，越高越好
   - 反映：生成文本与参考答案的语义层面相似度，比BLEU更能捕捉语义信息

输入输出：
========
输入：
- outputs目录下的实验结果JSON文件（run_exps.py生成）
- qa.json文件（auto_qa.py生成的标准问答对）

输出：
- metrics.csv：包含所有指标的评估结果表格
- bertscore_bar.png：BERTScore_F1指标的可视化柱状图

使用方式：
========
直接运行脚本即可：
python scoring.py

注意事项：
========
- 确保已安装依赖包：pip install evaluate bert-score rouge-chinese pandas seaborn tqdm
- 脚本会自动定位outputs目录，无需手动指定路径
- Hit指标使用Rouge-L >= 0.5作为命中阈值，可根据需要调整

作者: AI Assistant
日期: 2025-10-29
版本: 1.2.0 (添加指标详细说明)
"""

#pip install evaluate bert-score rouge-chinese pandas seaborn tqdm rouge_score
import json, pandas as pd, numpy as np, os
from pathlib import Path
from tqdm import tqdm
import evaluate
from bert_score import score as bert_score

def _get_default_outputs_dir():
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

# ---------- 1. 路径 ----------
OUTPUTS_BASE = _get_default_outputs_dir()
EXP_DIR  = Path(OUTPUTS_BASE)               # run_exps.py 生成的 json
QA_FILE  = Path(OUTPUTS_BASE) / "qa.json"  # auto_qa.py 生成的问答对
CSV_OUT  = Path(OUTPUTS_BASE) / "metrics.csv"

# ---------- 2. 载入标准问答 ----------
qa_map = {item["question"]: item["answer"] for item in json.loads(QA_FILE.read_text())}

# ---------- 3. 初始化 metric 函数 ----------
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")          # 用于 Hit 判断

def hit_at_k(contexts, answer, k):
    """只要任一 top-k 上下文与答案 Rouge-L >= 0.5 即算命中"""
    if not contexts or k == 0:
        return 0
    refs = [answer] * k
    hyps = contexts[:k]
    scores = []
    for h, r in zip(hyps, refs):
        result = rouge.compute(predictions=[h], references=[r])
        if result and "rougeL" in result:
            scores.append(result["rougeL"])
        else:
            scores.append(0.0)
    return int(max(scores) >= 0.5)

def redundancy_ratio(contexts):
    """去重 token 数 / 总 token 数"""
    from collections import Counter
    tokens_all = [tok for c in contexts for tok in c.split()]
    tokens_set = set(tokens_all)
    return 1 - len(tokens_set) / max(len(tokens_all), 1)

# ---------- 4. 逐实验打分 ----------
records = []
for exp_json in tqdm(list(EXP_DIR.glob("*.json")), desc="Scoring"):
    if exp_json.name == "qa.json":
        continue
    data = json.loads(exp_json.read_text())
    exp_name = exp_json.stem

    hits_1, hits_3, hits_5, mrr, redunds = [], [], [], [], []
    bleu_scores, bert_p, bert_r, bert_f1 = [], [], [], []

    for row in data:
        q, gen_ans, ctxs = row["question"], row["answer"], row["contexts"]
        ans = qa_map.get(q, "")
        if not ans:
            continue

        # ---- 检索指标 ----
        hits_1.append(hit_at_k(ctxs, ans, 1))
        hits_3.append(hit_at_k(ctxs, ans, 3))
        hits_5.append(hit_at_k(ctxs, ans, 5))
        # MRR
        first_hit = next((i + 1 for i, c in enumerate(ctxs) if hit_at_k([c], ans, 1)), 0)
        mrr.append(1 / first_hit if first_hit else 0.)
        redunds.append(redundancy_ratio(ctxs))

        # ---- 生成指标 ----
        bleu_result = bleu.compute(predictions=[gen_ans], references=[ans])
        bleu_score = bleu_result["bleu"] if bleu_result and "bleu" in bleu_result else 0.0
        bleu_scores.append(bleu_score)
        
        try:
            # BERTScore 返回三个张量的元组
            bert_scores = bert_score([gen_ans], [ans], lang="zh", verbose=False, rescale_with_baseline=True)
            P, R, F1 = bert_scores
            
            # 将张量转换为Python数值
            bert_p.append(float(P[0]))
            bert_r.append(float(R[0])) 
            bert_f1.append(float(F1[0]))
        except Exception as e:
            print(f"BERTScore计算错误: {e}")
            bert_p.append(0.0)
            bert_r.append(0.0)
            bert_f1.append(0.0)

    records.append({
        "exp": exp_name,
        "Hit@1": np.mean(hits_1),
        "Hit@3": np.mean(hits_3),
        "Hit@5": np.mean(hits_5),
        "MRR": np.mean(mrr),
        "Redundancy": np.mean(redunds),
        "BLEU": np.mean(bleu_scores),
        "BERTScore_F1": np.mean(bert_f1),
    })

# ---------- 5. 保存 ----------
df = pd.DataFrame(records)
df.to_csv(CSV_OUT, index=False, float_format="%.4f")
print(f"✅ 已写入 {CSV_OUT}  共 {len(df)} 组实验")

# ---------- 6. 快速可视化 ----------
import seaborn as sns, matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="exp", y="BERTScore_F1", color="steelblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("BERTScore-F1")
plt.tight_layout()
chart_path = Path(OUTPUTS_BASE) / "bertscore_bar.png"
plt.savefig(chart_path, dpi=300)
print(f"📊 快速柱状图已保存 → {chart_path}")