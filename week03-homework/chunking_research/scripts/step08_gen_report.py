#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from jinja2 import Template
from datetime import datetime
import os, json


def _get_project_root() -> str:
    """获取项目根目录路径
    
    动态计算相对于当前脚本的chunking_research目录路径，
    确保无论在哪里运行都能正确找到项目根目录
    
    Returns:
        chunking_research目录的绝对路径
    """
    # 获取当前脚本所在目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从scripts目录向上找到chunking_research目录
    # scripts -> chunking_research
    chunking_research_dir = os.path.dirname(current_script_dir)
    
    return chunking_research_dir


# ---------- 1. 路径 ----------
PROJECT_ROOT = Path(_get_project_root())
CSV        = PROJECT_ROOT / "outputs/metrics.csv"
FIG_DIR    = PROJECT_ROOT / "report/img"; FIG_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE   = PROJECT_ROOT / "templates/report_template.md"
REPORT_MD  = PROJECT_ROOT / "report/report.md"

# ---------- 2. 读数据 ----------
df = pd.read_csv(CSV)
df = df.round(4)

# ---------- 3. 生成所有指标的图表 ----------
# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 定义所有指标及其配置
metrics_config = {
    'Hit@1': {'color': '#FF6B6B', 'title': '检索命中率 Hit@1'},
    'Hit@3': {'color': '#4ECDC4', 'title': '检索命中率 Hit@3'}, 
    'Hit@5': {'color': '#45B7D1', 'title': '检索命中率 Hit@5'},
    'MRR': {'color': '#96CEB4', 'title': '平均倒数排名 MRR'},
    'Redundancy': {'color': '#FFEAA7', 'title': '上下文冗余率'},
    'BLEU': {'color': '#DDA0DD', 'title': 'BLEU 生成质量'},
    'BERTScore_F1': {'color': '#98D8C8', 'title': 'BERTScore F1 语义相似度'}
}

# 生成单个指标图表
chart_files = {}
for metric, config in metrics_config.items():
    plt.figure(figsize=(12, 6))
    
    # 创建柱状图
    bars = plt.bar(range(len(df)), df[metric], color=config['color'], alpha=0.8, edgecolor='white', linewidth=0.7)
    
    # 设置标题和标签
    plt.title(config['title'], fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('实验配置', fontsize=12, fontweight='bold')
    plt.ylabel(metric, fontsize=12, fontweight='bold')
    
    # 设置x轴标签
    plt.xticks(range(len(df)), df['exp'].tolist(), rotation=45, ha='right', fontsize=9)
    
    # 在柱子上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:  # 只在有值的柱子上显示标签
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold')
    
    # 美化图表
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    chart_file = FIG_DIR / f"{metric.lower().replace('@', '_at_')}_chart.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    chart_files[metric] = chart_file

# ---------- 4. 生成综合对比图 ----------
# 4.1 检索指标对比图
plt.figure(figsize=(15, 8))
retrieval_metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'MRR']
x = range(len(df))
width = 0.2

for i, metric in enumerate(retrieval_metrics):
    offset = (i - len(retrieval_metrics)/2 + 0.5) * width
    plt.bar([pos + offset for pos in x], df[metric], width, 
            label=metric, alpha=0.8, color=metrics_config[metric]['color'])

plt.title('检索性能指标对比', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('实验配置', fontsize=12, fontweight='bold')
plt.ylabel('得分', fontsize=12, fontweight='bold')
plt.xticks(x, df['exp'].tolist(), rotation=45, ha='right', fontsize=9)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

retrieval_comparison = FIG_DIR / "retrieval_comparison.png"
plt.savefig(retrieval_comparison, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 4.2 生成质量指标对比图
plt.figure(figsize=(15, 8))
generation_metrics = ['BLEU', 'BERTScore_F1']
x = range(len(df))
width = 0.35

for i, metric in enumerate(generation_metrics):
    offset = (i - len(generation_metrics)/2 + 0.5) * width
    plt.bar([pos + offset for pos in x], df[metric], width,
            label=metric, alpha=0.8, color=metrics_config[metric]['color'])

plt.title('生成质量指标对比', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('实验配置', fontsize=12, fontweight='bold')
plt.ylabel('得分', fontsize=12, fontweight='bold')
plt.xticks(x, df['exp'].tolist(), rotation=45, ha='right', fontsize=9)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

generation_comparison = FIG_DIR / "generation_comparison.png"
plt.savefig(generation_comparison, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ 已生成 {len(chart_files)} 个单指标图表")
print(f"✅ 已生成 2 个综合对比图表")

# ---------- 4. 关键数值 ----------
best_gen  = df.loc[df["BERTScore_F1"].idxmax()]
best_ret  = df.loc[df["Hit@5"].idxmax()]
best_redu = df.loc[df["Redundancy"].idxmin()]
delta_bert = df[df["exp"].str.contains("chunk_overlap_100")]["BERTScore_F1"].mean() - \
             df[df["exp"].str.contains("chunk_overlap_0")]["BERTScore_F1"].mean()
delta_hit  = df[df["exp"].str.contains("window_size_3")]["Hit@5"].mean() - \
             df[df["exp"].str.contains("window_size_1")]["Hit@5"].mean()

# ---------- 5. 渲染 ----------
tmpl = Template(open(TEMPLATE).read())
md = tmpl.render(
    gen_date    = datetime.now().strftime("%Y-%m-%d %H:%M"),
    exp_num     = len(df),
    tbl         = df.to_dict(orient="records"),
    # 所有单指标图表
    hit1_chart  = os.path.relpath(chart_files['Hit@1'], REPORT_MD.parent),
    hit3_chart  = os.path.relpath(chart_files['Hit@3'], REPORT_MD.parent),
    hit5_chart  = os.path.relpath(chart_files['Hit@5'], REPORT_MD.parent),
    mrr_chart   = os.path.relpath(chart_files['MRR'], REPORT_MD.parent),
    redundancy_chart = os.path.relpath(chart_files['Redundancy'], REPORT_MD.parent),
    bleu_chart  = os.path.relpath(chart_files['BLEU'], REPORT_MD.parent),
    bertscore_chart = os.path.relpath(chart_files['BERTScore_F1'], REPORT_MD.parent),
    # 综合对比图
    retrieval_comparison = os.path.relpath(retrieval_comparison, REPORT_MD.parent),
    generation_comparison = os.path.relpath(generation_comparison, REPORT_MD.parent),
    # 向后兼容的变量
    bertscore_png = os.path.relpath(chart_files['BERTScore_F1'], REPORT_MD.parent),
    hit5_png    = os.path.relpath(chart_files['Hit@5'], REPORT_MD.parent),
    best_gen    = best_gen,
    best_ret    = best_ret,
    best_redu   = best_redu,
    delta_bert  = round(delta_bert, 3),
    delta_hit   = round(delta_hit, 3),
)

REPORT_MD.write_text(md, encoding="utf-8")
print(f"✅ 报告已生成 → {REPORT_MD}")

# ---------- 6. 可选：转 PDF ----------
# pip install markdown pdfkit
# import markdown, pdfkit
# html = markdown.markdown(md, extensions=['tables'])
# pdfkit.from_string(html, "report/report.pdf")