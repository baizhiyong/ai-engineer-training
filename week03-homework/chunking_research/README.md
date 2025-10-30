# 文本分块策略对比研究项目

## 项目概述

本项目是一个完整的 RAG（检索增强生成）系统文本分块策略评估框架，覆盖了从数据准备到最终报告生成的全流程实验管道。项目基于 LlamaIndex 框架，通过系统性对比不同文本分块策略的效果，为 RAG 系统优化提供数据驱动的参数选择指导。

## 核心特性

- **全流程自动化**：从数据获取到报告生成的完整实验管道
- **多维度评估**：检索效果、生成质量、效率指标的综合评估体系
- **参数网格搜索**：基于 YAML 配置的参数组合自动生成与测试
- **可视化报告**：自动生成包含图表和分析的详细评估报告
- **模块化设计**：每个步骤独立可执行，便于调试和扩展

## 技术架构

### 工作流程

```
数据准备 → 文档处理 → 问答生成 → 实验执行 → 指标评估 → 报告生成
    ↓           ↓           ↓           ↓           ↓           ↓
  step01      step02      step05      step06      step07      step08
   ↓           ↓           ↓           ↓           ↓           ↓
数据下载 → PDF转文本 → 自动问答 → 分块实验 → 效果评分 → 图表报告
```

### 目录结构

```
chunking_research/
├── main.py                    # 项目入口文件
├── experiments.yaml           # 实验参数配置
├── README.md                  # 项目说明文档
├── data/
│   ├── raw/                   # 原始数据目录
│   └── clean/                 # 清理后数据目录
├── outputs/                   # 实验结果输出目录
│   ├── qa.json               # 标准问答对
│   ├── metrics.csv           # 评估指标汇总
│   └── *.json                # 各实验配置的结果
├── scripts/                   # 工作流脚本目录
│   ├── step01_data_downloader.py    # 数据下载器
│   ├── step02_pdf_to_txt.py         # PDF文本提取
│   ├── step03_generate_synth_docs.py # 合成文档生成
│   ├── step04_clean_txt.py          # 文本清理
│   ├── step05_auto_qa.py            # 自动问答生成
│   ├── step06_run_exps.py           # 分块实验执行
│   ├── step07_scoring.py            # 指标计算
│   └── step08_gen_report.py         # 报告生成
├── templates/
│   └── report_template.md     # 报告模板
└── report/
    ├── report.md             # 最终评估报告
    └── img/                  # 图表文件目录
```

## 实验管道详解

### 阶段一：数据准备（Steps 01-04）

#### Step 01: 数据下载器 (`step01_data_downloader.py`)
- **功能**：从多个来源自动下载研究数据
- **数据源**：
  - 维基百科文章（量子计算、气候变化、人工智能等）
  - arXiv 学术论文
- **特性**：
  - 面向对象设计，易于扩展
  - 完整的错误处理和日志记录
  - 批量下载功能
  - 下载状态监控

#### Step 02: PDF文本提取器 (`step02_pdf_to_txt.py`)
- **功能**：将 PDF 文件批量转换为文本文件
- **技术**：基于 PyMuPDF (fitz) 库
- **特性**：
  - 动态路径计算，无论在哪里运行都能正确找到目标文件夹
  - 兼容 PyMuPDF 1.26.x 版本的新 API
  - 自动遍历 PDF 的每一页进行文本提取

#### Step 03: 合成文档生成器 (`step03_generate_synth_docs.py`)
- **功能**：使用大模型生成高质量的实验文档
- **生成类型**：
  - 产品说明书（智能降噪耳机）
  - 科幻小说节选（2077年火星城市）
  - 公司年报（科技公司财报致股东信）
- **要求**：每类文档不少于 1000 字，连贯叙述，包含复杂句式和段落结构

#### Step 04: 文本清理器 (`step04_clean_txt.py`)
- **功能**：对原始文本进行清理和标准化
- **清理规则**：
  - 去除页眉页脚（页码、文档标识等）
  - 合并多余空行（≥3行空行合并为1行）
  - 去除网址、邮件地址
  - 长度过滤（保留1200-3000字符的文档）

### 阶段二：问答对生成（Step 05）

#### Step 05: 自动问答生成器 (`step05_auto_qa.py`)
- **功能**：基于清理后的文本自动生成高质量问答对
- **技术**：使用 DashScope API (qwen-plus模型)
- **生成策略**：
  - 每个文本生成1-2个问题和对应答案
  - 问题覆盖：核心观点、关键技术、数据案例、潜在影响、结构逻辑
  - 答案要求：简短精准（≤50字），可直接定位到原文
- **输出格式**：JSON数组，包含问题、答案、文档来源信息

### 阶段三：分块实验执行（Step 06）

#### Step 06: 分块实验运行器 (`step06_run_exps.py`)
- **功能**：系统性执行不同分块策略的对比实验
- **支持的分块策略**：
  - **SentenceSplitter**：基于句子的分块
  - **TokenTextSplitter**：基于 Token 的分块
  - **SentenceWindowNodeParser**：句子窗口分块
  - **MarkdownNodeParser**：Markdown 结构分块（可选）
  
- **参数配置**（基于 `experiments.yaml`）：
  ```yaml
  splitters:
    - name: Sentence
      cls: SentenceSplitter
      params:
        - {chunk_size: [256,512,1024], chunk_overlap: [0,50,100]}
    - name: Token
      cls: TokenTextSplitter
      params:
        - {chunk_size: [128,256,512], chunk_overlap: [20,40,80]}
    - name: SentenceWindow
      cls: SentenceWindowNodeParser
      params:
        - {window_size: [1,3,5]}
  ```

- **执行模式**：
  - **sequential**：顺序执行（默认，稳定可靠）
  - **parallel**：并行执行（速度快，但可能遇到兼容性问题）
  - **auto**：自动模式（先测试再选择最佳方式）

- **实验流程**：
  1. 根据配置生成参数笛卡尔积（共20组实验）
  2. 为每组参数创建对应的分块器
  3. 将文档分块并构建向量索引
  4. 使用问答对进行查询测试
  5. 保存每组实验的查询结果

### 阶段四：效果评估（Step 07）

#### Step 07: 指标计算器 (`step07_scoring.py`)
- **功能**：计算 RAG 系统的多维度评估指标

- **检索效果指标**：
  - **Hit@1**：检索的Top-1文档中是否包含答案（Rouge-L ≥ 0.5）
  - **Hit@3**：检索的Top-3文档中是否包含答案
  - **Hit@5**：检索的Top-5文档中是否包含答案
  - **MRR (Mean Reciprocal Rank)**：平均倒数排名，衡量相关文档排序质量
  - **Redundancy**：检索文档间的冗余度（1 - 去重token数/总token数）

- **生成质量指标**：
  - **BLEU**：基于n-gram重叠的生成文本质量评估
  - **BERTScore_F1**：基于BERT嵌入的语义相似度评估

- **输出**：
  - `metrics.csv`：包含所有指标的评估结果表格
  - `bertscore_bar.png`：BERTScore_F1指标的可视化柱状图

### 阶段五：报告生成（Step 08）

#### Step 08: 报告生成器 (`step08_gen_report.py`)
- **功能**：自动生成包含图表和分析的详细评估报告
- **生成内容**：
  - 实验概览和主要结果表格
  - 综合性能对比图（检索性能 vs 生成质量）
  - 详细指标分析图（每个指标单独的柱状图）
  - 自动结论生成（最佳配置识别）
  - 参数显著性分析
  - 深度参数分析和实施建议

- **图表类型**：
  - 单指标柱状图（7个指标）
  - 检索性能综合对比图
  - 生成质量综合对比图
  - 所有图表支持高分辨率输出（300 DPI）

## 实验结果概览

### 主要发现

基于20组不同分块策略的对比实验，我们发现：

1. **最佳生成质量**：`SentenceWindow_window_size_3`（BERTScore-F1 = 0.4297）
2. **最佳检索命中率**：`Sentence_chunk_size_256_chunk_overlap_0`（Hit@5 = 0.0556）
3. **上下文冗余最低**：`Token_chunk_size_128_chunk_overlap_20`（Redundancy = 0.2853）

### 参数影响分析

- **chunk_size**：小切片（256）在检索精确度上表现最佳
- **chunk_overlap**：适中重叠（50-100）在生成质量上效果更好
- **策略类型**：SentenceWindow 在语义相似度上有优势，Sentence 在检索命中率上更强

## 快速开始

### 环境要求

```bash
# Python 3.11+
pip install llama-index
pip install llama-index-llms-openai-like
pip install llama-index-embeddings-dashscope
pip install dashscope
pip install evaluate bert-score
pip install pandas seaborn matplotlib
pip install tqdm jinja2
pip install pymupdf  # for PDF processing
pip install wikipedia-api arxiv  # for data downloading
```

### 配置环境变量

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

### 执行完整实验

```bash
# 方式一：逐步执行（推荐用于调试）
cd scripts/
python step01_data_downloader.py
python step02_pdf_to_txt.py
python step03_generate_synth_docs.py
python step04_clean_txt.py
python step05_auto_qa.py
python step06_run_exps.py
python step07_scoring.py
python step08_gen_report.py

# 方式二：从项目根目录运行主入口
python -m chunking_research.main
```

### 自定义实验配置

编辑 `experiments.yaml` 文件来调整实验参数：

```yaml
splitters:
  - name: CustomSentence
    cls: SentenceSplitter
    params:
      - {chunk_size: [128, 256], chunk_overlap: [10, 20]}
```

## 扩展指南

### 添加新的分块策略

1. 在 `step06_run_exps.py` 的 `get_splitter()` 函数中添加新策略
2. 在 `experiments.yaml` 中配置参数
3. 运行实验获得对比结果

### 添加新的评估指标

1. 在 `step07_scoring.py` 中实现新指标计算逻辑
2. 在 `step08_gen_report.py` 中添加对应的可视化代码
3. 更新 `templates/report_template.md` 模板

### 添加新的数据源

1. 扩展 `step01_data_downloader.py` 的 `DataDownloader` 类
2. 添加对应的文档处理逻辑
3. 确保生成的问答对质量

## 注意事项

- 确保 `DASHSCOPE_API_KEY` 环境变量正确设置
- 首次运行会下载 NLTK 数据，需要网络连接
- 实验过程中会调用 API，请注意使用量和频率限制
- 大量文档处理时建议分批执行
- 所有脚本都支持动态路径计算，可在任意目录运行

## 技术债务与改进方向

1. **性能优化**：向量索引构建可缓存复用
2. **评估指标扩展**：增加更多语义理解相关指标
3. **实验配置**：支持更复杂的参数组合和约束
4. **报告增强**：添加交互式图表和统计显著性检验
5. **数据质量**：加入文档质量评估和过滤机制

## 许可证

本项目为教学用途，遵循 MIT 许可证。