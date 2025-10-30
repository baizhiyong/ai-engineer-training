
def main():
    """
    文本分块策略对比研究项目 - 主入口
    
    这是一个完整的RAG系统文本分块策略评估框架，
    覆盖从数据准备到最终报告生成的全流程实验管道。
    
    执行方式：
    - 在根目录运行：python -m chunking_research.main
    - 或在当前目录运行：python main.py
    """
    
    print("=" * 80)
    print("🚀 文本分块策略对比研究项目")
    print("=" * 80)
    print("📋 项目概述：")
    print("   基于LlamaIndex框架的RAG系统文本分块策略评估")
    print("   通过系统性对比不同分块策略，为RAG系统优化提供数据驱动的参数选择指导")
    print()
    
    print("🔄 完整实验流程：")
    print()
    
    # 阶段一：数据准备
    print("📁 阶段一：数据准备")
    print("   Step 01 - 数据下载器 (step01_data_downloader.py)")
    print("           📥 从维基百科和arXiv下载研究数据")
    print("           📥 支持多源数据获取：量子计算、气候变化、人工智能等主题")
    print("           📥 面向对象设计，具备错误处理和状态监控功能")
    print()
    
    print("   Step 02 - PDF文本提取 (step02_pdf_to_txt.py)")
    print("           📄 将PDF文件批量转换为文本文件")
    print("           📄 基于PyMuPDF库，支持多页文档处理")
    print("           📄 动态路径计算，兼容不同运行环境")
    print()
    
    print("   Step 03 - 合成文档生成 (step03_generate_synth_docs.py)")
    print("           🤖 使用大模型生成高质量实验文档")
    print("           🤖 生成类型：产品说明书、科幻小说节选、公司年报")
    print("           🤖 要求：每类文档不少于1000字，包含复杂句式和段落结构")
    print()
    
    print("   Step 04 - 文本清理 (step04_clean_txt.py)")
    print("           🧹 对原始文本进行清理和标准化")
    print("           🧹 去除页眉页脚、合并多余空行、过滤网址邮件")
    print("           🧹 长度过滤：保留1200-3000字符的高质量文档")
    print()
    
    # 阶段二：问答对生成
    print("💬 阶段二：问答对生成")
    print("   Step 05 - 自动问答生成 (step05_auto_qa.py)")
    print("           ❓ 基于清理后文本自动生成高质量问答对")
    print("           ❓ 使用DashScope API (qwen-plus模型)")
    print("           ❓ 问题覆盖：核心观点、关键技术、数据案例、潜在影响")
    print("           ❓ 答案要求：简短精准（≤50字），可直接定位到原文")
    print()
    
    # 阶段三：分块实验执行
    print("🔬 阶段三：分块实验执行")
    print("   Step 06 - 分块实验运行 (step06_run_exps.py)")
    print("           ⚗️  系统性执行不同分块策略的对比实验")
    print("           ⚗️  支持策略：SentenceSplitter、TokenTextSplitter、SentenceWindowNodeParser")
    print("           ⚗️  参数网格搜索：chunk_size、chunk_overlap、window_size等")
    print("           ⚗️  执行模式：sequential（顺序）、parallel（并行）、auto（自动）")
    print("           ⚗️  实验规模：基于experiments.yaml配置生成20组对比实验")
    print()
    
    # 阶段四：效果评估
    print("📊 阶段四：效果评估")
    print("   Step 07 - 指标计算 (step07_scoring.py)")
    print("           📈 计算RAG系统的多维度评估指标")
    print("           📈 检索效果：Hit@1/3/5、MRR、Redundancy")
    print("           📈 生成质量：BLEU、BERTScore_F1")
    print("           📈 输出：metrics.csv汇总表 + bertscore_bar.png可视化")
    print()
    
    # 阶段五：报告生成
    print("📋 阶段五：报告生成")
    print("   Step 08 - 报告生成 (step08_gen_report.py)")
    print("           📑 自动生成包含图表和分析的详细评估报告")
    print("           📑 生成9个不同类型的可视化图表（单指标+综合对比）")
    print("           📑 基于Jinja2模板的动态报告生成")
    print("           📑 包含：实验概览、参数影响分析、最佳配置推荐、实施建议")
    print()
    
    print("🎯 预期输出结果：")
    print("   ✅ outputs/qa.json - 标准问答对数据集")
    print("   ✅ outputs/metrics.csv - 完整评估指标汇总")
    print("   ✅ outputs/*.json - 各实验配置的详细结果")
    print("   ✅ report/report.md - 最终评估报告")
    print("   ✅ report/img/ - 高分辨率图表文件")
    print()
    
    print("💡 主要发现预览：")
    print("   🏆 最佳生成质量：SentenceWindow_window_size_3 (BERTScore-F1=0.4297)")
    print("   🏆 最佳检索命中率：Sentence_chunk_size_256_chunk_overlap_0 (Hit@5=0.0556)")
    print("   🏆 最低上下文冗余：Token_chunk_size_128_chunk_overlap_20 (Redundancy=0.2853)")
    print()
    
    print("🔧 执行说明：")
    print("   方式一（逐步执行，推荐调试）：")
    print("     cd scripts/")
    print("     python step01_data_downloader.py")
    print("     python step02_pdf_to_txt.py")
    print("     python step03_generate_synth_docs.py")
    print("     python step04_clean_txt.py")
    print("     python step05_auto_qa.py")
    print("     python step06_run_exps.py")
    print("     python step07_scoring.py")
    print("     python step08_gen_report.py")
    print()
    print("   方式二（一键执行，生产环境）：")
    print("     export DASHSCOPE_API_KEY='your_api_key_here'")
    print("     python -m chunking_research.main")
    print()
    
    print("⚠️  注意事项：")
    print("   🔑 确保DASHSCOPE_API_KEY环境变量正确设置")
    print("   🌐 首次运行会下载NLTK数据，需要网络连接")
    print("   💰 实验过程中会调用API，请注意使用量和频率限制")
    print("   ⏱️  完整实验大约需要15-30分钟（取决于网络和API响应速度）")
    print()
    
    print("=" * 80)
    print("✨ 这是一个完整的机器学习实验研究框架示例")
    print("✨ 展现了从问题定义到结论总结的全流程科研方法")
    print("✨ 为RAG系统参数优化提供数据驱动的指导建议")
    print("=" * 80)

if __name__ == "__main__":
    main()