"""
查询脚本 - 从磁盘加载索引到内存并查询售后问题
"""
import os
from data_loader import FAQDataLoader
from config import settings


def main():
    """主函数：从磁盘加载索引并查询售后问题"""
    print("FAQ查询系统")
    print("=" * 40)
    
    # 检查索引文件是否存在
    if not os.path.exists(settings.faiss_index_path):
        print(f"错误: 索引文件不存在: {settings.faiss_index_path}")
        print("请先运行 train.py 生成索引")
        return
    
    # 从磁盘加载索引到内存
    print("正在从磁盘加载索引到内存...")
    data_loader = FAQDataLoader()
    index = data_loader.load_index()
    
    if index is None:
        print("索引加载失败")
        return
    
    print("索引已成功加载到内存")
    
    # 查询售后问题
    query = "如何退货？"
    print(f"查询问题: {query}")
    print("-" * 40)
    
    query_engine = index.as_query_engine(similarity_top_k=settings.top_k)
    response = query_engine.query(query)
    
    print(f"查询结果: {response}")
    print("\n" + "=" * 40)
    print("最相关的FAQ条目:")
    print("=" * 40)
    
    # 输出最相关的FAQ条目
    for i, node in enumerate(response.source_nodes, 1):
        metadata = node.node.metadata
        score = node.score
        print(f"\n【FAQ条目 {i}】(相似度: {score:.4f})")
        print(f"问题: {metadata.get('question', 'N/A')}")
        print(f"答案: {metadata.get('answer', 'N/A')}")
        print("-" * 30)


if __name__ == "__main__":
    main()