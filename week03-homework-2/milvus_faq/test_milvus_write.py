#!/usr/bin/env python3
"""直接测试 MilvusVectorStore 写入能力

这个脚本会：
1. 手动创建带 embedding 的 TextNode
2. 直接调用 vector_store.add() 写入
3. 检查 Milvus collection 的 num_entities
4. 验证是否是 VectorStoreIndex.from_documents 的问题
"""
import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding

# 配置
COLLECTION = "test_faq_write"
DIM = 1024

# 设置 embedding
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v3", 
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 创建 vector store (overwrite=True 强制重建)
vector_store = MilvusVectorStore(
    uri="tcp://127.0.0.1:19530",
    collection_name=COLLECTION,
    dim=DIM,
    overwrite=True
)

print(f"Testing write to collection '{COLLECTION}'...")

# 手动创建一些测试节点
test_nodes = [
    TextNode(
        text="这是测试问题1",
        metadata={"question": "测试1", "answer": "答案1"},
        id_="test_node_1"
    ),
    TextNode(
        text="这是测试问题2",
        metadata={"question": "测试2", "answer": "答案2"},
        id_="test_node_2"
    ),
    TextNode(
        text="这是测试问题3",
        metadata={"question": "测试3", "answer": "答案3"},
        id_="test_node_3"
    )
]

print(f"Created {len(test_nodes)} test nodes")

# 为节点生成 embeddings
print("Generating embeddings...")
for node in test_nodes:
    embedding = Settings.embed_model.get_text_embedding(node.get_content())
    node.embedding = embedding
    print(f"  Node {node.id_}: embedding dim = {len(embedding)}")

# 直接调用 vector_store.add()
print(f"\nCalling vector_store.add() with {len(test_nodes)} nodes...")
try:
    vector_store.add(test_nodes)
    print("✓ vector_store.add() completed without exception")
except Exception as e:
    print(f"✗ vector_store.add() failed: {e}")
    import traceback
    traceback.print_exc()

# 检查 Milvus 中的实体数量
print("\nChecking Milvus collection...")
try:
    from pymilvus import connections, Collection, utility
    # Reconnect (llama_index may have closed the connection)
    connections.connect(host="127.0.0.1", port=19530)
    
    c = Collection(COLLECTION)
    
    # Force flush to ensure data is persisted
    print(f"Flushing collection '{COLLECTION}'...")
    c.flush()
    
    print(f"Loading collection '{COLLECTION}'...")
    c.load()
    
    # Check num_entities
    num = c.num_entities
    print(f"Collection '{COLLECTION}' num_entities = {num}")
    
    # Also check via utility
    try:
        stats = c.num_entities
        print(f"Via Collection.num_entities: {stats}")
    except:
        pass
    
    if num > 0:
        print("\n✓✓✓ SUCCESS: Vectors were written to Milvus!")
        print("This means the issue is with VectorStoreIndex.from_documents, not MilvusVectorStore itself.")
        print("\nSOLUTION: Need to call flush() after vector_store.add() in main.py")
    else:
        print("\n⚠ num_entities is 0, but you see data in Attu")
        print("This means data was written but not flushed/visible via Python API yet.")
        print("SOLUTION: Add explicit flush() calls in main.py after index building.")
except Exception as e:
    print(f"Failed to check collection: {e}")
    import traceback
    traceback.print_exc()
