import os, json
import logging
import hashlib
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

from dotenv import load_dotenv
load_dotenv()

# 参数配置
COLLECTION = "faq_demo"
FAQ_TXT = "faq.txt"
DIM = 1024
TOP_K = 3
STATE_FILE = ".index.json"

# 配置嵌入模型和 LLM
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v3", 
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

# 向量存储
vector_store = MilvusVectorStore(
    uri="tcp://127.0.0.1:19530",
    collection_name=COLLECTION, 
    dim=DIM, 
    overwrite=False
)

# 构建/加载索引
def build_or_load_index() -> VectorStoreIndex:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if os.path.exists(".index.json"):
        logger.info("加载已有索引...")
        return VectorStoreIndex.from_vector_store(vector_store)
    
    # 读取 FAQ 文件
    logger.info("从 %s 构建索引...", FAQ_TXT)
    docs: List[Document] = []
    for line in open(FAQ_TXT, encoding="utf8"):
        q, a = line.strip().split("\t")
        doc_text = f"问题: {q}\n答案: {a}"
        docs.append(Document(text=doc_text, metadata={"question": q, "answer": a}))
    
    logger.info("读取了 %d 个文档", len(docs))
    
    # 使用语义切分器
    semantic_splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=Settings.embed_model
    )
    
    # 使用 StorageContext 确保正确写入
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    logger.info("使用语义切分器构建索引...")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[semantic_splitter]
    )
    logger.info("索引构建完成")
    
    # 验证写入
    from pymilvus import connections, Collection
    import time
    
    connections.connect(host="127.0.0.1", port=19530)
    c = Collection(COLLECTION)
    c.flush()
    time.sleep(1)
    c.load()
    logger.info("Collection '%s' 已写入 %d 个实体", COLLECTION, c.num_entities)
    
    # 保存状态（包含已索引的问题列表）
    save_state(docs)
    return index

def save_state(docs: List[Document]):
    """保存已索引的 FAQ 问题列表"""
    state = {
        "built": True,
        "questions": [doc.metadata["question"] for doc in docs]
    }
    json.dump(state, open(STATE_FILE, "w"), ensure_ascii=False, indent=2)

def load_existing_questions() -> set:
    """加载已索引的问题集合"""
    if not os.path.exists(STATE_FILE):
        return set()
    try:
        state = json.load(open(STATE_FILE))
        return set(state.get("questions", []))
    except:
        return set()

def add_new_faqs(new_faqs: List[tuple]) -> int:
    """增量添加新的 FAQ
    
    Args:
        new_faqs: [(question, answer), ...] 列表
    
    Returns:
        添加的数量
    """
    if not new_faqs:
        return 0
    
    logger = logging.getLogger(__name__)
    logger.info("开始添加 %d 个新 FAQ...", len(new_faqs))
    
    # 构建 Document
    docs = [
        Document(
            text=f"问题: {q}\n答案: {a}",
            metadata={"question": q, "answer": a}
        )
        for q, a in new_faqs
    ]
    
    # 使用语义切分器
    semantic_splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=Settings.embed_model
    )
    
    # 使用 StorageContext 确保正确写入（和初始构建一样的方式）
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 创建临时索引来处理新文档（会自动生成嵌入并写入）
    logger.info("使用 StorageContext 添加新文档...")
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[semantic_splitter]
    )
    logger.info("新文档已写入 Milvus")
    
    # 刷新
    from pymilvus import connections, Collection
    import time
    
    connections.connect(host="127.0.0.1", port=19530)
    c = Collection(COLLECTION)
    c.flush()
    time.sleep(0.5)
    c.load()
    
    logger.info("当前总实体数: %d", c.num_entities)
    
    # 更新状态文件
    existing = load_existing_questions()
    existing.update([q for q, _ in new_faqs])
    state = {
        "built": True,
        "questions": list(existing)
    }
    json.dump(state, open(STATE_FILE, "w"), ensure_ascii=False, indent=2)
    
    return len(new_faqs)

index = build_or_load_index()

# 全局 query_engine，会在 reindex 后更新
_query_engine = index.as_query_engine(similarity_top_k=TOP_K)

def get_query_engine():
    """获取 query_engine（使用缓存，reindex 后会更新）"""
    return _query_engine

def refresh_query_engine():
    """刷新 query_engine（在 reindex 后调用）"""
    global _query_engine
    current_index = VectorStoreIndex.from_vector_store(vector_store)
    _query_engine = current_index.as_query_engine(similarity_top_k=TOP_K)

# FastAPI 接口
app = FastAPI()

class AskReq(BaseModel): 
    question: str

class AskResp(BaseModel): 
    answer: str
    retrieved_faqs: List[dict]

class ReindexReq(BaseModel):
    question: str
    answer: str

class ReindexResp(BaseModel):
    status: str
    message: str
    added_count: int

@app.post("/ask", response_model=AskResp)
#curl -sS -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "运费怎么计算"}' | jq
def ask(req: AskReq):
    logger = logging.getLogger(__name__)
    logger.info("用户问题: %s", req.question)
    
    # 动态获取最新的 query_engine
    query_engine = get_query_engine()
    
    # 召回相关 FAQ
    nodes = query_engine.retrieve(req.question)
    logger.info("召回 %d 个相关FAQ", len(nodes))
    
    # 构建召回列表
    retrieved_faqs = []
    for i, n in enumerate(nodes):
        faq_item = {
            "question": n.node.metadata.get("question", ""),
            "answer": n.node.metadata.get("answer", ""),
            "score": float(n.score)
        }
        retrieved_faqs.append(faq_item)
        logger.info("  #%d (%.4f): %s", i+1, n.score, faq_item["question"])
    
    # 构建提示词
    context = "\n".join([
        f"FAQ{i+1}: 问题：{faq['question']} 答案：{faq['answer']}"
        for i, faq in enumerate(retrieved_faqs)
    ])
    
    prompt = f"""你是一个智能客服助手。请根据以下相关FAQ信息，回答用户的问题。
如果FAQ中没有相关信息，请礼貌地告知用户。

相关FAQ：
{context}

用户问题：{req.question}

请给出准确、友好的回答："""
    
    # 调用 LLM 生成答案
    response = Settings.llm.complete(prompt)
    final_answer = response.text.strip()
    logger.info("生成答案: %s", final_answer)
    
    return AskResp(answer=final_answer, retrieved_faqs=retrieved_faqs)

@app.post("/reindex", response_model=ReindexResp)
def reindex(req: ReindexReq):
    '''

        curl -X POST http://127.0.0.1:8000/reindex \
        -H "Content-Type: application/json" \
        -d '{
            "question": "支持国际配送吗？",
            "answer": "目前仅支持中国大陆地区配送，暂不支持国际配送"
        }'


        curl -sS -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "国际配送吗"}' | jq

    '''
    """增量添加新的 FAQ"""
    logger = logging.getLogger(__name__)
    logger.info("收到 reindex 请求: Q='%s'", req.question)
    
    # 检查是否已存在
    existing_questions = load_existing_questions()
    if req.question in existing_questions:
        return ReindexResp(
            status="skipped",
            message=f"问题已存在: {req.question}",
            added_count=0
        )
    
    try:
        # 添加新 FAQ
        added = add_new_faqs([(req.question, req.answer)])
        
        # 刷新 query_engine 以使用最新数据
        refresh_query_engine()
        logger.info("query_engine 已刷新")
        
        return ReindexResp(
            status="success",
            message=f"成功添加 {added} 个FAQ",
            added_count=added
        )
    except Exception as e:
        logger.exception("添加 FAQ 失败: %s", e)
        return ReindexResp(
            status="failed",
            message=f"添加失败: {str(e)}",
            added_count=0
        )

def main():
    """作业入口，启动 FAQ 问答服务"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()