import os
import logging
import pandas as pd
from fastapi import FastAPI
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext,SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
# 配置嵌入模型和 LLM
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v3", 
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    embed_batch_size=10
)

aliyun_llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

Settings.llm = aliyun_llm

# --- 多跳查询主函数 ---
def multi_hop_query(question: str):
    """
    执行多跳查询：RAG -> KG -> LLM
    """
    # 1. 从 RAG 中获取相关信息
    rag_results = rag_query(question)

    # 2. 从 KG 中获取相关知识
    kg_results = kg_query(rag_results)

    # 3. 使用 LLM 生成最终答案
    final_answer = llm_generate(question, kg_results)

    return final_answer

def rag_query(question: str):
    # 使用llamaindex的RAG查询引擎,从本地读取数据
    df = pd.read_csv("knowledge_points.csv")
    documents = []
    for _, row in df.iterrows():
        doc_text = f"知识点: {row['name']}\n描述: {row['description']}\n学科: {row['field']}"
        documents.append(Document(text=doc_text, metadata={"name": row['name']}))
    logging.info("读取了 %d 个文档", len(documents))

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    context = query_engine.retrieve(question)
    logging.info("召回信息：" + ", ".join(map(str, context)))
    # 通过大模型，从问题中提取出知识点名字，使用本地RAG
    prompt = f"""从用户问题中提取出知识点。根据以下知识点的相关信息。

知识点相关信息：
{context}

用户问题：{question}

请给出准确、友好的回答：直接返回知识点的名字，不要返回其他"""
    
    logging.info("LLM 提取知识点的提示词: %s", prompt)
    # 调用 LLM 生成答案
    response = Settings.llm.complete(prompt)
    final_answer = response.text.strip()
    logging.info("提取出的知识点: %s", final_answer)

    return final_answer

def kg_query(rag_results: str):
    # 调用 KG 查询引擎
    entity_name = rag_results  # 假设 RAG 返回的就是实体名称
    # 构建推理路径记录
    reasoning_path = []
    reasoning_path.append(f"步骤 1: 从 RAG 获取到的实体名称为 '{entity_name}'。")
    # 根据实体名称构造 Cypher 查询
    #查询指数函数在所有学科中的应用
    cypher_query = f"""
    MATCH (math:Math {{name: '{entity_name}'}})<-[r:CROSS_DISCIPLINE_LINK]-(other)
    RETURN math.name AS 数学概念,
       other.name AS 应用领域,
       labels(other) AS 学科,
       r.type AS 应用类型
    """
    reasoning_path.append(f"步骤 2: 构造 Cypher 查询在图谱中查找。")
    reasoning_path.append(f"   - Cypher 查询: {cypher_query.strip()}")
    # 直接执行 Cypher
    graph_store = Neo4jGraphStore(
        username="neo4j",
        password="test1234",
        url="bolt://localhost:7687",
        database="neo4j",
    )
    kg_query_engine = KnowledgeGraphQueryEngine(
            storage_context=StorageContext.from_defaults(graph_store=graph_store),
            llm=aliyun_llm,
            verbose=True, # 打印生成的 Cypher 查询，便于调试
    )
    #graph_store = kg_query_engine.storage_context.graph_store
    graph_response = graph_store.query(cypher_query)
    kg_result_text = str(graph_response)
    #kg_response = kg_query_engine.query(rag_results)
    reasoning_path.append(f"步骤 3: 从图谱中获取到的结果为 '{kg_result_text}'。")
    logging.info("\n".join(reasoning_path))
    return kg_result_text
def llm_generate(question: str, kg_results: str):
    # 使用 LLM 生成答案
    prompt = f"""根据以下图谱信息，回答用户的问题：{question}
    用户问题涉及的知识点和相关应用信息如下：
    {kg_results}
    请给出准确、友好的回答：
    """
    logging.info("LLM 生成答案的提示词: %s", prompt)
    response = Settings.llm.complete(prompt)
    final_answer = response.text.strip()
    return final_answer

# FastAPI 接口
app = FastAPI()


class AskReq(BaseModel): 
    question: str

class AskResp(BaseModel): 
    answer: str

@app.post("/multi-hop-query",response_model=AskResp)
# eg. curl -X POST "http://localhost:8000/multi-hop-query" -H "Content-Type: application/json" -d "{\"question\": \"指数函数在物理学中的应用有哪些？\"}"
def multi_hop_query_endpoint(req: AskReq):
    logging.info("收到多跳查询请求，问题: %s", req.question)
    question = req.question
    answer = multi_hop_query(question)
    return AskResp(answer=answer)


def main():
# 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
# 在根目录可以通过python -m graph_rag.main 运行
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()