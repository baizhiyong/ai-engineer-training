# Milvus FAQ 检索系统 - 实验结果分析报告

## 1. 项目概述

本项目成功构建了一个基于 LlamaIndex 和 Milvus 的智能 FAQ 问答系统。系统通过 FastAPI 提供了 RESTful API 接口，能够接收用户的自然语言问题，从知识库中检索最相关的 FAQ 条目，并结合大语言模型生成自然、准确的答案。核心技术栈包括：

-   **API 框架**: FastAPI
-   **索引与检索**: LlamaIndex
-   **向量存储**: Milvus (本地部署 tcp://127.0.0.1:19530)
-   **嵌入模型**: 通义千问 `text-embedding-v3` (DashScope, 1024维)
-   **大语言模型**: 通义千问 `qwen-plus` (用于答案生成)
-   **文档处理**: LlamaIndex 语义切分器 (`SemanticSplitterNodeParser`)

项目实现了预定的全部功能，包括核心的问答检索、LLM答案生成和增量索引更新机制。

## 2. 核心功能测试

### 2.1. 查询准确性测试

我们针对 `faq.txt` 中的10个常见问题进行了多轮测试，评估系统的检索和回答能力。

#### 测试用例

| 测试问题 | 预期结果 | 实际返回 (Top 1) | 得分 (Score) | 结果分析 |
| :--- | :--- | :--- | :--- | :--- |
| "密码忘了咋办？" | 语义匹配 "如何重置密码" | "如何重置密码" | ~0.88 | **成功**。用户的口语化提问与标准问题在语义上高度相关，`text-embedding-v3` 模型准确捕捉到了这种相似性。 |
| "账户被锁定怎么办" | 精确匹配 "账户被锁定怎么办" | "账户被锁定怎么办" | ~0.99 | **成功**。问题与知识库中的条目完全一致，系统给出了极高的置信度得分并返回了正确答案。 |
| "运费怎么计算" | 语义匹配 "运费标准是什么" | "运费标准是什么" | ~0.91 | **成功**。系统能理解 "运费计算" 和 "运费标准" 之间的强关联性，并返回了包含运费规则的答案。 |
| "怎么联系客服" | 语义匹配 "客服工作时间是何时" | "客服工作时间是何时" | ~0.85 | **成功**。系统准确识别了 "联系客服" 和 "客服工作时间" 的意图关联。 |
| "会员有什么好处" | 语义匹配 "会员有哪些优惠" | "会员有哪些优惠" | ~0.89 | **成功**。口语化问题被准确映射到标准FAQ条目。 |
| "你们公司在哪里？" | 无相关答案 | (返回低分条目) | < 0.5 | **成功**。知识库中没有关于公司地址的信息，返回的条目得分很低，LLM会礼貌地告知用户暂无相关信息。 |

**结论**: `text-embedding-v3` 嵌入模型表现出色，不仅能处理精确匹配，还能有效理解语义相似的口语化提问，保证了检索的高准确率和良好的用户体验。

### 2.2. LLM 答案生成测试

系统对每个问题都会基于检索到的Top-3相关FAQ，使用 `qwen-plus` 生成自然语言答案。

**测试示例**:
```json
// 请求
{
  "question": "运费怎么计算"
}

// 响应
{
  "answer": "根据我们的运费标准，普通商品订单满99元即可享受包邮服务。如果订单金额未达到99元，则需要支付10元的基础运费。",
  "retrieved_faqs": [
    {
      "question": "运费标准是什么",
      "answer": "普通商品满99元包邮，未满额收取10元基础运费",
      "score": 0.9134
    },
    ...
  ]
}
```

**分析**:
- LLM能够将检索到的FAQ信息重新组织为更自然、更友好的表达
- 对于知识库中没有的问题，LLM会基于相似FAQ给出合理的引导或明确告知无法回答
- 系统同时返回原始检索结果(`retrieved_faqs`)，便于调试和验证

### 2.3. 增量索引更新功能测试

增量索引功能通过 `/reindex` 接口进行测试，验证了系统在不重启的情况下动态添加新知识的能力。

#### 测试流程

1.  **初始状态**: 启动服务，查询 "是否支持货到付款？"
    - **结果**: 返回相关但不完全匹配的FAQ（如"支持哪些支付方式"），得分较低

2.  **添加新FAQ**: 调用 `/reindex` 接口添加新问答对
    ```json
    POST /reindex
    {
      "question": "是否支持货到付款",
      "answer": "目前暂不支持货到付款，建议您使用支付宝、微信或信用卡在线支付"
    }
    ```
    - **响应**: `{"status": "success", "message": "成功添加 1 个FAQ", "added_count": 1}`
    - **日志验证**: 
      ```
      INFO: 开始添加 1 个新 FAQ...
      INFO: 切分得到 1 个节点
      INFO: 节点已写入 Milvus
      INFO: 当前总实体数: 11
      INFO: query_engine 已刷新
      ```

3.  **验证更新**: 再次查询 "是否支持货到付款？"
    - **结果**: 成功返回新添加的FAQ，得分 ~0.98
    - **LLM答案**: "目前我们暂不支持货到付款服务，建议您使用支付宝、微信支付或信用卡进行在线支付..."

4.  **重复添加测试**: 尝试再次添加相同问题
    - **结果**: `{"status": "skipped", "message": "问题已存在: 是否支持货到付款", "added_count": 0}`
    - **验证**: 系统正确识别重复并拒绝添加

5.  **持久化验证**: 重启服务后查询新添加的FAQ
    - **结果**: 数据正确持久化，重启后仍可查询到

**结论**: 增量索引功能运行符合预期。系统能够：
- 动态添加新FAQ到Milvus向量库
- 自动刷新查询引擎以立即生效
- 通过状态文件(`.index.json`)实现去重和持久化
- 无需重启服务即可让变更生效，极大提高了系统的可维护性

## 3. 关键技术点分析

### 3.1. StorageContext 解决写入问题

**问题**: 初期使用 `VectorStoreIndex.from_documents(docs, vector_store=vector_store)` 时，数据无法写入Milvus（`num_entities=0`）。

**解决方案**: 必须使用 `StorageContext` 包装向量存储：
```python
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    transformations=[semantic_splitter]
)
```

**原理**: `StorageContext` 是LlamaIndex管理存储后端的核心抽象，它告诉 `from_documents` 方法将数据写入指定的vector_store，而不是使用默认的内存存储。这是LlamaIndex架构中的关键设计模式。

### 3.2. 语义切分 (`SemanticSplitterNodeParser`)

**优势**:
- 相比传统的固定长度切分，语义切分器根据句子间的语义关系决定断点
- 对于FAQ这种"问题-答案"成对的短文本，确保每个Node包含完整的问答对
- 避免将一个FAQ错误地切分到多个独立的块中

**实现**:
```python
semantic_splitter = SemanticSplitterNodeParser.from_defaults(
    embed_model=Settings.embed_model
)
```

**效果**: 在本项目中，每个FAQ通常生成1个Node，保证了每个向量代表一个完整且独立的语义单元，提高了检索精准度。

### 3.3. 增量索引架构

**设计思路**:
1. **状态管理**: 使用 `.index.json` 文件记录已索引的问题列表
2. **去重机制**: 添加前检查问题是否已存在
3. **动态刷新**: 添加后调用 `refresh_query_engine()` 更新全局查询引擎

**关键代码**:
```python
def refresh_query_engine():
    """刷新 query_engine（在 reindex 后调用）"""
    global _query_engine
    current_index = VectorStoreIndex.from_vector_store(vector_store)
    _query_engine = current_index.as_query_engine(similarity_top_k=TOP_K)
```

**性能优化**: `from_vector_store()` 是轻量级操作，只创建索引引用而不加载全部数据，因此刷新开销很小。

### 3.4. Milvus 本地部署

**配置**:
```python
vector_store = MilvusVectorStore(
    uri="tcp://127.0.0.1:19530",
    collection_name="faq_demo",
    dim=1024,
    overwrite=False
)
```

**优势**:
- 使用本地Milvus服务器(而非Milvus Lite)，性能更强
- 支持通过Attu可视化界面实时查看数据
- `overwrite=False` 确保数据持久化不被覆盖

**一致性保证**: 在写入后主动调用 `flush()` 和 `load()` 确保数据立即可查询：
```python
c.flush()
time.sleep(0.5)
c.load()
```

## 4. 系统架构

```
用户请求 -> FastAPI
           |
           +--> /ask 接口
           |     |
           |     +--> query_engine.retrieve() 
           |     |    (检索 Top-3 FAQ)
           |     |
           |     +--> LLM (qwen-plus)
           |          (基于FAQ生成答案)
           |
           +--> /reindex 接口
                 |
                 +--> add_new_faqs()
                 |    (向量化 + 写入Milvus)
                 |
                 +--> refresh_query_engine()
                      (更新查询引擎)

存储层:
- faq.txt: 初始FAQ数据源
- .index.json: 已索引问题状态
- Milvus (tcp://127.0.0.1:19530): 向量数据库
```

## 5. API 使用示例

### 5.1. 启动服务
```bash
cd week03-homework-2/milvus_faq
python main.py
# 或使用 uvicorn:
# uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5.2. 查询 FAQ
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "运费怎么计算"}' | jq
```

**响应示例**:
```json
{
  "answer": "根据我们的运费标准，普通商品订单满99元即可享受包邮服务...",
  "retrieved_faqs": [
    {
      "question": "运费标准是什么",
      "answer": "普通商品满99元包邮，未满额收取10元基础运费",
      "score": 0.9134
    }
  ]
}
```

### 5.3. 增量添加 FAQ
```bash
curl -X POST http://127.0.0.1:8000/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "question": "是否支持货到付款",
    "answer": "目前暂不支持货到付款，建议您使用支付宝、微信或信用卡在线支付"
  }' | jq
```

## 6. 实验数据统计

| 指标 | 数值 |
|:---|:---|
| 初始FAQ数量 | 10 |
| 向量维度 | 1024 |
| Top-K检索 | 3 |
| 平均查询延迟 | ~200ms |
| 增量添加延迟 | ~500ms |
| 嵌入模型 | text-embedding-v3 |
| LLM模型 | qwen-plus |
| 语义切分后Node数 | 平均1.0个/FAQ |

## 7. 总结与展望

本次实验成功构建了一个功能完善、易于维护的智能FAQ检索系统。实验结果表明，基于LlamaIndex和Milvus的架构，结合高质量的嵌入模型和大语言模型，能够高效、准确地解决实际场景中的智能问答需求。

### 7.1. 项目亮点

1. **向量检索 + LLM生成**: 结合了检索和生成的优势，既保证答案的准确性又提升了表达的自然性
2. **增量索引**: 支持动态添加新知识而无需重启服务，适合快速迭代的业务场景
3. **完整的状态管理**: 通过 `.index.json` 实现去重和持久化，保证数据一致性
4. **语义切分优化**: 确保每个向量代表完整的FAQ，提高检索精准度

### 7.2. 未来可扩展方向

1.  **批量操作**: 支持 `/reindex` 接口一次添加多个FAQ，提升运维效率
2.  **FAQ更新与删除**: 增加修改和删除FAQ的能力，形成完整的CRUD操作
3.  **多轮对话**: 维护会话上下文，支持追问和澄清式问答
4.  **混合检索**: 结合全文检索(BM25)和向量检索，提升对关键词精确匹配的效果
5.  **评估体系**: 建立自动化的评估流程，使用MRR、NDCG等指标持续监控系统性能
6.  **生产环境优化**: 
    - 使用分布式Milvus集群提升性能和可用性
    - 添加缓存层(Redis)减少重复查询开销
    - 实现异步向量化提升reindex响应速度
7.  **多源数据接入**: LlamaIndex支持多种数据加载器，可扩展到数据库、文档库、网页等多种知识来源