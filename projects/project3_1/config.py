"""
配置文件
"""
import os


# DashScope配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v3"

# 向量配置
FAISS_INDEX_PATH = "./data/faiss_index"
VECTOR_DIMENSION = 512

# FAQ文件
FAQ_FILE_PATH = "./FAQ.txt"

# 检索配置
TOP_K = 3


class Settings:
    """系统配置"""
    dashscope_api_key = DASHSCOPE_API_KEY
    dashscope_embedding_model = DASHSCOPE_EMBEDDING_MODEL
    faiss_index_path = FAISS_INDEX_PATH
    vector_dimension = VECTOR_DIMENSION
    faq_file_path = FAQ_FILE_PATH
    top_k = TOP_K


settings = Settings()