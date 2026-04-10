import os
from dotenv import load_dotenv
import logging

# 加载 .env 文件
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("RAG-System")

# 基础配置
md5_path = os.getenv("MD5_PATH", "./md5.text")

# Chroma
collection_name = os.getenv("COLLECTION_NAME", "rag")
persist_directory = os.getenv("PERSIST_DIRECTORY", "./chroma_db")

# Spliter
chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_spliter_char_number = int(os.getenv("MAX_SPLITTER_CHAR_NUMBER", 1000))

# 相似度检索数量
similarity_k = int(os.getenv("SIMILARITY_K", 3))

# 模型配置
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
chat_model_name = os.getenv("CHAT_MODEL_NAME", "qwen-max")

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}