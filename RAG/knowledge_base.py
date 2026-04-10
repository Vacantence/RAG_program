"""
知识库
"""

import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger("RAG-System.KnowledgeBase")

def check_md5(md5_str: str) -> bool:
    """检查传入的md5字符串是否重复"""
    try:
        if not os.path.exists(config.md5_path):
            with open(config.md5_path, 'w', encoding="utf-8") as f:
                pass
            return False
        
        with open(config.md5_path, 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_str:
                    return True
        return False
    except Exception as e:
        logger.error(f"Error checking MD5: {e}")
        return False

def save_md5(md5_str: str):
    """将传入的md5字符串记录到文件内保存"""
    try:
        with open(config.md5_path, 'a', encoding='utf-8') as f:
            f.write(md5_str + '\n')
    except Exception as e:
        logger.error(f"Error saving MD5: {e}")

def get_string_md5(input_str: str, encoding='utf-8') -> str:
    """将传入的字符串转换为md5字符串"""
    return hashlib.md5(input_str.encode(encoding)).hexdigest()

class KnowledgeBaseService:
    def __init__(self):
        logger.info("Initializing KnowledgeBaseService...")
        os.makedirs(config.persist_directory, exist_ok=True)
        
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len
        )
        
    def upload_by_str(self, data: str, filename: str) -> str:
        """将传入的字符串向量化，存入向量数据库中"""
        try:
            md5_hex = get_string_md5(data)
            
            if check_md5(md5_hex):
                logger.info(f"Content from {filename} already exists, skipping.")
                return f"[跳过] {filename} 的内容已经存在于数据库中"
            
            # 分割文本
            if len(data) > config.max_spliter_char_number:
                knowledge_chunks = self.spliter.split_text(data)
            else:
                knowledge_chunks = [data]
                
            metadata = {
                "source": filename,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator": "Admin",
            }
            
            self.chroma.add_texts(
                knowledge_chunks,
                metadatas=[metadata for _ in knowledge_chunks],
            )
            
            save_md5(md5_hex)
            logger.info(f"Successfully uploaded {filename} to vector store.")
            return f"[成功] {filename} 已成功载入向量库"
        except Exception as e:
            logger.error(f"Error uploading string: {e}")
            return f"[失败] 载入 {filename} 时发生错误: {str(e)}"
    
    def upload_by_file(self, file_content: bytes, filename: str) -> str:
        """根据文件名后缀处理不同格式的文件并上传"""
        try:
            # 基础文本处理
            if filename.endswith('.txt'):
                text = file_content.decode('utf-8')
                return self.upload_by_str(text, filename)
            
            # 如果是 PDF，建议使用 PyPDFLoader (此处展示逻辑，需确保环境有 pypdf)
            elif filename.endswith('.pdf'):
                # 在 Streamlit 环境中，通常先保存为临时文件再读取
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    
                    # 提取文本并合并（或者直接 add_documents）
                    full_text = "\n".join([doc.page_content for doc in docs])
                    return self.upload_by_str(full_text, filename)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            else:
                return f"[失败] 目前不支持的文件格式: {filename.split('.')[-1]}"
                
        except Exception as e:
            logger.error(f"Error uploading file {filename}: {e}")
            return f"[失败] 处理文件 {filename} 时出错: {str(e)}"
    