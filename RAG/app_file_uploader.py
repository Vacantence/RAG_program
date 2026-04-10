"""
基于Streamlit完成web网页上传服务

streamlit当页面元素发生变化,代码会重新执行一遍,状态会重置
"""

import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

# 添加网页标题
st.title("知识库更新服务")

# file_uploader
uploader_file = st.file_uploader(
    "📤 上传知识文件 (支持 .txt, .pdf)",
    type=['txt', 'pdf'],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024 # KB
    
    st.info(f"📄 文件名: {file_name} | 📏 大小: {file_size:.2f} KB")
    
    with st.spinner("⏳ 正在载入向量库，请稍候..."):
        # 直接使用 getvalue 获取字节内容
        file_content = uploader_file.getvalue()
        result = st.session_state["service"].upload_by_file(file_content, file_name)
        
        if "[成功]" in result:
            st.success(result)
        elif "[跳过]" in result:
            st.warning(result)
        else:
            st.error(result)
    