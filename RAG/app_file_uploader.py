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
    "请上传txt文件",
    type=['txt'],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024 #KB
    
    st.subheader(f"文件名:{file_name}")
    st.write(f"格式:{file_type} | 大小:{file_size:.2f} KB")
    
    #获取文件内容 getvalue -> bytes -> decode(utf-8)
    text = uploader_file.getvalue().decode('utf-8')
    
    with st.spinner("载入向量库中..."):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text,file_name)
        st.write(result)
    