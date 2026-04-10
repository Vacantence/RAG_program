import streamlit as st
import time
from rag import RagService
import config_data as config

# 标题
st.title("🚀 智能 RAG 客服系统")
st.caption("基于 LangChain + ChromaDB + 通义千问")
st.divider()

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置管理")
    
    # 会话 ID 管理
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "user_001"
    
    session_id = st.text_input("当前会话 ID", value=st.session_state["session_id"])
    if session_id != st.session_state["session_id"]:
        st.session_state["session_id"] = session_id
        st.session_state["message"] = [{"role": "assistant", "content": "你好，我是新会话，有什么可以帮您？"}]
        st.rerun()

    # 清除历史
    if st.button("🗑️ 清空当前对话"):
        from file_history_store import get_history
        history = get_history(st.session_state["session_id"])
        history.clear()
        st.session_state["message"] = [{"role": "assistant", "content": "对话已清空，请开始新问题。"}]
        st.success("对话记录已清空")
        st.rerun()

    st.divider()
    st.info("💡 提示: 本系统支持多轮对话、来源追溯以及多查询检索优化。")

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好,有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 渲染历史消息
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("🔍 查看参考来源"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**来源 {i+1}: {doc.metadata.get('source', '未知')}**")
                    st.caption(doc.page_content)

# 在页面中添加输入框
prompt = st.chat_input("请输入您的问题...")

if prompt:
    # 在页面输出用户的问题
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    
    with st.spinner("🧠 AI 正在分析并生成回答..."):
        # 调用 RAG 链
        # 注意：使用用户输入的 session_id
        session_config = {
            "configurable": {
                "session_id": st.session_state["session_id"],
            }
        }
        
        response = st.session_state["rag"].chain.invoke(
            {"input": prompt}, 
            config=session_config
        )
        
        answer = response["answer"]
        sources = response["sources"]
        
        # 显示回答
        with st.chat_message("assistant"):
            st.write(answer)
            
            # 显示来源追溯
            if sources:
                with st.expander("查看参考来源"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**来源 {i+1}: {doc.metadata.get('source', '未知')}**")
                        st.caption(doc.page_content)
        
        st.session_state["message"].append({
            "role": "assistant", 
            "content": answer,
            "sources": sources # 保存来源到会话状态
        })
