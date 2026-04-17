import streamlit as st
import time
from rag import RagService
import config_data as config

# 标题
st.title("🕸️ LangGraph 智能客服")
st.caption("基于 LangGraph 状态机 + Tool Calling + 记忆持久化")
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
    st.info("💡 提示: 本系统已升级为 LangGraph 架构，支持复杂的循环思考与工具调用。")

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好,我是基于 LangGraph 的智能助手，有什么可以帮您？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 渲染历史消息
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "thought" in message and message["thought"]:
            with st.expander("💭 思考过程 (工具调用)"):
                st.caption(message["thought"])

# 在页面中添加输入框
prompt = st.chat_input("请输入您的问题...")

if prompt:
    # 在页面输出用户的问题
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    
    with st.spinner("🧠 LangGraph 思考中..."):
        # 调用新版 RagService (LangGraph 版)
        response = st.session_state["rag"].invoke(
            prompt, 
            session_id=st.session_state["session_id"]
        )
        
        answer = response["output"]
        
        # 提取思考过程（中间的消息，如 Tool 消息或带有 ToolCalls 的 AI 消息）
        all_msgs = response["messages"]
        thought_process = []
        for msg in all_msgs:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                thought_process.append(f"🤖 决定调用工具: {msg.tool_calls[0]['name']}")
            elif msg.type == "tool":
                thought_process.append(f"🛠️ 工具返回结果 (长度: {len(msg.content)} 字符)")
        
        thought_str = "\n\n".join(thought_process) if thought_process else ""
        
        # 显示回答
        with st.chat_message("assistant"):
            st.write(answer)
            if thought_str:
                with st.expander("💭 思考过程 (工具调用)"):
                    st.caption(thought_str)
        
        st.session_state["message"].append({
            "role": "assistant", 
            "content": answer,
            "thought": thought_str
        })
