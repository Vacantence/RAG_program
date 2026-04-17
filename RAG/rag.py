from typing import List, Dict, Any, Union, Annotated, Sequence, TypedDict
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from file_history_store import get_history
import logging

logger = logging.getLogger("RAG-System.RagService")

# 定义图的状态
class AgentState(TypedDict):
    # add_messages 会将新消息追加到现有列表中，而不是覆盖它
    messages: Annotated[Sequence[BaseMessage], add_messages]

class RagService:
    def __init__(self):
        logger.info("Initializing LangGraph Agentic RagService...")
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        # 必须绑定工具到模型上，才能使用 tool calling
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        
        # 1. 定义工具
        self.tools = [self._create_knowledge_tool()]
        self.model_with_tools = self.chat_model.bind_tools(self.tools)
        
        # 2. 构建图
        self.workflow = self._build_graph()
        
        # 3. 编译图 (这里暂不使用内置 Checkpointer，因为用户已有文件历史存储逻辑)
        self.app = self.workflow.compile()

    def _create_knowledge_tool(self):
        """将 RAG 检索逻辑封装为 Tool"""
        retriever = self.vector_service.get_retriever()

        @tool
        def company_knowledge_search(query: str) -> str:
            """
            当用户询问关于公司内部政策、产品信息、尺码推荐、洗涤养护等专业知识时，调用此工具。
            输入应该是用户查询的关键词或问题描述。
            """
            docs = retriever.invoke(query)
            if not docs:
                return "在知识库中未找到相关信息。"
            
            formatted_docs = []
            for doc in docs:
                source = doc.metadata.get("source", "未知来源")
                formatted_docs.append(f"内容: {doc.page_content}\n来源: {source}")
            
            return "\n\n---\n\n".join(formatted_docs)
            
        return company_knowledge_search

    def _build_graph(self):
        """构建 LangGraph 工作流图"""
        builder = StateGraph(AgentState)

        # 定义节点：调用模型
        def call_model(state: AgentState, config: RunnableConfig):
            # 获取历史记录并与当前输入合并 (由外部 invoke 传入历史)
            messages = state['messages']
            response = self.model_with_tools.invoke(messages, config)
            return {"messages": [response]}

        # 定义节点：工具执行器 (使用预置的 ToolNode)
        tool_node = ToolNode(self.tools)

        # 添加节点到图中
        builder.add_node("agent", call_model)
        builder.add_node("tools", tool_node)

        # 设置入口点
        builder.set_entry_point("agent")

        # 定义边：条件边决定是否调用工具
        def should_continue(state: AgentState):
            messages = state['messages']
            last_message = messages[-1]
            # 如果模型输出了 tool_calls，则跳转到 tools 节点
            if last_message.tool_calls:
                return "tools"
            # 否则结束
            return END

        builder.add_conditional_edges(
            "agent",
            should_continue,
        )

        # 定义边：工具执行完后必须回到 agent 再次判断
        builder.add_edge("tools", "agent")

        return builder

    def invoke(self, input_text: str, session_id: str = "user_001"):
        """封装调用方法，兼容原有的历史记录逻辑"""
        # 1. 获取历史消息
        history_store = get_history(session_id)
        history_messages = history_store.messages
        
        # 2. 构造当前输入
        current_message = HumanMessage(content=input_text)
        
        # 3. 运行图
        inputs = {"messages": history_messages + [current_message]}
        config = {"configurable": {"thread_id": session_id}}
        
        final_state = self.app.invoke(inputs, config)
        
        # 4. 提取 AI 的最终回答并更新历史记录
        final_ai_message = final_state["messages"][-1]
        
        # 将新消息（Human + AI + 过程中的 Tool 消息）存入历史
        # 注意：add_messages 只会返回增量或全量，这里我们需要计算出本轮新增的消息
        new_messages = final_state["messages"][len(history_messages):]
        history_store.add_messages(new_messages)
        
        return {
            "output": final_ai_message.content,
            "messages": final_state["messages"]
        }

if __name__ == "__main__":
    # 测试代码
    service = RagService()
    res = service.invoke("你好，请问我150斤该穿什么尺码？", session_id="user_001")
    print(f"回答: {res['output']}")