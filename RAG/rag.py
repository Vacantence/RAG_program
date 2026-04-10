from typing import List, Dict, Any
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history
import logging

logger = logging.getLogger("RAG-System.RagService")

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.language_models import BaseChatModel

class RagService:
    def __init__(self):
        logger.info("Initializing RagService...")
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个专业且简洁的智能客服。请优先基于提供的参考资料回答用户问题。"
                 "如果参考资料中没有相关信息，请诚实告知用户。\n\n参考资料:\n{context}"),
                MessagesPlaceholder("history"),
                ("user", "{input}")
            ]
        )
        
        self.chain = self.__get_chain()
    
    def __get_chain(self):
        """构建 RAG 执行链条"""
        base_retriever = self.vector_service.get_retriever()
        
        # 引入 MultiQueryRetriever 提升检索召回率
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, 
            llm=self.chat_model
        )
        
        def format_docs(docs: List[Document]) -> str:
            if not docs:
                return "未找到相关参考资料。"
            return "\n\n".join([f"内容: {doc.page_content}\n来源: {doc.metadata.get('source', '未知')}" for doc in docs])

        # 1. 检索上下文
        retrieve_context = {
            "context": (lambda x: x["input"]) | retriever,
            "input": lambda x: x["input"],
            "history": lambda x: x["history"]
        }

        # 2. 生成回答
        # 注意：RunnableWithMessageHistory 要求输入是一个字典，且包含 input_messages_key 指定的键
        # 我们在这里构建一个能够同时返回回答和上下文的链
        
        def process_input(inputs):
            # inputs: {context: List[Document], input: str, history: List}
            formatted_context = format_docs(inputs["context"])
            return {
                "input": inputs["input"],
                "context": formatted_context,
                "history": inputs["history"],
                "raw_context": inputs["context"] # 保留原始文档用于后续展示
            }

        core_chain = (
            RunnableParallel(retrieve_context)
            | RunnableLambda(process_input)
            | {
                "answer": self.prompt_template | self.chat_model | StrOutputParser(),
                "sources": lambda x: x["raw_context"]
            }
        )

        # 包装历史记录
        # 由于 RunnableWithMessageHistory 默认处理的是单输出，我们需要确保它能正确处理 dict 输出
        # 或者我们手动管理历史记录以获得更多控制权。为了保持简单且符合 LangChain 规范：
        chain_with_history = RunnableWithMessageHistory(
            core_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer" # 指定哪个键是回答，用于存入历史
        )
        
        return chain_with_history

if __name__ == "__main__":
    # 测试代码
    service = RagService()
    res = service.chain.invoke(
        {"input": "我体重150斤,尺码推荐"},
        config=config.session_config
    )
    print(f"回答: {res['answer']}")
    print(f"来源数量: {len(res['sources'])}")