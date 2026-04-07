# RAG案例项目

一个基于 LangChain、Chroma 和 Streamlit 的轻量级 RAG（检索增强生成）问答项目。项目支持将本地 TXT 文档写入向量库，并通过大模型结合知识库内容进行多轮问答，适合作为 RAG 入门与演示案例。

## 项目特点

- 基于 LangChain 构建完整 RAG 流程
- 使用 Chroma 作为本地向量数据库
- 使用 DashScope Embedding 模型完成文本向量化
- 使用通义千问聊天模型完成知识问答
- 支持基于文件的多轮对话历史持久化
- 提供知识库上传页面和问答页面两个 Streamlit 应用
- 支持基于 MD5 的文本去重，避免重复入库

## 项目结构

```text
RAG案例项目/
├─ app_file_uploader.py      # 知识库上传页面
├─ app_qa.py                 # 智能问答页面
├─ config_data.py            # 全局配置
├─ file_history_store.py     # 对话历史文件存储
├─ knowledge_base.py         # 文档切分与向量入库
├─ rag.py                    # RAG 问答链核心逻辑
├─ vector_stores.py          # 向量检索封装
├─ data/                     # 示例知识库文本
├─ chroma_db/                # 本地向量数据库目录
└─ chat_history/             # 会话历史记录目录
```

## 核心流程

1. 通过上传页面导入 TXT 文档
2. 对文本进行切分，并调用 Embedding 模型生成向量
3. 将文本片段及元数据写入 Chroma 向量库
4. 用户在问答页面输入问题
5. 系统先从向量库检索相关片段，再将检索结果和历史消息一起传给大模型
6. 大模型基于知识库上下文生成回答

## 主要技术栈

- Python
- LangChain
- Streamlit
- Chroma
- DashScope Embeddings
- Tongyi / Qwen Chat Model

## 运行前准备

### 1. 安装依赖

建议使用 Python 3.9 及以上版本。

```bash
pip install streamlit langchain langchain-community langchain-core langchain-chroma langchain-text-splitters dashscope
```

### 2. 配置 API Key

本项目使用阿里云百炼相关模型，运行前请先配置环境变量：

```bash
set DASHSCOPE_API_KEY=你的API_KEY
```

如果你使用 PowerShell，也可以执行：

```powershell
$env:DASHSCOPE_API_KEY="你的API_KEY"
```

## 启动方式

### 启动知识库上传页面

```bash
streamlit run app_file_uploader.py
```

### 启动智能问答页面

```bash
streamlit run app_qa.py
```

## 配置说明

项目默认配置位于 `config_data.py`，包括：

- 向量库目录 `persist_directory`
- 集合名称 `collection_name`
- 文本切分参数 `chunk_size`、`chunk_overlap`
- 检索数量 `similarity_threshold`
- Embedding 模型名称 `embedding_model_name`
- 对话模型名称 `chat_model_name`

## 示例数据

`data/` 目录中提供了示例知识文件：

- 尺码推荐
- 洗涤养护
- 颜色选择

可以先通过上传页面将这些文本导入知识库，再到问答页面进行测试。

## 项目亮点

- 代码结构清晰，适合理解 RAG 基础实现
- 覆盖“知识入库 + 检索问答 + 历史会话”完整链路
- 适合用作课程作业、项目演示或简历项目

## 后续可扩展方向

- 支持 PDF、Word 等更多文件格式
- 增加更细粒度的召回与重排策略
- 增加后台管理与知识库删除能力
- 接入更完整的用户系统与多会话管理

