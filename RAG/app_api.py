from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from rag import RagService
from knowledge_base import KnowledgeBaseService
import uvicorn
import config_data as config

app = FastAPI(
    title="RAG Service API",
    description="提供知识库管理与问答服务的后端 API",
    version="1.0.0"
)

# 初始化服务
rag_service = RagService()
kb_service = KnowledgeBaseService()

class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default_user"

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG 问答接口
    """
    try:
        session_config = {
            "configurable": {
                "session_id": request.session_id,
            }
        }
        res = rag_service.chain.invoke({"input": request.prompt}, config=session_config)
        
        # 转换 sources 为字典列表
        sources_list = [
            {"content": doc.page_content, "metadata": doc.metadata} 
            for doc in res.get("sources", [])
        ]
        
        return ChatResponse(
            answer=res["answer"],
            sources=sources_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件到知识库
    """
    try:
        content = await file.read()
        result = kb_service.upload_by_file(content, file.filename)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
