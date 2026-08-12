from fastapi import APIRouter,Request,HTTPException
from pydantic import BaseModel, Field
from app.services.rag import RAGService
router=APIRouter(prefix="/api/v1/rag")
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    conversation_id: str = Field(default="default", min_length=1, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=20)
@router.post("/query")
def query(body:QueryRequest,request:Request):
    if not body.query.strip(): raise HTTPException(422,"query must not be empty")
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(503, "RAG service is not ready")
    try:
        return service.query(body.query, body.conversation_id, body.top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
