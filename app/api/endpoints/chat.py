# app/api/endpoints/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.rag_engine import rag_engine

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5

@router.post("/ask")
def ask_bank_bot(body: ChatQuery):
    try:
        result = rag_engine.ask(body.query, top_k=body.top_k or 5)
        return {"answer": result.get("answer", "I couldn’t find this in the provided documents.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
