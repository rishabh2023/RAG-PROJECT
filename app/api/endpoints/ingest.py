
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Optional


from app.rag.ingest import ingest_directory
from app.config import settings

router = APIRouter(prefix="/ingest", tags=["ingest"])

class IngestRequest(BaseModel):
    path: Optional[str] = None

@router.post("")
def ingest(req: IngestRequest= Body(default=IngestRequest())):
    try:
        pdf_dir = req.path or "app/data/documents"
        result = ingest_directory(pdf_dir)
        return {"status": "ok", "ingested": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
