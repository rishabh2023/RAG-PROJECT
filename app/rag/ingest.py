
import os
from typing import List, Dict
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

def load_pdfs(pdf_dir: str) -> List[Dict]:
    """Load PDFs and return list of dicts with text per page."""
    docs = []
    for path in Path(pdf_dir).glob("*.pdf"):
        try:
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                bank = infer_bank_from_name(path.name)
                docs.append({
                    "content": text,
                    "metadata": {
                        "source": str(path),
                        "page": i + 1,
                        "bank": bank,
                        "title": path.stem,
                    }
                })
        except Exception as e:
            print(f"[ingest] Failed to read {path}: {e}")
    return docs

def infer_bank_from_name(filename: str) -> str:
    lower = filename.lower()
    if "axis" in lower: return "Axis Bank"
    if "sbi" in lower or "state bank" in lower: return "State Bank of India"
    if "hdfc" in lower: return "HDFC Bank"
    if "icici" in lower: return "ICICI Bank"
    if "kotak" in lower: return "Kotak Mahindra Bank"
    return "Unknown"

def chunk_docs(docs: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for d in docs:
        for chunk in splitter.split_text(d["content"]):
            if not chunk.strip():
                continue
            meta = dict(d["metadata"])
            meta["length"] = len(chunk)
            chunks.append({"content": chunk, "metadata": meta})
    return chunks

def ensure_index(pc: Pinecone, index_name: str, dim: int):
    indexes = [i["name"] for i in pc.list_indexes()]
    if index_name not in indexes:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def upsert_chunks(chunks: List[Dict]):
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set in environment.")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME or "loan-docs"
    model_name = settings.SBERT_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
    namespace = settings.PINECONE_NAMESPACE or "default"

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    ensure_index(pc, index_name, dim)
    index = pc.Index(index_name)

    # batch upsert
    vectors = []
    for i, ch in enumerate(chunks):
        emb = model.encode(ch["content"]).tolist()
        vid = f"{ch['metadata'].get('source','doc')}#p{ch['metadata'].get('page',0)}#{i}"
        vectors.append({
            "id": vid,
            "values": emb,
            "metadata": ch["metadata"] | {"text": ch["content"]},
        })
        if len(vectors) >= 100:
            index.upsert(vectors=vectors, namespace=namespace)
            vectors = []
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)

def ingest_directory(pdf_dir: str):
    docs = load_pdfs(pdf_dir)
    chunks = chunk_docs(docs)
    upsert_chunks(chunks)
    return {"pages": len(docs), "chunks": len(chunks)}
