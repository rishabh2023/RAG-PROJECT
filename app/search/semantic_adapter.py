
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import os

from pinecone import Pinecone
from app.config import settings

# This adapter expects your vectors (embeddings) to be upserted with ids that map to chunk ids.
# It performs a vector query (using text-embedding model you used at ingest time).

# NOTE: We assume the embedder used at ingest time is the same you'll use for queries,
# and that you have a "text-embedding" function available in your stack (SentenceTransformer etc.).
# If you already have a function elsewhere, import and use that.

try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast
except Exception:
    _EMBEDDER = None

def _embed(text: str):
    if _EMBEDDER is None:
        raise RuntimeError("SentenceTransformer not available. Please install or wire your embedder.")
    return _EMBEDDER.encode([text])[0].tolist()

def make_client() -> Optional[Pinecone]:
    if not settings.PINECONE_API_KEY:
        return None
    return Pinecone(api_key=settings.PINECONE_API_KEY)

def semantic_search(
    query: str,
    top_k: int = 20,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Returns list of (chunk_id, score) ranked desc by score.
    """
    pc = make_client()
    if pc is None:
        return []
    index_name = index_name or getattr(settings, "PINECONE_INDEX_NAME", None)
    namespace = namespace or getattr(settings, "PINECONE_NAMESPACE", None)
    if not index_name:
        return []

    index = pc.Index(index_name)
    emb = _embed(query)
    res = index.query(
        vector=emb,
        top_k=top_k,
        namespace=namespace,
        include_values=False,
        include_metadata=True,
    )
    # Pinecone v3 returns matches as objects with id and score
    out: List[Tuple[str, float]] = []
    for m in res.matches or []:
        cid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
        score = float(getattr(m, "score", 0.0) or (m.get("score") if isinstance(m, dict) else 0.0))
        if cid:
            out.append((cid, score))
    # already in rank order
    return out
