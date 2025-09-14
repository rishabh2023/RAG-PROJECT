
from __future__ import annotations
from typing import List, Dict, Optional
import os

from app.search.hybrid_search import HybridSearch
from app.search.semantic_adapter import semantic_search
from app.config import settings

# In-memory registry for chunks so hybrid can return text+metadata
_CHUNKS: List[Dict] = []
_HYBRID: Optional[HybridSearch] = None

def register_chunks(chunks: List[Dict]):
    """
    Call this after ingest to provide chunk texts and metadata to hybrid.
    Expected fields per chunk: {"id": str, "text": str, "metadata": {...}}
    """
    global _CHUNKS, _HYBRID
    _CHUNKS = chunks
    _HYBRID = HybridSearch(semantic_fn=lambda q, top_k=20: semantic_search(q, top_k=top_k))
    # Persist lexical index alongside your other data
    persist_path = os.path.join(os.path.dirname(__file__), "..", "data", "bm25_lex.pkl")
    persist_path = os.path.abspath(persist_path)
    _HYBRID.register_chunks(chunks, persist_lex_to=persist_path)

def retrieve(query: str, top_k: int = 6, mode: str = "hybrid") -> List[Dict]:
    """
    mode: "semantic" | "lexical" | "hybrid"
    Returns list of chunks with text+metadata for the answer chain.
    """
    if not query:
        return []
    if _HYBRID is None:
        # Fall back to semantic only (no lexical index yet)
        sem = semantic_search(query, top_k=top_k)
        # Minimal mapping to dicts
        id_to_chunk = {c["id"]: c for c in _CHUNKS}
        return [{
            "id": cid,
            "score": s,
            "text": id_to_chunk.get(cid, {}).get("text", ""),
            "metadata": id_to_chunk.get(cid, {}).get("metadata", {})
        } for cid, s in sem]
    return _HYBRID.search(query, top_k=top_k, mode=mode)
