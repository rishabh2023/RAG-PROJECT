
"""
Hybrid search: BM25 (lexical) + semantic (embedding) with Reciprocal Rank Fusion.
Drop-in module. Minimal assumptions about your existing stack.
- Lexical: rank_bm25 over chunk texts
- Semantic: delegate to your existing vector store search function
- Fusion: Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import os
import re
import pickle
from dataclasses import dataclass

# Lexical
from rank_bm25 import BM25Okapi

# Tokenization (very simple; replace with spaCy etc. if needed)
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict

class BM25LexicalIndex:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.doc_tokens: List[List[str]] = []
        self.chunks: List[Chunk] = []
        self.id_to_idx: Dict[str, int] = {}

    def build(self, chunks: List[Dict], persist_path: Optional[str] = None):
        """chunks: list of dicts with keys: id, text, metadata"""
        self.chunks = [Chunk(id=c["id"], text=c["text"], metadata=c.get("metadata", {})) for c in chunks]
        self.doc_tokens = [_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.doc_tokens)
        self.id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}
        if persist_path:
            with open(persist_path, "wb") as f:
                pickle.dump({"chunks": self.chunks, "doc_tokens": self.doc_tokens}, f)

    def load(self, persist_path: str):
        with open(persist_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.doc_tokens = data["doc_tokens"]
        self.bm25 = BM25Okapi(self.doc_tokens)
        self.id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Returns list of (chunk_id, score) sorted desc by score"""
        if not self.bm25:
            return []
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # top indices
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i].id, float(scores[i])) for i in idxs]

# RRF fusion
def reciprocal_rank_fusion(
    lists: List[List[Tuple[str, float]]],
    kappa: int = 60,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    lists: list of ranked lists; each list is [(id, score), ...] in rank order (best first).
    Returns fused list [(id, fused_score)] sorted by fused_score desc.
    """
    rank_maps: List[Dict[str, int]] = []
    for L in lists:
        rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(L, start=1)}
        rank_maps.append(rank_map)

    fused: Dict[str, float] = {}
    # Collect unique ids
    all_ids = set()
    for L in lists:
        all_ids.update([doc_id for doc_id, _ in L])

    for doc_id in all_ids:
        score = 0.0
        for rank_map in rank_maps:
            if doc_id in rank_map:
                r = rank_map[doc_id]
                score += 1.0 / (kappa + r)
        fused[doc_id] = score

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return ranked

class HybridSearch:
    """
    Plug in your existing semantic search function:
    - pass a callable semantic_fn(query: str, top_k: int) -> List[Tuple[str, float]]
      returning [(chunk_id, similarity_score), ...] in rank order
    - maintain a mapping from chunk_id -> (text, metadata) via `register_chunks`
    """
    def __init__(self, semantic_fn):
        self.semantic_fn = semantic_fn
        self.lex = BM25LexicalIndex()
        self.chunk_store: Dict[str, Chunk] = {}

    def register_chunks(self, chunks: List[Dict], persist_lex_to: Optional[str] = None):
        """
        chunks: [{"id": "chunk_123", "text": "...", "metadata": {...}}, ...]
        """
        self.lex.build(chunks, persist_path=persist_lex_to)
        self.chunk_store = {c["id"]: Chunk(id=c["id"], text=c["text"], metadata=c.get("metadata", {})) for c in chunks}

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid") -> List[Dict]:
        """
        mode: "semantic" | "lexical" | "hybrid"
        returns: list of {"id": id, "score": score, "text": text, "metadata": metadata}
        """
        if mode == "semantic":
            sem = self.semantic_fn(query, top_k=top_k)
            ids = [i for (i, s) in sem]
            return [{
                "id": cid,
                "score": s,
                "text": self.chunk_store.get(cid, Chunk(cid, "", {})).text,
                "metadata": self.chunk_store.get(cid, Chunk(cid, "", {})).metadata
            } for cid, s in sem]

        if mode == "lexical":
            lex = self.lex.search(query, top_k=top_k)
            return [{
                "id": cid,
                "score": s,
                "text": self.chunk_store.get(cid, Chunk(cid, "", {})).text,
                "metadata": self.chunk_store.get(cid, Chunk(cid, "", {})).metadata
            } for cid, s in lex]

        # hybrid
        sem = self.semantic_fn(query, top_k=max(top_k, 20))
        lex = self.lex.search(query, top_k=max(top_k, 50))  # get a slightly larger pool for fusion
        fused = reciprocal_rank_fusion([sem, lex], kappa=60, top_k=top_k)

        return [{
            "id": cid,
            "score": s,
            "text": self.chunk_store.get(cid, Chunk(cid, "", {})).text,
            "metadata": self.chunk_store.get(cid, Chunk(cid, "", {})).metadata
        } for cid, s in fused]
