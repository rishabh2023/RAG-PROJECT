# app/core/rag_engine.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import pickle
import re

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import google.generativeai as genai

from app.config import settings


@dataclass
class SearchHit:
    text: str
    score: float
    bank: str
    source: str
    page: int


# ---------------------------- Lexical (BM25) ----------------------------

def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())

class BM25LexicalIndex:
    """
    Loads a persisted lexical index built at ingest time.
    Expected pickle schema:
      {"chunks": [{"id": str, "text": str, "metadata": {...}}, ...],
       "doc_tokens": List[List[str]]}
    """
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self.id_to_idx: Dict[str, int] = {}

    def available(self) -> bool:
        return os.path.exists(self.persist_path)

    def load(self) -> None:
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        doc_tokens = data["doc_tokens"]
        self.bm25 = BM25Okapi(doc_tokens)
        self.id_to_idx = {c["id"]: i for i, c in enumerate(self.chunks)}

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_simple_tokenize(query))
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i]["id"], float(scores[i])) for i in idxs]

    def chunk_by_id(self, cid: str) -> Optional[Dict]:
        i = self.id_to_idx.get(cid)
        return self.chunks[i] if i is not None else None


# ----------------------- Reciprocal Rank Fusion -------------------------

def reciprocal_rank_fusion(
    lists: List[List[Tuple[str, float]]],
    kappa: int = 60,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    ranks: List[Dict[str, int]] = []
    for L in lists:
        ranks.append({cid: r for r, (cid, _) in enumerate(L, start=1)})

    all_ids = set()
    for L in lists:
        all_ids.update([cid for cid, _ in L])

    fused: Dict[str, float] = {}
    for cid in all_ids:
        score = 0.0
        for rm in ranks:
            if cid in rm:
                score += 1.0 / (kappa + rm[cid])
        fused[cid] = score

    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


# ---------------------------- RAG Engine --------------------------------

class PineconeRAGEngine:
    def __init__(self):
        # Pinecone / Embeddings
        if not settings.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = getattr(settings, "PINECONE_INDEX_NAME", "loan-docs")
        self.namespace = getattr(settings, "PINECONE_NAMESPACE", "default")
        self.model_name = getattr(settings, "SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(self.model_name)
        self.index = self.pc.Index(self.index_name)

        # Gemini LLM
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            settings.GEMINI_MODEL or "gemini-1.5-flash",
            generation_config={"temperature": 0.4, "top_p": 0.9, "max_output_tokens": 600},
        )

        # Optional BM25 (Hybrid)
        # This file is written by ingest (register_chunks) if you followed earlier step.
        bm25_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bm25_lex.pkl"))
        self.lex: Optional[BM25LexicalIndex] = BM25LexicalIndex(bm25_path)
        if self.lex.available():
            try:
                self.lex.load()
            except Exception:
                # If loading fails, continue with semantic-only
                self.lex = None

    # --- Embeddings ---
    def embed(self, text: str):
        return self.embedder.encode(text).tolist()

    # --- Retrieval (Hybrid when possible) ---
    def _retrieve_semantic(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        q = self.embed(query)
        res = self.index.query(
            namespace=self.namespace,
            vector=q,
            top_k=max(top_k, 20),   # slightly larger pool for fusion
            include_metadata=True
        )
        out: List[Tuple[str, float, Dict]] = []
        for m in res.matches or []:
            meta = m.metadata or {}
            out.append((m.id, float(m.score or 0.0), meta))
        return out

    def _retrieve_lexical(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        if not self.lex:
            return []
        hits = self.lex.search(query, top_k=max(top_k, 50))
        # Map ids to metadata via chunk store
        out: List[Tuple[str, float, Dict]] = []
        for cid, s in hits:
            ch = self.lex.chunk_by_id(cid) or {}
            meta = ch.get("metadata", {})
            # Ensure "text" in meta for synthesis (if persisted that way)
            if "text" not in meta:
                meta["text"] = ch.get("text", "")
            out.append((cid, s, meta))
        return out

    def retrieve(self, query: str, top_k: int = 6) -> List[SearchHit]:
        # If we have BM25 index, do Hybrid (RRF); else semantic-only
        sem = self._retrieve_semantic(query, top_k)
        if self.lex:
            lex = self._retrieve_lexical(query, top_k)
            # Prepare ranked lists of (id, score) for fusion
            sem_rank = [(cid, s) for (cid, s, _) in sem]
            lex_rank = [(cid, s) for (cid, s, _) in lex]
            fused = reciprocal_rank_fusion([sem_rank, lex_rank], kappa=60, top_k=top_k)

            # Build final hits using metadata from whichever side we have
            meta_by_id: Dict[str, Dict] = {}
            for cid, _, meta in sem:
                meta_by_id[cid] = meta
            for cid, _, meta in lex:
                meta_by_id.setdefault(cid, meta)

            hits: List[SearchHit] = []
            for cid, fscore in fused:
                meta = meta_by_id.get(cid, {})
                hits.append(SearchHit(
                    text=meta.get("text", ""),
                    score=float(fscore),
                    bank=meta.get("bank", "Unknown"),
                    source=meta.get("source", ""),
                    page=int(meta.get("page", 0)),
                ))
            return hits

        # Semantic-only fallback (original behavior)
        hits: List[SearchHit] = []
        for cid, s, meta in sem[:top_k]:
            hits.append(SearchHit(
                text=meta.get("text", ""),
                score=float(s),
                bank=meta.get("bank", "Unknown"),
                source=meta.get("source", ""),
                page=int(meta.get("page", 0)),
            ))
        return hits

    # --- Synthesis (unchanged answer shape) ---
    def synthesize(self, question: str, hits: List[SearchHit]) -> Dict[str, Any]:
        # Build compact grounded context
        context_blocks = []
        for h in hits:
            if h.text.strip():
                context_blocks.append(f"[{h.bank} | p{h.page}]\n{h.text}")
            if len(context_blocks) >= 8:
                break

        prompt = f"""
You are a helpful, precise assistant. Answer only using the CONTEXT below.
If something is not stated in the context, say: "Not specified in the provided documents."
Do not invent banks, figures, policies, dates, or fees.

Formatting rules:
- No code fences, no JSON, no markdown backticks.
- Use short paragraphs and bullet points.
- If multiple banks appear, show bank-wise bullets like: "• Axis Bank: …"
- Write numbers with units (e.g., 9.65% p.a., ₹10,000 + GST).
- Keep it concise and clear for a layperson; add a one-line summary at the end.

QUESTION:
{question}

CONTEXT:
{chr(10).join(context_blocks)}
""".strip()

        resp = self.model.generate_content(prompt)
        text = (resp.text or "").strip() if resp else ""

        if not text:
            text = "I couldn’t find this in the provided documents."

        return {
            "answer": text,
            "banks": [{"name": h.bank, "confidence": h.score} for h in hits],
            "citations": [{"bank": h.bank, "page": h.page, "score": h.score, "source": h.source} for h in hits],
        }

    def ask(self, question: str, top_k: int = 6) -> Dict[str, Any]:
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return {"answer": "I couldn’t find this in the provided documents.", "banks": [], "citations": []}
        out = self.synthesize(question, hits)
        if "citations" not in out:
            out["citations"] = [{"bank": h.bank, "page": h.page, "score": h.score, "source": h.source} for h in hits]
        return out


# Singleton
rag_engine = PineconeRAGEngine()
