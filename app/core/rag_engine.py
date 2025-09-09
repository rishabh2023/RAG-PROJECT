# app/core/rag_engine.py
from typing import List, Dict, Any
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

from app.config import settings


@dataclass
class SearchHit:
    text: str
    score: float
    bank: str
    source: str
    page: int


class PineconeRAGEngine:
    def __init__(self):
        if not settings.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = getattr(settings, "PINECONE_INDEX_NAME", "loan-docs")
        self.namespace = getattr(settings, "PINECONE_NAMESPACE", "default")
        self.model_name = getattr(settings, "SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(self.model_name)

        # LLM config (Gemini)
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            settings.GEMINI_MODEL or "gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.9,
                "max_output_tokens": 700,
            },
        )

        self.index = self.pc.Index(self.index_name)

    def embed(self, text: str):
        return self.embedder.encode(text).tolist()

    def retrieve(self, query: str, top_k: int = 6) -> List[SearchHit]:
        q = self.embed(query)
        res = self.index.query(
            namespace=self.namespace,
            vector=q,
            top_k=top_k,
            include_metadata=True
        )
        hits: List[SearchHit] = []
        for m in res.matches or []:
            meta = m.metadata or {}
            hits.append(SearchHit(
                text=meta.get("text", ""),
                score=m.score or 0.0,
                bank=meta.get("bank", "Unknown"),
                source=meta.get("source", ""),
                page=int(meta.get("page", 0)),
            ))
        return hits

    def synthesize(self, question: str, hits: List[SearchHit]) -> Dict[str, Any]:
        # Build compact grounded context
        context_blocks = []
        for h in hits:
            if h.text.strip():
                context_blocks.append(f"[{h.bank} | p{h.page} | score={h.score:.3f}]\n{h.text}")
            if len(context_blocks) >= 8:
                break

        # General-purpose, grounded prompt (no JSON output)
        prompt = f"""
You are a helpful, precise assistant. Answer **only** using the CONTEXT below.
If something is not stated in the context, say: "Not specified in the provided documents."
Do not invent banks, figures, policies, dates, or fees.

Formatting rules:
- No code fences, no JSON, no markdown backticks.
- Use short paragraphs and bullet points.
- If multiple banks appear, show **bank-wise** bullets like: "• Axis Bank: …"
- Write numbers with units (e.g., 9.65% p.a., ₹10,000 + GST).
- Keep it concise and clear for a layperson; add a one-line summary at the end.
- Default to English unless the user’s question is in another language.

QUESTION:
{question}

CONTEXT:
{chr(10).join(context_blocks)}
"""

        resp = self.model.generate_content(prompt)
        text = (resp.text or "").strip()

        # Fallback if model returns nothing
        if not text:
            text = "I couldn’t find this in the provided documents."

        # Return a dict so upstream can still access citations if needed
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
