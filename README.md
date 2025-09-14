# 📘 Loan Support AI — RAG with Hybrid Search (Semantic + BM25)

An **AI-powered customer support backend for Home Loans**.
Built with **FastAPI**, **Pinecone**, **SentenceTransformers**, **BM25 (rank-bm25)**, and **Google Gemini**.

The system answers **loan-related queries** using a **Retrieval-Augmented Generation (RAG)** pipeline, optimized with **Hybrid Search** (Semantic + Lexical fusion).

---

## 🚀 Features

- **FastAPI** backend with clean API endpoints
- **Ingest pipeline**: PDF → Chunking → Embeddings → Pinecone + BM25 index
- **Hybrid Search**: Combines Semantic (dense) and Lexical (BM25) search
- **Reciprocal Rank Fusion (RRF)**: Balances semantic similarity and keyword precision
- **LLM integration** with **Google Gemini** for contextual, grounded answers
- **Bank-specific citations** with page references
- **Configurable via environment variables**

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **Vector DB**: Pinecone
- **Embeddings**: [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2` by default)
- **Lexical Index**: [rank-bm25](https://pypi.org/project/rank-bm25/) for BM25 scoring
- **LLM**: Google Gemini (1.5 Flash by default)
- **Deployment**: Uvicorn / Gunicorn

---

## 📂 Project Structure

```
app/
├── api/
│   └── endpoints/
│       ├── chat.py          # /chat/ask endpoint
│       └── ingest.py        # /ingest endpoint
├── core/
│   └── rag_engine.py        # Main RAG engine (Hybrid retrieval + LLM synthesis)
├── search/
│   ├── hybrid_search.py     # Hybrid search (BM25 + Semantic + RRF)
│   └── semantic_adapter.py  # Pinecone semantic search adapter
├── services/
│   └── retrieval.py         # Retrieval service (register_chunks + retrieve)
├── data/
│   └── bm25_lex.pkl         # BM25 lexical index (created at ingest time)
├── config.py                # Settings (API keys, model names, index names)
└── main.py                  # FastAPI app entrypoint
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd LoanSupportAI
```

### 2. Create virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables (`.env`)

Create a `.env` file with:

```env
# Pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=loan-docs
PINECONE_NAMESPACE=default

# Embeddings
SBERT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-1.5-flash

# API
PROJECT_NAME=Loan Support AI
API_PREFIX=/api/v1
USE_LOCAL_FAISS=false
```

---

## ▶️ Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

API will be available at:
🔗 [http://localhost:5000/docs](http://localhost:5000/docs)

---

## 📥 Ingest Documents

Upload PDFs into `app/data/documents/` or specify a path.
Then call:

```bash
curl -X POST "http://localhost:5000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{"path":"app/data/documents"}'
```

This will:

1. Load PDFs → Chunk text
2. Embed chunks → Pinecone
3. Build BM25 index → `app/data/bm25_lex.pkl`

---

## 💬 Ask Questions

```bash
curl -X POST "http://localhost:5000/api/v1/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the home loan eligibility for Axis Bank?", "top_k":6}'
```

Response:

```json
{
  "answer": "• Axis Bank: Eligibility requires a minimum monthly income of ₹25,000... \n\nSummary: Most banks require stable income and good credit history.",
  "banks": [{ "name": "Axis Bank", "confidence": 0.92 }],
  "citations": [
    {
      "bank": "Axis Bank",
      "page": 12,
      "score": 0.92,
      "source": "axis-loan.pdf"
    }
  ]
}
```

---

## 🔍 Retrieval Modes

- Default = **Hybrid** (Semantic + BM25 + RRF)
- If `bm25_lex.pkl` is missing → falls back to **Semantic-only**
- You can force modes by editing `rag_engine.ask(mode="semantic"|"lexical"|"hybrid")`

---

## 🧰 Troubleshooting

### ❌ `ImportError: cannot import name 'Pinecone'`

- Uninstall old client: `pip uninstall -y pinecone-client`
- Install new SDK: `pip install pinecone>=3.0.0`

### ❌ No `bm25_lex.pkl`

- Run `/ingest` again to rebuild lexical index.

### ❌ LLM not returning text

- Check `GEMINI_API_KEY` is set and valid.
- If Gemini fails, engine returns fallback: "I couldn’t find this in the provided documents."

---

## 📌 Next Steps

- Add **streaming responses** for real-time answers
- Deploy on **AWS/GCP** with auto-scaling
- Add **Auth (JWT / OAuth)** for production
- Integrate **evaluation harness** (precision\@k, recall)

---

## 👨‍💻 Author

**Loan Support AI** — Built with ❤️ using FastAPI, Pinecone, and Gemini.
