# Cognify : Compliance-Grade XAI Trust & Governance Framework

## Stack
- **Backend**: Python 3.11 + FastAPI
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Sparse retrieval**: rank_bm25
- **NLI / Hallucination detection**: cross-encoder/nli-deberta-v3-small (via sentence-transformers)
- **LLM**: Groq API (OpenAI-compatible SDK)
- **Vector store**: ChromaDB (local, persistent)
- **BRD parsing**: pdfplumber + python-docx
- **Frontend**: React + Tailwind (served separately or via FastAPI static)

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

If you want auto-reload while developing, use:

```bash
uvicorn main:app --reload --reload-exclude .venv --reload-exclude chroma_db --port 8000
```

## Environment Variables
```
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
CHROMA_PERSIST_DIR=./chroma_db
```

Backward compatibility: `GROK_API_KEY`, `GROK_MODEL`, and `GROK_BASE_URL`
are still accepted if `GROQ_*` variables are not set.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /ingest | Ingest a PDF/DOC/txt RBI publication |
| POST | /query  | Query with full XAI attribution |
| POST | /brd/validate | Upload & validate a BRD |
| GET  | /report/{session_id} | Retrieve audit report |
| GET  | /editions/conflicts | List detected edition conflicts |

## Architecture

```
Ingestion → Hybrid RAG → Hallucination Detection → Edition Conflict Check → Trust Gate → Audit Report
```
