"""
main.py — XAI Governance Framework
FastAPI application exposing all pipeline endpoints.
"""

from ingestion import get_collection
from brd_engine import validate_brd
from trust_gate import run_full_pipeline
from ingestion import ingest_document
import logging
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables before importing local modules
# that may read settings at import time.
# override=True prevents stale shell vars from shadowing .env values.
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PS1-XAI Governance Framework",
    description="Compliance-grade XAI Trust & Governance for RBI regulatory AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS, etc.) from the project root
static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
uploads_dir = static_dir / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

# In-memory report store (use Redis/DB in production)
_reports: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None  # e.g. {"pub_name": "FSR"}
    include_brd_path: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    trust_gate: str
    ragas_scorecard: dict
    report_url: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the main UI (index.html)."""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "XAI-Governance-Framework"}


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
):
    """
    Ingest an RBI publication (FSR / MPR / PSR / FER).
    Accepts PDF, DOCX, or TXT.
    """
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            400, f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}")

    safe_filename = Path(file.filename).name

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Save a persistent copy for clickable source links.
    source_path = uploads_dir / safe_filename
    source_path.write_bytes(content)
    source_url = f"/static/uploads/{quote(safe_filename)}"

    try:
        result = ingest_document(
            tmp_path,
            filename=file.filename,
            source_filename=safe_filename,
            source_url=source_url,
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "status": "ingested",
        **result,
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Query the RAG pipeline with full XAI attribution, NLI verification,
    trust gating, and audit report generation.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    try:
        result = run_full_pipeline(
            query=req.query,
            brd_file_path=req.include_brd_path,
        )
    except Exception as e:
        logger.exception("Pipeline error for query: %s", req.query)
        raise HTTPException(500, f"Pipeline error: {str(e)}") from e

    session_id = result["session_id"]
    _reports[session_id] = result["report"]

    return QueryResponse(
        session_id=session_id,
        answer=result["answer"],
        trust_gate=result["trust_gate"],
        ragas_scorecard=result["ragas_scorecard"],
        report_url=f"/report/{session_id}",
    )


@app.get("/report/{session_id}")
async def get_report(session_id: str):
    """Retrieve the full audit report for a session."""
    report = _reports.get(session_id)
    if not report:
        raise HTTPException(
            404, f"No report found for session '{session_id}'.")
    return JSONResponse(content=report)


@app.post("/brd/validate")
async def validate_brd_endpoint(file: UploadFile = File(...)):
    """
    Upload and validate a Business Requirement Document.
    Maps requirements to RBI sections and returns alignment scores.
    """
    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported BRD format '{suffix}'.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = validate_brd(tmp_path, filename=file.filename)
    except Exception as e:
        logger.exception("BRD validation error")
        raise HTTPException(500, f"BRD validation error: {str(e)}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.get("/editions/conflicts")
async def list_edition_conflicts():
    """
    List all detected edition conflicts in the ingested corpus.
    Returns sections where multiple editions exist with potentially contradictory content.
    """
    collection = get_collection()
    try:
        all_items = collection.get(
            where={"edition_conflict_flag": {"$eq": True}},
            include=["metadatas"],
        )
    except Exception as e:
        logger.warning("Could not query conflicts: %s", e)
        return {"conflicts": []}

    seen = set()
    conflicts = []
    for meta in (all_items.get("metadatas") or []):
        key = f"{meta.get('pub_name')}:{meta.get('section_id')}"
        if key in seen:
            continue
        seen.add(key)
        conflicts.append({
            "pub_name": meta.get("pub_name"),
            "section_id": meta.get("section_id"),
            "section_title": meta.get("section_title"),
            "edition": meta.get("edition"),
            "conflicts_with_edition": meta.get("conflicts_with_edition"),
        })

    return {
        "total_conflicts": len(conflicts),
        "conflicts": conflicts,
    }


@app.get("/corpus/stats")
async def corpus_stats():
    """
    Return statistics about the ingested corpus:
    total chunks, publications present, editions ingested.
    """
    collection = get_collection()
    all_items = collection.get(include=["metadatas"])
    metas = all_items.get("metadatas") or []

    pub_editions: dict[str, set] = {}
    for m in metas:
        pub = m.get("pub_name", "Unknown")
        edition = m.get("edition", "Unknown")
        pub_editions.setdefault(pub, set()).add(edition)

    return {
        "total_chunks": len(metas),
        "publications": {
            pub: {
                "editions": sorted(editions),
                "chunk_count": sum(1 for m in metas if m.get("pub_name") == pub),
            }
            for pub, editions in pub_editions.items()
        },
    }
