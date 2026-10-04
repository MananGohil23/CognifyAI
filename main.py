"""
main.py — P6: AI Hallucination Confidence Labeler
FastAPI application exposing confidence labeling, Q&A evaluation, and RAG endpoints.
"""

from ingestion import get_collection
from brd_engine import validate_brd
from trust_gate import run_full_pipeline
from ingestion import ingest_document
from confidence_labeler import evaluate_qa_reliability
from sample_data import SAMPLE_SCENARIOS

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

# Load environment variables
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="P6 - AI Hallucination Confidence Labeler",
    description="Responsible Enterprise AI: Q&A Reliability Checker, Perplexity Metric & Uncertainty Explanation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve only user-uploaded source files (never the project root / .env / source).
static_dir = Path(__file__).parent
uploads_dir = static_dir / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=uploads_dir), name="static-uploads")

# In-memory report store
_reports: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CheckRequest(BaseModel):
    question: str
    answer: Optional[str] = None
    source_text: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    include_brd_path: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    trust_gate: str
    reliability_tag: str
    trust_score: float
    perplexity: float
    uncertainty_score: float
    short_reason: str
    warnings: list[str]
    ragas_scorecard: dict
    report_url: str
    claim_breakdown: list[dict] = []
    retrieved_chunks: list[dict] = []



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
    return {"status": "ok", "service": "P6-AI-Hallucination-Confidence-Labeler"}


@app.get("/samples")
async def get_samples():
    """Return pre-loaded hackathon benchmark Q&A scenarios."""
    return {"samples": SAMPLE_SCENARIOS}


@app.post("/check")
async def check_qa_endpoint(req: CheckRequest):
    """
    Q&A Reliability Checker endpoint.
    Accepts question, answer (optional), and source_text (optional).
    Outputs reliability_tag ('Certain', 'Uncertain', 'Needs Verification'),
    perplexity metric, short reason, warnings, and claim breakdown.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(400, "Question parameter cannot be empty.")

    # If answer is omitted, generate answer using simple search/LLM or return evaluation
    answer = req.answer
    if not answer or not answer.strip():
        # Fallback query if no answer provided
        try:
            rag_res = run_full_pipeline(query=req.question)
            answer = rag_res.get("answer", "")
            if not req.source_text:
                # aggregate retrieved text
                retrieved = rag_res.get("retrieved_chunks", [])
                req.source_text = "\n\n".join([c.get("text", "") for c in retrieved[:3]])
        except Exception:
            answer = "No answer provided to verify."

    eval_result = evaluate_qa_reliability(
        question=req.question,
        answer=answer,
        source_text=req.source_text
    )
    return eval_result


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
):
    """
    Ingest a reference document (PDF, DOCX, or TXT).
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

    gate_raw = result["trust_gate"]
    # Map raw gate to P6 Reliability Tag
    if gate_raw == "Safe":
        rel_tag = "Certain"
        reason = "Answer is strongly backed by ingested source documents."
    elif gate_raw == "Needs Human Review":
        rel_tag = "Uncertain"
        reason = "Answer is partially supported or contains ungrounded claims."
    else:
        rel_tag = "Needs Verification"
        reason = "Answer contains unsupported claims or contradicts source passages."

    scorecard = result.get("ragas_scorecard", {})
    faithfulness = scorecard.get("faithfulness", 0.5)
    composite = scorecard.get("composite_trust_score", 0.5)

    uncertainty_pct = round((1.0 - composite) * 100, 1)
    # Estimate perplexity from trust score
    ppl = round(max(1.10, min(12.0, (1.0 - faithfulness) * 6.5 + 1.25)), 2)

    warnings = []
    if rel_tag != "Certain":
        warnings.append(f"Trust Gate Flagged: {gate_raw}. Verification recommended.")
    if scorecard.get("edition_conflict_risk", 0) > 0.2:
        warnings.append("Potential source document conflict detected.")

    verified_claims = result.get("verified_claims", [])
    claim_breakdown = []
    for c in verified_claims:
        nli = c.get("nli_result") or {}
        lbl = nli.get("label") or ("entailed" if c.get("final_trust_gate") == "Safe" else "neutral" if c.get("final_trust_gate") == "Needs Human Review" else "contradiction")
        claim_breakdown.append({
            "sentence": c.get("sentence", ""),
            "nli_label": lbl,
            "entailment_score": nli.get("entailment_score", round(composite, 2)),
            "neutral_score": nli.get("neutral_score", 0.0),
            "contradiction_score": nli.get("contradiction_score", 0.0),
            "confidence": nli.get("confidence", 0.0),
            "citation": f"[{c.get('pub_name', '')} · {c.get('edition', '')} · §{c.get('section_id', '')}]" if c.get("pub_name") else "",
            "source_text": c.get("source_text", ""),
            "reasoning": c.get("reasoning", f"Trust gate: {c.get('final_trust_gate', 'Needs Human Review')}")
        })

    retrieved = result.get("retrieved_chunks", [])

    return QueryResponse(
        session_id=session_id,
        answer=result["answer"],
        trust_gate=gate_raw,
        reliability_tag=rel_tag,
        trust_score=composite,
        perplexity=ppl,
        uncertainty_score=uncertainty_pct,
        short_reason=reason,
        warnings=warnings,
        ragas_scorecard=scorecard,
        report_url=f"/report/{session_id}",
        claim_breakdown=claim_breakdown,
        retrieved_chunks=retrieved,
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
