"""
ingestion.py — XAI Governance Framework
Handles ingestion of RBI half-yearly publications (FSR, MPR, PSR, FER).

Key responsibilities:
  - Parse PDF / DOCX / plain-text files
  - Chunk at section level, preserving pub name + edition date + section ID
  - Detect and tag edition conflicts (e.g. FSR June vs FSR December)
  - Upsert chunks into ChromaDB with full metadata
"""

import os
import re
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY__ANONYMIZED_TELEMETRY"] = "False"

import pdfplumber
import docx as python_docx
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_PUBLICATIONS = {"FSR", "MPR", "PSR", "FER"}

PUBLICATION_PATTERNS = {
    "FSR": r"financial stability report",
    "MPR": r"monetary policy report",
    "PSR": r"payment system report",
    "FER": r"report on foreign exchange reserves",
}

SECTION_HEADING_RE = re.compile(
    r"^(?:chapter|section|part|annex)?\s*(\d+(?:\.\d+)*)[.\s]+(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "rbi_publications"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking defaults tuned for policy-heavy RBI prose.
SECTION_SUBCHUNK_WORDS = 240
SECTION_SUBCHUNK_OVERLAP = 40
FALLBACK_WINDOW_WORDS = 260
FALLBACK_OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 40


# ---------------------------------------------------------------------------
# Singleton resources (loaded once)
# ---------------------------------------------------------------------------

_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model %s", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_pdf(path: Path) -> str:
    """Extract raw text from a PDF preserving page breaks."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_docx(path: Path) -> str:
    doc = python_docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_pdf(path)
    elif suffix in (".docx", ".doc"):
        return extract_text_docx(path)
    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Publication metadata detection
# ---------------------------------------------------------------------------

def detect_publication(text: str, filename: str) -> str:
    """
    Determine document/publication type or name.
    1. Checks if known publication acronyms (FSR, MPR, PSR, FER) or patterns match.
    2. Otherwise, derives a clean, readable identifier from the filename stem or document heading.
    Never rejects or raises an error — fully accepts any PDF, DOCX, or text file.
    """
    fname = Path(filename).stem
    fname_upper = fname.upper()

    # Check for known standard publications if present
    for pub in ALLOWED_PUBLICATIONS:
        if pub in fname_upper:
            return pub

    sample = text[:2000].lower() if text else ""
    for pub, pattern in PUBLICATION_PATTERNS.items():
        if re.search(pattern, sample):
            return pub

    # General document fallback: derive clean identifier from filename
    clean_name = re.sub(r"[^A-Za-z0-9_\-\s]", "", fname).strip()
    clean_name = re.sub(r"[\s\-]+", "_", clean_name).upper()
    if clean_name:
        return clean_name[:24]

    # Try extracting title from first non-empty line
    if text:
        first_line = text.strip().split("\n")[0].strip()
        if first_line:
            clean_title = re.sub(r"[^A-Za-z0-9_\-\s]", "", first_line).strip()
            clean_title = re.sub(r"[\s\-]+", "_", clean_title).upper()
            if clean_title:
                return clean_title[:24]

    return "DOC"


def detect_edition_date(text: str, filename: str) -> str:
    """
    Extract edition, date, or version (e.g. 'June 2023', '2024', 'v1.0').
    Falls back to current date if not explicitly specified.
    """
    patterns = [
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
        r"\b(20\d{2})\b",
        r"\bv(?:ersion)?\s*(\d+(?:\.\d+)*)\b",
    ]
    sample = filename + " " + (text[:3000] if text else "")
    for pat in patterns:
        m = re.search(pat, sample, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return datetime.now().strftime("%B %Y")


# ---------------------------------------------------------------------------
# Section-level chunking
# ---------------------------------------------------------------------------

def chunk_by_section(
    text: str,
    pub_name: str,
    edition: str,
    source_filename: str | None = None,
    source_url: str | None = None,
) -> list[dict]:
    """
    Split document into sections using heading detection, then subchunk long
    sections into paragraph/window-based chunks.

    Each chunk carries: text, section_id, section_title, pub_name, edition, chunk_id.

    For documents with no detectable headings, falls back to smaller sliding windows.
    """
    chunks = []
    lines = text.split("\n")
    current_section_id = "0"
    current_section_title = "Preamble"
    current_lines = []

    def flush(section_id, section_title, lines_buf):
        body = "\n".join(lines_buf).strip()
        if len(body.split()) < MIN_CHUNK_WORDS:  # skip near-empty sections
            return
        subchunks = _section_to_subchunks(
            body,
            pub_name=pub_name,
            edition=edition,
            section_id=section_id,
            section_title=section_title,
            max_words=SECTION_SUBCHUNK_WORDS,
            overlap=SECTION_SUBCHUNK_OVERLAP,
            source_filename=source_filename,
            source_url=source_url,
        )
        chunks.extend(subchunks)

    for line in lines:
        m = SECTION_HEADING_RE.match(line.strip())
        if m:
            flush(current_section_id, current_section_title, current_lines)
            current_section_id = m.group(1)
            current_section_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush(current_section_id, current_section_title, current_lines)

    # If no sections detected, fall back to sliding window
    if len(chunks) <= 1:
        logger.warning(
            "No section headings detected — using sliding window chunking")
        chunks = _sliding_window_chunks(
            text,
            pub_name,
            edition,
            source_filename=source_filename,
            source_url=source_url,
        )

    return chunks


def _sliding_window_chunks(text: str, pub_name: str, edition: str,
                           window: int = FALLBACK_WINDOW_WORDS,
                           overlap: int = FALLBACK_OVERLAP_WORDS,
                           source_filename: str | None = None,
                           source_url: str | None = None) -> list[dict]:
    """Fallback chunking when section headings are unavailable."""
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        window_words = words[i: i + window]
        body = " ".join(window_words)
        if len(window_words) < MIN_CHUNK_WORDS:
            break
        section_id = f"w{idx}"
        chunk_id = _make_chunk_id(pub_name, edition, section_id, body)
        chunks.append({
            "text": body,
            "section_id": section_id,
            "section_title": f"Window {idx}",
            "pub_name": pub_name,
            "edition": edition,
            "chunk_id": chunk_id,
            "edition_conflict_flag": False,
            "source_filename": source_filename or "",
            "source_url": source_url or "",
        })
        i += window - overlap
        idx += 1
    return chunks


def _section_to_subchunks(
    body: str,
    pub_name: str,
    edition: str,
    section_id: str,
    section_title: str,
    max_words: int,
    overlap: int,
    source_filename: str | None = None,
    source_url: str | None = None,
) -> list[dict]:
    """
    Split a section into paragraph-aware subchunks to improve retrieval precision
    while preserving section-level traceability.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    if not paragraphs:
        paragraphs = [body]

    chunks: list[dict] = []
    acc_words: list[str] = []
    sub_idx = 0

    def push_chunk(words: list[str]):
        nonlocal sub_idx
        if len(words) < MIN_CHUNK_WORDS:
            return

        text = " ".join(words).strip()
        sub_section_id = f"{section_id}.{sub_idx}" if section_id != "0" else f"0.{sub_idx}"
        chunk_id = _make_chunk_id(pub_name, edition, sub_section_id, text)
        chunks.append({
            "text": text,
            "section_id": sub_section_id,
            "section_title": section_title,
            "pub_name": pub_name,
            "edition": edition,
            "chunk_id": chunk_id,
            "edition_conflict_flag": False,
            "parent_section_id": section_id,
            "source_filename": source_filename or "",
            "source_url": source_url or "",
        })
        sub_idx += 1

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        # Hard-split unusually long paragraphs before appending.
        start = 0
        while start < len(para_words):
            piece = para_words[start:start + max_words]
            if acc_words and len(acc_words) + len(piece) > max_words:
                push_chunk(acc_words)
                acc_words = acc_words[-overlap:] if overlap > 0 else []

            acc_words.extend(piece)

            if len(acc_words) >= max_words:
                push_chunk(acc_words)
                acc_words = acc_words[-overlap:] if overlap > 0 else []

            start += max_words

    if acc_words:
        push_chunk(acc_words)

    # If everything was too small, keep one chunk instead of dropping content.
    if not chunks and body.split():
        base_words = body.split()
        text = " ".join(base_words)
        chunk_id = _make_chunk_id(pub_name, edition, section_id, text)
        chunks.append({
            "text": text,
            "section_id": section_id,
            "section_title": section_title,
            "pub_name": pub_name,
            "edition": edition,
            "chunk_id": chunk_id,
            "edition_conflict_flag": False,
            "parent_section_id": section_id,
            "source_filename": source_filename or "",
            "source_url": source_url or "",
        })

    return chunks


def _make_chunk_id(pub_name: str, edition: str, section_id: str, body: str) -> str:
    key = f"{pub_name}:{edition}:{section_id}:{body}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Edition conflict detection
# ---------------------------------------------------------------------------

def detect_edition_conflicts(new_chunks: list[dict], collection) -> list[dict]:
    """
    For each new chunk, check if a chunk with the same pub_name + section_id
    already exists from a different edition. If yes, set edition_conflict_flag=True
    on both and record the conflicting edition.

    Returns the new_chunks list (possibly mutated) and logs conflicts found.
    """
    conflicts_found = []
    for chunk in new_chunks:
        conflict_section_id = chunk.get(
            "parent_section_id") or chunk["section_id"]
        # Query existing chunks with same pub + section
        existing = collection.get(
            where={
                "$and": [
                    {"pub_name": {"$eq": chunk["pub_name"]}},
                    {
                        "$or": [
                            {"section_id": {"$eq": conflict_section_id}},
                            {"parent_section_id": {"$eq": conflict_section_id}},
                        ]
                    },
                ]
            },
            include=["metadatas"],
        )
        for meta in (existing.get("metadatas") or []):
            if meta.get("edition") != chunk["edition"]:
                chunk["edition_conflict_flag"] = True
                chunk["conflicts_with_edition"] = meta["edition"]
                conflicts_found.append({
                    "pub_name": chunk["pub_name"],
                    "section_id": conflict_section_id,
                    "editions": [meta["edition"], chunk["edition"]],
                })
                # Also tag the existing chunk
                existing_id = meta.get("chunk_id")
                if existing_id:
                    try:
                        collection.update(
                            ids=[existing_id],
                            metadatas=[{**meta, "edition_conflict_flag": True,
                                        "conflicts_with_edition": chunk["edition"]}],
                        )
                    except Exception:
                        pass

    if conflicts_found:
        logger.warning("Edition conflicts detected: %s", conflicts_found)
    return new_chunks


# ---------------------------------------------------------------------------
# Upsert to ChromaDB
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: list[dict], collection, embedder: SentenceTransformer):
    """Embed and upsert chunks into ChromaDB."""
    if not chunks:
        return

    # Deduplicate by chunk_id (keep first occurrence)
    seen_ids = set()
    unique_chunks = []
    for c in chunks:
        cid = c["chunk_id"]
        if cid not in seen_ids:
            unique_chunks.append(c)
            seen_ids.add(cid)

    if len(unique_chunks) < len(chunks):
        logger.warning(
            "Deduplicated %d duplicate chunks, keeping %d unique",
            len(chunks) - len(unique_chunks), len(unique_chunks)
        )

    texts = [c["text"] for c in unique_chunks]
    ids = [c["chunk_id"] for c in unique_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    metadatas = [
        {
            "pub_name": c["pub_name"],
            "edition": c["edition"],
            "section_id": c["section_id"],
            "parent_section_id": c.get("parent_section_id", c["section_id"]),
            "section_title": c["section_title"],
            "chunk_id": c["chunk_id"],
            "edition_conflict_flag": c.get("edition_conflict_flag", False),
            "conflicts_with_edition": c.get("conflicts_with_edition", ""),
            "source_filename": c.get("source_filename", ""),
            "source_url": c.get("source_url", ""),
        }
        for c in unique_chunks
    ]

    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=texts, metadatas=metadatas)
    logger.info("Upserted %d chunks from %s %s", len(unique_chunks),
                unique_chunks[0]["pub_name"], unique_chunks[0]["edition"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_document(
    file_path: str | Path,
    filename: str | None = None,
    source_filename: str | None = None,
    source_url: str | None = None,
) -> dict:
    """
    Full ingestion pipeline for a single RBI publication file.

    Returns a summary dict:
      {
        "pub_name": "FSR",
        "edition": "June 2023",
        "chunks_ingested": 42,
        "edition_conflicts": [...]
      }
    """
    path = Path(file_path)
    fname = filename or path.name

    logger.info("Starting ingestion: %s", fname)
    text = extract_text(path)
    pub_name = detect_publication(text, fname)
    edition = detect_edition_date(text, fname)
    chunks = chunk_by_section(
        text,
        pub_name,
        edition,
        source_filename=source_filename,
        source_url=source_url,
    )

    collection = get_collection()
    chunks = detect_edition_conflicts(chunks, collection)
    embedder = get_embedder()
    upsert_chunks(chunks, collection, embedder)

    conflicts = [
        c for c in chunks if c.get("edition_conflict_flag")
    ]

    return {
        "pub_name": pub_name,
        "edition": edition,
        "chunks_ingested": len(chunks),
        "edition_conflicts": [
            {"section_id": c["section_id"],
                "conflicts_with": c.get("conflicts_with_edition")}
            for c in conflicts
        ],
    }
