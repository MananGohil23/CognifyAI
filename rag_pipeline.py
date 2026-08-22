"""
rag_pipeline.py — XAI Governance Framework
Hybrid RAG: dense embeddings + BM25 sparse retrieval → rerank → LLM generation
Every generated claim is attributed to: publication · edition · section_id · paragraph.
"""

import logging
import os
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

from ingestion import get_collection, get_embedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_DENSE = 20   # initial dense retrieval pool
TOP_K_SPARSE = 20  # BM25 candidates
TOP_K_RERANK = 8   # after reranking

SYSTEM_PROMPT = """You are a compliance analyst AI for Indian banking regulation.
You ONLY answer using the provided source passages from RBI publications.
You MUST cite every claim with [PUB · EDITION · SECTION] immediately after the claim.
Do NOT produce meta commentary about the pipeline, confidence, or system behavior.
Do NOT use hedging/disclaimer phrases like "I cannot find", "insufficient information", or "cannot determine"
if any relevant facts are present in sources. Prefer extractive factual summaries.
If evidence exists, provide the latest available numeric/qualitative indicators directly from sources.
Only say evidence is unavailable when none of the sources contain relevant facts.
Never fabricate statistics, dates, or regulatory language."""


def _is_abstaining_answer(text: str) -> bool:
    t = (text or "").lower()
    abstain_markers = [
        "cannot find",
        "insufficient information",
        "cannot determine",
        "not explicitly mention",
        "provided sources do not",
        "unable to",
    ]
    return any(m in t for m in abstain_markers)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    logger.info("Loading reranker model %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL, max_length=512)


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def retrieve_dense(query: str, embedder: SentenceTransformer,
                   collection, filters: dict | None = None,
                   top_k: int = TOP_K_DENSE) -> list[dict]:
    """
    Cosine similarity search in ChromaDB.
    Optional filters: {"pub_name": "FSR"} etc.
    Returns list of {text, metadata, score}.
    """
    qvec = embedder.encode([query]).tolist()

    where = None
    if filters:
        where = {k: {"$eq": v} for k, v in filters.items()}
        if len(where) > 1:
            where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    results = collection.query(
        query_embeddings=qvec,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])
    metadatas = (results.get("metadatas") or [[]])
    distances = (results.get("distances") or [[]])
    if not documents or not documents[0]:
        return []

    hits = []
    for doc, meta, dist in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        hits.append({
            "text": doc,
            "metadata": meta,
            "dense_score": float(1 - dist),  # cosine similarity
        })
    return hits


# ---------------------------------------------------------------------------
# Sparse (BM25) retrieval
# ---------------------------------------------------------------------------

def retrieve_sparse(query: str, all_docs: list[dict], top_k: int = TOP_K_SPARSE) -> list[dict]:
    """
    BM25 over a candidate pool (typically all ingested docs or a pre-filtered subset).
    Returns top_k hits sorted by BM25 score.
    """
    if not all_docs:
        return []

    tokenised_corpus = [doc["text"].lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(query.lower().split())

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**all_docs[i], "sparse_score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]


# ---------------------------------------------------------------------------
# Hybrid fusion (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(dense_hits: list[dict], sparse_hits: list[dict],
                           k: int = 60) -> list[dict]:
    """
    Merge dense and sparse ranked lists using RRF.
    Returns unified list sorted by fused score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        cid = hit["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = hit

    for rank, hit in enumerate(sparse_hits):
        cid = hit["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in doc_map:
            doc_map[cid] = hit

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**doc_map[cid], "rrf_score": score}
        for cid, score in fused
    ]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank(query: str, hits: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """Cross-encoder reranking over fused candidates."""
    if not hits:
        return []
    reranker = get_reranker()
    pairs = [(query, h["text"]) for h in hits]
    ce_scores = reranker.predict(pairs)
    for hit, score in zip(hits, ce_scores):
        hit["rerank_score"] = float(score)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# LLM generation with per-claim attribution
# ---------------------------------------------------------------------------

def generate_with_attribution(query: str, context_hits: list[dict]) -> dict:
    """
    Call LLM with retrieved context.
    Instructs model to cite every claim as [PUB · EDITION · SECTION].
    Returns {answer, claims: [{text, citation}]}.
    """
    context_blocks = []
    for i, hit in enumerate(context_hits):
        m = hit["metadata"]
        citation = f"[{m['pub_name']} · {m['edition']} · §{m['section_id']}]"
        context_blocks.append(
            f"SOURCE {i+1} {citation}:\n{hit['text'][:800]}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    llm_model = os.getenv("GROQ_MODEL") or os.getenv(
        "GROK_MODEL", "groq/compound-mini")
    llm_base_url = os.getenv("GROQ_BASE_URL") or os.getenv(
        "GROK_BASE_URL", "https://api.groq.com/openai/v1")

    llm_api_key = (os.getenv("GROQ_API_KEY")
                   or os.getenv("GROK_API_KEY") or "").strip()
    llm_api_key = llm_api_key.strip('"').strip("'")
    if not llm_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it before querying.")

    client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    user_prompt = (
        f"QUERY: {query}\n\n"
        f"SOURCES:\n{context_str}\n\n"
        "Return only source-grounded factual claims. "
        "Each sentence MUST end with a citation in this exact format: [PUB · EDITION · §SECTION]. "
        "Prefer concise extractive statements with numbers/dates directly present in sources."
    )

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1500,
    )
    raw_answer = response.choices[0].message.content

    # Retry once with stricter extractive instructions if model abstains
    # despite relevant retrieved context.
    if context_hits and _is_abstaining_answer(raw_answer):
        retry_prompt = (
            f"QUERY: {query}\n\n"
            f"SOURCES:\n{context_str}\n\n"
            "Do NOT abstain. Extract concrete facts directly from sources. "
            "Use 3-6 short factual sentences. "
            "Every sentence MUST end with [PUB · EDITION · §SECTION]. "
            "Do not include any sentence without citation."
        )
        retry = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        raw_answer = retry.choices[0].message.content

    # Parse inline citations → structured claim list
    claims = _parse_claims(raw_answer, context_hits)

    return {
        "answer": raw_answer,
        "claims": claims,
        "model": llm_model,
        "sources_used": len(context_hits),
    }


def _parse_claims(answer: str, context_hits: list[dict]) -> list[dict]:
    """
    Extract individual claims and their citations from the answer.
    Returns list of {sentence, pub_name, edition, section_id, source_text, metadata}.
    """
    import re
    answer = (answer or "").replace("Â·", "·").replace("Â§", "§")
    citation_re = re.compile(
        r"(?P<sentence>[^\n\[]+?)\s*"
        r"\[(?P<pub>[A-Z]+)\s*·\s*(?P<edition>[^·]+)\s*·\s*§(?P<section>[^\]]+)\]\.?",
        re.MULTILINE,
    )
    claims = []
    for m in citation_re.finditer(answer):
        sentence = (m.group("sentence") or "").strip()
        pub = (m.group("pub") or "").strip()
        edition = (m.group("edition") or "").strip()
        section = (m.group("section") or "").strip()
        section = re.sub(r"^[^0-9A-Za-z]+", "", section)
        # Find the matching source chunk
        source_text = ""
        for hit in context_hits:
            meta = hit["metadata"]
            if (meta["pub_name"] == pub.strip() and
                    meta["section_id"] == section.strip()):
                source_text = hit["text"][:500]
                source_meta = meta
                break
        else:
            source_meta = {}
        claims.append({
            "sentence": sentence.strip(),
            "pub_name": pub.strip(),
            "edition": edition.strip(),
            "section_id": section.strip(),
            "source_text": source_text,
            "metadata": source_meta,
        })
    return claims


# ---------------------------------------------------------------------------
# Main retrieval + generation pipeline
# ---------------------------------------------------------------------------

def query_pipeline(
    query: str,
    filters: dict | None = None,
) -> dict:
    """
    Full hybrid RAG pipeline.
    Returns:
      {
        "answer": str,
        "claims": [{sentence, pub_name, edition, section_id, source_text}],
        "retrieved_chunks": [...],
        "retrieval_stats": {...}
      }
    """
    embedder = get_embedder()
    collection = get_collection()

    # 1. Dense retrieval
    dense_hits = retrieve_dense(query, embedder, collection, filters=filters)

    # 2. Sparse retrieval over the dense pool (avoids full corpus BM25)
    sparse_hits = retrieve_sparse(query, dense_hits)

    # 3. Hybrid fusion
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)

    # 4. Rerank
    reranked = rerank(query, fused)

    # 5. Generate
    generation = generate_with_attribution(query, reranked)

    return {
        **generation,
        "retrieved_chunks": [
            {
                "text": h["text"],
                "pub_name": h["metadata"]["pub_name"],
                "edition": h["metadata"]["edition"],
                "section_id": h["metadata"]["section_id"],
                "section_title": h["metadata"]["section_title"],
                "source_filename": h["metadata"].get("source_filename", ""),
                "source_url": h["metadata"].get("source_url", ""),
                "text_preview": h["text"][:200],
                "rerank_score": h.get("rerank_score"),
                "edition_conflict_flag": h["metadata"].get("edition_conflict_flag", False),
            }
            for h in reranked
        ],
        "retrieval_stats": {
            "dense_candidates": len(dense_hits),
            "sparse_candidates": len(sparse_hits),
            "after_fusion": len(fused),
            "after_rerank": len(reranked),
        },
    }
