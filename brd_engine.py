"""
brd_engine.py - XAI Governance Framework

Business Requirement Document (BRD) compliance engine:
  1. Ingest BRD (PDF / DOCX / txt)
  2. Extract individual requirements
  3. Map each requirement to the closest RBI publication section
  4. Score alignment with multi-signal evidence
  5. Flag gaps, violations, and compliance risks
  6. Return audit-ready compliance report
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hallucination_detector import check_entailment
from ingestion import extract_text, get_collection, get_embedder
from rag_pipeline import rerank, retrieve_dense

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------

REQ_PATTERNS = [
    # Numbered requirements: "1. The system shall..." / "R-001: ..."
    re.compile(
        r"(?:^|\n)\s*(?:R-?\d+|REQ-?\d+|\d+\.)\s*[:\.]?\s*(.+?)(?=\n|$)", re.MULTILINE),
    # SHALL / MUST / SHOULD requirements
    re.compile(
        r"([^.!?\n]*(?:shall|must|should|is required to|needs to)[^.!?\n]*[.!?])", re.IGNORECASE),
    # Bullet items
    re.compile(r"(?:^|\n)\s*[-•*]\s+(.+?)(?=\n|$)", re.MULTILINE),
]

MIN_REQ_LENGTH = 30  # characters - skip very short matches


@dataclass
class Requirement:
    req_id: str
    text: str
    source_line: int = 0


def extract_requirements(brd_text: str) -> list[Requirement]:
    """
    Extract individual requirement statements from BRD text.
    De-duplicates and assigns sequential IDs.
    """
    seen = set()
    reqs: list[Requirement] = []
    req_counter = 1

    for pattern in REQ_PATTERNS:
        for match in pattern.finditer(brd_text):
            text = match.group(1).strip(
            ) if match.lastindex else match.group(0).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) < MIN_REQ_LENGTH:
                continue
            key = text.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            reqs.append(
                Requirement(
                    req_id=f"REQ-{req_counter:03d}",
                    text=text,
                )
            )
            req_counter += 1

    logger.info("Extracted %d requirements from BRD", len(reqs))
    return reqs


# ---------------------------------------------------------------------------
# Section mapping
# ---------------------------------------------------------------------------


def map_requirement_to_sections(req: Requirement, top_k: int = 3) -> list[dict]:
    """
    For a given requirement, retrieve the top matching sections from the
    ingested RBI publication corpus using dense retrieval + reranking.
    Returns a list of matched chunks with scores.
    """
    embedder = get_embedder()
    collection = get_collection()
    dense_hits = retrieve_dense(req.text, embedder, collection, top_k=20)
    reranked = rerank(req.text, dense_hits, top_k=top_k)

    return [
        {
            "pub_name": hit["metadata"]["pub_name"],
            "edition": hit["metadata"]["edition"],
            "section_id": hit["metadata"]["section_id"],
            "section_title": hit["metadata"]["section_title"],
            "text_preview": hit["text"][:400],
            "rerank_score": round(hit.get("rerank_score", 0.0), 4),
            "edition_conflict_flag": hit["metadata"].get("edition_conflict_flag", False),
        }
        for hit in reranked
    ]


# ---------------------------------------------------------------------------
# Alignment scoring
# ---------------------------------------------------------------------------

ALIGNMENT_THRESHOLDS = {
    # Thresholds are applied on blended score, not pure entailment.
    "fully_aligned": 0.55,
    "partially_aligned": 0.35,
    "gap": 0.0,
}

RISK_LABELS = {
    "fully_aligned": "Low Risk",
    "partially_aligned": "Medium Risk",
    "gap": "High Risk - Possible Violation",
    "contradiction": "Critical - Regulatory Violation",
}


def _content_words(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "shall", "should", "must", "may", "has", "have", "had", "into", "within", "under",
        "such", "their", "there", "these", "those", "will", "been", "being", "also", "than",
    }
    return {tok for tok in tokens if tok not in stop_words}


def _lexical_overlap(requirement_text: str, section_text: str) -> float:
    req_terms = _content_words(requirement_text)
    sec_terms = _content_words(section_text)
    if not req_terms or not sec_terms:
        return 0.0

    overlap = len(req_terms & sec_terms)
    base = min(len(req_terms), len(sec_terms))
    if base <= 0:
        return 0.0
    return overlap / base


def score_alignment(req: Requirement, matched_sections: list[dict]) -> dict:
    """
    Run NLI across top matched sections and choose the best evidence-backed mapping.
    Returns alignment status, score, risk label, and reasoning.
    """
    if not matched_sections:
        return {
            "alignment_status": "gap",
            "alignment_score": 0.0,
            "risk_label": RISK_LABELS["gap"],
            "nli_result": None,
            "reasoning": "No matching section found in corpus - requirement may address uncovered topic.",
        }

    candidates = matched_sections[:3]
    rerank_values = [float(c.get("rerank_score", 0.0) or 0.0)
                     for c in candidates]
    max_rerank = max(rerank_values) if rerank_values else 0.0
    min_rerank = min(rerank_values) if rerank_values else 0.0

    scored: list[dict[str, Any]] = []
    for section in candidates:
        section_text = section.get("text_preview", "")
        nli = check_entailment(section_text, req.text)

        entailment = float(nli.get("entailment_score", 0.0) or 0.0)
        contradiction = float(nli.get("contradiction_score", 0.0) or 0.0)
        margin = entailment - contradiction

        rerank_raw = float(section.get("rerank_score", 0.0) or 0.0)
        if max_rerank > min_rerank:
            rerank_norm = (rerank_raw - min_rerank) / (max_rerank - min_rerank)
        else:
            rerank_norm = 1.0 if rerank_raw > 0 else 0.0

        lexical = _lexical_overlap(req.text, section_text)
        edition_penalty = 0.10 if section.get("edition_conflict_flag") else 0.0

        # Weighted blend to reduce false positives from any single signal.
        blended = (
            0.55 * entailment
            + 0.20 * rerank_norm
            + 0.20 * lexical
            - 0.35 * contradiction
            - edition_penalty
        )
        blended = max(0.0, min(1.0, blended))

        scored.append(
            {
                "section": section,
                "nli": nli,
                "entailment": entailment,
                "contradiction": contradiction,
                "margin": margin,
                "rerank_norm": rerank_norm,
                "lexical_overlap": lexical,
                "blended": blended,
            }
        )

    scored.sort(key=lambda item: item["blended"], reverse=True)
    best_pack = scored[0]
    best = best_pack["section"]

    entailment = float(best_pack["entailment"])
    contradiction = float(best_pack["contradiction"])
    margin = float(best_pack["margin"])
    blended_score = float(best_pack["blended"])

    if contradiction >= 0.55 and margin <= -0.15:
        status = "contradiction"
    elif (
        blended_score >= ALIGNMENT_THRESHOLDS["fully_aligned"]
        and entailment >= 0.72
        and margin >= 0.18
    ):
        status = "fully_aligned"
    elif (
        blended_score >= ALIGNMENT_THRESHOLDS["partially_aligned"]
        and entailment >= 0.45
        and margin >= -0.05
    ):
        status = "partially_aligned"
    else:
        status = "gap"

    return {
        "alignment_status": status,
        "alignment_score": round(blended_score, 3),
        "risk_label": RISK_LABELS.get(status, "Unknown"),
        "nli_result": best_pack["nli"],
        "mapped_section": {
            "pub_name": best.get("pub_name"),
            "edition": best.get("edition"),
            "section_id": best.get("section_id"),
            "section_title": best.get("section_title"),
        },
        "reasoning": (
            f"blended={blended_score:.2f}; entailment={entailment:.2f}, contradiction={contradiction:.2f}, "
            f"margin={margin:.2f}, lexical={best_pack['lexical_overlap']:.2f}, rerank={best_pack['rerank_norm']:.2f} "
            f"against [{best.get('pub_name')} | {best.get('edition')} | section {best.get('section_id')}] "
            f"'{best.get('section_title')}'"
        ),
        "evaluated_matches": [
            {
                "pub_name": item["section"].get("pub_name"),
                "edition": item["section"].get("edition"),
                "section_id": item["section"].get("section_id"),
                "blended_score": round(item["blended"], 3),
                "entailment_score": round(item["entailment"], 3),
                "contradiction_score": round(item["contradiction"], 3),
                "lexical_overlap": round(item["lexical_overlap"], 3),
                "rerank_norm": round(item["rerank_norm"], 3),
            }
            for item in scored
        ],
    }


# ---------------------------------------------------------------------------
# Remediation suggestions
# ---------------------------------------------------------------------------


def suggest_remediation(req: Requirement, alignment: dict, matched_sections: list[dict]) -> str:
    """
    For non-compliant or gap requirements, propose a specific section-aligned fix.
    """
    status = alignment["alignment_status"]
    if status == "fully_aligned":
        return "No remediation required."

    if not matched_sections:
        return (
            f"Requirement '{req.text[:80]}...' has no match in the RBI corpus. "
            "Consider whether this requirement is within regulatory scope or requires additional RBI guidance."
        )

    mapped = alignment.get("mapped_section") or {}
    best = mapped if mapped.get("section_id") else matched_sections[0]
    section_ref = (
        f"[{best.get('pub_name')} | {best.get('edition')} | section {best.get('section_id')} "
        f"'{best.get('section_title')}']"
    )

    if status == "contradiction":
        return (
            f"CRITICAL: This requirement directly contradicts {section_ref}. "
            "Revise the requirement to align with the RBI position in that section. "
            f"Preview: '{(matched_sections[0].get('text_preview', ''))[:200]}'"
        )
    if status == "partially_aligned":
        return (
            f"Partial alignment with {section_ref}. "
            "Strengthen requirement language to explicitly reflect the RBI criteria in the mapped section. "
            f"Current blended score: {alignment.get('alignment_score', 0.0):.2f}."
        )

    return (
        f"No strong alignment found. Closest match is {section_ref} "
        f"(score {alignment.get('alignment_score', 0.0):.2f}). "
        "Requirement may be out of scope or phrased in non-standard regulatory language. "
        "Recommend legal review against the full publication."
    )


# ---------------------------------------------------------------------------
# Full BRD validation pipeline
# ---------------------------------------------------------------------------


def validate_brd(file_path: str | Path, filename: str | None = None) -> dict:
    """
    End-to-end BRD validation:
      1. Extract text
      2. Extract requirements
      3. Map + score each requirement
      4. Aggregate compliance summary

    Returns:
      {
        "brd_filename": str,
        "total_requirements": int,
        "overall_alignment_score": float,
        "summary": {aligned, partial, gaps, violations},
        "requirements": [{req_id, text, sections, alignment, remediation}]
      }
    """
    path = Path(file_path)
    fname = filename or path.name

    logger.info("Validating BRD: %s", fname)
    brd_text = extract_text(path)
    requirements = extract_requirements(brd_text)

    if not requirements:
        return {
            "brd_filename": fname,
            "error": "No requirements could be extracted from this document.",
            "total_requirements": 0,
        }

    results = []
    summary = {"aligned": 0, "partial": 0, "gaps": 0, "violations": 0}

    for req in requirements:
        sections = map_requirement_to_sections(req)
        alignment = score_alignment(req, sections)
        remediation = suggest_remediation(req, alignment, sections)

        status = alignment["alignment_status"]
        if status == "fully_aligned":
            summary["aligned"] += 1
        elif status == "partially_aligned":
            summary["partial"] += 1
        elif status == "contradiction":
            summary["violations"] += 1
        else:
            summary["gaps"] += 1

        results.append(
            {
                "req_id": req.req_id,
                "text": req.text,
                "matched_sections": sections,
                "alignment": alignment,
                "remediation": remediation,
            }
        )

    total = len(requirements)
    overall_score = round(sum(r["alignment"]["alignment_score"]
                          for r in results) / total, 3)

    return {
        "brd_filename": fname,
        "total_requirements": total,
        "overall_alignment_score": overall_score,
        "compliance_grade": _grade(overall_score),
        "summary": summary,
        "requirements": results,
    }


def _grade(score: float) -> str:
    if score >= 0.8:
        return "A - Fully Compliant"
    if score >= 0.6:
        return "B - Substantially Compliant"
    if score >= 0.4:
        return "C - Partially Compliant"
    return "D - Non-Compliant"
