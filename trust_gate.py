"""
trust_gate.py — XAI Governance Framework

Aggregates NLI results, edition conflict flags, and BRD alignment scores
into a final trust gate decision.

Also computes the RAGAS-style trust scorecard:
  - Context relevance
  - Faithfulness (entailment rate)
  - Citation precision
  - Edition-conflict risk
  - Paraphrase stability

Generates the audit-ready report (compliance evidence registry).
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trust gate thresholds
# ---------------------------------------------------------------------------

TRUST_THRESHOLDS = {
    "Safe": {
        "min_faithfulness": 0.5,
        "max_conflict_rate": 0.2,
        "min_stability": 0.4,
        "min_citation_precision": 0.5,
        "max_contradiction_rate": 0.2,
        "max_review_flag_rate": 0.35,
    },
    "Needs Human Review": {
        "min_faithfulness": 0.15,
        "max_conflict_rate": 0.7,
        "min_stability": 0.15,
        "max_contradiction_rate": 0.6,
        "max_review_flag_rate": 0.9,
    },
    # Below these thresholds → Non-Compliant
}


# ---------------------------------------------------------------------------
# RAGAS-style scorecard
# ---------------------------------------------------------------------------

def compute_ragas_scorecard(
    query: str,
    retrieved_chunks: list[dict],
    verified_claims: list[dict],
) -> dict:
    """
    Compute the five-axis trust scorecard.
    All scores are 0.0–1.0.
    """
    _ = query
    # 1. Context relevance — how many chunks are edition-conflict-free
    total_chunks = len(retrieved_chunks)
    conflict_chunks = sum(
        1 for c in retrieved_chunks if c.get("edition_conflict_flag"))
    context_relevance = round(
        1.0 - (conflict_chunks / max(total_chunks, 1)), 3)

    # 2. Faithfulness — fraction of claims that are ENTAILED
    entailed = [
        c
        for c in verified_claims
        if (c.get("nli_result") or {}).get("label") == "entailed"
    ]
    faithfulness = round(len(entailed) / max(len(verified_claims), 1), 3)

    # 3. Citation precision — fraction of claims that cite a real retrieved chunk
    cited_section_ids = {c.get("section_id")
                         for c in retrieved_chunks if c.get("section_id")}
    claims_with_valid_citation = [
        c for c in verified_claims
        if c.get("section_id") in cited_section_ids
    ]
    citation_precision = round(
        len(claims_with_valid_citation) / max(len(verified_claims), 1), 3)

    # 4. Edition-conflict risk — fraction of claimed sections with conflicts
    claimed_sections = [c.get("section_id") for c in verified_claims]
    conflict_section_ids = {
        c.get("section_id") for c in retrieved_chunks if c.get("edition_conflict_flag") and c.get("section_id")
    }
    conflicted_claims = sum(
        1 for sid in claimed_sections if sid in conflict_section_ids)
    edition_conflict_risk = round(
        conflicted_claims / max(len(claimed_sections), 1), 3)

    # 5. Paraphrase stability — average stability score across verified claims
    stability_scores = [
        c["paraphrase_stability"]["stability_score"]
        for c in verified_claims
        if c.get("paraphrase_stability") and c["paraphrase_stability"].get("stability_score") is not None
    ]
    paraphrase_stability = round(
        sum(stability_scores) / len(stability_scores), 3) if stability_scores else None

    # Composite trust score (weighted average)
    weights = {"faithfulness": 0.35, "citation_precision": 0.25,
               "context_relevance": 0.2, "paraphrase_stability": 0.1,
               "edition_conflict_risk_inv": 0.1}
    ecr_inv = 1.0 - edition_conflict_risk
    stability_val = paraphrase_stability if paraphrase_stability is not None else 0.5
    composite = round(
        weights["faithfulness"] * faithfulness
        + weights["citation_precision"] * citation_precision
        + weights["context_relevance"] * context_relevance
        + weights["paraphrase_stability"] * stability_val
        + weights["edition_conflict_risk_inv"] * ecr_inv,
        3,
    )

    return {
        "context_relevance": context_relevance,
        "faithfulness": faithfulness,
        "citation_precision": citation_precision,
        "edition_conflict_risk": edition_conflict_risk,
        "paraphrase_stability": paraphrase_stability,
        "composite_trust_score": composite,
    }


# ---------------------------------------------------------------------------
# Final trust gate decision
# ---------------------------------------------------------------------------

def apply_trust_gate(
    verified_claims: list[dict],
    ragas_scorecard: dict,
    edition_conflicts: list[dict],
    query_premise_check: Optional[dict] = None,
) -> dict:
    """
    Apply thresholds to determine the overall trust gate decision.
    Per-claim decisions are also included for granular audit.

    Returns:
      {
        "overall_gate": "Safe" | "Needs Human Review" | "Non-Compliant",
        "reasoning": str,
        "per_claim_gates": [...]
      }
    """
    _ = edition_conflicts
    premise = query_premise_check or {}
    sc = ragas_scorecard
    t = TRUST_THRESHOLDS

    # Per-claim gates (already set by hallucination_detector)
    per_claim = [
        {
            "req_id": c.get("req_id"),
            "sentence": c.get("sentence", "")[:120],
            "gate": c.get("final_trust_gate", "Needs Human Review"),
            "nli_label": (c.get("nli_result") or {}).get("label"),
            "reasoning": c.get("reasoning"),
        }
        for c in verified_claims
    ]

    def _token_set(text: str) -> set[str]:
        tokens = re.findall(r"[a-z]{3,}", (text or "").lower())
        stop = {
            "the", "and", "with", "from", "that", "this", "were", "was",
            "for", "are", "per", "end", "as", "at", "into", "during",
            "under", "over", "across", "between",
        }
        return {t for t in tokens if t not in stop}

    def _number_set(text: str) -> set[str]:
        return set(re.findall(r"\d+(?:\.\d+)?", text or ""))

    def _compute_source_grounding_rate(claims: list[dict]) -> float:
        """
        Measures how well generated claims are directly grounded in retrieved source text.
        Score range 0..1 where higher is better.
        """
        if not claims:
            return 0.0

        scores = []
        for c in claims:
            sent = c.get("sentence", "")
            source = c.get("source_text", "")
            if not sent or not source:
                scores.append(0.0)
                continue

            sent_tokens = _token_set(sent)
            src_tokens = _token_set(source)
            overlap_precision = len(
                sent_tokens & src_tokens) / max(len(sent_tokens), 1)

            sent_nums = _number_set(sent)
            src_nums = _number_set(source)
            numeric_consistency = (
                len(sent_nums & src_nums) / len(sent_nums)
                if sent_nums
                else 1.0
            )

            # Blend lexical overlap and numeric consistency.
            grounded = 0.65 * overlap_precision + 0.35 * numeric_consistency
            scores.append(min(1.0, max(0.0, grounded)))

        return round(sum(scores) / len(scores), 3)

    # Contradiction rate (less strict than hard fail on a single contradiction)
    contradiction_count = sum(
        1 for p in per_claim if p["gate"] == "Non-Compliant")
    review_count = sum(
        1 for p in per_claim if p["gate"] == "Needs Human Review")
    contradiction_rate = contradiction_count / max(len(per_claim), 1)
    review_rate = review_count / max(len(per_claim), 1)
    neutral_count = sum(1 for p in per_claim if p["nli_label"] == "neutral")
    neutral_rate = neutral_count / max(len(per_claim), 1)
    source_grounding_rate = _compute_source_grounding_rate(verified_claims)
    reasons = []

    if contradiction_count:
        reasons.append(
            f"{contradiction_count}/{max(len(per_claim), 1)} claims contradict source passages "
            f"({contradiction_rate:.0%})."
        )
    if sc["faithfulness"] < t["Safe"]["min_faithfulness"]:
        reasons.append(
            f"Faithfulness {sc['faithfulness']:.0%} below Safe threshold {t['Safe']['min_faithfulness']:.0%}.")
    if sc["edition_conflict_risk"] > t["Safe"]["max_conflict_rate"]:
        reasons.append(
            f"Edition conflict risk {sc['edition_conflict_risk']:.0%} exceeds Safe threshold.")
    if sc["paraphrase_stability"] is not None and sc["paraphrase_stability"] < t["Safe"]["min_stability"]:
        reasons.append(
            f"Paraphrase stability {sc['paraphrase_stability']:.0%} below Safe threshold.")
    if sc["citation_precision"] < t["Safe"]["min_citation_precision"]:
        reasons.append(
            f"Citation precision {sc['citation_precision']:.0%} below Safe threshold.")
    if contradiction_rate > t["Safe"]["max_contradiction_rate"]:
        reasons.append(
            f"Contradiction rate {contradiction_rate:.0%} above Safe threshold {t['Safe']['max_contradiction_rate']:.0%}."
        )
    if review_rate > t["Safe"]["max_review_flag_rate"]:
        reasons.append(
            f"Needs Human Review flag rate {review_rate:.0%} above Safe threshold {t['Safe']['max_review_flag_rate']:.0%}."
        )
    if source_grounding_rate < 0.6:
        reasons.append(
            f"Source grounding rate {source_grounding_rate:.0%} below Safe threshold 60%."
        )

    if premise.get("risk_level") == "high":
        reasons.append(
            "Query premise conflicts with retrieved corpus evidence (numeric/edition mismatch)."
        )
    elif premise.get("risk_level") == "medium":
        reasons.append(
            "Query premise is partially unsupported by retrieved corpus; manual verification required."
        )

    # If model mostly abstains/returns neutral and there are no contradictions,
    # treat as review-needed instead of outright non-compliance.
    abstention_dominant = contradiction_count == 0 and (
        sc["faithfulness"] == 0 or neutral_rate >= 0.7
    )
    if abstention_dominant:
        reasons.append(
            "Claims are predominantly neutral/abstaining without direct contradictions; requires human review."
        )

    severe_low_grounding = (
        sc["faithfulness"] < t["Needs Human Review"]["min_faithfulness"]
        and sc["citation_precision"] < 0.35
        and source_grounding_rate < 0.25
    )

    safe_candidate = (
        contradiction_rate <= t["Safe"]["max_contradiction_rate"]
        and sc["faithfulness"] >= t["Safe"]["min_faithfulness"]
        and sc["citation_precision"] >= t["Safe"]["min_citation_precision"]
        and sc["edition_conflict_risk"] <= t["Safe"]["max_conflict_rate"]
        and source_grounding_rate >= 0.6
        and (sc["paraphrase_stability"] is None or sc["paraphrase_stability"] >= t["Safe"]["min_stability"])
    )

    forced_non_compliant = premise.get("risk_level") == "high"
    forced_review = premise.get("risk_level") == "medium"

    if forced_non_compliant:
        overall_gate = "Non-Compliant"
    elif safe_candidate and not forced_review:
        overall_gate = "Safe"
    elif abstention_dominant:
        overall_gate = "Needs Human Review"
    elif (
        severe_low_grounding
        or sc["edition_conflict_risk"] > t["Needs Human Review"]["max_conflict_rate"]
        or contradiction_rate > t["Needs Human Review"]["max_contradiction_rate"]
    ):
        overall_gate = "Non-Compliant"
    elif reasons:
        overall_gate = "Needs Human Review"
    else:
        overall_gate = "Safe"

    return {
        "overall_gate": overall_gate,
        "reasoning": " ".join(reasons) if reasons else "All thresholds met.",
        "per_claim_gates": per_claim,
    }


# ---------------------------------------------------------------------------
# Audit report generation
# ---------------------------------------------------------------------------

def generate_audit_report(
    session_id: str,
    query: str,
    answer: str,
    retrieved_chunks: list[dict],
    verified_claims: list[dict],
    ragas_scorecard: dict,
    trust_gate_result: dict,
    edition_conflicts: list[dict],
    brd_result: Optional[dict] = None,
) -> dict:
    """
    Assemble the complete audit-ready report.
    This is the compliance evidence registry — every decision is logged.
    """
    report = {
        "report_id": f"XAI-{session_id}",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "answer_preview": answer[:500] if answer else "",

        # Trust gate
        "overall_trust_gate": trust_gate_result["overall_gate"],
        "gate_reasoning": trust_gate_result["reasoning"],

        # RAGAS scorecard
        "ragas_scorecard": ragas_scorecard,

        # Sources used
        "sources": [
            {
                "pub_name": c["pub_name"],
                "edition": c["edition"],
                "section_id": c["section_id"],
                "section_title": c.get("section_title", ""),
                "source_filename": c.get("source_filename", ""),
                "source_url": c.get("source_url", ""),
                "rerank_score": c.get("rerank_score"),
                "edition_conflict_flag": c.get("edition_conflict_flag", False),
            }
            for c in retrieved_chunks
        ],

        # Claim-level evidence
        "claim_evidence": [
            {
                "sentence": c.get("sentence", ""),
                "citation": f"[{c.get('pub_name')} · {c.get('edition')} · §{c.get('section_id')}]",
                "nli_result": c.get("nli_result"),
                "paraphrase_stability": c.get("paraphrase_stability"),
                "trust_gate": c.get("final_trust_gate"),
                "reasoning": c.get("reasoning"),
            }
            for c in verified_claims
        ],

        # Edition conflicts
        "edition_conflicts": edition_conflicts,

        # BRD compliance (if validated)
        "brd_compliance": brd_result,

        # Explainability artifacts
        "explainability": {
            "retrieval_rationale": [
                f"Retrieved §{c['section_id']} from {c['pub_name']} {c['edition']} "
                f"(rerank_score={c.get('rerank_score', 'N/A')}) — used as evidence for claim attribution."
                for c in retrieved_chunks[:5]
            ],
            "rejection_log": [
                f"Claim rejected/flagged: '{c.get('sentence', '')[:80]}' — {c.get('reasoning')}"
                for c in verified_claims
                if c.get("final_trust_gate") != "Safe"
            ],
        },

        # Decision log
        "decision_log": [
            {
                "step": "Ingestion",
                "outcome": f"{len(retrieved_chunks)} chunks retrieved from corpus",
            },
            {
                "step": "Hybrid Retrieval",
                "outcome": "Dense + BM25 + RRF fusion applied",
            },
            {
                "step": "NLI Verification",
                "outcome": f"{len(verified_claims)} claims verified",
            },
            {
                "step": "Trust Gate",
                "outcome": trust_gate_result["overall_gate"],
            },
        ],
    }

    return report


# ---------------------------------------------------------------------------
# Claim parsing helper
# ---------------------------------------------------------------------------

def parse_claims(answer_text: str, context_hits: list[dict]) -> list[dict]:
    """
    Parse generated answer text into structured claims with source attribution.
    Expected citation format in answer text: [PUB · EDITION · §SECTION].
    """
    cleaned_answer = (answer_text or "").replace("Â·", "·").replace("Â§", "§")
    citation_re = re.compile(
        r"(?P<sentence>[^\n\[]+?)\s*"
        r"\[(?P<pub>[^\[\]·]+)\s*·\s*(?P<edition>[^\[\]·]+)\s*·\s*§?(?P<section>[^\]]+)\]\.?",
        re.MULTILINE,
    )

    def _norm(v: str) -> str:
        return re.sub(r"\s+", " ", str(v or "")).strip().lower()

    context_by_key = {}
    context_by_pub_section = {}
    context_by_section = {}
    for h in context_hits:
        m = h.get("metadata", {})
        pub = _norm(m.get("pub_name"))
        edition = _norm(m.get("edition"))
        section = _norm(m.get("section_id"))
        context_by_key[(pub, edition, section)] = h
        context_by_pub_section[(pub, section)] = h
        context_by_section[section] = h

    claims = []
    for m in citation_re.finditer(cleaned_answer):
        sentence = (m.group("sentence") or "").strip()
        pub_name = (m.group("pub") or "").strip()
        edition = (m.group("edition") or "").strip()
        section_id = (m.group("section") or "").strip()
        section_id = re.sub(r"^[^0-9A-Za-z]+", "", section_id)
        source_hit = (
            context_by_key.get(
                (_norm(pub_name), _norm(edition), _norm(section_id)))
            or context_by_pub_section.get((_norm(pub_name), _norm(section_id)))
            or context_by_section.get(_norm(section_id))
            or {}
        )
        source_text = source_hit.get("text", "")
        claims.append(
            {
                "sentence": sentence.strip(),
                "source_text": source_text,
                "section_id": section_id,
                "pub_name": pub_name,
                "edition": edition,
                "metadata": source_hit.get("metadata", {}),
            }
        )
    return claims


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_full_pipeline(
    query: str,
    brd_file_path: Optional[str] = None,
) -> dict:
    """
    Orchestrate the complete XAI governance pipeline:
      RAG → Hallucination Detection → Trust Gate → Audit Report

    Returns the full audit report dict.
    """
    from rag_pipeline import (
        retrieve_dense,
        retrieve_sparse,
        reciprocal_rank_fusion,
        rerank,
        generate_with_attribution,
    )
    from hallucination_detector import verify_claims
    from brd_engine import validate_brd
    from ingestion import get_collection, get_embedder

    def _normalize_chunk(hit: dict) -> dict:
        meta = hit.get("metadata", {})
        return {
            "text": hit.get("text", ""),
            "section_id": meta.get("section_id") or hit.get("section_id", ""),
            "section_title": meta.get("section_title") or hit.get("section_title", ""),
            "pub_name": meta.get("pub_name") or hit.get("pub_name", ""),
            "edition": meta.get("edition") or hit.get("edition", ""),
            "source_filename": meta.get("source_filename") or hit.get("source_filename", ""),
            "source_url": meta.get("source_url") or hit.get("source_url", ""),
            "edition_conflict_flag": bool(
                meta.get("edition_conflict_flag", hit.get(
                    "edition_conflict_flag", False))
            ),
            "rerank_score": hit.get("rerank_score"),
            "metadata": meta or {
                "section_id": hit.get("section_id", ""),
                "pub_name": hit.get("pub_name", ""),
                "edition": hit.get("edition", ""),
                "section_title": hit.get("section_title", ""),
                "source_filename": hit.get("source_filename", ""),
                "source_url": hit.get("source_url", ""),
                "edition_conflict_flag": bool(hit.get("edition_conflict_flag", False)),
            },
        }

    def _detect_edition_conflicts(retrieved_chunks: list[dict]) -> list[dict]:
        by_section: dict[tuple[str, str], set[str]] = {}
        for c in retrieved_chunks:
            pub = c.get("pub_name")
            sec = c.get("section_id")
            ed = c.get("edition")
            if not pub or not sec or not ed:
                continue
            by_section.setdefault((pub, sec), set()).add(ed)

        conflicts = []
        for chunk in retrieved_chunks:
            key = (chunk.get("pub_name"), chunk.get("section_id"))
            editions = sorted(by_section.get(key, set()))
            has_multi_edition = len(editions) > 1
            chunk["edition_conflict_flag"] = bool(
                chunk.get("edition_conflict_flag") or has_multi_edition)
            if has_multi_edition:
                conflicts.append(
                    {
                        "pub_name": key[0],
                        "section_id": key[1],
                        "editions": editions,
                        "conflict_note": "Multiple editions retrieved for same section",
                    }
                )

        deduped = {}
        for c in conflicts:
            deduped[(c["pub_name"], c["section_id"])] = c
        return list(deduped.values())

    def _to_external_gate(gate: str) -> str:
        if gate == "Needs Human Review":
            return "Needs Review"
        return gate

    def _infer_claims_from_answer(answer_text: str, chunks: list[dict], max_claims: int = 3) -> list[dict]:
        """
        Fallback when model omits citation tags.
        Splits answer into sentences and anchors them to top retrieved chunks.
        """
        if not answer_text or not chunks:
            return []

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", answer_text)
            if s and len(s.strip()) >= 30
        ]
        if not sentences:
            sentences = [answer_text[:300].strip()]

        claims = []
        for idx, sentence in enumerate(sentences[:max_claims]):
            chunk = chunks[idx % len(chunks)]
            claims.append(
                {
                    "sentence": sentence,
                    "source_text": chunk.get("text", "")[:1200],
                    "section_id": chunk.get("section_id", ""),
                    "pub_name": chunk.get("pub_name", ""),
                    "edition": chunk.get("edition", ""),
                    "metadata": chunk.get("metadata", {}),
                }
            )
        return claims

    def _analyze_query_premise(query_text: str, chunks: list[dict]) -> dict:
        """
        Detect whether explicit user premise details are unsupported by retrieved corpus.
        Used to catch hallucination traps such as outdated figures/editions not in corpus.
        """
        q = (query_text or "").lower()
        corpus_text = "\n".join((c.get("text", "") or "").lower()
                                for c in chunks)
        corpus_editions = [((c.get("edition") or "").lower()) for c in chunks]
        retrieved_pubs = {
            (c.get("pub_name") or "").strip().upper()
            for c in chunks
            if (c.get("pub_name") or "").strip()
        }

        query_numbers = set(re.findall(r"\d+(?:\.\d+)?", q))
        corpus_numbers = set(re.findall(r"\d+(?:\.\d+)?", corpus_text))
        unmatched_numbers = sorted(
            n for n in query_numbers if n not in corpus_numbers)

        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        query_month_refs = [m for m in months if m in q]
        missing_month_refs = [
            m for m in query_month_refs
            if not any(m in ed for ed in corpus_editions)
        ]

        has_unverified_numeric_premise = len(unmatched_numbers) > 0
        has_missing_edition_reference = len(missing_month_refs) > 0

        # Publication intent checks (e.g., query asks FSR but retrieved only MPR)
        query_publications = []
        if "fsr" in q or "financial stability report" in q:
            query_publications.append("FSR")
        if "mpr" in q or "monetary policy report" in q:
            query_publications.append("MPR")
        if "psr" in q or "payment system report" in q:
            query_publications.append("PSR")
        if "fer" in q or "foreign exchange" in q:
            query_publications.append("FER")

        missing_query_publications = [
            p for p in query_publications if p not in retrieved_pubs
        ]

        # Comparative/cross-edition intent checks
        comparative_markers = [
            "across", "between", "successive", "compare", "comparison",
            "revised", "revision", "changed", "change", "evolved", "trend",
        ]
        has_comparative_intent = (
            "edition" in q
            and any(m in q for m in comparative_markers)
        )

        insufficient_cross_edition_support = False
        if has_comparative_intent:
            if query_publications:
                for pub in query_publications:
                    pub_editions = {
                        (c.get("edition") or "").strip().lower()
                        for c in chunks
                        if (c.get("pub_name") or "").strip().upper() == pub
                    }
                    if len(pub_editions) < 2:
                        insufficient_cross_edition_support = True
                        break
            else:
                all_editions = {
                    (c.get("edition") or "").strip().lower()
                    for c in chunks
                    if (c.get("edition") or "").strip()
                }
                if len(all_editions) < 2:
                    insufficient_cross_edition_support = True

        if has_unverified_numeric_premise and has_missing_edition_reference:
            risk_level = "high"
        elif (
            has_unverified_numeric_premise
            or has_missing_edition_reference
            or bool(missing_query_publications)
            or insufficient_cross_edition_support
        ):
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "query_numbers": sorted(query_numbers),
            "unmatched_numbers": unmatched_numbers,
            "query_month_refs": query_month_refs,
            "missing_month_refs": missing_month_refs,
            "has_unverified_numeric_premise": has_unverified_numeric_premise,
            "has_missing_edition_reference": has_missing_edition_reference,
            "query_publications": query_publications,
            "retrieved_publications": sorted(retrieved_pubs),
            "missing_query_publications": missing_query_publications,
            "has_comparative_intent": has_comparative_intent,
            "insufficient_cross_edition_support": insufficient_cross_edition_support,
        }

    def _dedupe_claims(claims: list[dict]) -> list[dict]:
        """Remove duplicate claim-citation pairs while preserving order."""
        seen = set()
        deduped = []
        for c in claims:
            sentence = re.sub(
                r"\s+", " ", (c.get("sentence") or "")).strip().lower()
            key = (
                sentence,
                (c.get("pub_name") or "").strip().lower(),
                (c.get("edition") or "").strip().lower(),
                (c.get("section_id") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped

    def _apply_premise_penalty(scorecard: dict, premise_check: dict) -> dict:
        """
        Adjust composite trust when the user's query premise conflicts with corpus.
        Prevents inflated trust scores in hallucination-trap scenarios.
        """
        sc = dict(scorecard)
        risk = (premise_check or {}).get("risk_level", "low")

        premise_consistency = 1.0
        if risk == "medium":
            premise_consistency = 0.6
        elif risk == "high":
            premise_consistency = 0.15

        sc["premise_consistency"] = round(premise_consistency, 3)

        # Keep all displayed properties aligned with premise-grounding trust.
        # Positive metrics are damped; risk metrics are increased when premise
        # consistency is low.
        positive_metrics = [
            "faithfulness",
            "citation_precision",
            "context_relevance",
            "paraphrase_stability",
        ]
        for key in positive_metrics:
            if sc.get(key) is not None:
                sc[key] = round(float(sc[key]) * premise_consistency, 3)

        # Lower is better: push risk up when premise consistency is poor.
        base_risk = float(sc.get("edition_conflict_risk", 0.0))
        premise_risk = 1.0 - premise_consistency
        sc["edition_conflict_risk"] = round(max(base_risk, premise_risk), 3)

        base = float(sc.get("composite_trust_score", 0.0))
        penalty_mult = premise_consistency
        penalized = round(base * penalty_mult, 3)

        # Hard cap in explicit hallucination traps.
        if risk == "high":
            penalized = min(penalized, 0.35)

        sc["composite_trust_score"] = penalized
        return sc

    session_id = uuid.uuid4().hex[:12]
    logger.info("[%s] Starting full pipeline", session_id)

    # 1) Hybrid retrieval
    embedder = get_embedder()
    collection = get_collection()

    dense_hits = retrieve_dense(query, embedder, collection)
    sparse_hits = retrieve_sparse(query, dense_hits)
    fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits)
    reranked_hits = rerank(query, fused_hits)
    logger.info(
        "[%s] Retrieval complete: dense=%d sparse=%d fused=%d reranked=%d",
        session_id,
        len(dense_hits),
        len(sparse_hits),
        len(fused_hits),
        len(reranked_hits),
    )

    retrieved_chunks = [_normalize_chunk(h) for h in reranked_hits]

    # Edge case: empty retrieval fallback
    if not retrieved_chunks:
        fallback_answer = (
            "I could not find grounded evidence in the ingested RBI corpus for this query. "
            "Please ingest relevant documents or refine the query."
        )
        scorecard = {
            "context_relevance": 0.0,
            "faithfulness": 0.0,
            "citation_precision": 0.0,
            "edition_conflict_risk": 0.0,
            "paraphrase_stability": None,
            "composite_trust_score": 0.0,
        }
        gate_result = {
            "overall_gate": "Needs Human Review",
            "reasoning": "No chunks retrieved; answer cannot be grounded.",
            "per_claim_gates": [],
        }
        report = generate_audit_report(
            session_id=session_id,
            query=query,
            answer=fallback_answer,
            retrieved_chunks=[],
            verified_claims=[],
            ragas_scorecard=scorecard,
            trust_gate_result=gate_result,
            edition_conflicts=[],
            brd_result=None,
        )
        return {
            "session_id": session_id,
            "answer": fallback_answer,
            "trust_gate": _to_external_gate(gate_result["overall_gate"]),
            "ragas_scorecard": scorecard,
            "report": report,
        }

    # 2) Answer generation with attribution
    generation = generate_with_attribution(query, reranked_hits)
    answer = generation.get("answer", "")
    claims = generation.get("claims") or []
    if not claims:
        claims = parse_claims(answer, reranked_hits)
    logger.info("[%s] Generated answer with %d claims",
                session_id, len(claims))

    normalized_claims = []
    for c in claims:
        normalized_claims.append(
            {
                "sentence": c.get("sentence") or c.get("text") or "",
                "source_text": c.get("source_text", ""),
                "section_id": c.get("section_id", ""),
                "pub_name": c.get("pub_name", ""),
                "edition": c.get("edition", ""),
                "metadata": c.get("metadata", {}),
            }
        )

    # Remove repeated claim/citation entries caused by repetitive model output.
    normalized_claims = _dedupe_claims(normalized_claims)

    # Edge case: missing citations
    if answer and not normalized_claims:
        normalized_claims = _infer_claims_from_answer(answer, retrieved_chunks)
        if not normalized_claims:
            normalized_claims = [
                {
                    "sentence": answer[:500],
                    "source_text": "",
                    "section_id": "",
                    "pub_name": "",
                    "edition": "",
                    "metadata": {},
                    "final_trust_gate": "Needs Human Review",
                    "reasoning": "No citation tags were found in generated answer.",
                }
            ]

    # 3) Claim verification (NLI + paraphrase stability)
    try:
        verified = verify_claims(normalized_claims)
    except Exception as exc:
        logger.exception("NLI verification failed: %s", exc)
        verified = []
        for c in normalized_claims:
            verified.append(
                {
                    **c,
                    "nli_result": None,
                    "paraphrase_stability": None,
                    "final_trust_gate": "Needs Human Review",
                    "reasoning": "NLI verification unavailable; defaulting to review.",
                }
            )
    logger.info("[%s] Verification complete: %d claims",
                session_id, len(verified))

    # 4) Edition conflict detection
    edition_conflicts = _detect_edition_conflicts(retrieved_chunks)
    premise_check = _analyze_query_premise(query, retrieved_chunks)

    # If the user query premise itself is contradictory/unsupported by corpus,
    # do not present any claim-level citation as fully safe.
    if premise_check.get("risk_level") == "high":
        for c in verified:
            if c.get("final_trust_gate") == "Safe":
                c["final_trust_gate"] = "Needs Human Review"
                existing_reason = c.get("reasoning", "")
                prefix = "Query premise conflicts with corpus evidence."
                c["reasoning"] = (
                    f"{prefix} {existing_reason}".strip()
                    if existing_reason
                    else prefix
                )

    # 5) RAGAS scorecard
    scorecard = compute_ragas_scorecard(query, retrieved_chunks, verified)
    scorecard = _apply_premise_penalty(scorecard, premise_check)

    # 6) Trust gate
    gate_result = apply_trust_gate(
        verified,
        scorecard,
        edition_conflicts,
        query_premise_check=premise_check,
    )
    logger.info("[%s] Trust gate decided: %s", session_id,
                gate_result.get("overall_gate"))

    # 7) Optional BRD validation
    brd_result = None
    if brd_file_path:
        brd_result = validate_brd(brd_file_path)

    # 8) Audit report
    report = generate_audit_report(
        session_id=session_id,
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        verified_claims=verified,
        ragas_scorecard=scorecard,
        trust_gate_result=gate_result,
        edition_conflicts=edition_conflicts,
        brd_result=brd_result,
    )
    report["query_premise_check"] = premise_check

    return {
        "session_id": session_id,
        "answer": answer,
        "trust_gate": _to_external_gate(gate_result["overall_gate"]),
        "ragas_scorecard": scorecard,
        "report": report,
    }
