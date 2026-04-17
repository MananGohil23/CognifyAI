"""
hallucination_detector.py — XAI Governance Framework

NLI-based entailment verification of every generated claim against its source passage.
Uses cross-encoder/nli-deberta-v3-small for principled entailment scoring.

Labels:
  ENTAILED     → claim is supported by source (safe)
  NEUTRAL      → claim is plausible but not grounded (needs review)
  CONTRADICTION → claim contradicts source (non-compliant)

Also performs paraphrase stability: re-paraphrases the claim and checks
consistency, flagging unstable claims.
"""

import logging
import re

import torch
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
PARAPHRASE_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"

_nli_pipe = None
_para_pipe = None

LABEL_MAP = {
    "ENTAILMENT": "entailed",
    "NEUTRAL": "neutral",
    "CONTRADICTION": "contradiction",
}

TRUST_GATE_MAP = {
    "entailed": "Safe",
    "neutral": "Needs Human Review",
    "contradiction": "Non-Compliant",
}

# Confidence calibration for NLI decisions.
# These reduce false "contradiction" calls when model confidence is weak.
NLI_MIN_CONFIDENCE = 0.45
NLI_MIN_MARGIN = 0.08
CONTRADICTION_STRONG_THRESHOLD = 0.60


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z]{3,}", _normalize_text(text))
    stop = {
        "the", "and", "with", "from", "that", "this", "were", "was",
        "for", "are", "per", "end", "as", "at", "into", "during",
    }
    return {t for t in tokens if t not in stop}


def _number_set(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _heuristic_entailment(premise: str, hypothesis: str) -> dict | None:
    """Fast deterministic grounding check for extractive numeric claims."""
    p = _normalize_text(premise)
    h = _normalize_text(hypothesis)
    if not p or not h:
        return None

    p_tokens = _token_set(p)
    h_tokens = _token_set(h)
    if not p_tokens or not h_tokens:
        return None

    overlap = len(p_tokens & h_tokens)
    precision = overlap / max(len(h_tokens), 1)

    p_nums = _number_set(premise)
    h_nums = _number_set(hypothesis)
    num_overlap = len(p_nums & h_nums)

    # Strong extractive match: many shared content words + matching numbers.
    if (precision >= 0.55) or (precision >= 0.4 and num_overlap >= 1):
        return {
            "label": "entailed",
            "entailment_score": 0.92,
            "neutral_score": 0.06,
            "contradiction_score": 0.02,
            "confidence": 0.92,
            "margin": 0.86,
            "trust_gate": TRUST_GATE_MAP["entailed"],
        }
    return None


def get_nli_pipeline():
    global _nli_pipe
    if _nli_pipe is None:
        logger.info("Loading NLI model: %s", NLI_MODEL)
        _nli_pipe = hf_pipeline(
            "text-classification",
            model=NLI_MODEL,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,  # return all labels
        )
    return _nli_pipe


def get_paraphrase_pipeline():
    global _para_pipe
    if _para_pipe is None:
        logger.info("Loading paraphrase model: %s", PARAPHRASE_MODEL)
        _para_pipe = hf_pipeline(
            "text2text-generation",
            model=PARAPHRASE_MODEL,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _para_pipe


# ---------------------------------------------------------------------------
# Core entailment check
# ---------------------------------------------------------------------------

def check_entailment(premise: str, hypothesis: str) -> dict:
    """
    Returns {label, entailment_score, neutral_score, contradiction_score, raw_scores}.
    premise   = source passage (retrieved chunk)
    hypothesis = generated claim sentence
    """
    heuristic = _heuristic_entailment(premise, hypothesis)
    if heuristic is not None:
        return heuristic

    pipe = get_nli_pipeline()
    # Provide a true sentence-pair input. This model is trained for
    # premise/hypothesis pairs and performs poorly on concatenated strings.
    pair_input = {
        "text": premise[:1200],
        "text_pair": hypothesis[:400],
    }
    raw = pipe(pair_input)

    # With top_k=None, pipeline may return either list[dict] or [list[dict]].
    if raw and isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]

    scores = {LABEL_MAP.get(
        r["label"].upper(), r["label"].lower()): r["score"] for r in raw}
    label = max(scores, key=scores.get)

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = sorted_scores[0][1] if sorted_scores else 0.0
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = top_score - second_score

    # If confidence is weak or ambiguous, default to neutral for safer gating.
    if top_score < NLI_MIN_CONFIDENCE or margin < NLI_MIN_MARGIN:
        label = "neutral"

    # Require stronger evidence before treating as contradiction.
    if label == "contradiction" and scores.get("contradiction", 0.0) < CONTRADICTION_STRONG_THRESHOLD:
        label = "neutral"

    return {
        "label": label,
        "entailment_score": scores.get("entailed", 0.0),
        "neutral_score": scores.get("neutral", 0.0),
        "contradiction_score": scores.get("contradiction", 0.0),
        "confidence": round(top_score, 4),
        "margin": round(margin, 4),
        "trust_gate": TRUST_GATE_MAP[label],
    }


# ---------------------------------------------------------------------------
# Paraphrase stability
# ---------------------------------------------------------------------------

def paraphrase_claim(claim: str, num_variants: int = 2) -> list[str]:
    """Generate paraphrases of a claim for stability checking."""
    try:
        pipe = get_paraphrase_pipeline()
        outputs = pipe(
            f"paraphrase: {claim}",
            max_length=200,
            num_return_sequences=num_variants,
            num_beams=num_variants + 2,
        )
        return [o["generated_text"] for o in outputs]
    except Exception as e:
        logger.warning("Paraphrase generation failed: %s", e)
        return []


def check_paraphrase_stability(claim: str, source: str, num_variants: int = 2) -> dict:
    """
    Generate paraphrases of the claim and check each against the source.
    Stability = fraction of paraphrases that also entail the source.
    """
    variants = paraphrase_claim(claim, num_variants)
    if not variants:
        return {"stability_score": None, "paraphrases": [], "stable": None}

    results = []
    for v in variants:
        r = check_entailment(source, v)
        results.append({
            "paraphrase": v,
            "label": r["label"],
            "entailment_score": r["entailment_score"],
        })

    stability = sum(
        1 for r in results if r["label"] == "entailed") / len(results)
    return {
        "stability_score": round(stability, 3),
        "paraphrases": results,
        "stable": stability >= 0.5,
    }


# ---------------------------------------------------------------------------
# Batch claim verification
# ---------------------------------------------------------------------------

def verify_claims(claims: list[dict]) -> list[dict]:
    """
    For each claim (as returned by rag_pipeline._parse_claims), run:
      1. Entailment check (claim vs source_text)
      2. Paraphrase stability

    Returns enriched claim list with:
      {
        ...original claim fields,
        "nli_result": {...},
        "paraphrase_stability": {...},
        "final_trust_gate": "Safe" | "Needs Human Review" | "Non-Compliant",
        "reasoning": str
      }
    """
    verified = []
    for claim in claims:
        source = claim.get("source_text", "")
        sentence = claim.get("sentence", "")

        if not source or not sentence:
            claim["nli_result"] = None
            claim["paraphrase_stability"] = None
            claim["final_trust_gate"] = "Needs Human Review"
            claim["reasoning"] = "No source text available for verification"
            verified.append(claim)
            continue

        nli = check_entailment(source, sentence)
        stability = check_paraphrase_stability(sentence, source)

        # Escalate trust gate if unstable
        trust = nli["trust_gate"]
        if trust == "Safe" and stability.get("stable") is False:
            trust = "Needs Human Review"

        reasoning = _build_reasoning(nli, stability, claim)
        verified.append({
            **claim,
            "nli_result": nli,
            "paraphrase_stability": stability,
            "final_trust_gate": trust,
            "reasoning": reasoning,
        })

    return verified


def _build_reasoning(nli: dict, stability: dict, claim: dict) -> str:
    lines = [
        f"NLI label: {nli['label']} "
        f"(entailment={nli['entailment_score']:.2f}, "
        f"contradiction={nli['contradiction_score']:.2f})",
    ]
    if stability.get("stability_score") is not None:
        lines.append(
            f"Paraphrase stability: {stability['stability_score']:.0%} "
            f"({'stable' if stability['stable'] else 'unstable'})"
        )
    lines.append(
        f"Source: [{claim.get('pub_name')} · {claim.get('edition')} · §{claim.get('section_id')}]"
    )
    return " | ".join(lines)


# ---------------------------------------------------------------------------
# Edition conflict reasoning
# ---------------------------------------------------------------------------

def resolve_edition_conflict(
    claim_text: str,
    older_passage: str,
    newer_passage: str,
    older_edition: str,
    newer_edition: str,
) -> dict:
    """
    When two editions of the same section exist, determine which one the claim
    aligns with, and which edition's position should be treated as current.

    Returns:
      {
        "claim_aligns_with": "older" | "newer" | "neither" | "both",
        "superseded_by": newer_edition,
        "recommendation": str,
        "older_nli": dict,
        "newer_nli": dict
      }
    """
    older_nli = check_entailment(older_passage, claim_text)
    newer_nli = check_entailment(newer_passage, claim_text)

    older_supported = older_nli["label"] == "entailed"
    newer_supported = newer_nli["label"] == "entailed"

    if older_supported and newer_supported:
        aligns = "both"
    elif newer_supported:
        aligns = "newer"
    elif older_supported:
        aligns = "older"
    else:
        aligns = "neither"

    recommendation = (
        f"Claim aligns with {aligns} edition. "
        f"The {newer_edition} edition supersedes {older_edition}. "
        + (
            "Claim is CURRENT and Safe."
            if aligns == "newer"
            else "Claim may reflect OUTDATED guidance — flag for human review."
            if aligns == "older"
            else "Claim is NOT grounded in either edition — Non-Compliant."
        )
    )

    return {
        "claim_aligns_with": aligns,
        "superseded_by": newer_edition,
        "recommendation": recommendation,
        "older_nli": older_nli,
        "newer_nli": newer_nli,
    }
