"""
confidence_labeler.py — P6: AI Hallucination Confidence Labeler

Provides confidence labeling, perplexity metric calculation, and uncertainty explanation
for AI-generated Q&A outputs against available source text or support signals.

Reliability Tags:
  - Certain           → Strong evidence support, high entailment, minimal uncertainty.
  - Uncertain         → Partial evidence support, ambiguous details, or medium entropy.
  - Needs Verification → Unsupported by evidence, contradiction, missing sources, or high entropy.
"""

import math
import logging
import re
from typing import Optional
from hallucination_detector import check_entailment

logger = logging.getLogger(__name__)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into distinct sentences for claim-level verification."""
    if not text:
        return []
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def calculate_semantic_perplexity(label_scores: list[dict]) -> float:
    """
    Compute Perplexity metric based on NLI probability distribution entropy.
    PPL = exp(Average Shannon Entropy).
    Normalized range: ~1.00 (highest certainty) to 10.00+ (high uncertainty/hallucination).
    """
    if not label_scores:
        return 8.50

    entropies = []
    contradiction_penalty = 1.0

    for s in label_scores:
        pe = max(s.get("entailment_score", 0.001), 0.0001)
        pn = max(s.get("neutral_score", 0.001), 0.0001)
        pc = max(s.get("contradiction_score", 0.001), 0.0001)

        # Normalize to sum to 1
        total = pe + pn + pc
        pe, pn, pc = pe / total, pn / total, pc / total

        entropy = -(pe * math.log(pe) + pn * math.log(pn) + pc * math.log(pc))
        entropies.append(entropy)

        if s.get("label") == "contradiction" or pc > 0.4:
            contradiction_penalty += 1.8

    avg_entropy = sum(entropies) / len(entropies)
    base_ppl = math.exp(avg_entropy)

    # Scale to intuitive 1.0 - 10.0 scale
    ppl = round(base_ppl * contradiction_penalty * 1.2, 2)
    return max(1.05, min(ppl, 15.00))


def evaluate_qa_reliability(
    question: str,
    answer: str,
    source_text: Optional[str] = None
) -> dict:
    """
    Evaluate Q&A answer against source text or support signals.

    Returns:
      {
        "question": str,
        "answer": str,
        "source_text": str,
        "reliability_tag": "Certain" | "Uncertain" | "Needs Verification",
        "perplexity": float,
        "uncertainty_score": float (0-100%),
        "short_reason": str,
        "warnings": list[str],
        "claim_breakdown": list[dict],
        "metrics": {
            "faithfulness": float,
            "entailed_claims": int,
            "neutral_claims": int,
            "contradiction_claims": int,
            "total_claims": int
        }
      }
    """
    q_clean = (question or "").strip()
    a_clean = (answer or "").strip()
    s_clean = (source_text or "").strip()

    warnings = []

    # Case 1: Empty answer or question
    if not a_clean:
        return {
            "question": q_clean,
            "answer": "",
            "source_text": s_clean,
            "reliability_tag": "Needs Verification",
            "trust_score": 0.0,
            "perplexity": 10.00,
            "uncertainty_score": 100.0,
            "short_reason": "No answer content generated to verify.",
            "warnings": ["Answer is empty."],
            "claim_breakdown": [],
            "metrics": {
                "faithfulness": 0.0,
                "composite_trust_score": 0.0,
                "entailed_claims": 0,
                "neutral_claims": 0,
                "contradiction_claims": 0,
                "total_claims": 0
            }
        }

    # Case 2: No source text provided
    if not s_clean:
        warnings.append("No source text or grounding passage was provided for cross-verification.")
        return {
            "question": q_clean,
            "answer": a_clean,
            "source_text": "",
            "reliability_tag": "Needs Verification",
            "trust_score": 0.15,
            "perplexity": 7.50,
            "uncertainty_score": 85.0,
            "short_reason": "Evidence is missing. The answer cannot be cross-referenced against authoritative sources.",
            "warnings": warnings,
            "claim_breakdown": [
                {
                    "sentence": a_clean,
                    "nli_label": "unverified",
                    "entailment_score": 0.0,
                    "reasoning": "Missing source context for grounding."
                }
            ],
            "metrics": {
                "faithfulness": 0.0,
                "composite_trust_score": 0.15,
                "entailed_claims": 0,
                "neutral_claims": 0,
                "contradiction_claims": 0,
                "total_claims": 1
            }
        }

    # Case 3: Source text provided — Perform sentence-level entailment check
    sentences = _split_into_sentences(a_clean)
    if not sentences:
        sentences = [a_clean]

    claim_breakdown = []
    nli_results = []

    entailed_cnt = 0
    neutral_cnt = 0
    contradiction_cnt = 0

    for sent in sentences:
        nli = check_entailment(premise=s_clean, hypothesis=sent)
        nli_results.append(nli)

        label = nli.get("label", "neutral")
        if label == "entailed":
            entailed_cnt += 1
        elif label == "contradiction":
            contradiction_cnt += 1
        else:
            neutral_cnt += 1

        claim_breakdown.append({
            "sentence": sent,
            "nli_label": label,
            "entailment_score": nli.get("entailment_score", 0.0),
            "neutral_score": nli.get("neutral_score", 0.0),
            "contradiction_score": nli.get("contradiction_score", 0.0),
            "confidence": nli.get("confidence", 0.0),
            "trust_gate": nli.get("trust_gate", "Needs Human Review"),
            "reasoning": f"NLI label: {label.upper()} (entailment={nli.get('entailment_score', 0):.2f}, contradiction={nli.get('contradiction_score', 0):.2f})"
        })

    total_claims = len(sentences)
    faithfulness = round(entailed_cnt / max(total_claims, 1), 2)
    perplexity = calculate_semantic_perplexity(nli_results)
    uncertainty_score = round(min(1.0, (1.0 - faithfulness) + (contradiction_cnt * 0.4) + (perplexity / 15.0) * 0.3) * 100, 1)
    trust_score = round(max(0.02, min(0.99, (100.0 - uncertainty_score) / 100.0)), 2)

    # Determine Reliability Tag & Short Reason
    if contradiction_cnt > 0:
        reliability_tag = "Needs Verification"
        warnings.append(f"Contradiction detected: {contradiction_cnt} claim(s) directly conflict with the source text.")
        short_reason = f"The generated answer contains facts that contradict the provided source passage ({contradiction_cnt} contradiction found)."

    elif faithfulness >= 0.80 and perplexity <= 2.80:
        reliability_tag = "Certain"
        short_reason = f"Answer is strongly supported by evidence ({int(faithfulness * 100)}% claim entailment match)."

    elif faithfulness >= 0.40:
        reliability_tag = "Uncertain"
        warnings.append("Partial evidence support: some statements lack explicit backing in the source text.")
        short_reason = f"Answer is partially supported ({entailed_cnt}/{total_claims} claims entailed). Additional verification recommended for ungrounded details."

    else:
        reliability_tag = "Needs Verification"
        warnings.append("Weak evidence backing: majority of claims are not grounded in the source text.")
        short_reason = f"Low grounding fidelity ({int(faithfulness * 100)}% entailment). Most statements cannot be corroborated by the source passage."

    return {
        "question": q_clean,
        "answer": a_clean,
        "source_text": s_clean,
        "reliability_tag": reliability_tag,
        "trust_score": trust_score,
        "perplexity": perplexity,
        "uncertainty_score": uncertainty_score,
        "short_reason": short_reason,
        "warnings": warnings,
        "claim_breakdown": claim_breakdown,
        "metrics": {
            "faithfulness": faithfulness,
            "composite_trust_score": trust_score,
            "entailed_claims": entailed_cnt,
            "neutral_claims": neutral_cnt,
            "contradiction_claims": contradiction_cnt,
            "total_claims": total_claims
        }
    }
