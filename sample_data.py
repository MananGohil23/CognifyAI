"""
sample_data.py — P6: AI Hallucination Confidence Labeler

Pre-configured sample Q&A pairs, source text snippets, and expected reliability labels
for quick testing during the hackathon demo.
"""

SAMPLE_SCENARIOS = [
    {
        "id": "sample_1_certain",
        "title": "Supported (Certain)",
        "question": "Who invented Python?",
        "answer": "Guido van Rossum invented Python.",
        "source_text": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
        "expected_label": "Certain",
        "category": "Technology & History",
        "notes": "Answer is fully supported by the provided source snippet."
    },
    {
        "id": "sample_2_uncertain",
        "title": "Partially Supported (Uncertain)",
        "question": "When was Python created and who maintains it today?",
        "answer": "Python was created by Guido van Rossum in 1991 and is currently maintained by Microsoft Corporation.",
        "source_text": "Python is a programming language created by Guido van Rossum and released in 1991. The Python Software Foundation (PSF) manages open-source Python development.",
        "expected_label": "Uncertain",
        "category": "Technology & History",
        "notes": "Creation date is supported, but claim about Microsoft managing Python is unsupported / inaccurate."
    },
    {
        "id": "sample_3_needs_verification",
        "title": "Unsupported / Hallucinated (Needs Verification)",
        "question": "Who invented Python and where was it created?",
        "answer": "Dennis Ritchie invented Python in 1972 at Bell Labs during the development of Unix.",
        "source_text": "Python was created by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands in the late 1980s.",
        "expected_label": "Needs Verification",
        "category": "Technology & History",
        "notes": "Direct contradiction: claims Dennis Ritchie created Python at Bell Labs."
    },
    {
        "id": "sample_4_enterprise_policy",
        "title": "Enterprise Governance Policy (Certain)",
        "question": "What is the maximum data retention period for customer PII under company policy?",
        "answer": "Customer PII must be securely purged or anonymized within 90 days of account closure.",
        "source_text": "Enterprise Data Governance Policy §4.2: All personally identifiable information (PII) belonging to former customers must be permanently anonymized or deleted within 90 calendar days following official account closure.",
        "expected_label": "Certain",
        "category": "Enterprise Policy",
        "notes": "Exact numerical constraint (90 days) matches source specification."
    },
    {
        "id": "sample_5_missing_evidence",
        "title": "Missing Evidence (Needs Verification)",
        "question": "What is the penalty for late filing of financial audit reports?",
        "answer": "The regulatory penalty for late filing is 2% of annual revenue per month of delay.",
        "source_text": "Financial Audit Guidelines 2024: Companies must submit audited financial statements within 60 days of the fiscal year-end.",
        "expected_label": "Needs Verification",
        "category": "Financial Regulatory",
        "notes": "The source specifies submission deadline but mentions no penalty rate. Answer is fabricated."
    }
]
