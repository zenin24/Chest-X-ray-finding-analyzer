"""
Unit tests for Entity-Grounded Clinical Verification.
"""

import pytest

from app.rag.retriever import EvidenceRetriever
from app.verification.extractor import (
    Certainty,
    Polarity,
    extract_entities_from_text,
)
from app.verification.verifier import ClinicalVerifier, VerificationStatus


def test_entity_extraction_positive_and_negative():
    """Verify entity extractor captures findings, anatomy, polarity, and certainty."""
    text = (
        "There is patchy opacity suggestive of pneumonia in the lung fields. "
        "No evidence of pneumothorax or pleural effusion. Heart size is normal."
    )
    entities = extract_entities_from_text(text)
    assert len(entities) >= 3

    names = {e.canonical_name: e for e in entities}
    assert "pneumonia" in names
    assert names["pneumonia"].polarity == Polarity.POSITIVE
    assert names["pneumonia"].certainty in (Certainty.PROBABLE, Certainty.POSSIBLE)

    assert "pneumothorax" in names
    assert names["pneumothorax"].polarity == Polarity.NEGATIVE
    assert names["pneumothorax"].certainty == Certainty.ABSENT

    assert "pleural_effusion" in names
    assert names["pleural_effusion"].polarity == Polarity.NEGATIVE


def test_clinical_verification_concordant():
    """Verify that concordant findings between model and report pass verification."""
    model_findings = [
        {"finding": "pneumonia", "similarity_score": 0.245, "threshold": 0.22, "above_threshold": True, "status": "candidate"},
        {"finding": "pneumothorax", "similarity_score": 0.170, "threshold": 0.20, "above_threshold": False, "status": "unlikely"},
    ]
    report_text = "Findings suggestive of pneumonia. No evidence of pneumothorax."
    retriever = EvidenceRetriever()
    evidence_map = retriever.link_evidence_to_findings(["pneumonia"])

    audit = ClinicalVerifier.verify(model_findings, report_text, evidence_map)
    assert audit.overall_status == VerificationStatus.VERIFIED
    assert len(audit.contradictions) == 0
    assert len(audit.unsupported_assertions) == 0
    assert len(audit.verified_entities) == 2


def test_clinical_verification_detects_contradiction():
    """Verify contradiction is caught when report denies a finding that model detected."""
    model_findings = [
        {"finding": "pneumonia", "similarity_score": 0.250, "threshold": 0.22, "above_threshold": True, "status": "candidate"},
    ]
    # Report falsely claims NO pneumonia
    report_text = "No signs of pneumonia. Clear lungs."

    audit = ClinicalVerifier.verify(model_findings, report_text)
    assert audit.overall_status == VerificationStatus.FLAGGED_CONTRADICTIONS
    assert len(audit.contradictions) >= 1
    assert any("pneumonia" in c for c in audit.contradictions)


def test_clinical_verification_detects_unsupported():
    """Verify unsupported statement is caught when report claims finding that model did not detect."""
    model_findings = [
        {"finding": "consolidation", "similarity_score": 0.160, "threshold": 0.21, "above_threshold": False, "status": "unlikely"},
    ]
    # Report claims consolidation is present without model support
    report_text = "There is dense consolidation in the lung."

    audit = ClinicalVerifier.verify(model_findings, report_text)
    assert audit.overall_status == VerificationStatus.FLAGGED_UNSUPPORTED
    assert len(audit.unsupported_assertions) >= 1
    assert any("consolidation" in u for u in audit.unsupported_assertions)