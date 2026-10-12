"""
Unit tests for Structured Medical Report Generation.
"""

import pytest

from app.rag.retriever import EvidenceRetriever
from app.reporting.generator import MedicalReportGenerator, StructuredMedicalReport
from app.verification.verifier import ClinicalVerifier


def test_generate_structured_report():
    """Verify structured report is populated with findings, verification, RAG, and disclaimer."""
    model_findings = [
        {"finding": "pneumonia", "similarity_score": 0.245, "threshold": 0.22, "above_threshold": True, "status": "candidate"},
        {"finding": "pneumothorax", "similarity_score": 0.170, "threshold": 0.20, "above_threshold": False, "status": "unlikely"},
        {"finding": "pleural_effusion", "similarity_score": 0.180, "threshold": 0.23, "above_threshold": False, "status": "unlikely"},
    ]
    retriever = EvidenceRetriever()
    evidence_map = retriever.link_evidence_to_findings(["pneumonia"])

    # Run verification
    report_text = "Findings suggestive of pneumonia. No evidence of pneumothorax."
    verification_audit = ClinicalVerifier.verify(model_findings, report_text, evidence_map)

    # Generate full report
    report = MedicalReportGenerator.generate_report(
        image_findings=model_findings,
        verification_report=verification_audit,
        retrieved_evidence=evidence_map,
        accession_id="ACC-TEST-001",
    )

    assert isinstance(report, StructuredMedicalReport)
    assert report.examination.accession_id == "ACC-TEST-001"
    assert report.examination.routing_status == "SUPPORTED"
    assert len(report.findings_list) == 3
    assert len(report.pertinent_negatives) >= 1
    assert "pneumonia" in report.impression.lower()
    assert report.human_review_status == "PENDING_PHYSICIAN_OVERVIEW"
    assert "RESEARCH AND EDUCATIONAL PROTOTYPE" in report.regulatory_disclaimer
    assert len(report.formatted_report_text) > 300
    assert "ACR" in report.formatted_report_text or "FLEISCH" in report.formatted_report_text


def test_generate_normal_report():
    """Verify structured report for normal scan."""
    model_findings = [
        {"finding": "normal", "similarity_score": 0.255, "threshold": 0.24, "above_threshold": True, "status": "candidate"},
        {"finding": "pneumothorax", "similarity_score": 0.150, "threshold": 0.20, "above_threshold": False, "status": "unlikely"},
    ]
    retriever = EvidenceRetriever()
    evidence_map = retriever.link_evidence_to_findings(["normal"])

    report_text = "Clear bilateral lung fields. No pneumothorax."
    verification_audit = ClinicalVerifier.verify(model_findings, report_text, evidence_map)

    report = MedicalReportGenerator.generate_report(
        image_findings=model_findings,
        verification_report=verification_audit,
        retrieved_evidence=evidence_map,
    )

    assert "Clear bilateral lung fields" in report.impression
    assert report.clinical_verification["overall_status"] == "VERIFIED"