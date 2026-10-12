"""
Unit tests for Medical Evidence Retrieval (RAG).
"""

import pytest
from app.rag.retriever import EvidenceRetriever, EvidencePassage


def test_corpus_loading():
    """Verify evidence corpus loads completely with valid passage metadata."""
    retriever = EvidenceRetriever()
    assert len(retriever.passages) >= 9

    findings_in_corpus = {p.finding for p in retriever.passages}
    expected = {
        "normal",
        "pneumonia",
        "pleural_effusion",
        "pneumothorax",
        "cardiomegaly",
        "atelectasis",
        "pulmonary_edema",
        "consolidation",
    }
    assert expected.issubset(findings_in_corpus)


def test_get_passages_for_finding():
    """Verify specific finding retrieval returns correct guideline citations."""
    retriever = EvidenceRetriever()
    passages = retriever.get_passages_for_finding("pneumonia")
    assert len(passages) >= 1

    first = passages[0]
    assert first.finding == "pneumonia"
    assert "ACR" in first.identifier or "FLEISCH" in first.identifier
    assert len(first.passage_text) > 40
    assert first.anatomical_focus != ""


def test_keyword_search():
    """Verify search finds relevant clinical passages based on terminology."""
    retriever = EvidenceRetriever()
    results = retriever.search("cardiothoracic ratio heart enlargement", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top.passage.finding == "cardiomegaly"
    assert top.relevance_score > 0.0


def test_link_evidence_to_multiple_findings():
    """Verify mapping multiple findings to structured evidence."""
    retriever = EvidenceRetriever()
    evidence_map = retriever.link_evidence_to_findings(["pleural_effusion", "pneumothorax"])
    assert "pleural_effusion" in evidence_map
    assert "pneumothorax" in evidence_map
    assert len(evidence_map["pleural_effusion"]) >= 1
    assert any("BTS" in p.identifier for p in evidence_map["pleural_effusion"])