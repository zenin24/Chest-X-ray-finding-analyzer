"""
Medical Evidence Retrieval Engine (RAG).

Provides grounded retrieval of verified clinical literature and radiology guideline
passages for candidate medical findings, ensuring statements in generated reports
are explicitly linked to authoritative peer-reviewed references.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvidencePassage(BaseModel):
    """Structured medical guideline evidence passage."""
    id: str = Field(..., description="Unique evidence identifier")
    finding: str = Field(..., description="Associated finding key (e.g. pneumonia)")
    source_title: str = Field(..., description="Guideline or publication title")
    identifier: str = Field(..., description="Official standard ID: ACR, Fleischner, BTS, etc.")
    publication: str = Field(..., description="Journal or issuing authority")
    passage_text: str = Field(..., description="Exact clinical text passage")
    anatomical_focus: str = Field(..., description="Target anatomical zone")
    radiological_criteria: str = Field(..., description="Diagnostic radiographic criteria")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Differential diagnoses")


class EvidenceSearchResult(BaseModel):
    """Result of querying the evidence knowledge base."""
    passage: EvidencePassage
    relevance_score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    matched_terms: List[str] = Field(default_factory=list, description="Keywords matching query")


class EvidenceRetriever:
    """Retriever for clinical evidence from the authoritative radiology corpus."""

    _instance: Optional["EvidenceRetriever"] = None

    def __init__(self, corpus_path: Optional[str] = None):
        if corpus_path is None:
            corpus_path = os.path.join(os.path.dirname(__file__), "corpus.json")
        self.corpus_path = corpus_path
        self.passages: List[EvidencePassage] = []
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Load and parse the JSON evidence corpus."""
        if not os.path.exists(self.corpus_path):
            logger.error("Corpus file not found at '%s'", self.corpus_path)
            return

        with open(self.corpus_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        self.passages = [EvidencePassage(**item) for item in data]
        logger.info("Loaded %d medical evidence passages from '%s'", len(self.passages), self.corpus_path)

    @classmethod
    def get_instance(cls) -> "EvidenceRetriever":
        """Singleton accessor for EvidenceRetriever."""
        if cls._instance is None:
            cls._instance = EvidenceRetriever()
        return cls._instance

    def get_passages_for_finding(self, finding: str, limit: int = 2) -> List[EvidencePassage]:
        """Fetch all evidence passages matching a finding identifier."""
        norm_finding = finding.lower().strip().replace(" ", "_")
        matches = [p for p in self.passages if p.finding.lower() == norm_finding]
        return matches[:limit]

    def search(self, query: str, top_k: int = 3) -> List[EvidenceSearchResult]:
        """Lexical and semantic keyword search across passages."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        # Filter stop words
        stop_words = {"a", "an", "the", "in", "on", "of", "and", "or", "for", "with", "is", "are"}
        keywords = query_tokens - stop_words

        results = []
        for passage in self.passages:
            text_corpus = (
                f"{passage.finding} {passage.source_title} {passage.passage_text} "
                f"{passage.anatomical_focus} {passage.radiological_criteria}"
            ).lower()
            passage_tokens = set(re.findall(r"\w+", text_corpus))

            overlap = keywords.intersection(passage_tokens)
            if overlap:
                score = len(overlap) / max(len(keywords), 1)
                results.append(
                    EvidenceSearchResult(
                        passage=passage,
                        relevance_score=round(float(score), 3),
                        matched_terms=sorted(list(overlap)),
                    )
                )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    def link_evidence_to_findings(self, findings: List[str]) -> Dict[str, List[EvidencePassage]]:
        """Map a list of finding names to their corresponding clinical evidence passages."""
        evidence_map: Dict[str, List[EvidencePassage]] = {}
        for f in findings:
            passages = self.get_passages_for_finding(f)
            if passages:
                evidence_map[f] = passages
        return evidence_map