"""
Automated Structured Medical Report Generator.

Synthesizes multimodal image findings, pertinent negatives, clinical entity verification,
retrieved guideline literature, and uncertainty qualifications into hospital-grade
structured clinical reports (both JSON schema and readable radiology report text).
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.rag.retriever import EvidencePassage
from app.verification.verifier import ClinicalVerificationReport

logger = logging.getLogger(__name__)


# Standard Anatomical Locations for Supported Findings
FINDING_ANATOMY = {
    "normal": "Bilateral lung fields, pleural spaces, and cardiac silhouette",
    "pneumonia": "Bilateral pulmonary parenchyma / alveolar airspaces",
    "pleural_effusion": "Bilateral costophrenic angles and dependent pleural spaces",
    "pneumothorax": "Apicolateral pleural spaces and chest wall margins",
    "cardiomegaly": "Cardiac silhouette and cardiothoracic diameter ratio",
    "atelectasis": "Subsegmental lung bases and interlobar fissures",
    "pulmonary_edema": "Perihilar bronchovascular bundles and basilar interstitium",
    "consolidation": "Segmental pulmonary parenchyma and alveolar units",
}

FINDING_DISPLAY_NAMES = {
    "normal": "Normal Aerated Parenchyma",
    "pneumonia": "Infectious Pneumonia / Infiltrate",
    "pleural_effusion": "Pleural Effusion",
    "pneumothorax": "Pneumothorax",
    "cardiomegaly": "Cardiomegaly",
    "atelectasis": "Atelectasis / Volume Loss",
    "pulmonary_edema": "Pulmonary Edema / Vascular Congestion",
    "consolidation": "Airspace Consolidation",
}


class ExaminationMetadata(BaseModel):
    """Imaging examination technical metadata."""
    accession_id: str
    examination_date: str
    modality_type: str
    anatomical_region: str
    projection_view: str = "Frontal (PA/AP)"
    model_name: str
    model_version: str = "1.0.0"
    routing_status: str = "SUPPORTED"


class ReportFindingItem(BaseModel):
    """Itemized radiological finding in structured report."""
    finding_id: str
    finding_name: str
    display_name: str
    anatomical_location: str
    status: str  # candidate vs unlikely
    score: float
    calibrated_threshold: float
    confidence_interpretation: str
    above_threshold: bool
    spatial_explainability_available: bool = True


class StructuredMedicalReport(BaseModel):
    """Complete multimodal clinical research report schema."""
    report_id: str
    created_at: str
    examination: ExaminationMetadata
    clinical_history: str = "Diagnostic research analysis."
    findings_list: List[ReportFindingItem] = Field(default_factory=list)
    pertinent_negatives: List[str] = Field(default_factory=list)
    impression: str
    clinical_verification: Dict[str, Any]
    evidence_citations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_and_uncertainty_note: str
    limitations: List[str] = Field(default_factory=list)
    human_review_status: str = "PENDING_PHYSICIAN_OVERVIEW"
    ai_status_notice: str = "AI-GENERATED CANDIDATE OBSERVATIONS - NOT A CONFIRMED DIAGNOSIS"
    regulatory_disclaimer: str = (
        "RESEARCH AND EDUCATIONAL PROTOTYPE ONLY. All confidence and similarity scores "
        "represent normalized vector dot-products in PubMedCLIP joint embedding space and are "
        "NOT clinically validated diagnostic probabilities. This report must not replace qualified "
        "medical interpretation or licensed physician evaluation."
    )
    formatted_report_text: str = ""


class MedicalReportGenerator:
    """Constructs structured medical reports from analysis and verification pipelines."""

    @classmethod
    def generate_report(
        cls,
        image_findings: List[Dict[str, Any]],
        verification_report: ClinicalVerificationReport,
        retrieved_evidence: Dict[str, List[EvidencePassage]],
        modality_type: str = "Projection Radiography (X-ray)",
        anatomical_region: str = "Thorax / Pulmonary & Cardiac",
        model_name: str = "PubMedCLIP (flaviagiammarino/pubmed-clip-vit-base-patch32)",
        accession_id: Optional[str] = None,
    ) -> StructuredMedicalReport:
        """Synthesize analysis outputs into a complete structured clinical report."""
        now = datetime.now()
        report_id = f"RAD-REP-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"
        if not accession_id:
            accession_id = f"ACC-{now.strftime('%Y%m%d%H%M%S')}"

        exam_meta = ExaminationMetadata(
            accession_id=accession_id,
            examination_date=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            modality_type=modality_type,
            anatomical_region=anatomical_region,
            projection_view="Frontal (PA/AP)",
            model_name=model_name,
            routing_status="SUPPORTED",
        )

        findings_items: List[ReportFindingItem] = []
        pertinent_negatives: List[str] = []
        candidate_findings = []

        for f in image_findings:
            key = f.get("finding", "").lower()
            score = float(f.get("similarity_score", f.get("score", 0.0)))
            threshold = float(f.get("threshold", 0.25))
            status = f.get("status", "unlikely")
            above_threshold = f.get("above_threshold", False) or status == "candidate"

            # Formulate clinical interpretation
            margin = score - threshold
            if above_threshold:
                if margin >= 0.03:
                    interp = f"Definite candidate finding (Score {score:.4f} > Calibrated threshold {threshold:.2f})."
                else:
                    interp = f"Borderline candidate finding (Score {score:.4f} near Calibrated threshold {threshold:.2f}). Preserves uncertainty."
                candidate_findings.append(key)
            else:
                interp = f"Unlikely (Score {score:.4f} below Calibrated threshold {threshold:.2f})."

            item = ReportFindingItem(
                finding_id=f"FIND-{key.upper()}",
                finding_name=key,
                display_name=FINDING_DISPLAY_NAMES.get(key, key.replace("_", " ").title()),
                anatomical_location=FINDING_ANATOMY.get(key, "Thoracic cavity"),
                status=status,
                score=round(score, 4),
                calibrated_threshold=round(threshold, 2),
                confidence_interpretation=interp,
                above_threshold=above_threshold,
                spatial_explainability_available=True,
            )
            findings_items.append(item)

            # Record pertinent negative statements for critical acute pathologies
            if not above_threshold and key in ("pneumothorax", "pleural_effusion", "pulmonary_edema"):
                if key == "pneumothorax":
                    pertinent_negatives.append("No pneumothorax: Pleural lines intact without peripheral hyperlucency.")
                elif key == "pleural_effusion":
                    pertinent_negatives.append("No pleural effusion: Costophrenic sulci remain sharp and acute bilaterally.")
                elif key == "pulmonary_edema":
                    pertinent_negatives.append("No vascular congestion or alveolar pulmonary edema identified.")

        # Build evidence citations list
        citations_list = []
        for f_key, passages in retrieved_evidence.items():
            for p in passages:
                citations_list.append({
                    "finding": f_key,
                    "source_title": p.source_title,
                    "identifier": p.identifier,
                    "publication": p.publication,
                    "radiological_criteria": p.radiological_criteria,
                    "passage_excerpt": p.passage_text[:160] + "...",
                })

        # Formulate clinical impression
        if not candidate_findings or (len(candidate_findings) == 1 and candidate_findings[0] == "normal"):
            impression = (
                "IMPRESSION: Clear bilateral lung fields without focal consolidation, pneumothorax, "
                "or pleural effusion. Cardiothoracic silhouette within normal limits. No acute cardiopulmonary abnormality."
            )
        else:
            cand_names = [FINDING_DISPLAY_NAMES.get(c, c) for c in candidate_findings if c != "normal"]
            cand_str = ", ".join(cand_names)
            neg_str = " ".join(pertinent_negatives[:2])
            impression = (
                f"IMPRESSION: Radiographic features suggestive of {cand_str}. "
                f"{neg_str} Recommend clinical correlation and comparison with prior radiographs."
            )

        # Limitations list
        limitations = [
            "Coarse 7x7 spatial patch resolution: small focal micro-lesions (<32x32 px) cannot be localized with sub-patch precision.",
            "Single frontal view evaluation: lack of lateral radiograph restricts assessment of retrocardiac space and thoracic spine.",
            "Zero-shot cosine similarity: reflect semantic vector alignment in PubMedCLIP latent space, not true Bayesian posterior disease probabilities.",
        ]

        conf_note = (
            "Confidence Interpretation: Similarity scores are normalized dot-products calibrated against empirical "
            "operating thresholds derived from a multi-label validation sweep. Higher scores signify stronger multimodal "
            "semantic alignment between image features and controlled clinical prompt vectors."
        )

        # Generate formatted radiology report text
        formatted_text = cls._format_readable_text(
            report_id=report_id,
            meta=exam_meta,
            findings_items=findings_items,
            pertinent_negatives=pertinent_negatives,
            impression=impression,
            verification_report=verification_report,
            citations=citations_list,
            limitations=limitations,
        )

        return StructuredMedicalReport(
            report_id=report_id,
            created_at=exam_meta.examination_date,
            examination=exam_meta,
            findings_list=findings_items,
            pertinent_negatives=pertinent_negatives,
            impression=impression,
            clinical_verification=verification_report.model_dump(),
            evidence_citations=citations_list,
            confidence_and_uncertainty_note=conf_note,
            limitations=limitations,
            human_review_status="PENDING_PHYSICIAN_OVERVIEW",
            formatted_report_text=formatted_text,
        )

    @classmethod
    def _format_readable_text(
        cls,
        report_id: str,
        meta: ExaminationMetadata,
        findings_items: List[ReportFindingItem],
        pertinent_negatives: List[str],
        impression: str,
        verification_report: ClinicalVerificationReport,
        citations: List[Dict[str, Any]],
        limitations: List[str],
    ) -> str:
        """Render a readable clinical report document."""
        lines = [
            "=" * 72,
            "DEPARTMENT OF RADIOLOGY & CLINICAL AI RESEARCH",
            "STRUCTURED MEDICAL REPORT (AI-ASSISTED PRELIMINARY AUDIT)",
            "=" * 72,
            f"REPORT ID:         {report_id}",
            f"DATE & TIME:       {meta.examination_date}",
            f"ACCESSION ID:      {meta.accession_id}",
            f"EXAMINATION:       {meta.modality_type} - {meta.anatomical_region}",
            f"PROJECTION:        {meta.projection_view}",
            f"ANALYSIS MODEL:    {meta.model_name}",
            f"REVIEW STATUS:     PENDING QUALIFIED RADIOLOGIST OVERVIEW",
            "-" * 72,
            "",
            "CLINICAL FINDINGS:",
        ]

        for item in findings_items:
            flag = "[POSITIVE CANDIDATE]" if item.above_threshold else "[UNLIKELY / NEGATIVE]"
            lines.append(
                f"  * {item.display_name:<34} {flag:<22} "
                f"(Score: {item.score:.4f} | Calibrated Cutoff: {item.calibrated_threshold:.2f})"
            )
            lines.append(f"    Location: {item.anatomical_location}")
            lines.append(f"    Interpretation: {item.confidence_interpretation}")

        if pertinent_negatives:
            lines.append("")
            lines.append("PERTINENT NEGATIVE OBSERVATIONS:")
            for neg in pertinent_negatives:
                lines.append(f"  * {neg}")

        lines.extend([
            "",
            "-" * 72,
            impression,
            "-" * 72,
            "",
            "CLINICAL ENTITY VERIFICATION AUDIT:",
            f"  * Status: {verification_report.overall_status.value}",
            f"  * Entities Evaluated: {verification_report.total_entities_extracted}",
            f"  * Summary: {verification_report.summary}",
        ])

        if verification_report.contradictions:
            lines.append("  * CONTRADICTIONS FLAGGED:")
            for c in verification_report.contradictions:
                lines.append(f"    - {c}")

        if verification_report.uncertainty_warnings:
            lines.append("  * UNCERTAINTY WARNINGS:")
            for w in verification_report.uncertainty_warnings:
                lines.append(f"    - {w}")

        if citations:
            lines.extend([
                "",
                "EVIDENCE-BASED GUIDELINE REFERENCES (RAG):",
            ])
            for cit in citations[:3]:
                lines.append(f"  * [{cit['identifier']}] {cit['source_title']}")
                lines.append(f"    Criteria: {cit['radiological_criteria']}")

        lines.extend([
            "",
            "TECHNICAL LIMITATIONS & UNCERTAINTY:",
        ])
        for lim in limitations:
            lines.append(f"  - {lim}")

        lines.extend([
            "",
            "=" * 72,
            "CAUTION & REGULATORY DISCLAIMER:",
            "RESEARCH & EDUCATIONAL PROTOTYPE ONLY. NOT FDA OR CE CLEARED.",
            "All findings require verification and sign-off by a qualified licensed radiologist.",
            "=" * 72,
        ])

        return "\n".join(lines)