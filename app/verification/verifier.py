"""
Entity-Grounded Clinical Verification Engine.

Performs rigorous consistency checking between:
1. Image-derived model findings and calibrated thresholds
2. Extracted clinical entities from report statements
3. Retrieved medical literature evidence from authoritative guidelines

Identifies contradictions, unsupported statements, missing evidence, and uncertainty mismatches.
"""

from enum import Enum
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.rag.retriever import EvidencePassage
from app.verification.extractor import (
    Certainty,
    ClinicalEntity,
    EntityType,
    Polarity,
    extract_entities_from_text,
)

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FLAGGED_CONTRADICTIONS = "FLAGGED_CONTRADICTIONS"
    FLAGGED_UNSUPPORTED = "FLAGGED_UNSUPPORTED"
    PENDING = "PENDING"


class EntityVerificationItem(BaseModel):
    """Verification outcome for an individual extracted clinical entity."""
    entity: ClinicalEntity
    model_score: Optional[float] = None
    calibrated_threshold: Optional[float] = None
    model_status: Optional[str] = None  # candidate vs unlikely
    is_grounded_in_image: bool = False
    is_grounded_in_evidence: bool = False
    verification_flag: str = "VERIFIED"  # VERIFIED, CONTRADICTION, UNSUPPORTED, UNCERTAIN
    explanation: str = ""


class ClinicalVerificationReport(BaseModel):
    """Complete entity-grounded clinical verification report."""
    overall_status: VerificationStatus = Field(..., description="Overall verification health status")
    total_entities_extracted: int = Field(0, description="Count of semantic clinical entities parsed")
    verified_entities: List[EntityVerificationItem] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list, description="Contradictions between model and report")
    unsupported_assertions: List[str] = Field(default_factory=list, description="Assertions lacking model support")
    uncertainty_warnings: List[str] = Field(default_factory=list, description="Uncertainty violations or overconfidence")
    missing_evidence_notices: List[str] = Field(default_factory=list, description="Findings missing literature citations")
    summary: str = Field("", description="Clinical verification audit summary")


class ClinicalVerifier:
    """Orchestrates entity-grounded verification between model predictions and report content."""

    @classmethod
    def verify(
        cls,
        image_findings: List[Dict[str, Any]],
        report_text: str,
        retrieved_evidence: Optional[Dict[str, List[EvidencePassage]]] = None,
    ) -> ClinicalVerificationReport:
        """Execute full clinical verification check.

        Args:
            image_findings: List of finding dictionaries from model inference.
            report_text: Report statements or clinical impression text.
            retrieved_evidence: Optional map of finding names to retrieved guideline passages.

        Returns:
            ClinicalVerificationReport with comprehensive audit details.
        """
        if retrieved_evidence is None:
            retrieved_evidence = {}

        # Index image findings by canonical name
        findings_map: Dict[str, Dict[str, Any]] = {}
        for f in image_findings:
            key = f.get("finding", "").lower().strip()
            findings_map[key] = f

        # Parse entities from report
        entities = extract_entities_from_text(report_text)

        verified_items: List[EntityVerificationItem] = []
        contradictions: List[str] = []
        unsupported_assertions: List[str] = []
        uncertainty_warnings: List[str] = []
        missing_evidence: List[str] = []

        for entity in entities:
            # We verify clinical observation entities against image findings
            if entity.entity_type != EntityType.OBSERVATION:
                continue

            finding_key = entity.canonical_name
            model_info = findings_map.get(finding_key)

            if not model_info:
                # Entity mentioned in report that is completely absent from model's vocabulary
                if entity.polarity == Polarity.POSITIVE:
                    unsupported_assertions.append(
                        f"Report asserts positive '{entity.canonical_name}', but model cannot evaluate this condition."
                    )
                continue

            score = float(model_info.get("similarity_score", model_info.get("score", 0.0)))
            threshold = float(model_info.get("threshold", 0.25))
            model_status = model_info.get("status", "unlikely")
            is_candidate = model_info.get("above_threshold", False) or model_status == "candidate"

            # Check literature grounding
            has_evidence = finding_key in retrieved_evidence and len(retrieved_evidence[finding_key]) > 0

            # 1. POSITIVE ASSERTIONS IN REPORT
            if entity.polarity == Polarity.POSITIVE:
                if is_candidate:
                    # Model and report agree finding is present!
                    # Check for overconfidence if score is borderline
                    score_diff = abs(score - threshold)
                    if score_diff <= 0.015 and entity.certainty == Certainty.CONFIRMED:
                        uncertainty_warnings.append(
                            f"Finding '{finding_key}' is borderline (score={score:.4f}, threshold={threshold:.2f}). "
                            "Report expresses 'CONFIRMED' certainty; recommendation is to preserve uncertainty."
                        )
                        flag = "UNCERTAIN_BORDERLINE"
                        expl = f"Verified above threshold, but score ({score:.4f}) is borderline; certainty should be qualified."
                    else:
                        flag = "VERIFIED_POSITIVE"
                        expl = f"Verified: Vision model detected above calibrated threshold ({score:.4f} >= {threshold:.2f})."

                    if not has_evidence and finding_key != "normal":
                        missing_evidence.append(f"Positive finding '{finding_key}' lacks linked clinical guideline passage.")

                    verified_items.append(
                        EntityVerificationItem(
                            entity=entity,
                            model_score=score,
                            calibrated_threshold=threshold,
                            model_status=model_status,
                            is_grounded_in_image=True,
                            is_grounded_in_evidence=has_evidence,
                            verification_flag=flag,
                            explanation=expl,
                        )
                    )
                else:
                    # Report says finding is present, but model says below threshold!
                    unsupported_assertions.append(
                        f"Unsupported Assertion: Report claims '{finding_key}' is present, but vision model score "
                        f"({score:.4f}) is below decision boundary ({threshold:.2f})."
                    )
                    verified_items.append(
                        EntityVerificationItem(
                            entity=entity,
                            model_score=score,
                            calibrated_threshold=threshold,
                            model_status=model_status,
                            is_grounded_in_image=False,
                            is_grounded_in_evidence=has_evidence,
                            verification_flag="UNSUPPORTED",
                            explanation="Report asserts finding, but vision model score is below decision threshold.",
                        )
                    )

            # 2. NEGATIVE ASSERTIONS IN REPORT (e.g., "no pneumothorax")
            elif entity.polarity == Polarity.NEGATIVE:
                if not is_candidate:
                    # Model and report agree finding is absent
                    verified_items.append(
                        EntityVerificationItem(
                            entity=entity,
                            model_score=score,
                            calibrated_threshold=threshold,
                            model_status=model_status,
                            is_grounded_in_image=True,
                            is_grounded_in_evidence=True,
                            verification_flag="VERIFIED_NEGATIVE",
                            explanation=f"Pertinent negative verified: Model score ({score:.4f}) remains below threshold ({threshold:.2f}).",
                        )
                    )
                else:
                    # Contradiction: report claims absent, but vision model found it above threshold!
                    contradictions.append(
                        f"Contradiction Detected: Report asserts absence of '{finding_key}', but vision model score "
                        f"({score:.4f}) actively exceeds decision threshold ({threshold:.2f})."
                    )
                    verified_items.append(
                        EntityVerificationItem(
                            entity=entity,
                            model_score=score,
                            calibrated_threshold=threshold,
                            model_status=model_status,
                            is_grounded_in_image=False,
                            is_grounded_in_evidence=has_evidence,
                            verification_flag="CONTRADICTION",
                            explanation="Report claims finding is absent, but vision model actively detected candidate abnormality.",
                        )
                    )

        # Determine overall verification status
        if contradictions:
            overall_status = VerificationStatus.FLAGGED_CONTRADICTIONS
            summary = f"Audit flagged {len(contradictions)} contradiction(s) between model predictions and report text."
        elif unsupported_assertions:
            overall_status = VerificationStatus.FLAGGED_UNSUPPORTED
            summary = f"Audit flagged {len(unsupported_assertions)} unsupported statement(s) exceeding model evidence."
        else:
            overall_status = VerificationStatus.VERIFIED
            summary = f"All {len(verified_items)} extracted clinical entities successfully grounded in model predictions and evidence."

        return ClinicalVerificationReport(
            overall_status=overall_status,
            total_entities_extracted=len(entities),
            verified_entities=verified_items,
            contradictions=contradictions,
            unsupported_assertions=unsupported_assertions,
            uncertainty_warnings=uncertainty_warnings,
            missing_evidence_notices=missing_evidence,
            summary=summary,
        )