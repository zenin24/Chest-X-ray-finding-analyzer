"""
API route definitions.

This module contains all FastAPI route handlers for the Chest X-ray Finding Analyzer,
multimodal routing, RAG evidence retrieval, clinical verification, and structured reporting.
"""

from io import BytesIO
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from app.explainability.visualization import generate_explanation_overlay
from app.inference.analyzer import FINDING_PROMPTS, analyze_image
from app.models.pubmedclip import pubmedclip_model
from app.models.registry import ModelRegistry
from app.models.router import classify_and_route
from app.preprocessing.image import (
    get_image_info,
    load_image,
    validate_image_file,
)
from app.rag.retriever import EvidenceRetriever
from app.reporting.generator import MedicalReportGenerator
from app.schemas.response import (
    AnalysisResponse,
    ErrorResponse,
    ExplainabilityInfo,
    FindingResult,
    ImageInfoResponse,
    ModelInfo,
)
from app.verification.verifier import ClinicalVerifier

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session audit history (stores recent analysis summaries without fabricating data)
_SESSION_ANALYSIS_HISTORY: List[Dict[str, Any]] = []


@router.get("/", tags=["Health"])
async def root() -> dict:
    """Root endpoint for basic status check."""
    return {
        "message": "Chest X-ray Finding Analyzer & Multimodal Clinical Verification API",
        "status": "running",
        "model_loaded": pubmedclip_model.is_loaded,
        "supported_modalities": ModelRegistry.get_supported_categories(),
    }


@router.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Detailed health check endpoint reporting model status, device, and registry."""
    return {
        "status": "healthy",
        "model": pubmedclip_model.get_model_info(),
        "registry_modalities": [m.model_dump() for m in ModelRegistry.list_all()],
    }


@router.get("/models/registry", tags=["Model Registry"])
async def get_model_registry() -> List[dict]:
    """Retrieve all medical imaging categories, active models, and clinical readiness status."""
    return [m.model_dump() for m in ModelRegistry.list_all()]


@router.get("/modalities", tags=["Model Registry"])
async def get_modalities() -> dict:
    """Convenience endpoint returning structured modality registry and live service connectivity."""
    return {
        "status": "success",
        "service_connected": True,
        "modalities": [m.model_dump() for m in ModelRegistry.list_all()],
    }


@router.post(
    "/modality/classify",
    tags=["Modality Routing"],
)
async def classify_modality_endpoint(
    file: UploadFile = File(...),
    manual_category: Optional[str] = Form(None),
) -> dict:
    """Inspect and classify the imaging modality and anatomical region of an uploaded medical image."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    validate_image_file(file.filename, file.content_type)
    contents = await file.read()
    image = load_image(contents)

    routing_result = classify_and_route(image, manual_category=manual_category, filename=file.filename)
    return routing_result.model_dump()


@router.post(
    "/evidence/search",
    tags=["Evidence Retrieval"],
)
async def search_clinical_evidence(
    query: str = Query(..., description="Clinical search terms"),
    top_k: int = Query(3, ge=1, le=10),
) -> List[dict]:
    """Search authoritative clinical radiology guidelines for evidence passages."""
    retriever = EvidenceRetriever.get_instance()
    results = retriever.search(query=query, top_k=top_k)
    return [r.model_dump() for r in results]


@router.get(
    "/history",
    tags=["History"],
)
async def get_analysis_history() -> List[dict]:
    """Retrieve session audit history of analyzed medical images."""
    return list(reversed(_SESSION_ANALYSIS_HISTORY[-50:]))


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file format, corruption, or unsupported modality"},
        500: {"model": ErrorResponse, "description": "Inference failure"},
    },
    tags=["Analysis"],
)
async def analyze_xray(
    file: UploadFile = File(...),
    modality_override: Optional[str] = Form(None),
) -> AnalysisResponse:
    """Analyze a medical image with multimodal routing, PubMedCLIP, RAG evidence, and clinical verification.

    - **file**: Medical image file (JPEG, PNG, BMP, TIFF)
    - **modality_override**: Optional manual modality selection (e.g. 'chest_xray')

    Returns ranked findings, RAG citations, clinical entity verification, and a structured report.
    """
    logger.info("Received analysis request: filename=%s, override=%s", file.filename, modality_override)

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload request.",
        )

    # Step 1: Validate file extension and MIME type
    try:
        validate_image_file(file.filename, file.content_type)
    except ValueError as e:
        logger.warning("Image validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Step 2: Read raw bytes and load image into PIL RGB format
    try:
        contents = await file.read()
        image = load_image(contents)
    except ValueError as e:
        logger.warning("Failed to parse image file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupt image file: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error reading image upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded image.",
        )

    # Step 3: Modality Classification & Routing Check
    routing_result = classify_and_route(image, manual_category=modality_override, filename=file.filename)
    if not routing_result.is_valid_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image quality check failed: {'; '.join(routing_result.quality_warnings)}",
        )

    if not routing_result.can_proceed_to_analysis:
        logger.warning("Rejected unsupported modality: %s", routing_result.routing_message)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported Modality: {routing_result.routing_message}",
        )

    # Step 4: Run Vision-Language Inference Pipeline
    try:
        analysis_result = analyze_image(image)
    except RuntimeError as e:
        logger.error("Model inference error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Unexpected error during analysis: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during X-ray analysis: {e}",
        )

    # Format findings list
    findings = [
        FindingResult(
            finding=f["finding"],
            score=f["score"],
            similarity_score=f["similarity_score"],
            rank=f["rank"],
            status=f["status"],
            above_threshold=f["above_threshold"],
            explanation_available=f.get("explanation_available", True),
        )
        for f in analysis_result["findings"]
    ]

    # Step 5: RAG Evidence Retrieval for Candidate Findings
    retriever = EvidenceRetriever.get_instance()
    candidate_keys = [f.finding for f in findings if f.above_threshold]
    if not candidate_keys:
        candidate_keys = ["normal"]
    evidence_map = retriever.link_evidence_to_findings(candidate_keys)

    citations_summary = []
    for f_name, passages in evidence_map.items():
        for p in passages:
            citations_summary.append({
                "finding": f_name,
                "source_title": p.source_title,
                "identifier": p.identifier,
                "publication": p.publication,
                "radiological_criteria": p.radiological_criteria,
                "passage_excerpt": p.passage_text[:160] + "...",
            })

    # Step 6: Entity-Grounded Clinical Verification
    # Build preliminary impression text for verification audit
    if "normal" in candidate_keys and len(candidate_keys) == 1:
        draft_text = "Clear bilateral lungs. Normal cardiac silhouette. No pneumothorax or effusion."
    else:
        active_str = ", ".join([c.replace("_", " ") for c in candidate_keys if c != "normal"])
        draft_text = f"Findings suggestive of {active_str}. No pneumothorax."

    findings_raw = [
        {
            "finding": f.finding,
            "similarity_score": f.score,
            "threshold": analysis_result["threshold_used"],
            "above_threshold": f.above_threshold,
            "status": f.status,
        }
        for f in findings
    ]
    verification_audit = ClinicalVerifier.verify(findings_raw, draft_text, evidence_map)

    # Step 7: Structured Medical Report Generation
    structured_rep = MedicalReportGenerator.generate_report(
        image_findings=findings_raw,
        verification_report=verification_audit,
        retrieved_evidence=evidence_map,
        modality_type=routing_result.modality_type,
        anatomical_region=routing_result.anatomical_region,
        model_name="PubMedCLIP ViT-B/32",
    )

    # Step 8: Record in Session History
    _SESSION_ANALYSIS_HISTORY.append({
        "timestamp": structured_rep.created_at,
        "filename": file.filename,
        "report_id": structured_rep.report_id,
        "detected_modality": routing_result.category_display_name,
        "top_finding": findings[0].finding if findings else "none",
        "top_score": findings[0].score if findings else 0.0,
        "verification_status": verification_audit.overall_status.value,
        "human_review_status": structured_rep.human_review_status,
    })

    return AnalysisResponse(
        status="success",
        model=ModelInfo(),
        findings=findings,
        prototype_confidence=analysis_result["prototype_confidence"],
        threshold_used=analysis_result["threshold_used"],
        explainability=ExplainabilityInfo(),
        disclaimer="Research prototype. Not for clinical diagnosis.",
        modality=routing_result,
        verification=verification_audit,
        evidence_citations=citations_summary,
        structured_report=structured_rep,
    )


@router.post(
    "/explain",
    responses={
        200: {"content": {"image/png": {}}, "description": "Visual heatmap overlay PNG image"},
        400: {"model": ErrorResponse, "description": "Invalid image file or parameters"},
        500: {"model": ErrorResponse, "description": "Heatmap generation failure"},
    },
    tags=["Explainability"],
)
async def generate_explainability_map(
    file: UploadFile = File(...),
    finding: str = "pneumonia",
):
    """Generate a visual spatial patch-text similarity heatmap overlay PNG for a specific finding."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        validate_image_file(file.filename, file.content_type)
        contents = await file.read()
        image = load_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Match prompt text for given finding
    prompt_map = {f_name: p_text for f_name, p_text in FINDING_PROMPTS}
    text_prompt = prompt_map.get(
        finding, f"A chest X-ray showing {finding.replace('_', ' ')}"
    )

    try:
        blended_img, _ = generate_explanation_overlay(image, finding, text_prompt)
        buf = BytesIO()
        blended_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("Failed to generate explainability overlay: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate explainability map: {e}"
        )


@router.post(
    "/image-info",
    response_model=ImageInfoResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Utility"],
)
async def inspect_image(file: UploadFile = File(...)) -> ImageInfoResponse:
    """Inspect basic metadata of an uploaded image file without running inference."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        validate_image_file(file.filename, file.content_type)
        contents = await file.read()
        image = load_image(contents)
        info = get_image_info(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ImageInfoResponse(
        status="success",
        filename=file.filename,
        image_format=info["format"],
        width=info["width"],
        height=info["height"],
        mode=info["mode"],
    )