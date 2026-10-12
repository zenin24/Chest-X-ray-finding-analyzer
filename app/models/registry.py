"""
Model Registry & Modality Configuration.

Maintains formal records of all medical imaging categories, associated models,
supported findings, clinical status (SUPPORTED, EXPERIMENTAL, PLANNED, UNSUPPORTED),
and technical capability specifications.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModalityStatus(str, Enum):
    """Support tier for medical imaging categories."""
    SUPPORTED = "SUPPORTED"
    EXPERIMENTAL = "EXPERIMENTAL"
    PLANNED = "PLANNED"
    UNSUPPORTED = "UNSUPPORTED"


class ModalityMetadata(BaseModel):
    """Metadata specification for an imaging modality / anatomical category."""
    category_id: str = Field(..., description="Unique machine-readable identifier")
    display_name: str = Field(..., description="Human-readable modality and anatomical region")
    modality_type: str = Field(..., description="Imaging technique: X-Ray, CT Scan, MRI, etc.")
    anatomical_region: str = Field(..., description="Target anatomical zone")
    status: ModalityStatus = Field(..., description="Operational readiness tier")
    active_model_name: Optional[str] = Field(None, description="Pretrained model identifier")
    model_version: Optional[str] = Field(None, description="Model release or checkpoint tag")
    supported_findings: List[str] = Field(default_factory=list, description="Labels verifiable by this model")
    zero_shot_prompt: str = Field(..., description="Disambiguation prompt for zero-shot classification")
    explainability_supported: bool = Field(False, description="Whether spatial heatmap overlay is available")
    explainability_type: Optional[str] = Field(None, description="e.g. 7x7 ViT spatial patch attribution")
    calibration_method: Optional[str] = Field(None, description="e.g. Empirical per-finding threshold calibration")
    notes: str = Field("", description="Clinical context and current development status")


# Canonical Modality Registry Definitions
MODALITY_REGISTRY: Dict[str, ModalityMetadata] = {
    "chest_xray": ModalityMetadata(
        category_id="chest_xray",
        display_name="Chest Radiography (PA/AP X-ray)",
        modality_type="Projection Radiography (X-ray)",
        anatomical_region="Thorax / Pulmonary & Cardiac",
        status=ModalityStatus.SUPPORTED,
        active_model_name="flaviagiammarino/pubmed-clip-vit-base-patch32",
        model_version="1.0.0",
        supported_findings=[
            "normal",
            "pneumonia",
            "pleural_effusion",
            "pneumothorax",
            "cardiomegaly",
            "atelectasis",
            "pulmonary_edema",
            "consolidation",
        ],
        zero_shot_prompt="a frontal projection chest radiograph showing thorax, lungs, ribs, and heart",
        explainability_supported=True,
        explainability_type="7x7 ViT-B/32 spatial patch-text cosine attribution overlay",
        calibration_method="Empirical grid sweep per-finding threshold calibration on benchmark validation split",
        notes="Fully supported research prototype pipeline with zero-shot similarity, calibration, and 7x7 patch explainability.",
    ),
    "bone_xray": ModalityMetadata(
        category_id="bone_xray",
        display_name="Bone Radiography (Extremity Fracture)",
        modality_type="Projection Radiography (X-ray)",
        anatomical_region="Extremities (Hand, Wrist, Arm, Leg)",
        status=ModalityStatus.PLANNED,
        active_model_name="MURA-DenseNet121 / Musculoskeletal Fracture Detector",
        model_version="Planned (Architecture Review)",
        supported_findings=[
            "fracture",
            "dislocation",
            "periosteal_reaction",
            "intact_cortical_bone",
        ],
        zero_shot_prompt="a plain radiograph of bone fracture in the wrist, hand, arm, tibia, or extremity",
        explainability_supported=False,
        explainability_type=None,
        calibration_method=None,
        notes="Planned modality. Candidate model under evaluation on Stanford MURA dataset. Routing prevents accidental CXR inference.",
    ),
    "knee_xray": ModalityMetadata(
        category_id="knee_xray",
        display_name="Knee Radiography (Degenerative & Fracture)",
        modality_type="Projection Radiography (X-ray)",
        anatomical_region="Knee Joint / Patellofemoral",
        status=ModalityStatus.PLANNED,
        active_model_name="OAI Kellgren-Lawrence Classifier",
        model_version="Planned",
        supported_findings=[
            "osteoarthritis_kl_grade",
            "joint_space_narrowing",
            "osteophytes",
            "patellar_fracture",
        ],
        zero_shot_prompt="a plain radiograph of the knee joint with femur, tibia, and patella",
        explainability_supported=False,
        explainability_type=None,
        calibration_method=None,
        notes="Planned modality. Requires specialized joint space analysis model and Kellgren-Lawrence grading dataset.",
    ),
    "other_xray": ModalityMetadata(
        category_id="other_xray",
        display_name="Musculoskeletal Radiography (Shoulder/Pelvis/Spine)",
        modality_type="Projection Radiography (X-ray)",
        anatomical_region="Axial Skeleton (Spine, Pelvis, Shoulder)",
        status=ModalityStatus.PLANNED,
        active_model_name="Orthopedic Multi-Region Detector",
        model_version="Planned",
        supported_findings=["fracture", "degenerative_disc_disease", "scoliosis"],
        zero_shot_prompt="a radiograph of the shoulder, hip, pelvis, cervical spine, or lumbar vertebrae",
        explainability_supported=False,
        notes="Planned modality. Awaiting multi-center orthopedic radiograph validation.",
    ),
    "brain_ct": ModalityMetadata(
        category_id="brain_ct",
        display_name="Brain Computed Tomography (Head CT)",
        modality_type="Computed Tomography (CT)",
        anatomical_region="Neuro / Cranial",
        status=ModalityStatus.PLANNED,
        active_model_name="RSNA-ICH Intracranial Hemorrhage Classifier",
        model_version="Planned",
        supported_findings=[
            "intracranial_hemorrhage",
            "subdural_hematoma",
            "subarachnoid_hemorrhage",
            "midline_shift",
        ],
        zero_shot_prompt="an axial computed tomography CT scan slice of the brain, cerebral hemispheres, and skull",
        explainability_supported=False,
        notes="Planned modality. Cross-sectional slice windowing (brain/bone/subdural) requires volumetric CT preprocessing pipeline.",
    ),
    "chest_ct": ModalityMetadata(
        category_id="chest_ct",
        display_name="Chest Computed Tomography (Thoracic CT)",
        modality_type="Computed Tomography (CT)",
        anatomical_region="Thorax (Axial Cross-Section)",
        status=ModalityStatus.PLANNED,
        active_model_name="LIDC-IDRI Volumetric Nodule Detector",
        model_version="Planned",
        supported_findings=["pulmonary_nodule", "ground_glass_opacity", "bronchiectasis"],
        zero_shot_prompt="a cross-sectional axial computed tomography CT scan of the lung parenchyma",
        explainability_supported=False,
        notes="Planned modality. 3D volumetric DICOM reconstruction required before inference.",
    ),
    "non_medical": ModalityMetadata(
        category_id="non_medical",
        display_name="Non-Medical / Non-Radiological Image",
        modality_type="Natural Photography / Document",
        anatomical_region="None",
        status=ModalityStatus.UNSUPPORTED,
        active_model_name=None,
        supported_findings=[],
        zero_shot_prompt="a natural photograph, scenery, portrait, document, chart, or non-medical graphic",
        explainability_supported=False,
        notes="Unsupported input category. Rejection protocol prevents erroneous clinical analysis.",
    ),
}


class ModelRegistry:
    """Central registry for querying and managing medical imaging models and capabilities."""

    @classmethod
    def get_modality(cls, category_id: str) -> Optional[ModalityMetadata]:
        """Fetch metadata for a given modality ID."""
        return MODALITY_REGISTRY.get(category_id)

    @classmethod
    def list_all(cls) -> List[ModalityMetadata]:
        """List all registered imaging categories and their statuses."""
        return list(MODALITY_REGISTRY.values())

    @classmethod
    def is_supported(cls, category_id: str) -> bool:
        """Check whether a category is currently operational."""
        meta = cls.get_modality(category_id)
        return meta is not None and meta.status == ModalityStatus.SUPPORTED

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """Return all category IDs with active SUPPORTED status."""
        return [k for k, v in MODALITY_REGISTRY.items() if v.status == ModalityStatus.SUPPORTED]