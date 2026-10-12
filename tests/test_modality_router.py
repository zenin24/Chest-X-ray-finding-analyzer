"""
Unit tests for Modality Classifier, Model Registry, and Model Routing.
"""

import os
from PIL import Image
import numpy as np
import pytest

from app.models.registry import ModelRegistry, ModalityStatus
from app.models.router import classify_and_route, validate_image_quality


def test_registry_modalities():
    """Verify registry categories and operational status."""
    modalities = ModelRegistry.list_all()
    assert len(modalities) >= 6

    cxr = ModelRegistry.get_modality("chest_xray")
    assert cxr is not None
    assert cxr.status == ModalityStatus.SUPPORTED
    assert "pneumonia" in cxr.supported_findings
    assert ModelRegistry.is_supported("chest_xray") is True

    bone = ModelRegistry.get_modality("bone_xray")
    assert bone is not None
    assert bone.status == ModalityStatus.PLANNED
    assert ModelRegistry.is_supported("bone_xray") is False

    knee = ModelRegistry.get_modality("knee_xray")
    assert knee is not None
    assert knee.status == ModalityStatus.PLANNED

    ct = ModelRegistry.get_modality("brain_ct")
    assert ct is not None
    assert ct.status == ModalityStatus.PLANNED


def test_image_quality_validation():
    """Verify image quality checking handles small, flat, and standard images."""
    # Tiny image (<64x64)
    tiny = Image.new("RGB", (32, 32), color="gray")
    is_valid, warnings = validate_image_quality(tiny)
    assert not is_valid
    assert any("too small" in w for w in warnings)

    # Flat image with zero contrast produces quality warning
    flat = Image.new("RGB", (256, 256), color=(128, 128, 128))
    is_valid, warnings = validate_image_quality(flat)
    assert any("contrast" in w for w in warnings)

    # Valid image with realistic gradient variance
    arr = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)
    valid_img = Image.fromarray(arr)
    is_valid, warnings = validate_image_quality(valid_img)
    assert is_valid


def test_manual_modality_override():
    """Verify manual modality selection routes supported vs unsupported categories correctly."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    # Give it texture to pass flat check
    arr = np.linspace(0, 255, 224 * 224, dtype=np.uint8).reshape((224, 224))
    img = Image.fromarray(arr).convert("RGB")

    # Manual CXR override
    res_cxr = classify_and_route(img, manual_category="chest_xray")
    assert res_cxr.manual_override_applied is True
    assert res_cxr.can_proceed_to_analysis is True
    assert res_cxr.detected_category_id == "chest_xray"
    assert res_cxr.status == ModalityStatus.SUPPORTED

    # Manual Bone override (PLANNED)
    res_bone = classify_and_route(img, manual_category="bone_xray")
    assert res_bone.manual_override_applied is True
    assert res_bone.can_proceed_to_analysis is False
    assert res_bone.detected_category_id == "bone_xray"
    assert res_bone.status == ModalityStatus.PLANNED
    assert "strictly prohibited" in res_bone.routing_message or "not currently active" in res_bone.routing_message


def test_auto_classification_on_eval_image():
    """Verify automated classification on a real sample chest radiograph."""
    sample_path = os.path.join("data", "evaluation", "images", "eval_001.png")
    assert os.path.exists(sample_path)
    img = Image.open(sample_path)

    result = classify_and_route(img)
    assert result.is_valid_image is True
    assert result.confidence > 0.0
    assert len(result.category_scores) > 0
    assert "chest_xray" in result.category_scores