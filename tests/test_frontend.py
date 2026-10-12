"""
Tests for Phase 20 frontend UI components and static file structure.
"""

import os
import pytest


def test_frontend_files_exist():
    """Verify frontend HTML, CSS, and JS files exist."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    html_path = os.path.join(base_dir, "index.html")
    css_path = os.path.join(base_dir, "styles.css")
    js_path = os.path.join(base_dir, "app.js")

    assert os.path.exists(html_path), "index.html is missing"
    assert os.path.exists(css_path), "styles.css is missing"
    assert os.path.exists(js_path), "app.js is missing"


def test_frontend_html_contents():
    """Verify HTML contains required UI sections and safety disclaimers."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    html_path = os.path.join(base_dir, "index.html")

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Title & Header
    assert "Chest X-ray Finding Analyzer" in content
    # Safety Disclaimer Banner
    assert "Not for Clinical Diagnosis" in content or "Research Prototype" in content
    # Upload Dropzone & File Input
    assert 'id="dropzone"' in content
    assert 'id="fileInput"' in content
    # Analyze Action Button
    assert 'id="analyzeBtn"' in content
    # Findings Table
    assert 'class="findings-table"' in content
    # Explainability Card
    assert 'id="explainabilityCard"' in content


def test_frontend_js_api_integration():
    """Verify app.js references /analyze endpoint and required DOM elements."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    js_path = os.path.join(base_dir, "app.js")

    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    assert "/analyze" in js_content, "app.js must call the /analyze endpoint"
    assert "prototype_confidence" in js_content, "app.js must display prototype_confidence"
    assert "findings" in js_content, "app.js must process findings array"
    assert "status-candidate" in js_content, "app.js must render status-candidate class"
    assert "status-unlikely" in js_content, "app.js must render status-unlikely class"


def test_static_files_route(client):
    """Test that static files are accessible via FastAPI TestClient."""
    response = client.get("/static/index.html")
    assert response.status_code == 200
    assert "Chest X-ray Finding Analyzer" in response.text

    css_response = client.get("/static/styles.css")
    assert css_response.status_code == 200

    js_response = client.get("/static/app.js")
    assert js_response.status_code == 200
    assert "/analyze" in js_response.text


def test_modality_intake_and_capability_components():
    """Verify 01 / INTAKE and Modality Capability modal components are in frontend files."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    html_path = os.path.join(base_dir, "index.html")
    css_path = os.path.join(base_dir, "styles.css")
    js_path = os.path.join(base_dir, "app.js")

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Panel 01 INTAKE card & tiles
    assert 'id="modalityIntakeCard"' in html
    assert '01 / INTAKE' in html
    assert 'Imaging category' in html
    assert 'id="tileChestXray"' in html
    assert 'id="tileBoneXray"' in html
    assert 'id="tileKneeXray"' in html
    assert 'id="tileCT"' in html
    assert 'id="intakeCalloutBanner"' in html
    assert 'id="intakeCalloutText"' in html
    assert 'id="intakeOfflineBanner"' in html

    # Modality Capability Modal
    assert 'id="modalityCapabilityModal"' in html
    assert 'Modality capability' in html
    assert 'id="modalLiveBadge"' in html
    assert 'Planned coverage · none available in this prototype' in html
    assert 'id="modalCapabilityBanner"' in html

    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    assert ".panel-intake" in css
    assert ".intake-category-grid" in css
    assert ".category-tile" in css
    assert ".category-tile.active" in css
    assert ".modal-capability" in css
    assert ".modality-list-vertical" in css

    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    assert "MODALITY_CATALOG" in js
    assert "selectModalityCategory" in js
    assert "modalityCapabilityModal" in js


def test_clinical_evidence_and_verification_components():
    """Verify Panel 04, RAG evidence stream, entity verification table, and report sign-off exist."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    html_path = os.path.join(base_dir, "index.html")
    css_path = os.path.join(base_dir, "styles.css")
    js_path = os.path.join(base_dir, "app.js")

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Panel 04 Evidence & Verification
    assert 'id="evidenceVerificationCard"' in html
    assert 'Clinical Evidence & Verification' in html
    assert 'id="tabRagEvidence"' in html
    assert 'id="tabClinicalVerification"' in html
    assert 'id="paneRagEvidence"' in html
    assert 'id="paneClinicalVerification"' in html
    assert 'id="evidenceCardsStream"' in html
    assert 'id="verificationSummaryBanner"' in html
    assert 'id="verificationEntityTbody"' in html
    assert 'id="auditDigestHash"' in html

    # Panel 05 & 06 renumbering check
    assert 'Spatial Attribution' in html
    assert 'Calibration & Boundary' in html

    # Enhanced Report Modal Sections 5, 6, 7 & Sign-off button
    assert 'id="rptEvidenceList"' in html
    assert 'id="rptVerificationAudit"' in html
    assert 'id="rptAuditStatus"' in html
    assert 'id="rptSignoffBox"' in html
    assert 'id="rptSignStatusBadge"' in html
    assert 'id="btnSignReport"' in html

    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    assert ".panel-evidence-verification" in css
    assert ".evidence-citation-card" in css
    assert ".verification-summary-banner" in css
    assert ".verification-table" in css
    assert ".report-signoff-card" in css
    assert ".btn-sign-cta" in css

    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    assert "renderEvidenceCitations" in js
    assert "renderVerificationAudit" in js
    assert "getFallbackEvidenceCitations" in js
    assert "getFallbackVerificationData"
    assert "btnSignReport" in js


