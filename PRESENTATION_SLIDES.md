# CHEST X-RAY FINDING ANALYZER: ZERO-SHOT MEDICAL VISION-LANGUAGE ANALYSIS
## 15-Slide Presentation Outline & Slide Content

---

### Slide 1: Title & Project Overview
- **Project Title:** Chest X-ray Finding Analyzer
- **Subtitle:** Zero-Shot Medical Vision-Language Analysis, Spatial Attribution, and Threshold Calibration
- **Foundation Model:** PubMedCLIP (`flaviagiammarino/pubmed-clip-vit-base-patch32`)
- **Backend & Frontend:** FastAPI (Asynchronous ASGI) + RAD·LAB Research Console (HTML5/CSS3/ES6)
- **Key Focus:** Open-vocabulary finding detection, $7 \times 7$ ViT patch attribution, and calibrated zero-shot decision boundaries.
- **Academic Disclaimer:** Research & Educational Prototype — Not for Clinical Diagnosis.

---

### Slide 2: Clinical Context & Problem Statement
- **Clinical Challenge:** High global volume of chest radiographs; interpretation backlogs lead to triage delays in emergency and outpatient care.
- **Deep Learning Bottleneck:**
  - Traditional supervised CNNs (DenseNet, ResNet) require hundreds of thousands of expert-annotated radiographs.
  - Closed-set vocabulary: Cannot evaluate any finding outside fixed training classes without complete retraining.
  - Black-box opacity: Lack intuitive spatial reasoning grounded in medical text.
- **Core Research Question:** Can biomedical vision-language foundation models enable zero-shot finding candidate ranking with visual spatial explainability and calibrated decision thresholds?

---

### Slide 3: Existing Systems vs. Proposed Solution
- **Existing Supervised Systems:**
  - Fixed output dimension (e.g., 14 sigmoid heads).
  - Unimodal visual representations; isolated from biomedical literature.
  - Retraining required for new clinical findings.
- **Proposed PubMedCLIP System:**
  - Joint 512-dimensional multimodal latent space.
  - Open-vocabulary query capability via natural language clinical prompts.
  - Direct spatial patch attribution from Vision Transformer backbone.
  - Empirical multi-label threshold calibration resolving baseline shift.

---

### Slide 4: Foundation Model — PubMedCLIP Architecture
- **Model Checkpoint:** `flaviagiammarino/pubmed-clip-vit-base-patch32`
- **Pre-training Domain:** Fine-tuned on biomedical image-text pairs extracted from PubMed Central articles.
- **Dual Encoder Architecture:**
  - **Vision Encoder (ViT-B/32):** Divides $224 \times 224$ images into 49 non-overlapping $32 \times 32$ spatial patches + 1 CLS token.
  - **Text Encoder:** Multi-layer Transformer tokenizing and embedding clinical prompts.
- **Latent Alignment:** Linear projection layers map both visual and textual features into an $L_2$-normalized 512-dimensional unit hypersphere.

---

### Slide 5: End-to-End System Architecture
- **Presentation Layer:** RAD·LAB Research Console & Interactive Swagger UI (`/docs`).
- **Web API Layer:** FastAPI ASGI server with CORS middleware and structured Pydantic v2 schemas.
- **Preprocessing Pipeline:** Format validation (PNG, JPEG, TIFF, BMP), RGB standardization, ImageNet tensor normalization.
- **Inference Pipeline:** Pre-computed prompt embeddings, PyTorch matrix dot-product ranking, prototype confidence extraction.
- **Explainability Pipeline:** Spatial ViT patch extraction, localized dot-product attribution, bilinear interpolation, alpha blending.
- **Evaluation Pipeline:** Benchmark dataset ($N=16$) with strict validation/test split separation.

---

### Slide 6: Controlled Medical Prompt Engineering
Standardized clinical prompts formulated for 8 cardinal radiographic findings:
1. **Normal:** *"A chest X-ray showing a normal healthy lung"*
2. **Pneumonia:** *"A chest X-ray showing pneumonia"*
3. **Pleural Effusion:** *"A chest X-ray showing pleural effusion"*
4. **Pneumothorax:** *"A chest X-ray showing pneumothorax"*
5. **Cardiomegaly:** *"A chest X-ray showing cardiomegaly"*
6. **Atelectasis:** *"A chest X-ray showing atelectasis"*
7. **Pulmonary Edema:** *"A chest X-ray showing pulmonary edema"*
8. **Consolidation:** *"A chest X-ray showing consolidation"*
- **Optimization:** Text embeddings ($8 \times 512$) are computed once during startup and cached in memory.

---

### Slide 7: Input Validation & Image Preprocessing
- **Extension & MIME Validation:** Enforces accepted formats (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`) and rejects corrupted byte sequences.
- **Color Mode Normalization:** Safely converts single-channel Grayscale, 4-channel RGBA, or CMYK inputs into 3-channel 8-bit RGB.
- **Tensor Preprocessing:**
  - Resized to $224 \times 224$ pixels via bilinear interpolation.
  - Scaled to $[0, 1]$ floating-point range.
  - Standardized with ImageNet statistics: $\mu = [0.481, 0.457, 0.408]$, $\sigma = [0.268, 0.261, 0.275]$.

---

### Slide 8: Zero-Shot Cosine Similarity & Ranking Engine
- **Mathematical Principle:** Because visual vector $\mathbf{v}$ and text vector $\mathbf{t}_k$ are unit-normalized ($\|\mathbf{v}\|_2 = \|\mathbf{t}_k\|_2 = 1$), cosine similarity equals the vector dot-product:
  $$S_k = \mathbf{v}^\top \mathbf{t}_k = \sum_{j=1}^{512} v_j \cdot t_{k, j} \in [-1, 1]$$
- **Rank Ordering:** Sorted in descending order of similarity score.
- **Prototype Confidence:** Maximum similarity score across all candidate findings:
  $$C_{\text{proto}} = \max_k S_k$$
- **Classification Status:** Labeled as `Candidate` (score $\ge \tau$) vs `Unlikely` (score $< \tau$).

---

### Slide 9: Visual Spatial Explainability (ViT Patch Attribution)
- **Patch Extraction:** Extracts 49 spatial patch representations from ViT-B/32 `last_hidden_state` (excluding CLS token at index 0).
- **Multimodal Projection:** Multiplies patch tokens by visual projection weights $W_{\text{proj}}$ to enter the 512-dim joint space.
- **Patch-Text Cosine Similarity:** Computes dot-product between each of the 49 patch vectors and the target finding prompt vector.
- **Visualization:**
  - Reshaped to $7 \times 7$ 2D spatial grid.
  - Min-Max normalized to range $[0, 1]$.
  - Upsampled via bilinear interpolation to match radiograph resolution.
  - Alpha-blended ($\alpha = 0.45$) with Jet colormap overlay and streamed as PNG.

---

### Slide 10: Benchmark Evaluation Dataset & Split Design
- **Dataset Structure:** 16 curated chest radiographs (`eval_001.png` to `eval_016.png`) with verified multi-label ground truth annotations.
- **Finding Representation:** Balanced ground truth coverage across all 8 target conditions.
- **Strict Partitioning (No Data Leakage):**
  - **Validation Split (`val`, $N=8$):** Exclusively utilized for grid sweep and optimal threshold parameter discovery.
  - **Test Split (`test`, $N=8$):** Completely held out to evaluate true generalization.

---

### Slide 11: Quantitative Evaluation & Metric Analysis
- **Uncalibrated Baseline (Default Threshold $\tau = 0.25$):**
  - Raw similarity scores cluster between $0.20$ and $0.24$.
  - Fixed $\tau = 0.25$ threshold results in zero true positives on the held-out test split.
  - Test Macro Precision: $0.0000$ | Recall: $0.0000$ | F1-Score: $0.0000$.
- **Per-Finding ROC-AUC Performance:**
  - Cardiomegaly: **$1.0000$** | Pneumonia: **$0.8571$** | Pleural Effusion: **$0.8571$**
  - Normal: $0.5714$ | Atelectasis: $0.4286$ | Consolidation: $0.2857$ | Pneumothorax: $0.1429$.

---

### Slide 12: Decision Threshold Calibration Pipeline
- **Calibration Methodology:** Systematic grid sweep over range $[0.15, 0.40]$ with step $0.01$ on Validation split.
- **Optimal Discovered Thresholds:**
  - Global Macro Optimal: $\tau^* = 0.2000$.
  - Dynamic Per-Finding: `pulmonary_edema` ($0.15$), `pneumothorax` ($0.20$), `consolidation` ($0.21$), `pneumonia` ($0.22$), `atelectasis` ($0.22$), `cardiomegaly` ($0.23$), `pleural_effusion` ($0.23$), `normal` ($0.24$).
- **Held-Out Test Set Improvement:**
  - Uncalibrated ($\tau = 0.25$): F1 = **$0.0000$**
  - Calibrated Macro ($\tau = 0.20$): F1 = **$0.2292$** (Recall = $1.0000$)
  - **Calibrated Per-Finding (Dynamic):** F1 = **$0.3843$** (Recall = $1.0000$, Specificity = $0.4107$).

---

### Slide 13: RAD·LAB Interactive Web Console
- **Frontend Architecture:** Vanilla HTML5, CSS3, and JavaScript (ES6+), zero heavy build dependencies.
- **Visual Design:** Dark-mode clinical PACS aesthetic with glassmorphic cards, KPI summary ribbon, and micro-animations.
- **Core Interactive Features:**
  - Drag-and-drop file upload with format check and `/image-info` dimension inspection.
  - "Demo Sample" instant synthetic radiograph generation for rapid demonstration.
  - Live ranked findings table with color-coded rank pills and status badges.
  - Dual explainability viewer: Toggle between $7 \times 7$ Patch Attention Matrix and real Grad-CAM style heatmap overlays.
  - Interactive threshold sweep bar with dynamic candidate recount.

---

### Slide 14: System Testing & Production Reliability
- **Automated Verification:** 38 automated test cases across 8 test suites executed via `pytest`:
  - `test_api.py`: Route status, payloads, and response schemas.
  - `test_preprocessing.py`: Multi-format image loading and channel conversion.
  - `test_pubmedclip.py`: Model singleton, embedding dimensions, and dot-products.
  - `test_calibration.py`: Sweep logic and optimal parameter selection.
  - `test_explainability.py`: $7 \times 7$ spatial heatmap generation.
  - `test_frontend.py`: Static file routes and UI element integrity.
  - `test_system_complete.py`: Multi-step end-to-end inference workflows.
- **Result:** **38 Passed, 0 Failed** in 11.60 seconds.

---

### Slide 15: Limitations, Safety & Future Enhancements
- **Known Limitations:**
  - Coarse $7 \times 7$ patch grid limits localization of tiny focal lesions.
  - Modest benchmark dataset ($N=16$) requires broader clinical validation.
  - Dot-product similarities are latent distances, not Bayesian diagnostic probabilities.
- **Safety Disclaimers:** Strictly an educational and research prototype. Not approved by FDA/CE.
- **Future Roadmap:**
  - High-resolution vision transformers (ViT-L/14, BioViL).
  - Bi-lateral multi-projection synthesis (Frontal + Lateral views).
  - Automated radiology report generation using fine-tuned medical LLMs.
