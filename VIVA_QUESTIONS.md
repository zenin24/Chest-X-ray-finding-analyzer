# CHEST X-RAY FINDING ANALYZER: COMPREHENSIVE VIVA QUESTIONS & ANSWERS (25 Q&As)

This reference contains **25 examiner viva questions and detailed technical answers** covering deep learning, vision-language models, model architecture, mathematical formulations, calibration, web engineering, and medical ethics.

---

### Q1: What is a Vision-Language Model (VLM), and how does it differ from a standard Convolutional Neural Network (CNN)?
**Answer:**  
A standard CNN (like ResNet-50 or DenseNet-121) is a unimodal architecture trained to map an image directly to a discrete set of class integers via a fixed linear classification head. It cannot process natural language and has no semantic understanding of relationships between labels.  
A Vision-Language Model (like CLIP or PubMedCLIP) is a multimodal dual-encoder architecture that projects both visual images and text sequences into a shared, unified latent embedding space. It learns via contrastive objectives to pull matching image-text pairs close together while pushing mismatched pairs apart. This enables open-vocabulary "zero-shot" classification, where finding detection is achieved by measuring vector alignment with arbitrary descriptive text prompts without retraining.

---

### Q2: Why did you choose PubMedCLIP over vanilla OpenAI CLIP?
**Answer:**  
OpenAI's original CLIP was pre-trained on WebImageText (WIT), a generic dataset of 400 million internet image-caption pairs containing dogs, landscapes, cars, and consumer goods. Consequently, vanilla CLIP exhibits weak domain representation for specialized medical radiography, where findings consist of subtle grayscale opacities, anatomical contours, and complex clinical jargon.  
PubMedCLIP (`flaviagiammarino/pubmed-clip-vit-base-patch32`) was fine-tuned specifically on biomedical image-text pairs extracted from open-access scientific papers in PubMed Central. It has specialized biomedical semantic representations for radiographic pathology terms such as "cardiomegaly", "pleural effusion", and "pneumothorax".

---

### Q3: What is the specific architecture of the vision encoder in PubMedCLIP?
**Answer:**  
The vision backbone is a **Vision Transformer (ViT-B/32)**:
- **Input Resolution:** $224 \times 224$ pixels, 3 channels (RGB).
- **Patch Size:** $32 \times 32$ pixels.
- **Patch Token Grid:** $\left(\frac{224}{32}\right) \times \left(\frac{224}{32}\right) = 7 \times 7 = 49$ spatial tokens, plus 1 learnable Class Token (CLS), totaling 50 tokens.
- **Transformer Hidden Dimension:** 768 dimensions across 12 transformer encoder layers.
- **Visual Projection Head:** A linear layer $W_{\text{proj}} \in \mathbb{R}^{768 \times 512}$ that maps the transformer output into the 512-dimensional multimodal joint space, followed by $L_2$ vector normalization.

---

### Q4: Why is cosine similarity used instead of a Softmax cross-entropy classification layer?
**Answer:**  
In standard supervised learning, a Softmax layer enforces a mutually exclusive probability distribution summing to 1.0 across classes ($\sum_i P_i = 1$). In chest radiography, patients frequently present with **multiple concurrent co-morbidities** (e.g., both pneumonia and pleural effusion simultaneously). A Softmax layer artificially forces competing probabilities.  
Furthermore, because PubMedCLIP embeds both image and text vectors onto a unit hypersphere where $\|\mathbf{v}\|_2 = 1$ and $\|\mathbf{t}_k\|_2 = 1$, the cosine similarity simplifies directly to the algebraic dot-product:
$$S_k = \cos(\mathbf{v}, \mathbf{t}_k) = \mathbf{v}^\top \mathbf{t}_k = \sum_{j=1}^{512} v_j \cdot t_{k, j}$$
Each finding is evaluated independently against its own text prompt in $[-1, 1]$, enabling flexible multi-label candidate ranking without architectural retraining.

---

### Q5: What is "Prototype Confidence" in your system, and how is it calculated?
**Answer:**  
In `app/inference/analyzer.py`, Prototype Confidence is defined as the maximum cosine similarity score obtained by the input image across all candidate finding prompts:
$$C_{\text{proto}} = \max_{k \in \{1,\dots,8\}} S_k$$
It represents the peak semantic alignment between the visual radiograph embedding and the strongest matching clinical prototype description in the prompt library. We explicitly emphasize in the UI and API documentation that Prototype Confidence is a normalized vector distance on a hypersphere and **not** a calibrated posterior diagnostic probability.

---

### Q6: How does your system generate visual explainability from a Vision Transformer without convolutional feature maps?
**Answer:**  
In convolutional networks, explainability commonly relies on Grad-CAM computed from the final convolutional layer. Since ViT-B/32 does not have convolutional feature maps, our engine (`app/explainability/visualization.py`) extracts spatial attention directly from the transformer:
1. It accesses the `last_hidden_state` tensor of shape $(1, 50, 768)$.
2. It strips the CLS token (index 0) to isolate the 49 spatial patch tokens $\mathbf{H}_{\text{spatial}} \in \mathbb{R}^{49 \times 768}$, each corresponding to a $32 \times 32$ pixel region.
3. It projects each spatial patch token through the visual projection head $W_{\text{proj}}$ into the 512-dim multimodal space and normalizes it.
4. It calculates the localized dot-product between each of the 49 patch vectors and the target clinical finding prompt vector.
5. The resulting $7 \times 7$ similarity grid is min-max normalized, upsampled via bilinear interpolation to match the original image resolution, color-mapped using the Jet palette, and alpha-blended ($\alpha = 0.45$) over the radiograph.

---

### Q7: What are the eight controlled medical findings evaluated in your project?
**Answer:**  
The eight findings and their canonical prompt strings in `app/inference/analyzer.py` are:
1. `normal`: *"A chest X-ray showing a normal healthy lung"*
2. `pneumonia`: *"A chest X-ray showing pneumonia"*
3. `pleural_effusion`: *"A chest X-ray showing pleural effusion"*
4. `pneumothorax`: *"A chest X-ray showing pneumothorax"*
5. `cardiomegaly`: *"A chest X-ray showing cardiomegaly"*
6. `atelectasis`: *"A chest X-ray showing atelectasis"*
7. `pulmonary_edema`: *"A chest X-ray showing pulmonary edema"*
8. `consolidation`: *"A chest X-ray showing consolidation"*

---

### Q8: What is the "zero-shot baseline shift" problem, and why was threshold calibration necessary?
**Answer:**  
In uncalibrated zero-shot vision-language models, visual and textual embeddings do not span the entire hypersphere evenly; they reside in a narrow conical sub-space. Consequently, raw cosine similarities between chest X-rays and medical text prompts cluster tightly between $0.18$ and $0.25$.  
If one applies a standard threshold like $0.50$ or heuristic $0.25$, the model fails to predict any positive candidates, resulting in $0\%$ recall and an F1-score of $0.0000$. Threshold calibration systematically analyzes validation data to identify the true empirical decision boundary (e.g., $0.2000$ or per-finding thresholds from $0.15$ to $0.24$) where true positives separate from negatives.

---

### Q9: Describe your evaluation dataset and explain how you prevented data leakage.
**Answer:**  
Our evaluation pipeline (`data/evaluation/dataset.py`) utilizes 16 curated chest radiographs (`eval_001.png` to `eval_016.png`) with verified multi-label annotations in `labels.csv`.  
To prevent data leakage, the dataset was strictly partitioned into two disjoint splits of 8 images each:
- **Validation Split (`val`, $N=8$):** Used solely for the grid sweep and discovering optimal threshold values.
- **Test Split (`test`, $N=8$):** Completely held out during parameter selection. The final evaluation metrics were computed strictly on this unseen test split using the thresholds learned from the validation split.

---

### Q10: What were the quantitative results of your threshold calibration on the held-out test split?
**Answer:**  
As documented in `data/evaluation/results/calibration_report.txt`:
1. **Uncalibrated Baseline (Threshold $\tau = 0.2500$):**  
   Precision = $0.0000$, Recall = $0.0000$, Specificity = $1.0000$, **F1-Score = $0.0000$**.
2. **Calibrated Macro Threshold ($\tau = 0.2000$):**  
   Precision = $0.1295$, Recall = $1.0000$, Specificity = $0.0357$, **F1-Score = $0.2292$**.
3. **Calibrated Dynamic Per-Finding Thresholds ($\tau_k \in [0.15, 0.24]$):**  
   Precision = **$0.2534$**, Recall = **$1.0000$**, Specificity = **$0.4107$**, **F1-Score = $0.3843$**.  
Calibrating per finding elevated the test F1-score from $0.0000$ to **$0.3843$** with $100\%$ sensitivity.

---

### Q11: Which findings achieved the highest ROC-AUC in your zero-shot evaluation?
**Answer:**  
In our held-out test evaluation (`evaluation_report.txt`):
- **Cardiomegaly:** ROC-AUC = **$1.0000$** (Perfect discrimination of enlarged cardiac silhouette).
- **Pneumonia:** ROC-AUC = **$0.8571$**.
- **Pleural Effusion:** ROC-AUC = **$0.8571$**.
- **Normal:** ROC-AUC = **$0.5714$**.
- **Atelectasis:** ROC-AUC = **$0.4286$**.
- **Consolidation:** ROC-AUC = **$0.2857$**.
- **Pneumothorax:** ROC-AUC = **$0.1429$**.  
Cardiomegaly, pneumonia, and pleural effusion performed strongly because they present with prominent gross geometric and density alterations that align well with PubMed literature captions.

---

### Q12: What image preprocessing steps are performed before feeding an X-ray to the model?
**Answer:**  
In `app/preprocessing/image.py`:
1. **Validation:** Checks file extension against `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif` and verifies MIME type.
2. **Color Mode Normalization:** Safely loads bytes into PIL and converts Grayscale (1-channel L), RGBA (4-channel), or CMYK into standardized 3-channel 8-bit RGB.
3. **Resizing:** Scales the image to $224 \times 224$ pixels using bilinear interpolation.
4. **Tensor Normalization:** Converts pixel values to PyTorch float tensors in $[0.0, 1.0]$ and standardizes using ImageNet mean ($\mu = [0.48145466, 0.4578275, 0.40821073]$) and standard deviation ($\sigma = [0.26862954, 0.26130258, 0.27577711]$).

---

### Q13: What design pattern did you use for loading PubMedCLIP in FastAPI, and why?
**Answer:**  
We implemented the **Singleton Pattern** in `app/models/pubmedclip.py`.  
Vision-language models require hundreds of megabytes of RAM and take 5 to 10 seconds to instantiate weights and parse configurations. If a model were instantiated per incoming HTTP request, the server would exhaust memory and experience massive latency spikes.  
With the singleton pattern and FastAPI's modern `lifespan` context manager, the model weights are loaded into memory exactly once during application startup. Subsequent requests share the pre-loaded model instance concurrently across threads.

---

### Q14: How are prompt embeddings optimized for fast inference during runtime?
**Answer:**  
In `PubMedCLIPModel.load()`, our application tokenizes and encodes all eight canonical medical prompts into 512-dimensional $L_2$-normalized vectors during server initialization and caches them in a single tensor of shape $(8, 512)$.  
When an image analysis request arrives at `POST /analyze`, the system only needs to run the visual encoder once to produce an image tensor of shape $(1, 512)$. The similarity computation is a single PyTorch matrix multiplication:
$$\text{similarities} = \mathbf{V} \times \mathbf{T}^\top \quad ((1, 512) \times (512, 8) \rightarrow (1, 8))$$
This tensor dot-product executes in less than 5 milliseconds on CPU.

---

### Q15: What API endpoints does your FastAPI backend expose?
**Answer:**  
Our backend exposes five primary routes:
1. `GET /`: Basic service health check confirming API uptime and model loaded status.
2. `GET /health`: Comprehensive diagnostic status reporting model checkpoint name, device type (CPU/CUDA), and initialization flags.
3. `POST /image-info`: Ingests a radiograph file and returns format, dimensions ($W \times H$), and channel mode without running model inference.
4. `POST /analyze`: Ingests an image and returns ranked findings, prototype confidence, rank numbers, and candidate status.
5. `POST /explain`: Ingests an image and finding name, returning a binary PNG stream of the spatial heatmap overlay.

---

### Q16: How did you solve the frontend "TypeError: Failed to fetch" error?
**Answer:**  
The issue occurred due to two factors:
1. The frontend was opened as a local file (`file:///...`), causing relative URLs like `fetch('/analyze')` to resolve to the invalid path `file:///analyze`.
2. The FastAPI backend lacked `CORSMiddleware`, causing browser security policies to reject cross-origin requests.

We resolved this by:
- Adding `fastapi.middleware.cors.CORSMiddleware` in `app/main.py` allowing origins, methods, and headers.
- Introducing dynamic `API_BASE` resolution in `frontend/app.js`: if `window.location.protocol` is `http:` or `https:`, it uses relative paths (`''`); if opened via `file://`, it automatically routes requests to `http://127.0.0.1:8000`.
- Serving the frontend directly through FastAPI via static file mounting at `http://127.0.0.1:8000/static/index.html`.

---

### Q17: What is the difference between Macro-Average and Micro-Average F1 scores in multi-label evaluation?
**Answer:**  
- **Macro-Average F1:** Computes the F1-score independently for each finding class and then takes the unweighted arithmetic mean across all classes:
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
  Macro-F1 treats all findings equally, making it sensitive to poor performance on rare or difficult pathology classes.
- **Micro-Average F1:** Globally aggregates total True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$) across all classes before calculating a single overall F1-score:
  $$\text{Micro-F1} = \frac{2 \sum TP}{2 \sum TP + \sum FP + \sum FN}$$
  Micro-F1 is heavily dominated by frequent classes and can mask poor performance on minority findings.

---

### Q18: What is the limitation of the $7 \times 7$ spatial patch resolution in ViT-B/32?
**Answer:**  
ViT-B/32 divides a $224 \times 224$ image into $32 \times 32$ pixel patches, yielding a coarse $7 \times 7$ grid of 49 spatial tokens.  
While this is effective for large, diffuse pathologies (like cardiomegaly across the heart silhouette or pleural effusion at the lung bases), it cannot localize fine, small-scale anatomical details—such as small apical pneumothorax lines, hairline rib fractures, or solitary pulmonary nodules under 1 cm—with sub-patch spatial precision.

---

### Q19: Why are cosine similarity scores in this project not interpreted as diagnostic probabilities?
**Answer:**  
A true diagnostic probability represents a calibrated Bayesian posterior probability $P(Y=1 \mid X)$ trained with cross-entropy to minimize log-loss against ground-truth prevalence.  
PubMedCLIP was trained with a contrastive ranking objective, not logistic regression. Its scores represent the cosine of the angle between two high-dimensional vectors on a unit hypersphere. A high similarity score indicates semantic vector closeness, which can be influenced by anatomical density, radiograph contrast, or patient positioning. Treating vector distances as clinical probabilities without regulatory approval would be medically dangerous.

---

### Q20: How does your frontend render the synthetic chest X-ray in the "Demo Sample" feature?
**Answer:**  
In `frontend/app.js`, when a user clicks "Demo Sample", the script uses an HTML5 Canvas API in memory:
1. It draws a $256 \times 256$ radial dark gradient simulating the thoracic cavity.
2. It strokes bilateral semi-elliptical white curves representing thoracic rib cage contours.
3. It draws a rounded translucent white mass in the lower-left mediastinum representing the cardiac silhouette.
4. It calls `canvas.toBlob()` to produce a real PNG file object named `sample_chest_xray.png` and passes it through the exact same file handler as a real upload.

---

### Q21: What tests are included in your 38-case test suite?
**Answer:**  
Executed via `pytest`, our test suite covers eight distinct modules:
1. `test_api.py` (6 tests): Endpoints `/`, `/health`, `/image-info`, `/analyze`, `/explain`, and invalid extension handling.
2. `test_calibration.py` (3 tests): Sweep metric calculations, optimal threshold selection, and end-to-end calibration execution.
3. `test_evaluation_dataset.py` (4 tests): Label map structure, split partitioning, preprocessing consistency, and binary ground truth validation.
4. `test_evaluation_metrics.py` (4 tests): Perfect score metrics, zero-division error resilience, macro/micro calculations, and evaluation pipeline output.
5. `test_explainability.py` (3 tests): $7 \times 7$ spatial similarity heatmap generation, overlay image saving, and direct function integration.
6. `test_frontend.py` (4 tests): File existence, HTML DOM element checks, JS API integration patterns, and static file route accessibility.
7. `test_preprocessing.py` (7 tests): Valid file extensions, invalid extensions, invalid MIME types, RGB loading, grayscale conversion, corrupt bytes resilience, and image metadata extraction.
8. `test_pubmedclip.py` (4 tests): Model initialization, image embedding dimensions ($1 \times 512$), text embedding dimensions ($8 \times 512$), and dot-product calculations.
9. `test_system_complete.py` (3 tests): Complete end-to-end multi-step workflow verification.

---

### Q22: What happens if an uploaded file is corrupted or not a valid image?
**Answer:**  
The system implements defensive validation across multiple layers:
1. In `app/preprocessing/image.py`, `validate_image_file()` checks the file extension and MIME type. If invalid, it raises a `ValueError`.
2. Inside `load_image()`, the byte buffer is parsed using PIL's `Image.open()`. If the buffer contains corrupt or incomplete byte sequences, PIL raises an `UnidentifiedImageError`.
3. In `app/api/routes.py`, these exceptions are caught and mapped cleanly to an HTTP `400 Bad Request` status with a structured JSON error response (`ErrorResponse` schema), preventing unhandled 500 server crashes.

---

### Q23: How is production containerization handled for this application?
**Answer:**  
The repository includes a production-ready `Dockerfile` and `docker-compose.yml`:
- **Base Image:** `python:3.11-slim` for minimal container footprint.
- **Dependencies:** Installs necessary system libraries (`libgl1-mesa-glx`, `libglib2.0-0` for OpenCV) while caching pip packages.
- **Security:** Creates a non-root system user (`appuser`) to execute the ASGI server, adhering to container security best practices.
- **Process Management:** Configures Gunicorn with `uvicorn.workers.UvicornWorker` workers specified in `gunicorn.conf.py`.
- **Orchestration:** `docker-compose.yml` mounts a named volume for persistent Hugging Face cache storage, preventing redundant model weight downloads across container restarts.

---

### Q24: What are the main ethical and regulatory considerations of deploying this system?
**Answer:**  
1. **Not a Clinical Device:** Under FDA (21 CFR Part 820) and EU MDR regulations, software providing diagnostic outputs for medical imaging is classified as Software as a Medical Device (SaMD) requiring clinical trials, 510(k) clearance, and ISO 13485 certification. This system is strictly an educational research prototype.
2. **Algorithmic Bias:** Vision-language models trained on medical literature may inherit demographic or geographic reporting biases present in published research.
3. **Over-Reliance Hazard:** Clinicians might mistakenly assume a high similarity score implies certainty. The system mitigates this by mandating prominent visual disclaimers, color-coding scores as "Prototype Similarity", and requiring human-in-the-loop verification.

---

### Q25: If you had 6 more months to expand this project, what would be your top technical priorities?
**Answer:**  
1. **Higher-Resolution Vision Backbones:** Replace ViT-B/32 with ViT-L/14 or a specialized multi-scale architecture (like BioViL or MedCLIP) to achieve $14 \times 14$ or $28 \times 28$ patch explainability.
2. **Generative Radiology Report Synthesis:** Connect visual embeddings to an open-source medical LLM (e.g., Llama-3-Med, BioMistral) to generate structured radiology reports (Findings and Impression sections).
3. **Multi-View Integration:** Incorporate orthogonal lateral radiographs alongside frontal views to evaluate the retrocardiac space and posterior costophrenic angles.
4. **PACS & DICOM Integration:** Implement native DICOM network protocols (DIMSE C-STORE, C-FIND) and 16-bit windowing/leveling presets directly in the web console.
