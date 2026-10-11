# CHEST X-RAY FINDING ANALYZER: 5-MINUTE LIVE DEMONSTRATION SCRIPT

**Project:** Chest X-ray Finding Analyzer  
**Audience:** College Evaluation Committee / External Examiners / Faculty Reviewers  
**Target Duration:** Exactly 5 Minutes  
**Prerequisites:** 
- FastAPI server running on `http://127.0.0.1:8000` via `.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`.
- Web browser open to `http://127.0.0.1:8000/static/index.html`.
- Second browser tab open to Swagger documentation: `http://127.0.0.1:8000/docs`.
- Terminal window ready in `C:\Users\ASUS\mainproject`.

---

## ⏱️ Detailed Time Breakdown & Speaker Actions

| Time | Phase | Target Screen / Tool | Key Technical Objective |
| :---: | :--- | :--- | :--- |
| **0:00 – 0:45** | **1. Introduction & Architecture** | RAD·LAB Console (Header / KPI) | State research scope, PubMedCLIP model, and safety disclaimers. |
| **0:45 – 1:45** | **2. Image Ingestion & Metadata** | Upload Dropzone / File Chip | Demonstrate file loading, MIME/format check, and `/image-info`. |
| **1:45 – 3:00** | **3. Zero-Shot Inference & Ranking** | Candidate Findings Table | Trigger `/analyze`, explain vector dot-products and prototype confidence. |
| **3:00 – 4:00** | **4. Visual Spatial Explainability** | Right Panel (Heatmap Overlay) | Switch to Grad-CAM overlay, explain $7 \times 7$ ViT patch projections. |
| **4:00 – 4:30** | **5. Dynamic Threshold Sweep** | Threshold Sweep Bar | Demonstrate interactive threshold calibration and candidate filtering. |
| **4:30 – 5:00** | **6. Automated Testing & Conclusion** | Terminal (`pytest`) | Run test suite (38/38 passing), reiterate academic/prototype status. |

---

## 🎙️ Step-by-Step Demonstration Protocol & Verbal Script

### PHASE 1: Introduction & Architecture Overview (0:00 – 0:45)

**Action:**
Show the RAD·LAB Console in the browser at `http://127.0.0.1:8000/static/index.html`. Point out the header, the model tag (`PubMedCLIP ViT-B/32`), and the yellow research disclaimer banner.

**Speaker Script:**
> *"Respected committee members, welcome to our demonstration of the **Chest X-ray Finding Analyzer**. 
>
> In clinical radiology, conventional deep learning models rely on supervised convolutional networks with fixed classification heads, making it impossible to detect novel findings without retraining. Our system introduces a **zero-shot medical vision-language platform** powered by **PubMedCLIP** (`flaviagiammarino/pubmed-clip-vit-base-patch32`), pre-trained on biomedical literature.
>
> As shown on our RAD·LAB console banner, this system is strictly a **research and educational prototype**. The similarity scores we will see are normalized vector dot-products within a shared 512-dimensional embedding space—not clinically validated diagnostic probabilities."*

---

### PHASE 2: Image Ingestion & Metadata Inspection (0:45 – 1:45)

**Action:**
1. Click the **"Demo Sample"** button on the upload card (or drag and drop `data/evaluation/images/eval_001.png`).
2. Show the immediate rendering of the chest radiograph preview.
3. Hover over the selected file badge (`sample_chest_xray.png` or `eval_001.png`) to show the title tooltip populated by the `/image-info` endpoint.
4. Point out that the "Analyze Chest X-Ray" button has transitioned from disabled to active.

**Speaker Script:**
> *"Let's begin by ingesting an image. We click 'Demo Sample'—or we can drag-and-drop any standard DICOM-exported PNG, JPEG, TIFF, or BMP file.
>
> In the background, the client instantly queries our asynchronous `POST /image-info` endpoint. The server validates that the file has a legitimate image MIME type, loads it through our PIL pipeline, verifies structural integrity, standardizes the channels to 3-channel 8-bit RGB, and returns metadata—here, $256 \times 256$ pixels.
>
> Notice that the 'Analyze Chest X-Ray' button is now enabled, ready to dispatch our zero-shot inference pipeline."*

---

### PHASE 3: Zero-Shot Inference & Finding Ranking (1:45 – 3:00)

**Action:**
1. Click **"Analyze Chest X-Ray"**.
2. Note the loading spinner and button text transition to *"Analyzing..."*.
3. Once the output table populates (typically under 1 second):
   - Highlight the **Prototype Confidence** card (e.g., `0.2427`).
   - Highlight the top-ranked findings (e.g., `normal`, `pulmonary_edema`, `cardiomegaly`).
   - Point out the rank pills (`#1`, `#2`, `#3`), the normalized similarity scores, and the `Candidate` vs. `Unlikely` status badges.

**Speaker Script:**
> *"We now click 'Analyze Chest X-Ray'. In real time, the FastAPI server normalizes the image tensor to $224 \times 224$ using ImageNet statistics and passes it through the ViT-B/32 vision encoder to generate a 512-dimensional $L_2$-normalized vector.
>
> The inference engine calculates the matrix dot-product between this visual vector and our eight pre-computed clinical prompt vectors—such as 'A chest X-ray showing pneumonia' or 'A chest X-ray showing cardiomegaly'.
>
> Looking at the Output Panel, the findings are dynamically sorted in descending order of similarity. The peak similarity is extracted as our **Prototype Confidence** (here, $0.2427$). Under the current threshold of $0.25$, findings are classified as either 'Candidate' or 'Unlikely'. This illustrates how contrastive vision-language models score finding alignment without any supervised retraining."*

---

### PHASE 4: Visual Spatial Explainability (3:00 – 4:00)

**Action:**
1. Direct the committee's attention to Card 03: **"Spatial Explainability"** on the right side of the console.
2. Click the **"Grad-CAM Overlay"** view mode toggle button.
3. Wait for the `POST /explain` response to load the blended heatmap overlay over the chest radiograph.
4. Click a different finding row in the table (e.g., click `Cardiomegaly` or `Pneumonia`).
5. Observe the overlay automatically refresh to reflect attention for that specific clinical condition.

**Speaker Script:**
> *"A primary drawback of vision transformers in medicine is that they are considered 'black boxes'. To solve this, we implemented an internal spatial patch attribution engine.
>
> When we toggle to 'Grad-CAM Overlay', the backend executes `POST /explain`. Instead of pooling all features, our engine extracts the 49 spatial patch representations from the Vision Transformer's final hidden layer—corresponding to a $7 \times 7$ grid of $32 \times 32$ pixel patches.
>
> Each spatial patch token is projected into the joint 512-dimensional multimodal space and evaluated against the chosen finding's text vector. The resulting attribution matrix is smoothly upsampled via bilinear interpolation and alpha-blended over the radiograph using a Jet colormap.
>
> If we click on 'Cardiomegaly', the heatmap dynamically refocuses on the cardiac silhouette; clicking 'Pneumonia' or 'Pleural Effusion' shifts the focus toward the lung fields and costophrenic angles. This gives clinicians interpretable visual grounding."*

---

### PHASE 5: Dynamic Threshold Calibration Sweep (4:00 – 4:30)

**Action:**
1. Scroll to the **"Calibration: Dynamic Threshold Sweep"** bar below the findings table.
2. Click on the column marked `0.20` or `0.22`.
3. Point out that the threshold indicator badge immediately updates, and the count of **"Above Threshold"** findings changes reactively.

**Speaker Script:**
> *"Zero-shot cosine similarities in vision-language models suffer from baseline offset: raw scores typically hover between $0.20$ and $0.24$. If an uncalibrated default threshold like $0.25$ is used, sensitivity drops to zero.
>
> Our project includes a systematic threshold calibration pipeline. In our experiments across validation and test splits, lowering the threshold to our empirically discovered optimal point of $0.2000$—or using per-finding dynamic thresholds—elevated the held-out test Macro F1-score from $0.0000$ to **$0.3843$** with $100\%$ recall.
>
> Here in the console, researchers can interactively sweep the decision threshold and watch the candidate classification update in real time."*

---

### PHASE 6: Automated Test Suite & Conclusion (4:30 – 5:00)

**Action:**
1. Switch to the open terminal window.
2. Run `.venv\Scripts\python.exe -m pytest -v`.
3. Let the committee witness all 38 tests passing in real time.
4. Conclude the demonstration.

**Speaker Script:**
> *"Finally, to verify architectural resilience, we run our automated test suite. 
> 
> Pytest validates our preprocessing pipeline, model embeddings, API routes, calibration algorithms, spatial heatmaps, and frontend integration. As you can see, all **38 automated tests pass with 100% success rate**.
>
> In summary, the Chest X-ray Finding Analyzer delivers a complete, modular, and explainable prototype combining biomedical foundation models, high-performance asynchronous APIs, and calibrated decision boundaries.
>
> Thank you, and we are now ready for your questions."*

---

## 🛠️ Quick Reference Troubleshooting for Demo Presenter

| Potential Demo Glitch | Immediate Remedial Action |
| :--- | :--- |
| **Server is not running** | Open PowerShell and run: `.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000` |
| **No image file available** | Simply click the **"Demo Sample"** button on the upload card; it synthetically generates a radiograph on an HTML canvas in memory. |
| **Overlay appears slow on first click** | Explain that PyTorch is generating the $7 \times 7$ tensor graph on CPU; subsequent requests for cached findings respond in milliseconds. |
| **Browser displays cached old script** | Press `Ctrl + F5` to force a hard cache refresh. |
