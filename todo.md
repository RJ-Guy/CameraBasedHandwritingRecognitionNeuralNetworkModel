# Project Presentation & Viva Preparation Checklist (TODO.md)
## Project: Camera-Based Handwriting Recognition System
**Deck Format:** 11 Crisp, High-Impact Slides with Real Images, Formulas & Benchmark Proofs

---

## Phase 1: Pre-Generation Sanity Verification
- [x] **Verify `content.md`:** Ensure all 11 crisp slides, line-by-line bullets, speaker scripts, and viva defense Q&As are finalized.
- [x] **Verify `plan.md`:** Ensure the 11-slide time budget, color palette, asset directory map, and AI generation prompt template are set.
- [ ] **Run Unit Tests Locally to Verify Codebase Health:**
  ```powershell
  pytest Model/tests/ -v
  ```
  *(Verify that all 65 tests pass with 100% green).*
- [ ] **Verify Real Output Artifacts Exist on Disk:**
  - [ ] Raw test captures: `data/samples/MachineLearning.jpg`, `NetworkFlow.jpg`, `ABC.jpeg`
  - [ ] Recognized text files: `Model/outputs/text/MachineLearning_recognized.txt`, `NetworkFlow_recognized.txt`, `ABC_recognized.txt`
  - [ ] Visual HUD annotated images: `Model/outputs/annotated/MachineLearning_annotated.png`, `NetworkFlow_annotated.png`, `ABC_annotated.png`
  - [ ] Intermediate debug visual stages:
    - [ ] `Model/outputs/debug/MachineLearning_01_ingested.png` (Phase 1)
    - [ ] `Model/outputs/debug/MachineLearning_03_gray_enhanced.png` (Phase 2 CLAHE)
    - [ ] `Model/outputs/debug/MachineLearning_02_binary_mask.png` (Phase 2 Sauvola)
    - [ ] `Model/outputs/debug/MachineLearning_04_segmentation.png` (Phase 3 lines/words)
    - [ ] `Model/outputs/debug/extracted_real_chars.png` (Phase 4 normalized chars)
    - [ ] `Model/outputs/debug/emnist_balanced_samples.png` (Phase 6 dataset)
    - [ ] `Model/outputs/debug/confusion_matrix.png` (Phase 6 confusion matrix)
    - [ ] `Model/outputs/debug/MachineLearning_05_comparison.png` (Phase 7 comparison)
- [ ] **Update Slide 1 with Your Personal Details in `content.md`:**
  - [ ] Student Name
  - [ ] Roll Number / Student ID
  - [ ] Department / University Name
  - [ ] Guide / Professor Name

---

## Phase 2: Git Versioning & Branch Push
Run these commands in PowerShell to commit and push your presentation assets to a new git branch:

```powershell
# 1. Check current status
git status

# 2. Create and switch to a dedicated presentation branch
git checkout -b presentation-deck

# 3. Stage all presentation markdown files
git add content.md plan.md todo.md presentation_content.md

# 4. Commit the changes
git commit -m "docs: finalize 11-slide presentation content, plan, and viva checklist"

# 5. Push the new branch to GitHub
git push -u origin presentation-deck
```

---

## Phase 3: AI Slide Deck Generation (Using Another AI)
Feed `content.md` and `plan.md` into your slide generator of choice:

### Option A: Gamma App (Recommended — Most Visual & Automated)
- [ ] Go to [gamma.app](https://gamma.app) $\to$ Click **Create New** $\to$ **Paste in text**.
- [ ] Paste the Master Prompt from Section 4 of `plan.md` together with `content.md`.
- [ ] Select **11 Cards/Slides** and choose a clean dark navy / white professional theme.
- [ ] Drag-and-drop the real repository images (from `Model/outputs/debug/` and `Model/outputs/annotated/`) into the corresponding slides.
- [ ] Export to `.pptx` or `.pdf`.

### Option B: ChatGPT / Claude with Python-PPTX Script
- [ ] Open ChatGPT or Claude.
- [ ] Attach `content.md` and `plan.md`.
- [ ] Prompt: *"Write a complete `generate_presentation.py` script using `python-pptx` that builds an 11-slide presentation following all instructions in `plan.md` and content in `content.md`, embedding the images from the repository paths."*
- [ ] Run the generated Python script locally: `python generate_presentation.py`.

### Option C: Marp Markdown Presentation
- [ ] Install Marp CLI or the Marp extension in VS Code.
- [ ] Add `--- marp: true` to the top of `content.md`.
- [ ] Export directly to high-resolution PDF slides.

---

## Phase 4: Slide Deck Review & Visual Polish
Verify these items before walking into the viva room:
- [ ] **Slide Count:** Exactly 11 slides (ensuring you can finish in 12–14 minutes).
- [ ] **Real Images Visible:**
  - Slide 1: Raw photo vs. Annotated HUD overlay
  - Slide 4: 3-stage visual strip (Ingested $\to$ Enhanced $\to$ Sauvola Mask)
  - Slide 5: Segmentation bounding boxes (`MachineLearning_04_segmentation.png`)
  - Slide 6: Normalized characters (`extracted_real_chars.png`)
  - Slide 8: Confusion matrix & EMNIST samples
  - Slide 9: Reconstructed text preview + Annotated HUD image
  - Slide 10: Gallery of real test captures (`MachineLearning`, `NetworkFlow`, `ABC`)
- [ ] **Mathematical Equations:** Homography $\mathbf{H}$, Sauvola $T=m(1-k(1-s/R))$, and Dual-Centroid $\theta = \arctan(\Delta x/\Delta y)$ rendered cleanly.
- [ ] **Presenter Notes:** The speaker script for each slide is placed into the PPT presenter notes pane.

---

## Phase 5: 12-Minute Viva Rehearsal & Defense Mastery
- [ ] **Pacing Practice:** Rehearse the 11 slides with a timer, aiming for **12–14 minutes total**:
  - Slides 1–3 (Title, Problem, Architecture): $\approx 3.5$ minutes
  - Slides 4–7 (Phases 1 through 5 — CV & Features): $\approx 5.5$ minutes
  - Slides 8–10 (MLP Classifier, Results & Benchmarks): $\approx 3.5$ minutes
  - Slide 11 (Unit Tests, Defense & Conclusion): $\approx 1.5$ minutes
- [ ] **Memorize the 5 Key Project Numbers:**
  1. Trainable Parameters: **39,067**
  2. Dataset: **EMNIST Balanced (47 classes, 131,600 samples)**
  3. Accuracy: **82.59% Train, 77.08% Test**
  4. CPU Training Time: **12.2 seconds (33 epochs)**
  5. Total CPU Inference Latency: **~1.65 seconds**
- [ ] **Master the Top 3 Professor Defense Answers:**
  - [ ] *"Why strictly zero deep learning?"* $\to$ Complete explainability, 39K parameters, runs on CPU without GPU, auditable failure modes, trains in 12 seconds.
  - [ ] *"Why Sauvola over Otsu?"* $\to$ Local standard deviation $s$ suppresses paper grain in flat regions while isolating ink edges under uneven shadows.
  - [ ] *"Why 256 dimensions in Circular DDD?"* $\to$ 8 concentric rings $\times$ 8 compass rays $\times$ 4 metrics (2 White cavity, 2 Black stroke) $= 256$-D topological vector.
