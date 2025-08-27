# spoken-digit-classifier
# FSDD Spoken Digit Classifier — Lightweight MFCC + LinearSVC

> Audio in → digit out (0–9), optimized for simplicity, speed, and clarity.

## Why this approach
- **Lightweight & fast:** classic features (MFCC + Δ + ΔΔ) + **LinearSVC** → sub-ms inference per clip on CPU.
- **Reproducible:** deterministic seeds; single dependencies; small artifact footprint.
- **Extendable:** swap features or model via CLI flags; optional Gradio app for mic testing.

## Dataset
- **Free Spoken Digit Dataset (FSDD)** via Hugging Face (`mteb/free-spoken-digit-dataset`) with **8 kHz WAV** audio and predefined **train/test** splits.

## Quickstart
```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# On Windows: py -m venv .venv && .venv\\Scripts\\activate
pip install -r requirements.txt

# 2) Train + evaluate (creates ./artifacts with model + results.json)
python -m src.train --out_dir artifacts

# 3) Optional: separate eval/inspection
python -m src.eval --artifacts artifacts

# 4) Local demo (microphone / upload)
python app.py --model artifacts/model.joblib
```

## Results
Results are written to `artifacts/results.json` and include CV mean±std and test-set metrics. 
On a typical laptop CPU, **LinearSVC** on MFCC features obtains *high accuracy* (commonly >95% on FSDD) with **very low latency**. Your exact numbers may vary by run and environment.

## Project structure
```text
fsdd-digit-classifier/
├── requirements.txt
├── README.md
├── app.py
└── src/
    ├── data.py         # load HuggingFace FSDD, build X/y from audio
    ├── features.py     # MFCC + Δ + ΔΔ pooled stats
    ├── train.py        # CV + final fit, saves model + results.json
    ├── eval.py         # reload & evaluate saved artifacts
    └── infer.py        # CLI prediction on a .wav file
```

## Design notes
- **Features:** 20 MFCCs + delta + delta²; pooled by mean/std/percentiles for robustness & compactness.
- **Model:** `StandardScaler → LinearSVC`. Linear margins work well on MFCCs for small speech tasks.
- **Responsiveness:** feature extraction dominates cost; model inference is ~O(d) with tiny d (~300–600).
- **Leakage & splits:** we use the **provided train/test** split to avoid speaker leakage.
- **Reproducibility:** fixed seed (42), explicit sample rate (8 kHz), saved artifacts.

## Extend (Ideas)
- **Augmentations:** time shift, background noise, pitch/speed jitter (librosa.effects).
- **Robustness:** add SNR sweeps; measure accuracy vs. noise; calibrate decision function.
- **Models:** swap `LinearSVC` for `LogisticRegression`, `SVC(rbf)`, or a small 1D-CNN.
- **Export:** ONNX/TF-Lite for edge devices.

## LLM Collaboration (what I recorded)
1. Draft repo plan and file skeletons.
2. Ask LLM for MFCC pooling choices + latency trade-offs.
3. Code templates for HF datasets + Gradio mic.
4. Debug feature dimensions, add CV, and write results to JSON.
5. Brainstorm quick robustness checks (noise, shifts).

## Ethical & licensing
- FSDD is an open dataset; review its license before redistribution.
- This repo is for a technical challenge; use responsibly.

---
**Author:** Vedant Thaker
