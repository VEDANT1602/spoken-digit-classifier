# 30‑Minute Devlog — What to Record

You don't need narration. Just screen-capture key moments of your workflow:

1. **Scaffold (2–3 min)**
   - Paste the challenge statement.
   - Ask your coding LLM to propose a minimal architecture and file tree.
   - Create `features.py`, `data.py`, and `train.py` stubs.

2. **Feature extraction (6–8 min)**
   - Prompt: “Given 8 kHz spoken digits, suggest MFCC-based features with deltas and compact pooling (<= 1k dims).”
   - Implement and unit-test `extract_features` on a single sample.

3. **Data loader (4–5 min)**
    - Prompt: “Load `mteb/free-spoken-digit-dataset` with audio decoding at 8kHz and build X/y.”
    - Show a quick print of shapes and label distribution.

4. **Model + CV (6–8 min)**
    - Prompt: “Create sklearn pipeline StandardScaler→LinearSVC with 5-fold StratifiedKFold; save results.json.”
    - Run training and highlight CV metrics.

5. **App (5–6 min)**
    - Prompt: “Gradio app that records microphone, resamples to 8kHz, extracts features, prints prediction + latency.”

6. **Wrap‑up (1–2 min)**
    - Show `artifacts/` with `model.joblib` and `results.json`.
    - Add a short README update.

### Optional Prompts (Robustness)
- “Add SNR-controlled white noise augmentation and re-train; compare accuracy vs. SNR in results.json.”
- “Time‑shift augmentation up to ±50 ms; does it help?”
