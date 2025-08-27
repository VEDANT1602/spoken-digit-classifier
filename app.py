'''
Gradio demo: speak/upload a digit (0-9) -> model predicts the number.
Run after training: python app.py --model artifacts/model.joblib
'''
from __future__ import annotations
import argparse, time
import gradio as gr
import numpy as np
import librosa
from joblib import load
from src.features import extract_features

TARGET_SR = 8000

def load_model(path: str):
    return load(path)

def infer_fn(audio, model_path):
    '''
    `audio` is (sr, data) when type='numpy' and source='microphone' in Gradio.
    '''
    if audio is None:
        return "No audio", None
    sr, data = audio
    data = np.array(data, dtype=np.float32)
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    t0 = time.perf_counter()
    feat = extract_features(data, sr)
    model = load_model(model_path)
    pred = int(model.predict(feat[None, :])[0])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return f"{pred}", latency_ms

def main(args):
    with gr.Blocks(title="Spoken Digit Classifier (FSDD)") as demo:
        gr.Markdown("# 🎙️ Spoken Digit Classifier\nSpeak a digit (0–9) and get a prediction.")
        model_path = gr.State(args.model)

        with gr.Row():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Say a digit (0-9)")
        with gr.Row():
            pred = gr.Textbox(label="Prediction", interactive=False)
            latency = gr.Number(label="Latency (ms)", interactive=False, precision=2)
        btn = gr.Button("Predict")
        btn.click(infer_fn, inputs=[audio_in, model_path], outputs=[pred, latency])
    demo.launch(server_name=args.host, server_port=args.port)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="artifacts/model.joblib")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    main(args)
