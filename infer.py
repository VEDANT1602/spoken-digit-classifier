from __future__ import annotations
import argparse
import soundfile as sf
import numpy as np
import librosa
from joblib import load
from .features import extract_features

TARGET_SR = 8000

def predict(model_path: str, wav_path: str) -> int:
    model = load(model_path)
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    feat = extract_features(y, sr)
    pred = int(model.predict(feat[None, :])[0])
    return pred

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="artifacts/model.joblib")
    ap.add_argument("--wav", type=str, required=True)
    args = ap.parse_args()
    print(predict(args.model, args.wav))
