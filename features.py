'''
Audio feature extraction helpers for FSDD digit classification.
Keeps it lightweight: MFCCs + Δ + ΔΔ pooled by simple statistics.
'''
from __future__ import annotations
import numpy as np
import librosa

def extract_features(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    '''
    Compute compact, robust features for speech digits.
    Returns a 1D feature vector.
    '''
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    # Trim silence (safety; FSDD is already trimmed)
    yt, _ = librosa.effects.trim(y, top_db=30)
    if yt.size == 0:  # pathological edge-case
        yt = y

    # Normalize peak to avoid amplitude variance
    peak = np.max(np.abs(yt)) or 1.0
    yt = yt / peak

    # MFCCs + deltas
    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    def pool(mat: np.ndarray) -> np.ndarray:
        # time aggregation
        mean = np.mean(mat, axis=1)
        std = np.std(mat, axis=1)
        p25 = np.percentile(mat, 25, axis=1)
        p50 = np.percentile(mat, 50, axis=1)
        p75 = np.percentile(mat, 75, axis=1)
        return np.concatenate([mean, std, p25, p50, p75], axis=0)

    feat = np.concatenate([pool(mfcc), pool(d1), pool(d2)], axis=0)
    return feat.astype(np.float32)

def batch_extract(wavs: list[np.ndarray], srs: list[int], n_mfcc: int = 20) -> np.ndarray:
    feats = [extract_features(y, sr, n_mfcc=n_mfcc) for y, sr in zip(wavs, srs)]
    return np.vstack(feats)
