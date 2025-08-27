'''
Data loading for FSDD using Hugging Face `datasets`.
Provides simple utilities to build X, y for train/test splits.
'''
from __future__ import annotations
import numpy as np
from datasets import load_dataset, Audio
from typing import Tuple, Dict
from .features import batch_extract

TARGET_SR = 8000

def load_fsdd(split: str = "train", cache_dir: str | None = None):
    '''
    Load the FSDD dataset split with decoded audio @ 8 kHz.
    Returns a Hugging Face Dataset object with columns: 'audio', 'label'.
    '''
    ds = load_dataset("mteb/free-spoken-digit-dataset", split=split, cache_dir=cache_dir)
    # Cast audio to 8kHz for consistency
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    return ds

def build_xy(split: str = "train", cache_dir: str | None = None, n_mfcc: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    ds = load_fsdd(split=split, cache_dir=cache_dir)
    # Decode audio
    wavs = []
    srs = []
    labels = []
    for ex in ds:
        audio = ex['audio']
        wavs.append(audio['array'])
        srs.append(audio['sampling_rate'])
        labels.append(int(ex['label']))
    X = batch_extract(wavs, srs, n_mfcc=n_mfcc)
    y = np.array(labels, dtype=np.int64)
    return X, y

def label_names() -> Dict[int, str]:
    # Digits 0..9
    return {i: str(i) for i in range(10)}
