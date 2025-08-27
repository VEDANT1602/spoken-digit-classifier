from __future__ import annotations
import argparse, json, os, random
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from joblib import dump
from . import data as D

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def build_model() -> Pipeline:
    # Lightweight, fast, and strong for MFCC features
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, loss="squared_hinge", dual=True))
    ])
    return model

def main(args):
    set_seeds(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    X_train, y_train = D.build_xy("train", cache_dir=args.cache_dir, n_mfcc=args.n_mfcc)
    X_test,  y_test  = D.build_xy("test",  cache_dir=args.cache_dir, n_mfcc=args.n_mfcc)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # CV on train
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    oof_preds = np.zeros_like(y_train)
    cv_acc, cv_f1 = [], []

    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        clf = build_model()
        clf.fit(X_train[tr], y_train[tr])
        va_pred = clf.predict(X_train[va])
        acc = accuracy_score(y_train[va], va_pred)
        f1m = f1_score(y_train[va], va_pred, average="macro")
        cv_acc.append(acc); cv_f1.append(f1m)
        oof_preds[va] = va_pred
        print(f"[Fold {fold}] acc={acc:.4f} f1_macro={f1m:.4f}")

    # Fit final on full train
    final_model = build_model()
    final_model.fit(X_train, y_train)
    test_pred = final_model.predict(X_test)

    test_acc = accuracy_score(y_test, test_pred)
    test_f1m = f1_score(y_test, test_pred, average="macro")

    print("\nTest metrics:")
    print(f"accuracy={test_acc:.4f}, f1_macro={test_f1m:.4f}")
    print("\nClassification report (test):")
    print(classification_report(y_test, test_pred, digits=4))

    # Save artifacts
    dump(final_model, out / "model.joblib")
    import numpy as np
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_test.npy", y_test)

    results = {
        "cv": {"n_splits": args.cv, "acc_mean": float(np.mean(cv_acc)), "acc_std": float(np.std(cv_acc)),
               "f1_macro_mean": float(np.mean(cv_f1)), "f1_macro_std": float(np.std(cv_f1))},
        "test": {"accuracy": float(test_acc), "f1_macro": float(test_f1m)},
        "features": {"n_mfcc": args.n_mfcc, "vector_dim": int(X_train.shape[1])},
        "model": "StandardScaler + LinearSVC",
        "seed": args.seed,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved model and results to: {out.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--n_mfcc", type=int, default=20)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
