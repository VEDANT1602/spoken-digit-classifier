from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from joblib import load

def main(args):
    artifacts = Path(args.artifacts)
    X_test = np.load(artifacts / "X_test.npy")
    y_test = np.load(artifacts / "y_test.npy")
    model = load(artifacts / "model.joblib")

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    print(f"Test accuracy={acc:.4f}, f1_macro={f1m:.4f}")
    print(classification_report(y_test, pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    (artifacts / "eval.json").write_text(json.dumps({"accuracy": float(acc), "f1_macro": float(f1m)}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="artifacts")
    main(ap.parse_args())
