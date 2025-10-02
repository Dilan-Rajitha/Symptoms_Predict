from pathlib import Path
import argparse, pandas as pd, numpy as np, joblib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "mega_symptom_dataset_500k.csv"
MODEL = ROOT / "models" / "model.joblib"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["labels"] = df["labels"].apply(lambda s: s.split("|"))

    saved = joblib.load(MODEL)
    pipe, mlb = saved["pipeline"], saved["mlb"]

    proba = pipe.predict_proba(df["text"])
    top1_idx = proba.argmax(axis=1)
    preds = mlb.classes_[top1_idx]

    y = mlb.transform(df["labels"]).toarray()
    correct = int(sum(y[i, top1_idx[i]] == 1 for i in range(len(df))))
    print(f"Top-1 accuracy: {correct/len(df)*100:.2f}% on {len(df)} samples")

    # show a few examples
    for i in range(3):
        print("•", df['text'].iloc[i][:80], "→", preds[i])

if __name__ == "__main__":
    main()
