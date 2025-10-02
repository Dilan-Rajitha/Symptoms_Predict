from pathlib import Path
import argparse, json, time, platform
import pandas as pd, numpy as np, joblib, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "mega_symptom_dataset_500k.csv"  # your big CSV
DEFAULT_OUT  = ROOT / "models" / "model.joblib"

def build_pipeline(ngram_low, ngram_high, min_df, max_features):
    vec = TfidfVectorizer(
        analyzer="char", ngram_range=(ngram_low, ngram_high),
        min_df=min_df, max_features=max_features
    )
    clf = OneVsRestClassifier(MultinomialNB())
    return Pipeline([("tfidf", vec), ("clf", clf)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--model_out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ngram_low", type=int, default=3)
    ap.add_argument("--ngram_high", type=int, default=5)
    ap.add_argument("--min_df", type=int, default=3)
    ap.add_argument("--max_features", type=int, default=300000)  # cap for memory
    args = ap.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    df["labels"] = df["labels"].apply(lambda s: s.split("|"))

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].values, df["labels"].values,
        test_size=args.val_split, random_state=args.seed, shuffle=True
    )

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(y_train)
    Y_val   = mlb.transform(y_val)

    pipe = build_pipeline(args.ngram_low, args.ngram_high, args.min_df, args.max_features)
    pipe.fit(X_train, Y_train)

    # quick top-1 val accuracy
    proba = pipe.predict_proba(X_val)
    top1_idx = proba.argmax(axis=1)
    correct = int(sum(Y_val[i, top1_idx[i]] == 1 for i in range(len(X_val))))
    acc = correct / len(X_val)

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(data_path.resolve()),
        "n_samples": int(len(df)),
        "val_split": args.val_split,
        "top1_val_acc": round(acc, 4),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "tfidf": {
            "ngram_range": [args.ngram_low, args.ngram_high],
            "min_df": args.min_df,
            "max_features": args.max_features
        }
    }

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "mlb": mlb, "meta": meta}, out)
    print(f"[OK] Saved model -> {out}")
    print("[META]", json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
