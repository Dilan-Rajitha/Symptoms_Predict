from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import joblib, numpy as np, traceback, json

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model.joblib"

app = FastAPI(title="Symptoms Predict API")

class Request(BaseModel):
    lang: Optional[str] = "en"
    text: str
    age: Optional[int] = None
    sex: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None

# Load model once
PIPELINE = None
MLB = None
META = {}
try:
    saved = joblib.load(MODEL)
    PIPELINE = saved["pipeline"]
    MLB = saved["mlb"]
    META = saved.get("meta", {})
    print("[MODEL META]", json.dumps(META, ensure_ascii=False))
except Exception as e:
    print("[MODEL LOAD ERROR]", e)
    print(traceback.format_exc())

@app.get("/")
def health():
    return {"ok": True, "message": "POST /ai/symptom-check", "meta": META}

def simple_triage(top):
    if not top:
        return {"level": "SELF_CARE", "why": ["No signal detected"]}
    t0 = top[0]
    # EMERGENCY patterns
    if t0["id"] in {"ami", "meningitis", "heatstroke", "dka", "stroke", "seizure"} and t0["prob"] > 0.35:
        return {"level": "EMERGENCY", "why": ["Potential life-threatening pattern"]}
    # Urgent same-day
    if t0["id"] in {"appendicitis", "angina", "dengue_fever", "kidney_stones", "cholera", "typhoid"} and t0["prob"] > 0.35:
        return {"level": "URGENT_TODAY", "why": [f"{t0['name']} suspicion"]}
    # Fallbacks
    if t0["prob"] < 0.25:
        return {"level": "SELF_CARE", "why": ["Low-risk pattern; monitor"]}
    return {"level": "GP_24_48H", "why": ["Moderate risk pattern"]}

@app.post("/ai/symptom-check")
def check(req: Request):
    try:
        if PIPELINE is None or MLB is None:
            return {"error": "Model not loaded. Place models/model.joblib and restart."}
        proba = PIPELINE.predict_proba([req.text])[0]
        idx = np.argsort(proba)[::-1][:3]
        top = []
        for i in idx:
            cid = str(MLB.classes_[i])
            p = float(proba[i])
            top.append({"id": cid, "name": cid.replace("_", " ").title(), "prob": p, "prob_pct": round(p*100, 2)})
        tri = simple_triage(top)
        return {"top_conditions": top, "triage": tri, "disclaimer": "Educational aid; not a medical diagnosis."}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
