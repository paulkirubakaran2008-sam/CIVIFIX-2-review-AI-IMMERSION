"""
Command-line predictor.   Usage:  python predict.py "Water is leaking from the tap in the lab"
"""
import sys
import joblib
from preprocess import clean_text


def predict(text: str, model=None):
    model = model or joblib.load("model.joblib")
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: -x[1])
    return {
        "complaint": text,
        "category": ranked[0][0],
        "confidence": round(float(ranked[0][1]) * 100, 1),
        "top3": [{"category": c, "confidence": round(float(p) * 100, 1)} for c, p in ranked[:3]],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python predict.py "your complaint"')
    result = predict(" ".join(sys.argv[1:]))
    print(f"Category   : {result['category']}  ({result['confidence']}%)")
    for r in result["top3"][1:]:
        print(f"  also possible: {r['category']} ({r['confidence']}%)")
