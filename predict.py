"""
Command-line predictor (Review 2: full analysis).
Usage:  python predict.py "Water is leaking from the tap in the lab"
"""
import sys
import joblib
from preprocess import clean_text
from engine import analyse, load_models


def predict(text: str, model=None):
    """Category-only prediction (Review 2 behaviour, kept for compatibility)."""
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
    r = analyse(" ".join(sys.argv[1:]), *load_models())
    print(f"Category   : {r['category']}  ({r['category_confidence']}%)")
    print(f"Severity   : {r['severity']}" + (f"  (ML said {r['ml_severity']}, raised by safety rules)" if r["escalated_by_rules"] else f"  ({r['severity_confidence']}%)"))
    print(f"Priority   : {r['priority']} {r['priority_label']}  (score {r['priority_score']}/100)")
    print(f"SLA        : {r['sla']}")
    print(f"Department : {r['department']}")
    for rule in r["safety_rules"]:
        print(f"  safety rule: {rule['reason']}")
    ex = r["explanation"]
    print("Key words  : category -> " + ", ".join(w["word"] for w in ex["category_words"]) +
          " | severity -> " + ", ".join(w["word"] for w in ex["severity_words"]))
    if r["needs_human_review"]:
        print(f"!! Needs human review: {r['review_reason']}")
