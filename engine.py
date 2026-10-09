"""
Module 5 - Hybrid decision engine  (CivicFix AI, Review 2)

Combines the two ML models with hand-written SAFETY RULES:
  1. ML predicts category and severity (with confidence).
  2. Safety rules scan the text for danger words (exposed wire, fire, collapse, emergency access ...).
     A matching rule can only RAISE severity, never lower it, so a dangerous complaint is never
     under-rated because the ML model was unsure.
  3. A priority score (0-100) is built from severity + rule bonus + category risk.
  4. The score gives a priority level P1-P4, an SLA (time to fix) and the responsible department.
  5. Explainability: the words that pushed the model towards its answer, plus the rules that fired.
  6. Low-confidence flag: if either model is unsure the complaint is marked for human review.
"""
import re
import joblib

from preprocess import clean_text

RANK = {"Low": 0, "Medium": 1, "High": 2}
NAME = {0: "Low", 1: "Medium", 2: "High"}

# (regex, minimum severity forced, score bonus, reason shown to the user)
SAFETY_RULES = [
    (r"\b(fire|burning|burnt|smoke|spark\w*|short circuit)\b", "High", 25, "Fire / electrical-fire hazard"),
    (r"\b(exposed|bare|open|hanging|live|loose)\s+(electrical\s+)?(wire|wiring|cable|socket)s?\b|\b(wiring|wire|cable) is exposed\b|\blive (wire|cable)\b|\b(cable|wire)s? (is |are )?hanging\b",
     "High", 25, "Exposed live wiring"),
    (r"\b(electric shock|electrocut\w*|shock)\b", "High", 25, "Electric shock risk"),
    (r"\b(collaps\w*|structural|falling|about to fall|may fall|could fall)\b", "High", 20, "Risk of structural collapse / falling object"),
    (r"\bemergency\b", "High", 15, "Emergency access or vehicle affected"),
    (r"\b(sewage|sewer)\b", "High", 15, "Sewage / health hazard"),
    (r"\b(burst|flood\w*)\b", "High", 15, "Flooding / burst pipe"),
    (r"\b(gas leak|gas smell)\b", "High", 25, "Gas leak"),
    (r"\b(injur\w*|accident|unsafe|immediate|serious)\b", "High", 15, "Injury / safety risk mentioned"),
    (r"\b(health risk|infection|disease|mosquito\w*)\b", "Medium", 10, "Public-health concern"),
    (r"\b(students? (can|may|might|could) (fall|slip|trip)|slippery|skid\w*)\b", "Medium", 8, "Slip / fall risk"),
]

SEVERITY_POINTS = {"Low": 15, "Medium": 40, "High": 65}
CATEGORY_RISK = {  # how dangerous this kind of fault can become if ignored
    "Electrical": 10, "Civil": 8, "Water Leakage": 7, "Toilet/Sanitation": 7, "Road/Pathway": 6,
    "Pothole": 6, "Plumbing": 5, "Carpentry": 4, "Garbage": 4, "Class Items": 2,
}
PRIORITY_LEVELS = [  # (minimum score, level, label, SLA)
    (80, "P1", "Critical", "Within 4 hours"),
    (55, "P2", "High", "Within 24 hours"),
    (30, "P3", "Normal", "Within 3 days"),
    (0, "P4", "Low", "Within 7 days"),
]
DEPARTMENT = {
    "Electrical": "Electrical Maintenance",
    "Plumbing": "Plumbing & Water Supply",
    "Water Leakage": "Plumbing & Water Supply",
    "Toilet/Sanitation": "Housekeeping & Sanitation",
    "Garbage": "Housekeeping & Sanitation",
    "Road/Pathway": "Civil & Estate Maintenance",
    "Pothole": "Civil & Estate Maintenance",
    "Civil": "Civil & Estate Maintenance",
    "Carpentry": "Carpentry & Furniture",
    "Class Items": "Academic Facilities / Lab Support",
}
LOW_CONF = 60.0  # percent; below this the result is flagged for human review


def load_models():
    return joblib.load("model.joblib"), joblib.load("severity_model.joblib")


def _probs(model, cleaned):
    p = model.predict_proba([cleaned])[0]
    return sorted(zip(model.classes_, p), key=lambda x: -x[1])


def explain(model, cleaned, label, top=4):
    """Occlusion: remove each word and see how much the probability of `label` drops."""
    words = cleaned.split()
    idx = list(model.classes_).index(label)
    base = model.predict_proba([cleaned])[0][idx]
    scores = []
    for i, w in enumerate(words):
        reduced = " ".join(words[:i] + words[i + 1:])
        drop = base - model.predict_proba([reduced])[0][idx]
        scores.append((w, drop))
    scores = [s for s in scores if s[1] > 0.005]
    scores.sort(key=lambda x: -x[1])
    seen, out = set(), []
    for w, d in scores:
        if w not in seen:
            seen.add(w)
            out.append({"word": w, "impact": round(d * 100, 1)})
    return out[:top]


def safety_check(text):
    t = text.lower()
    return [{"reason": why, "forces": sev, "bonus": bonus}
            for pattern, sev, bonus, why in SAFETY_RULES if re.search(pattern, t)]


def priority_from_score(score):
    for minimum, level, label, sla in PRIORITY_LEVELS:
        if score >= minimum:
            return level, label, sla


def analyse(text, cat_model=None, sev_model=None):
    if cat_model is None or sev_model is None:
        cat_model, sev_model = load_models()
    cleaned = clean_text(text)

    cat_ranked = _probs(cat_model, cleaned)
    sev_ranked = _probs(sev_model, cleaned)
    category, cat_conf = cat_ranked[0][0], float(cat_ranked[0][1]) * 100
    ml_sev, sev_conf = sev_ranked[0][0], float(sev_ranked[0][1]) * 100

    rules = safety_check(text)
    final_rank = max([RANK[ml_sev]] + [RANK[r["forces"]] for r in rules])
    final_sev = NAME[final_rank]
    bonus = min(30, sum(r["bonus"] for r in rules))

    score = min(100, SEVERITY_POINTS[final_sev] + bonus + CATEGORY_RISK.get(category, 0))
    level, label, sla = priority_from_score(score)

    low_conf = cat_conf < LOW_CONF or sev_conf < LOW_CONF
    notes = []
    if cat_conf < LOW_CONF:
        notes.append(f"Category confidence is only {cat_conf:.0f}%")
    if sev_conf < LOW_CONF:
        notes.append(f"Severity confidence is only {sev_conf:.0f}%")

    return {
        "complaint": text,
        "category": category,
        "category_confidence": round(cat_conf, 1),
        "category_top3": [{"category": c, "confidence": round(float(p) * 100, 1)} for c, p in cat_ranked[:3]],
        "ml_severity": ml_sev,
        "severity": final_sev,
        "severity_confidence": round(sev_conf, 1),
        "escalated_by_rules": RANK[final_sev] > RANK[ml_sev],
        "safety_rules": rules,
        "priority_score": score,
        "priority": level,
        "priority_label": label,
        "sla": sla,
        "department": DEPARTMENT.get(category, "General Maintenance"),
        "needs_human_review": low_conf,
        "review_reason": "; ".join(notes),
        "explanation": {
            "category_words": explain(cat_model, cleaned, category),
            "severity_words": explain(sev_model, cleaned, ml_sev),
        },
    }
