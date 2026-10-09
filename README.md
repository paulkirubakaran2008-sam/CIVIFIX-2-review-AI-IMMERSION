# CivicFix AI — Review 2 submission (≈70 % of the project)

**CivicFix AI** is an AI system that reads a campus complaint written in plain English and decides *what it is about*,
*how dangerous it is* and *who should fix it first*.

## Progress at Review 2

| # | Module | Weight | Status |
|---|--------|:------:|--------|
| 1 | Problem study, dataset collection (1,000 labelled complaints) | 10 % | ✅ Done |
| 2 | Text pre-processing + category classifier (TF-IDF + Logistic Regression) | 15 % | ✅ Done |
| 3 | Basic web app (Flask) with complaint form and prediction API | 10 % | ✅ Done |
| 4 | Severity classifier (Low / Medium / High) | 12 % | ✅ Done |
| 5 | Hybrid engine: safety rules + ML, priority score (P1–P4), SLA, department | 18 % | ✅ Done |
| 6a | Explainability (why this result) + low-confidence flag | 5 % | ✅ Done |
| 6b | Batch ranking of many complaints (`/batch` API is ready; dashboard view comes later) | 5 % | ⏳ Final |
| 7 | Advanced UI (dashboard, history, dark mode), deployment on Vercel | 10 % | ⏳ Final |
| 8 | Testing, performance tuning, final report | 15 % | ⏳ Final |
| | **Completed so far** | **70 %** | |

## What was added for the 70 % milestone
```
train_severity.py   Module 4 – severity model (class-balanced, 5-fold cross-validation)
engine.py           Module 5 – hybrid engine: safety rules + ML -> priority score, P1-P4, SLA, department,
                    explanation (word impact), low-confidence flag
tests.py            evaluation on the held-out split and on 24 hand-written complaints with new wording
app.py              /predict returns the full analysis, new /batch route ranks many complaints
templates/index.html  shows priority badge, SLA, department, safety rules, "why this result", review warning
```

### How the hybrid engine decides
1. **ML**: category model + severity model, each with a confidence.
2. **Safety rules**: danger phrases (exposed wire, fire/sparks, collapse, sewage, flooding, emergency access, injury risk …)
   can only **raise** severity, never lower it, so a dangerous complaint is not under-rated because the ML model was unsure.
3. **Priority score (0–100)** = severity points (15 / 40 / 65) + safety-rule bonus (max 30) + category risk (2–10).
4. **Priority → SLA**: P1 ≥ 80 (4 hours) · P2 ≥ 55 (24 hours) · P3 ≥ 30 (3 days) · P4 (7 days).
5. **Department** is chosen from the predicted category (e.g. Electrical → Electrical Maintenance).
6. **Explainability**: each word is removed in turn and the drop in probability is shown, so the user sees which words drove the result.
7. **Low-confidence flag**: if either model is below 60 % confidence the complaint is marked *Needs human review*.

## Project structure
```
dataset.csv            1,000 labelled complaints (complaint, category, severity)
preprocess.py          Module 1 – load, clean, remove duplicates, train/test split
train_category.py      Module 2 – category model, metrics + confusion matrix
train_severity.py      Module 4 – severity model, metrics + confusion matrix
engine.py              Module 5 – hybrid engine
predict.py             command-line prediction (full analysis)
app.py                 Flask web application (+ /batch)
tests.py               Review 2 evaluation script
templates/index.html   web page
outputs/               metrics.txt, severity_metrics.txt, hybrid_report.txt, confusion matrices
```

## How to run
```bash
pip install -r requirements.txt
python train_category.py      # category model   -> model.joblib
python train_severity.py      # severity model   -> severity_model.joblib
python tests.py               # evaluation       -> outputs/hybrid_report.txt
python predict.py "Electrical wire is exposed near students in the lab"
python app.py                 # open http://127.0.0.1:5000
```
Batch ranking example:
```bash
curl -X POST http://127.0.0.1:5000/batch -H "Content-Type: application/json" \
     -d '{"complaints": ["Fan makes a small noise", "Wire is sparking in lab 3", "Dustbin is overflowing"]}'
```

## Results so far
| Measure | Result |
|---|---|
| Category accuracy (83 held-out complaints) | **98.8 %** (`outputs/metrics.txt`) |
| Severity accuracy – held-out test / 5-fold CV | **95.2 % / 96.6 %** (`outputs/severity_metrics.txt`) |
| High-severity recall, ML only → hybrid | 86 % → **100 %** (7 test complaints) |
| Unseen hand-written complaints (24): category / priority at least as urgent as expected | 88 % / 92 % |
| Critical (P1) cases caught on unseen complaints | **8 / 8** |

Full details: `outputs/hybrid_report.txt`.

## Limitations (to be improved in the final review)
* The dataset is partly template-generated (1,000 rows → 415 unique), so held-out scores are higher than we should expect on real complaints.
  Only 7 High-severity complaints are in the test split, so the High-recall number is small-sample.
* The 24 unseen complaints are a small sanity check, not a benchmark. The ML models sometimes over-rate harmless complaints that use new words
  (e.g. a missing whiteboard marker came out as P2) and mislabel new vocabulary (e.g. "live cable … pole" as Water Leakage); the review flag and
  safety rules limit the damage, but the real fix is more real student complaints.
* Safety rules are keyword based and do not understand negation ("no fire hazard").
* Next: real complaint collection, dashboard + batch view, deployment, final report.
