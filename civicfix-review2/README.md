# CivicFix AI — Review 2 submission (≈35 % of the project)

**CivicFix AI** is an AI system that reads a campus complaint written in plain English and decides *what it is about*,
*how dangerous it is* and *who should fix it first*.

## Progress at Review 2

| # | Module | Weight | Status |
|---|--------|:------:|--------|
| 1 | Problem study, dataset collection (1,000 labelled complaints) | 10 % | ✅ Done |
| 2 | Text pre-processing + category classifier (TF-IDF + Logistic Regression) | 15 % | ✅ Done |
| 3 | Basic web app (Flask) with complaint form and prediction API | 10 % | ✅ Done |
| 4 | Severity classifier (Low / Medium / High) | 12 % | ⏳ Review 3 |
| 5 | Hybrid engine: safety rules + ML, priority score (P1–P4), SLA, department | 18 % | ⏳ Review 3 |
| 6 | Explainability (why this result), low-confidence flag, batch ranking | 10 % | ⏳ Final |
| 7 | Advanced UI (dashboard, history, dark mode), deployment on Vercel | 10 % | ⏳ Final |
| 8 | Testing, performance tuning, final report | 15 % | ⏳ Final |
| | **Completed so far** | **35 %** | |

## Project structure
```
dataset.csv          1,000 labelled complaints (complaint, category, severity)
preprocess.py        Module 1 – load, clean, remove duplicates, train/test split
train_category.py    Module 2 – trains the category model, saves metrics + confusion matrix
predict.py           command-line prediction
app.py               Module 3 – Flask web application
templates/index.html simple web page
outputs/             metrics.txt and confusion_matrix.png (results for the report)
```

## How to run
```bash
pip install -r requirements.txt
python preprocess.py          # shows data cleaning and category distribution
python train_category.py      # trains the model, prints accuracy, saves outputs/
python predict.py "Water is leaking from the tap in the lab"
python app.py                 # open http://127.0.0.1:5000
```

## Method
1. **Cleaning:** lower-case, remove symbols and template phrases, remove duplicate complaints (1,000 → 415 unique).
2. **Features:** TF-IDF on words and word pairs + character n-grams (handles spelling mistakes).
3. **Model:** Logistic Regression, 10 categories (Electrical, Plumbing, Water Leakage, Toilet/Sanitation, Garbage, Road/Pathway, Pothole, Civil, Carpentry, Class Items).
4. **Evaluation:** 80/20 stratified split → 332 training, 83 testing samples.

## Result so far
Category accuracy on the held-out test set: **98.8 %** (see `outputs/metrics.txt`, `outputs/confusion_matrix.png`).

**Limitation (to be improved in later reviews):** the dataset is partly template-generated, so many complaints are similar to each other and
the test score is higher than we should expect on completely new, real complaints. Next steps are collecting real student complaints,
adding the severity model and priority engine, and testing on unseen wording.
