"""
Module 4 - Severity classifier (Low / Medium / High)  (CivicFix AI, Review 2)

Pipeline:  clean text -> TF-IDF (words + character n-grams) -> Logistic Regression (class-balanced)
Class balancing matters because only ~12 % of complaints are High severity and those are the
ones we must not miss.
Outputs:   severity_model.joblib, outputs/severity_metrics.txt, outputs/severity_confusion_matrix.png
Run:       python train_severity.py
"""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, make_union

from preprocess import load_data, split_data_full

LABELS = ["Low", "Medium", "High"]
os.makedirs("outputs", exist_ok=True)


def build_model():
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )
    return make_pipeline(features, LogisticRegression(C=20, max_iter=3000, class_weight="balanced"))


def main():
    df = load_data()
    train, test = split_data_full(df)
    print(f"Training samples: {len(train)}   Testing samples: {len(test)}")

    model = build_model()
    model.fit(train["clean"], train["severity"])
    pred = model.predict(test["clean"])

    acc = accuracy_score(test["severity"], pred)
    report = classification_report(test["severity"], pred, labels=LABELS, zero_division=0)
    cv = cross_val_score(build_model(), df["clean"], df["severity"],
                         cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"\nSeverity test accuracy: {acc:.1%}\n5-fold CV accuracy: {cv.mean():.1%} (+/- {cv.std():.1%})\n")
    print(report)

    with open("outputs/severity_metrics.txt", "w") as f:
        f.write(f"Severity test accuracy: {acc:.1%}\n5-fold cross-validation: {cv.mean():.1%} +/- {cv.std():.1%}\n\n{report}")

    cm = confusion_matrix(test["severity"], pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, colorbar=False)
    ax.set_title("CivicFix AI - Severity classifier (test set)")
    plt.tight_layout()
    plt.savefig("outputs/severity_confusion_matrix.png", dpi=150)

    model.fit(df["clean"], df["severity"])      # final model uses all data
    joblib.dump(model, "severity_model.joblib")
    print("Saved severity_model.joblib and outputs/severity_confusion_matrix.png")


if __name__ == "__main__":
    main()
