"""
Module 2 - Complaint category classifier (baseline model)  (CivicFix AI, Review 2)

Pipeline:  clean text -> TF-IDF features (words + character n-grams) -> Logistic Regression
Outputs:   model.joblib, outputs/confusion_matrix.png, outputs/metrics.txt
Run:       python train_category.py
"""
import os
import joblib
import matplotlib
matplotlib.use("Agg")                      # draw charts without opening a window
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline, make_union

from preprocess import load_data, split_data

os.makedirs("outputs", exist_ok=True)


def build_model():
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),                 # whole words and word pairs
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),  # handles spelling mistakes
    )
    return make_pipeline(features, LogisticRegression(C=20, max_iter=2000))


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training samples: {len(X_train)}   Testing samples: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0)
    print(f"\nTest accuracy: {acc:.1%}\n")
    print(report)

    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.1%}\n\n{report}")

    # confusion matrix picture for the review report
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("CivicFix AI - Category classifier (test set)")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)

    model.fit(df["clean"], df["category"])          # final model uses all data
    joblib.dump(model, "model.joblib")
    print("Saved model.joblib and outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
