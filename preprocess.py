"""
Module 1 - Data loading and text pre-processing  (CivicFix AI, Review 2)

What this file does:
  1. Loads dataset.csv (complaint, category, severity).
  2. Cleans the text (lower-case, remove extra spaces and symbols).
  3. Removes duplicate complaints so the accuracy score is not inflated.
  4. Splits the data into training and testing sets.
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = "dataset.csv"


def clean_text(text: str) -> str:
    """Lower-case the text and keep only letters, numbers and spaces."""
    text = str(text).lower()
    text = re.sub(r"^please report:\s*", "", text)      # remove template prefix found in the dataset
    text = re.sub(r"\s+and needs attention$", "", text)  # remove template suffixes found in the dataset
    text = re.sub(r"\s+on campus$", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # remove symbols
    text = re.sub(r"\s+", " ", text).strip()           # remove extra spaces
    return text


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["complaint", "category"])
    df["clean"] = df["complaint"].apply(clean_text)
    before = len(df)
    df = df.drop_duplicates(subset=["clean", "category"]).reset_index(drop=True)
    print(f"Loaded {before} rows -> {len(df)} after removing duplicates")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split keeps the same category proportions in train and test."""
    return train_test_split(
        df["clean"], df["category"], test_size=test_size, random_state=seed, stratify=df["category"]
    )


def split_data_full(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Same split as split_data but keeps category AND severity (Review 2). Stratified on category,
    so the test rows are identical to the Review 2 category test set."""
    return train_test_split(
        df[["clean", "category", "severity"]], test_size=test_size, random_state=seed, stratify=df["category"]
    )


if __name__ == "__main__":
    data = load_data()
    print("\nCategory distribution:")
    print(data["category"].value_counts().to_string())
    print("\nSample cleaned complaints:")
    print(data[["complaint", "clean"]].head(5).to_string(index=False))
