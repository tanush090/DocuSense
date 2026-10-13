
"""Train the DocuSense Bill / Invoice / Receipt classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Paths are relative to the repository root.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "synthetic_indian_bill_invoice_receipt_3000.csv"
MODEL_PATH = ROOT_DIR / "document_classifier.pkl"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TEST_DATA_PATH = ARTIFACTS_DIR / "test_data.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the text-classification dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place synthetic_indian_bill_invoice_receipt_3000.csv "
            "inside the repository's data/ folder."
        )

    df = pd.read_csv(path)

    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if df["label"].nunique() < 2:
        raise ValueError("At least two different labels are required.")

    print(f"Dataset shape after cleaning: {df.shape}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df


def main():
    df = load_dataset(DATA_PATH)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTraining complete!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save the model expected by app.py.
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Preserve the held-out test set for evaluate.py.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.DataFrame({
        "text": X_test.to_numpy(),
        "label": y_test.to_numpy(),
    })
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"Test data saved to: {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
