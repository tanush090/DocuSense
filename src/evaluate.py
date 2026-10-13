
"""Evaluate the trained DocuSense document classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "document_classifier.pkl"
TEST_DATA_PATH = ROOT_DIR / "artifacts" / "test_data.csv"
REPORT_DIR = ROOT_DIR / "artifacts"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run src/train.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA_PATH}. Run src/train.py first."
        )

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    test_df = test_df.dropna(subset=["text", "label"])
    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(str)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("=" * 55)
    print("DOCUSENSE — MODEL EVALUATION")
    print("=" * 55)
    print(f"Test documents: {len(test_df)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    labels = sorted(set(y_test) | set(predictions))
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("Confusion Matrix:")
    print(pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    ))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = test_df.copy()
    results["predicted_label"] = predictions
    results.to_csv(
        REPORT_DIR / "evaluation_predictions.csv",
        index=False,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(xticks_rotation=30, values_format="d")
    plt.title("DocuSense — Confusion Matrix")
    plt.tight_layout()

    plot_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\nPrediction results saved to: {REPORT_DIR / 'evaluation_predictions.csv'}")
    print(f"Confusion matrix image saved to: {plot_path}")


if __name__ == "__main__":
    main()
