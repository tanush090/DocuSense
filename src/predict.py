
"""Predict whether document text is a Bill, Invoice, or Receipt."""

import argparse
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "document_classifier.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Add document_classifier.pkl to the repository root."
        )

    return joblib.load(MODEL_PATH)


def predict_document(text, model):
    """Return the predicted class and available confidence scores."""
    text = text.strip()

    if not text:
        raise ValueError("Document text cannot be empty.")

    prediction = model.predict([text])[0]
    confidence_scores = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]

        confidence_scores = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }

    return prediction, confidence_scores


def main():
    parser = argparse.ArgumentParser(
        description="Classify text as Bill, Invoice, or Receipt."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text extracted from the document.",
    )
    args = parser.parse_args()

    model = load_model()
    prediction, scores = predict_document(args.text, model)

    print(f"\nPredicted document type: {prediction}")

    if scores:
        print("\nConfidence scores:")
        for label, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {label}: {score:.2%}")


if __name__ == "__main__":
    main()
