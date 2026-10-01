"""
Emotion Prediction Module

Loads the validated emotion classification model from
MLflow Model Registry and performs emotion prediction.
"""

import os
import joblib
import mlflow
import mlflow.pytorch
import torch

from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://15.252.15.183:5000"
)

MODEL_URI = "models:/emotion-classifier@validated"

TOKENIZER_PATH = "models/emotion_model"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

MAX_LENGTH = 128


# ============================================================
# Device Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# Load Model
# ============================================================

print("\nLoading validated model from MLflow...")

model = mlflow.pytorch.load_model(
    MODEL_URI
)

model = model.to(DEVICE)

model.eval()

print(
    "Validated model loaded successfully."
)


# ============================================================
# Load Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH
)

print(
    "Tokenizer loaded successfully."
)


# ============================================================
# Load Label Encoder
# ============================================================

print("\nLoading label encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print(
    f"Emotion classes: {list(label_encoder.classes_)}"
)


# ============================================================
# Prediction Function
# ============================================================

def predict_emotion(text):
    """
    Predict emotion for a given text.

    Args:
        text (str):
            Input text.

    Returns:
        dict:
            prediction
            emotion
            confidence
    """

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------

    if not isinstance(text, str):
        raise TypeError(
            "Input text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Input text cannot be empty."
        )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    # --------------------------------------------------------
    # Move Inputs to Same Device as Model
    # --------------------------------------------------------

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    predicted_index = predicted_class.item()

    confidence_score = confidence.item()

    # --------------------------------------------------------
    # Decode Emotion
    # --------------------------------------------------------

    emotion = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "prediction": emotion,
        "emotion": emotion,
        "confidence": round(
            confidence_score * 100,
            2
        ),
    }

    return result


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":

    test_text = (
        "I am extremely happy today!"
    )

    result = predict_emotion(
        test_text
    )

    print("\nPrediction Result")
    print("-----------------")

    print(
        f"Text       : {test_text}"
    )

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Emotion    : {result['emotion']}"
    )

    print(
        f"Confidence : {result['confidence']}%"
    )
    