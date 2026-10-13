"""
Model Evaluation Module — Emotion Intelligence Platform

Evaluates a candidate emotion model, generates metrics and a confusion
matrix, logs the evaluation to MLflow, and registers the candidate only
when both accuracy and weighted F1 meet the required thresholds.

Does not overwrite or deploy the production model.
"""

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)

from db_connection import get_connection
from logger import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_MODEL_PATH = PROJECT_ROOT / "models" / "candidate_emotion_model"
CANDIDATE_ENCODER_PATH = PROJECT_ROOT / "models" / "candidate_label_encoder.pkl"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://15.252.15.183:5000",
)

MLFLOW_S3_BUCKET = os.getenv(
    "MLFLOW_S3_BUCKET",
    "emotion-intelligence-mlflow-901099689002-ap-south-1-an",
)

EXPERIMENT_NAME = "emotion_classification"
REGISTERED_MODEL_NAME = "emotion-classifier"

MIN_ACCURACY = 0.80
MIN_F1 = 0.80

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
METRICS_PATH = EVALUATION_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = EVALUATION_DIR / "confusion_matrix.png"


def load_data():
    """Load labeled records from PostgreSQL."""
    connection = None

    try:
        connection = get_connection()

        query = """
            SELECT clean_text, emotion_label
            FROM feature_store
            WHERE clean_text IS NOT NULL
              AND TRIM(clean_text) <> ''
              AND emotion_label IS NOT NULL
              AND TRIM(emotion_label) <> ''
              AND emotion_label <> 'disgust';
        """

        df = pd.read_sql_query(query, connection)
        df["clean_text"] = df["clean_text"].astype(str).str.strip()
        df["emotion_label"] = df["emotion_label"].astype(str).str.strip()

        df = df[
            (df["clean_text"] != "")
            & (df["emotion_label"] != "")
        ].drop_duplicates(
            subset=["clean_text", "emotion_label"]
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid evaluation records were found.")

        logger.info("Loaded %d evaluation records", len(df))
        print(f"Loaded {len(df)} evaluation records")

        return df

    finally:
        if connection is not None:
            connection.close()


def prepare_test_dataset(df, label_encoder):
    """Recreate the deterministic stratified split used during training."""
    df = df.copy()

    known_labels = set(label_encoder.classes_)
    observed_labels = set(df["emotion_label"].unique())

    unknown_labels = observed_labels - known_labels
    if unknown_labels:
        raise ValueError(
            "Evaluation data contains labels unknown to the candidate "
            f"encoder: {sorted(unknown_labels)}"
        )

    df["label"] = label_encoder.transform(df["emotion_label"])
    class_counts = df["label"].value_counts()

    if len(class_counts) != len(label_encoder.classes_):
        raise ValueError(
            "Evaluation data does not contain every candidate emotion class. "
            "Use a consistent evaluation dataset before registering the model."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each emotion class needs at least two records for "
            "stratified evaluation."
        )

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print(f"Test records: {len(test_df)}")
    return test_df.reset_index(drop=True)


def tokenize_test_data(test_df, tokenizer):
    """Tokenize test records for Transformers."""
    dataset = Dataset.from_pandas(
        test_df[["clean_text", "label"]],
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    return dataset


def calculate_metrics(y_true, y_pred, probabilities):
    """Calculate classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
    }

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=np.arange(probabilities.shape[1]),
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError as exc:
        metrics["roc_auc"] = None
        logger.warning("ROC-AUC could not be calculated: %s", exc)

    return metrics


def register_validated_model(run_id, metrics):
    """Register the exact candidate artifact only if quality gates pass."""
    accuracy = metrics["accuracy"]
    weighted_f1 = metrics["f1_score"]

    passed = accuracy >= MIN_ACCURACY and weighted_f1 >= MIN_F1

    print("\nModel Quality Gate")
    print("------------------")
    print(f"Accuracy: {accuracy:.4f} (required: {MIN_ACCURACY:.2f})")
    print(f"Weighted F1: {weighted_f1:.4f} (required: {MIN_F1:.2f})")

    if not passed:
        logger.warning(
            "Candidate failed quality gates; it will not be registered."
        )
        print("Candidate FAILED validation. Production model unchanged.")
        return None

    client = MlflowClient()
    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id

    model_source = (
        f"s3://{MLFLOW_S3_BUCKET}/"
        f"{experiment_id}/{run_id}/artifacts/emotion_model"
    )

    # Ensure the registered model exists.
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    version_info = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_source,
        run_id=run_id,
    )

    version = version_info.version

    for key, value in {
        "validation_status": "passed",
        "evaluation_accuracy": f"{accuracy:.6f}",
        "evaluation_f1": f"{weighted_f1:.6f}",
        "evaluation_run_id": run_id,
    }.items():
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            key,
            value,
        )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "validated",
        version,
    )

    logger.info(
        "Validated model registered: %s version %s",
        REGISTERED_MODEL_NAME,
        version,
    )

    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Version: {version}")
    print("Alias: @validated")

    return version


def main():
    logger.info("Candidate model evaluation started")

    if not CANDIDATE_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Candidate model not found: {CANDIDATE_MODEL_PATH}. "
            "Run src/train.py first."
        )

    if not CANDIDATE_ENCODER_PATH.is_file():
        raise FileNotFoundError(
            f"Candidate label encoder not found: {CANDIDATE_ENCODER_PATH}"
        )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    label_encoder = joblib.load(CANDIDATE_ENCODER_PATH)

    if not hasattr(label_encoder, "classes_"):
        raise ValueError("Candidate label encoder is invalid.")

    test_df = prepare_test_dataset(df, label_encoder)

    tokenizer = AutoTokenizer.from_pretrained(
        str(CANDIDATE_MODEL_PATH)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CANDIDATE_MODEL_PATH)
    )

    if model.config.num_labels != len(label_encoder.classes_):
        raise ValueError(
            "Candidate model output count does not match the label encoder."
        )

    expected_mapping = {
        index: label
        for index, label in enumerate(label_encoder.classes_)
    }

    if model.config.id2label and model.config.id2label != expected_mapping:
        logger.warning(
            "Model label names differ from the encoder mapping; "
            "the saved encoder defines the evaluation class order."
        )

    test_dataset = tokenize_test_data(test_df, tokenizer)

    trainer = Trainer(model=model)
    prediction_output = trainer.predict(test_dataset)

    logits = prediction_output.predictions
    y_true = prediction_output.label_ids
    probabilities = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=1
    ).cpu().numpy()
    y_pred = probabilities.argmax(axis=1)

    metrics = calculate_metrics(y_true, y_pred, probabilities)
    metrics["evaluated_records"] = int(len(y_true))
    metrics["min_accuracy"] = MIN_ACCURACY
    metrics["min_weighted_f1"] = MIN_F1

    quality_passed = (
        metrics["accuracy"] >= MIN_ACCURACY
        and metrics["f1_score"] >= MIN_F1
    )
    metrics["quality_gate_passed"] = bool(quality_passed)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, allow_nan=False)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=label_encoder.classes_,
    ).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Emotion Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(fig)

    registered_version = None

    with mlflow.start_run(run_name="candidate_model_evaluation") as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "phase": "candidate_model_evaluation",
            "model_path": str(CANDIDATE_MODEL_PATH),
            "registered_model": REGISTERED_MODEL_NAME,
            "quality_gate_passed": str(quality_passed).lower(),
        })

        mlflow.log_metrics({
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and value is not None
            and key not in {
                "evaluated_records",
                "min_accuracy",
                "min_weighted_f1",
            }
        })

        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(CONFUSION_MATRIX_PATH))
        mlflow.pytorch.log_model(model, "emotion_model")

        registered_version = register_validated_model(run_id, metrics)

    print("\nCandidate Evaluation Results")
    print("----------------------------")
    print(json.dumps(metrics, indent=2))
    print(f"\nMetrics: {METRICS_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")

    if registered_version is None:
        print("\nCandidate was not registered. Production model unchanged.")
    else:
        print(f"\nValidated registry version: {registered_version}")
        print("No ECS deployment was performed.")

    logger.info("Candidate evaluation completed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Candidate model evaluation failed")
        raise
