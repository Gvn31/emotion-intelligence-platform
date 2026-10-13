"""
Model Training Module — Emotion Intelligence Platform

Loads labeled data from PostgreSQL, fine-tunes an emotion classifier,
evaluates the candidate model, and saves candidate artifacts separately.

The existing production model is never overwritten by this script.
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.transformers
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from db_connection import get_connection
from logger import logger


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://15.252.15.183:5000",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_MODEL_PATH = PROJECT_ROOT / "models" / "candidate_emotion_model"
CANDIDATE_ENCODER_PATH = PROJECT_ROOT / "models" / "candidate_label_encoder.pkl"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "candidate_checkpoints"


def load_training_data():
    """Load and validate labeled training records from PostgreSQL."""
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
        df = df.drop_duplicates(subset=["clean_text", "emotion_label"])

        if df.empty:
            raise ValueError("No valid labeled training records found.")

        if df["emotion_label"].nunique() < 2:
            raise ValueError("At least two emotion classes are required.")

        logger.info("Loaded %d training records", len(df))
        print(f"Loaded {len(df)} training records")

        return df.reset_index(drop=True)

    except Exception:
        logger.exception("Failed to load training data")
        raise

    finally:
        if connection is not None:
            connection.close()


def prepare_dataset(df):
    """Encode labels and create a stratified train/test split."""
    df = df.copy()

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["emotion_label"])

    class_counts = df["label"].value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each emotion class must contain at least two records "
            "for stratified splitting."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    CANDIDATE_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, CANDIDATE_ENCODER_PATH)

    logger.info(
        "Candidate label encoder saved to %s",
        CANDIDATE_ENCODER_PATH,
    )

    return train_df, test_df, label_encoder


def tokenize_data(train_df, test_df):
    """Tokenize training and test data using the pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def build_dataset(frame):
        dataset = Dataset.from_pandas(
            frame[["clean_text", "label"]].reset_index(drop=True)
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

    return build_dataset(train_df), build_dataset(test_df), tokenizer


def compute_metrics(eval_pred):
    """Calculate accuracy and weighted F1."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def main():
    """Train a candidate model without overwriting the production model."""
    logger.info("Candidate model training started")

    df = load_training_data()
    train_df, test_df, label_encoder = prepare_dataset(df)

    train_dataset, test_dataset, tokenizer = tokenize_data(
        train_df, test_df
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True,
    )

    # Store the class-to-index mapping in the model configuration.
    model.config.id2label = {
        index: label
        for index, label in enumerate(label_encoder.classes_)
    }
    model.config.label2id = {
        label: index
        for index, label in enumerate(label_encoder.classes_)
    }

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_PATH),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("emotion_classification")

    with mlflow.start_run(run_name="candidate_emotion_training") as run:
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_labels": len(label_encoder.classes_),
            "training_records": len(train_df),
            "test_records": len(test_df),
            "candidate_model_path": str(CANDIDATE_MODEL_PATH),
        })

        mlflow.log_artifact(str(CANDIDATE_ENCODER_PATH))

        trainer.train()
        metrics = trainer.evaluate()

        accuracy = float(metrics["eval_accuracy"])
        weighted_f1 = float(metrics["eval_f1"])

        mlflow.log_metrics({
            "final_accuracy": accuracy,
            "final_weighted_f1": weighted_f1,
        })

        # Save only to the candidate directory.
        # Do not overwrite models/emotion_model here.
        CANDIDATE_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        trainer.save_model(str(CANDIDATE_MODEL_PATH))
        tokenizer.save_pretrained(str(CANDIDATE_MODEL_PATH))

        # Log the saved Hugging Face model to MLflow.
        # Explicit requirements prevent MLflow from probing for TensorFlow.
        mlflow.transformers.log_model(
            transformers_model=str(CANDIDATE_MODEL_PATH),
            artifact_path="emotion_model",
            task="text-classification",
            pip_requirements=[
                "mlflow==2.22.1",
                "torch",
                "transformers",
                "sentencepiece",
            ],
        )

        mlflow.set_tags({
            "model_role": "candidate",
            "validation_status": "pending",
        })

        print("\nCandidate Training Results")
        print("--------------------------")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Candidate model: {CANDIDATE_MODEL_PATH}")
        print(f"Candidate encoder: {CANDIDATE_ENCODER_PATH}")

    logger.info("Training completed successfully")
    print("\nTraining completed successfully.")
    print("The existing production model was not overwritten.")
    print("Run evaluation before promoting this candidate.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Candidate model training failed")
        raise
