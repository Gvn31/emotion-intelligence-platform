"""
Model Training Module

Loads labeled data from PostgreSQL,
fine-tunes an emotion classification model,
logs metrics to MLflow,
and saves the trained model.
"""

import os

import joblib
import mlflow
import mlflow.pytorch
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


def load_training_data():
    """Load labeled records from PostgreSQL."""
    conn = None

    try:
        conn = get_connection()

        query = """
        SELECT
            clean_text,
            emotion_label
        FROM feature_store
        WHERE emotion_label IS NOT NULL
        AND emotion_label != 'disgust';
        """

        df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} training records")
        print(f"Loaded {len(df)} training records")

        return df

    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise

    finally:
        if conn:
            conn.close()


def prepare_dataset(df):
    """Encode labels and split the dataset."""
    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(
        df["emotion_label"]
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    joblib.dump(
        label_encoder,
        "models/label_encoder.pkl"
    )

    logger.info("Label encoder saved")

    return train_df, test_df, label_encoder


def tokenize_data(train_df, test_df):
    """Create Hugging Face datasets and tokenize the text."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = Dataset.from_pandas(
        train_df
    )

    test_dataset = Dataset.from_pandas(
        test_df
    )

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True
    )

    columns = [
        "input_ids",
        "attention_mask",
        "label"
    ]

    train_dataset.set_format(
        type="torch",
        columns=columns
    )

    test_dataset.set_format(
        type="torch",
        columns=columns
    )

    return (
        train_dataset,
        test_dataset,
        tokenizer
    )


def compute_metrics(eval_pred):
    """Calculate accuracy and weighted F1."""
    logits, labels = eval_pred

    predictions = logits.argmax(
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


if __name__ == "__main__":

    try:

        logger.info(
            "Training started"
        )

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------

        df = load_training_data()

        train_df, test_df, label_encoder = prepare_dataset(
            df
        )

        # --------------------------------------------------
        # Tokenization
        # --------------------------------------------------

        (
            train_dataset,
            test_dataset,
            tokenizer
        ) = tokenize_data(
            train_df,
            test_df
        )

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {device}"
        )

        if torch.cuda.is_available():

            print(
                torch.cuda.get_device_name(0)
            )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(
                label_encoder.classes_
            ),
            ignore_mismatched_sizes=True,
        )

        model.to(device)

        # --------------------------------------------------
        # Training Arguments
        # --------------------------------------------------

        training_args = TrainingArguments(
            output_dir="models/checkpoints",
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
        )

        # --------------------------------------------------
        # Trainer
        # --------------------------------------------------

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        # --------------------------------------------------
        # MLflow
        # --------------------------------------------------

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            "emotion_classification"
        )

        with mlflow.start_run():

            mlflow.log_param(
                "model_name",
                MODEL_NAME
            )

            mlflow.log_param(
                "epochs",
                3
            )

            mlflow.log_param(
                "batch_size",
                16
            )

            mlflow.log_param(
                "learning_rate",
                2e-5
            )

            mlflow.log_param(
                "num_labels",
                len(
                    label_encoder.classes_
                )
            )

            mlflow.log_artifact(
                "models/label_encoder.pkl"
            )

            # --------------------------------------------------
            # Train
            # --------------------------------------------------

            trainer.train()

            # --------------------------------------------------
            # Evaluate during training
            # --------------------------------------------------

            metrics = trainer.evaluate()

            mlflow.log_metrics(
                metrics
            )

            # --------------------------------------------------
            # Remove mixed-precision wrapper
            # --------------------------------------------------

            model_to_save = (
                trainer.accelerator.unwrap_model(
                    trainer.model_wrapped,
                    keep_fp32_wrapper=False,
                )
            )

            # --------------------------------------------------
            # Save trained model locally
            # --------------------------------------------------

            model_to_save.save_pretrained(
                "models/emotion_model"
            )

            tokenizer.save_pretrained(
                "models/emotion_model"
            )

            # --------------------------------------------------
            # Log model artifact to MLflow
            # --------------------------------------------------

            mlflow.pytorch.log_model(
                model_to_save,
                "emotion_model",
            )

        # --------------------------------------------------
        # Completion
        # --------------------------------------------------

        logger.info(
            "Model training completed"
        )

        print(
            "\nModel training completed successfully."
        )

        print(
            f"Accuracy: "
            f"{metrics['eval_accuracy']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{metrics['eval_f1']:.4f}"
        )

    except Exception as e:

        logger.error(
            f"Training failed: {e}"
        )

        print(
            f"ERROR: {e}"
        )