"""
Model Evaluation Module

Evaluates the fine-tuned emotion classification model
using the same stratified test split used during training.

Metrics:
- Accuracy
- Precision
- Recall
- Weighted F1
- ROC-AUC
- Confusion Matrix

Results:
- evaluation/metrics.json
- evaluation/confusion_matrix.png
- MLflow evaluation run
- Model Registry registration after validation
- @validated alias for validated models
"""

import json
import os

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

import matplotlib.pyplot as plt
import pandas as pd
import torch

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)

from db_connection import get_connection
from logger import logger


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/emotion_model"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://15.252.15.183:5000",
)

EXPERIMENT_NAME = "emotion_classification"

REGISTERED_MODEL_NAME = "emotion-classifier"

# MLflow S3 artifact bucket
MLFLOW_S3_BUCKET = os.getenv(
    "MLFLOW_S3_BUCKET",
    "emotion-intelligence-mlflow-901099689002-ap-south-1-an",
)

# Model validation thresholds
MIN_ACCURACY = 0.80
MIN_F1 = 0.80

EVALUATION_DIR = "evaluation"

METRICS_PATH = "evaluation/metrics.json"

CONFUSION_MATRIX_PATH = "evaluation/confusion_matrix.png"


# ============================================================
# Load Evaluation Data
# ============================================================

def load_data():
    """Load labeled data from the feature store."""

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

        df = pd.read_sql(
            query,
            conn
        )

        logger.info(
            f"Loaded {len(df)} records for evaluation"
        )

        print(
            f"Loaded {len(df)} evaluation records"
        )

        return df

    finally:
        if conn:
            conn.close()


# ============================================================
# Prepare Test Dataset
# ============================================================

def prepare_test_dataset(df):
    """Create the stratified test split."""

    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(
        df["emotion_label"]
    )

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print(
        f"Test records: {len(test_df)}"
    )

    return (
        test_df,
        label_encoder
    )


# ============================================================
# Prepare Hugging Face Dataset
# ============================================================

def prepare_dataset(
    test_df,
    tokenizer
):
    """Tokenize the evaluation dataset."""

    dataset = Dataset.from_pandas(
        test_df,
        preserve_index=False
    )

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    dataset = dataset.map(
        tokenize,
        batched=True
    )

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label",
        ],
    )

    return dataset


# ============================================================
# Register and Validate Model
# ============================================================

def register_and_validate_model(
    run_id,
    accuracy,
    f1,
):
    """
    Register the exact model evaluated in the current
    MLflow run.

    Registration happens only when the quality gate passes.

    The registered model version uses the direct S3 artifact
    location instead of mlflow-artifacts:/ so that clients
    can download the production model directly from S3.
    """

    client = MlflowClient()

    # --------------------------------------------------------
    # Quality Gate
    # --------------------------------------------------------

    validation_passed = (
        accuracy >= MIN_ACCURACY
        and
        f1 >= MIN_F1
    )

    print(
        "\nModel Quality Gate"
    )

    print(
        "------------------"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Required : {MIN_ACCURACY:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"Required : {MIN_F1:.4f}"
    )

    # --------------------------------------------------------
    # If validation fails
    # --------------------------------------------------------

    if not validation_passed:

        print(
            "\nModel FAILED validation."
        )

        logger.warning(
            "Model failed validation. "
            "Model will not be registered."
        )

        return None

    # --------------------------------------------------------
    # Register evaluated model using direct S3 artifact source
    # --------------------------------------------------------

    print(
        "\nModel passed validation."
    )

    print(
        "Registering evaluated model with direct S3 artifact source..."
    )

    # The MLflow experiment stores artifacts using:
    #
    # <experiment_id>/<run_id>/artifacts/emotion_model
    #
    # Retrieve the experiment ID from the current evaluation run.

    client_run = client.get_run(run_id)

    experiment_id = client_run.info.experiment_id

    # Build the direct S3 location of the model logged
    # by mlflow.pytorch.log_model(..., "emotion_model").

    s3_model_uri = (
        f"s3://{MLFLOW_S3_BUCKET}/"
        f"{experiment_id}/"
        f"{run_id}/"
        f"artifacts/emotion_model"
    )

    print(
        f"S3 model source: {s3_model_uri}"
    )

    # Create the registry version directly from S3.
    #
    # IMPORTANT:
    # This avoids registering the model as:
    #
    # mlflow-artifacts:/...
    #
    # and instead stores:
    #
    # s3://...
    #
    # as the model source.

    registered_model = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=s3_model_uri,
        run_id=run_id,
    )

    version = registered_model.version

    print(
        f"Registered model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version: {version}"
    )

    # --------------------------------------------------------
    # Add evaluation metadata
    # --------------------------------------------------------

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "validation_status",
        "passed",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "evaluation_accuracy",
        f"{accuracy:.4f}",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "evaluation_f1",
        f"{f1:.4f}",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "evaluation_run_id",
        run_id,
    )

    # --------------------------------------------------------
    # Assign validated alias
    # --------------------------------------------------------

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "validated",
        version,
    )

    print(
        f"Alias '@validated' assigned "
        f"to version {version}."
    )

    logger.info(
        f"Model {REGISTERED_MODEL_NAME} "
        f"version {version} registered successfully"
    )

    logger.info(
        f"Model version {version} "
        f"received @validated alias"
    )

    return version


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def main():

    try:

        logger.info(
            "Model evaluation started"
        )

        # ----------------------------------------------------
        # Create evaluation directory
        # ----------------------------------------------------

        os.makedirs(
            EVALUATION_DIR,
            exist_ok=True
        )

        # ----------------------------------------------------
        # MLflow configuration
        # ----------------------------------------------------

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            EXPERIMENT_NAME
        )

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        df = load_data()

        # ----------------------------------------------------
        # Prepare test dataset
        # ----------------------------------------------------

        (
            test_df,
            label_encoder
        ) = prepare_test_dataset(
            df
        )

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        # ----------------------------------------------------
        # Prepare test dataset
        # ----------------------------------------------------

        test_dataset = prepare_dataset(
            test_df,
            tokenizer
        )

        # ----------------------------------------------------
        # Trainer
        # ----------------------------------------------------

        trainer = Trainer(
            model=model
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = trainer.predict(
            test_dataset
        )

        logits = predictions.predictions

        y_true = predictions.label_ids

        probabilities = torch.softmax(
            torch.tensor(logits),
            dim=1
        ).numpy()

        y_pred = probabilities.argmax(
            axis=1
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="weighted"
        )

        metrics = {
            "accuracy": float(
                accuracy
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1_score": float(
                f1
            ),
            "roc_auc": float(
                roc_auc
            ),
        }

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        with open(
            METRICS_PATH,
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        labels = label_encoder.classes_

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels,
        ).plot(
            ax=ax,
            xticks_rotation=45,
        )

        plt.title(
            "Emotion Classification Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            CONFUSION_MATRIX_PATH,
            dpi=300
        )

        plt.close()

        # ----------------------------------------------------
        # MLflow Evaluation Run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name="model_evaluation"
        ) as run:

            run_id = run.info.run_id

            mlflow.set_tag(
                "phase",
                "model_evaluation"
            )

            mlflow.set_tag(
                "model_path",
                MODEL_PATH
            )

            mlflow.set_tag(
                "registered_model",
                REGISTERED_MODEL_NAME
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.log_artifact(
                METRICS_PATH
            )

            mlflow.log_artifact(
                CONFUSION_MATRIX_PATH
            )

            # ------------------------------------------------
            # Log the exact model being evaluated
            # ------------------------------------------------

            mlflow.pytorch.log_model(
                model,
                "emotion_model"
            )

            # ----------------------------------------------------
            # Console Output
            # ----------------------------------------------------

            print(
                "\nModel Evaluation Results"
            )

            print(
                "------------------------"
            )

            print(
                f"Accuracy : {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall   : {recall:.4f}"
            )

            print(
                f"F1 Score : {f1:.4f}"
            )

            print(
                f"ROC-AUC  : {roc_auc:.4f}"
            )

            # ----------------------------------------------------
            # Phase 7 - Model Registry
            # ----------------------------------------------------

            registered_version = register_and_validate_model(
                run_id=run_id,
                accuracy=accuracy,
                f1=f1,
            )

        # ----------------------------------------------------
        # Final Result
        # ----------------------------------------------------

        if registered_version:

            print(
                "\nModel Registry update successful."
            )

            print(
                f"Model    : {REGISTERED_MODEL_NAME}"
            )

            print(
                f"Version  : {registered_version}"
            )

            print(
                "Alias    : @validated"
            )

        else:

            print(
                "\nModel was not registered "
                "because validation failed."
            )

        print(
            "\nEvaluation completed successfully."
        )

        logger.info(
            "Model evaluation completed successfully"
        )

    except Exception as e:

        logger.error(
            f"Model evaluation failed: {e}"
        )

        print(
            f"ERROR: {e}"
        )

        raise


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()