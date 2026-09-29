"""
Emotion Inference Module

Loads feature_store records,
predicts emotions using a pretrained model,
updates emotion_label,
sentiment_score,
emotion_confidence.
"""

from transformers import pipeline

from db_connection import get_connection
from logger import logger

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def load_model():
    """
    Load pretrained emotion model.
    """

    try:

        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None
        )

        logger.info(
            "Emotion model loaded successfully"
        )

        return classifier

    except Exception as e:

        logger.error(
            f"Failed to load model: {e}"
        )

        raise e


def load_records():
    """
    Load records that need emotion prediction.
    """

    conn = None
    cursor = None

    try:

        conn = get_connection()

        query = """
        SELECT
            id,
            clean_text
        FROM feature_store
        WHERE
            emotion_label IS NULL
            OR emotion_label = 'unknown';
        """

        cursor = conn.cursor()

        cursor.execute(query)

        rows = cursor.fetchall()

        logger.info(
            f"Loaded {len(rows)} records"
        )

        print(
            f"Loaded {len(rows)} records"
        )

        return rows

    except Exception as e:

        logger.error(
            f"Failed to load records: {e}"
        )

        raise e

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


def predict_emotion(
    classifier,
    text
):
    """
    Predict emotion from text.
    """

    try:

        if not text:

            return (
                "unknown",
                0.0
            )

        result = classifier(
            text[:512]
        )[0]

        best_prediction = max(
            result,
            key=lambda x: x["score"]
        )

        return (
            best_prediction["label"],
            float(best_prediction["score"])
        )

    except Exception as e:

        logger.error(
            f"Prediction failed: {e}"
        )

        return (
            "unknown",
            0.0
        )


def update_predictions(
    predictions
):
    """
    Update predictions in feature_store.
    """

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        update_query = """
        UPDATE feature_store
        SET
            emotion_label = %s,
            sentiment_score = %s,
            emotion_confidence = %s
        WHERE id = %s;
        """

        cursor.executemany(
            update_query,
            predictions
        )

        conn.commit()

        logger.info(
            f"{len(predictions)} records updated"
        )

        print(
            f"{len(predictions)} records updated"
        )

    except Exception as e:

        logger.error(
            f"Update failed: {e}"
        )

        raise e

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


if __name__ == "__main__":

    try:

        logger.info(
            "Emotion inference started"
        )

        classifier = load_model()

        rows = load_records()

        if len(rows) == 0:

            print(
                "No records available."
            )

        else:

            predictions = []

            for row in rows:

                record_id = row[0]

                text = row[1]

                emotion_label, confidence = (
                    predict_emotion(
                        classifier,
                        text
                    )
                )

                predictions.append(
                    (
                        emotion_label,
                        0.0,
                        confidence,
                        record_id
                    )
                )

            update_predictions(
                predictions
            )

            logger.info(
                "Emotion inference completed"
            )

            print(
                "\nEmotion inference completed successfully."
            )

    except Exception as e:

        logger.error(
            f"Pipeline failed: {e}"
        )

        print(
            f"ERROR: {e}"
        )