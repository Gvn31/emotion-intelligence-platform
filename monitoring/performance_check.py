"""
Prediction monitoring for the Emotion Intelligence Platform.

Monitors prediction volume, confidence, and emotion distribution
from PostgreSQL. This is not a ground-truth model accuracy evaluation.

Does not modify the database, retrain the model, or deploy anything.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db_connection import get_connection  # noqa: E402

REPORT_PATH = (
    PROJECT_ROOT / "monitoring" / "reports" / "performance_report.json"
)

LOW_CONFIDENCE_THRESHOLD = 0.60
RECENT_WINDOW_HOURS = 24

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_prediction_data():
    """Read prediction records from feature_store without modifying it."""
    connection = None

    try:
        connection = get_connection()

        query = """
            SELECT
                created_at,
                emotion_label,
                emotion_confidence
            FROM feature_store
            WHERE created_at IS NOT NULL
            ORDER BY created_at ASC;
        """

        return pd.read_sql_query(query, connection)

    finally:
        if connection is not None:
            connection.close()


def main():
    logger.info("Starting prediction performance monitoring")

    df = load_prediction_data()

    now = pd.Timestamp.now(tz="UTC")
    report_time = datetime.now(timezone.utc).isoformat()

    report = {
        "generated_at": report_time,
        "monitoring_type": "prediction_operational_monitoring",
        "overall_status": "unavailable",
        "total_records": int(len(df)),
        "recent_window_hours": RECENT_WINDOW_HOURS,
        "recent_records": 0,
        "latest_record_at": None,
        "data_age_hours": None,
        "metrics": {},
        "emotion_distribution": {},
        "warnings": [],
        "note": (
            "Prediction confidence and distribution are operational "
            "indicators, not measurements of model accuracy. Accuracy "
            "and F1 require independent ground-truth labels."
        ),
    }

    if df.empty:
        report["warnings"].append(
            "No prediction records with timestamps were found."
        )
    else:
        df["created_at"] = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True
        )
        df = df.dropna(subset=["created_at"]).copy()

        if df.empty:
            report["warnings"].append(
                "No valid prediction timestamps were found."
            )
        else:
            latest_record = df["created_at"].max()
            data_age_hours = max(
                0.0, (now - latest_record).total_seconds() / 3600
            )

            recent_start = now - pd.Timedelta(
                hours=RECENT_WINDOW_HOURS
            )
            recent = df[df["created_at"] >= recent_start].copy()

            report["latest_record_at"] = latest_record.isoformat()
            report["data_age_hours"] = round(data_age_hours, 2)
            report["recent_records"] = int(len(recent))

            labels = (
                recent["emotion_label"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            labels = labels[labels != ""]

            report["emotion_distribution"] = {
                str(label): int(count)
                for label, count in labels.value_counts().items()
            }

            confidence = pd.to_numeric(
                recent["emotion_confidence"], errors="coerce"
            )
            confidence = confidence.replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            # Support either fractional confidence (0–1)
            # or percentage confidence (0–100).
            if not confidence.empty and confidence.max() > 1:
                if confidence.max() <= 100:
                    confidence = confidence / 100
                else:
                    report["warnings"].append(
                        "Confidence values exceed the expected range."
                    )
                    confidence = pd.Series(dtype=float)

            confidence = confidence[
                confidence.between(0, 1, inclusive="both")
            ]

            if len(recent) == 0:
                report["overall_status"] = "no_recent_predictions"
                report["warnings"].append(
                    "No records were created in the last 24 hours. "
                    "Prediction monitoring cannot assess recent activity."
                )
            else:
                report["overall_status"] = "monitoring_available"

                if confidence.empty:
                    report["warnings"].append(
                        "No valid confidence values were available "
                        "for recent predictions."
                    )
                else:
                    low_confidence_count = int(
                        (confidence < LOW_CONFIDENCE_THRESHOLD).sum()
                    )

                    report["metrics"] = {
                        "confidence_records": int(len(confidence)),
                        "mean_confidence": round(
                            float(confidence.mean()), 4
                        ),
                        "median_confidence": round(
                            float(confidence.median()), 4
                        ),
                        "low_confidence_threshold": (
                            LOW_CONFIDENCE_THRESHOLD
                        ),
                        "low_confidence_count": low_confidence_count,
                        "low_confidence_rate": round(
                            float(
                                (confidence < LOW_CONFIDENCE_THRESHOLD)
                                .mean()
                            ),
                            4,
                        ),
                    }

            if data_age_hours > RECENT_WINDOW_HOURS:
                report["warnings"].append(
                    "Latest prediction data is older than 24 hours."
                )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    logger.info("Report saved to %s", REPORT_PATH)
    logger.info("Overall status: %s", report["overall_status"])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Prediction monitoring failed")
        raise
