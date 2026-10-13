
"""
Feature drift detection for the Emotion Intelligence Platform.

Compares older reference records with newer records from PostgreSQL.
Generates a JSON report only. Does not retrain or deploy anything.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db_connection import get_connection  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "monitoring" / "reports" / "drift_report.json"

# Start with measurable numeric features available in feature_store.
FEATURES = [
    "text_length",
    "word_count",
    "avg_word_length",
    "unique_word_count",
    "sentiment_score",
]

REFERENCE_DAYS = 30
CURRENT_DAYS = 3
MIN_ROWS_PER_WINDOW = 30
P_VALUE_THRESHOLD = 0.05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_feature_data():
    """Read feature records without modifying the database."""
    connection = None

    try:
        connection = get_connection()

        query = """
            SELECT
                created_at,
                text_length,
                word_count,
                avg_word_length,
                unique_word_count,
                sentiment_score
            FROM feature_store
            WHERE created_at IS NOT NULL
            ORDER BY created_at ASC;
        """

        return pd.read_sql_query(query, connection)

    finally:
        if connection is not None:
            connection.close()


def detect_feature_drift(reference, current):
    """Use a two-sample Kolmogorov-Smirnov test for numeric features."""
    results = {}

    for feature in FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            results[feature] = {
                "status": "unavailable",
                "reason": "Feature column is missing",
            }
            continue

        ref_values = pd.to_numeric(
            reference[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        cur_values = pd.to_numeric(
            current[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(ref_values) < MIN_ROWS_PER_WINDOW or len(cur_values) < MIN_ROWS_PER_WINDOW:
            results[feature] = {
                "status": "insufficient_data",
                "reference_count": int(len(ref_values)),
                "current_count": int(len(cur_values)),
            }
            continue

        test_result = ks_2samp(ref_values, cur_values)

        drift_detected = bool(test_result.pvalue < P_VALUE_THRESHOLD)

        results[feature] = {
            "status": "drift_detected" if drift_detected else "no_drift_detected",
            "reference_count": int(len(ref_values)),
            "current_count": int(len(cur_values)),
            "ks_statistic": float(test_result.statistic),
            "p_value": float(test_result.pvalue),
            "p_value_threshold": P_VALUE_THRESHOLD,
        }

    return results


def main():
    logger.info("Starting feature drift check")

    df = load_feature_data()

    if df.empty:
        raise RuntimeError("feature_store has no records to analyse")

    df["created_at"] = pd.to_datetime(
        df["created_at"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["created_at"]).sort_values("created_at")

    now = pd.Timestamp.now(tz="UTC")
    latest_record = df["created_at"].max()
    data_age_days = (now - latest_record).total_seconds() / 86400

    current_start = now - pd.Timedelta(days=CURRENT_DAYS)
    reference_start = now - pd.Timedelta(days=REFERENCE_DAYS)

    reference = df[
        (df["created_at"] >= reference_start)
        & (df["created_at"] < current_start)
    ]

    current = df[df["created_at"] >= current_start]

    # Do not treat stale data as current production data.
    if data_age_days > CURRENT_DAYS:
        overall_status = "stale_data"
        feature_results = {}
        reason = (
            f"Latest record is {data_age_days:.1f} days old; "
            "no records were found in the current time window."
        )
    elif len(reference) < MIN_ROWS_PER_WINDOW or len(current) < MIN_ROWS_PER_WINDOW:
        overall_status = "insufficient_data"
        feature_results = {}
        reason = "Not enough records in the reference/current windows."
    else:
        feature_results = detect_feature_drift(reference, current)
        detected = any(
            item.get("status") == "drift_detected"
            for item in feature_results.values()
        )
        overall_status = "drift_detected" if detected else "no_drift_detected"
        reason = None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "reason": reason,
        "reference_window_days": REFERENCE_DAYS,
        "current_window_days": CURRENT_DAYS,
        "latest_record_at": latest_record.isoformat(),
        "data_age_days": round(data_age_days, 2),
        "total_records": int(len(df)),
        "reference_records": int(len(reference)),
        "current_records": int(len(current)),
        "method": "two-sample Kolmogorov-Smirnov test",
        "note": (
            "Feature drift is not the same as model performance degradation. "
            "This report does not trigger retraining or deployment."
        ),
        "features": feature_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    logger.info("Drift report saved to %s", REPORT_PATH)
    logger.info("Overall status: %s", overall_status)
    logger.info(
        "Records: total=%d, reference=%d, current=%d",
        len(df), len(reference), len(current),
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Feature drift check failed")
        raise
