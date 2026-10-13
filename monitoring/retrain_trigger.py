"""
Retraining trigger for the Emotion Intelligence Platform.

Reads drift and performance monitoring reports and decides whether
retraining should be recommended.

This script does NOT start training, modify the model registry,
or deploy anything.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"

DRIFT_REPORT = REPORTS_DIR / "drift_report.json"
PERFORMANCE_REPORT = REPORTS_DIR / "performance_report.json"
TRIGGER_REPORT = REPORTS_DIR / "retrain_trigger_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_report(path):
    """Load a JSON monitoring report safely."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required monitoring report not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(f"Invalid report format: {path.name}")

    return report


def evaluate_retraining(drift, performance):
    """Evaluate monitoring results without launching retraining."""

    reasons = []
    blockers = []

    drift_status = drift.get("overall_status", "unavailable")
    performance_status = performance.get(
        "overall_status", "unavailable"
    )

    # Check that both monitoring reports contain usable data.
    if drift_status in {"stale_data", "insufficient_data", "unavailable"}:
        blockers.append(
            f"Drift monitoring is not actionable: {drift_status}"
        )

    if performance_status in {
        "no_recent_predictions",
        "unavailable",
    }:
        blockers.append(
            "Recent prediction monitoring data is unavailable."
        )

    # Check whether the current data is fresh enough.
    if drift.get("data_age_days") is not None:
        try:
            if float(drift["data_age_days"]) > 3:
                blockers.append(
                    "Latest feature data is older than 3 days."
                )
        except (TypeError, ValueError):
            blockers.append("Invalid feature data age in drift report.")

    # Do not recommend retraining if monitoring is blocked.
    if blockers:
        decision = "blocked"
        reasons.extend(blockers)

    elif drift_status == "drift_detected":
        decision = "retraining_recommended"
        reasons.append(
            "Feature drift was detected in fresh, sufficiently large data."
        )

    elif drift_status == "no_drift_detected":
        decision = "not_required"
        reasons.append(
            "No statistically significant feature drift was detected."
        )

    else:
        decision = "blocked"
        reasons.append(
            f"Unrecognized drift monitoring status: {drift_status}"
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "retraining_started": False,
        "deployment_started": False,
        "drift_status": drift_status,
        "performance_status": performance_status,
        "reasons": reasons,
        "blockers": blockers,
        "note": (
            "A retraining recommendation is not proof of model "
            "performance degradation. Validate data quality and model "
            "quality gates before training or deployment."
        ),
    }


def main():
    logger.info("Starting retraining trigger evaluation")

    drift = load_report(DRIFT_REPORT)
    performance = load_report(PERFORMANCE_REPORT)

    result = evaluate_retraining(drift, performance)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TRIGGER_REPORT.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    logger.info("Trigger report saved to %s", TRIGGER_REPORT)
    logger.info("Decision: %s", result["decision"])

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Retraining trigger evaluation failed")
        raise