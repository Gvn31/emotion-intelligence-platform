"""Run monitoring checks and retrain only when recommended."""

import json
import subprocess
import sys
from pathlib import Path


# Find the project folder (the parent of src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"


def run_script(script_path: Path) -> None:
    """Run one project script and stop if it fails."""
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    # Check for feature drift and recent model performance.
    run_script(PROJECT_ROOT / "monitoring" / "drift_check.py")
    run_script(PROJECT_ROOT / "monitoring" / "performance_check.py")

    # Use both monitoring reports to decide whether retraining is needed.
    run_script(PROJECT_ROOT / "monitoring" / "retrain_trigger.py")

    # Read the decision made by the retraining trigger.
    trigger_report = REPORTS_DIR / "retrain_trigger_report.json"
    if not trigger_report.is_file():
        raise FileNotFoundError(f"Trigger report not found: {trigger_report}")

    trigger = json.loads(trigger_report.read_text(encoding="utf-8"))
    decision = trigger.get("decision")

    # Stop here when the trigger does not recommend retraining.
    if decision != "retraining_recommended":
        print(f"Retraining skipped. Decision: {decision or 'unknown'}")
        return

    # Train a candidate model using the existing training script.
    print("Retraining recommended. Starting model training.")
    run_script(PROJECT_ROOT / "src" / "train.py")

    # Evaluate the candidate model. The evaluation script applies its quality checks.
    print("Training finished. Starting model evaluation.")
    run_script(PROJECT_ROOT / "src" / "evaluate.py")

    print("Retraining and evaluation finished.")


if __name__ == "__main__":
    main()
    