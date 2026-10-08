import subprocess
import sys
from pathlib import Path


def run_stage(script_name, marker_name):
    print(f"\n{'=' * 60}")
    print(f"Running stage: {script_name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, script_name],
        check=False
    )

    if result.returncode != 0:
        print(f"\nStage failed: {script_name}")
        sys.exit(result.returncode)

    marker = Path(marker_name)
    marker.touch()

    print(f"\nStage completed: {script_name}")
    print(f"Marker created: {marker_name}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/run_stage.py "
            "<script> <marker>"
        )
        sys.exit(1)

    run_stage(
        sys.argv[1],
        sys.argv[2]
    )