"""Recompute final direct-RUL evaluation and maintenance tables without training."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_maintenance_replay.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
