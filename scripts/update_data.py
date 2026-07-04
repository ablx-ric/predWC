import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

SCRIPTS_TO_RUN = [
    ("fetch_elo.py",           "ELO rankings"),
    ("fetch_elo_history.py",   "ELO history (per-match)"),
]


def run_script(script_name, label):
    path = SCRIPTS / script_name
    print(f"\n{'=' * 50}")
    print(f"  [{label}] Running {script_name}...")
    print(f"{'=' * 50}")
    result = subprocess.run(
        ["uv", "run", "python", str(path)],
        cwd=str(ROOT),
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
        return True
    else:
        print(f"  ERROR:\n{result.stderr}")
        return False


def main():
    print("=" * 50)
    print("  UPDATE ALL DATA")
    print(f"  Root: {ROOT}")
    print("=" * 50)

    success = 0
    failed = 0
    for script_name, label in SCRIPTS_TO_RUN:
        ok = run_script(script_name, label)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Done: {success} ok, {failed} failed")
    print(f"{'=' * 50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
