#!/usr/bin/env python3
"""Task 12 — Lansează antrenament scurt (5 epoci) ca în experimentul ghidat.

    python scripts_lab/t12_antrenare_scurta.py
    python scripts_lab/t12_antrenare_scurta.py --epochs 10 --device mps
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from _paths import DATA_YAML, LAB2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="lab_s3_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(LAB2 / "train_yolo_simplu.py"),
        "--data",
        str(DATA_YAML),
        "--project",
        str(LAB2 / "runs" / "detect"),
        "--name",
        args.name,
        "--versiune",
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--imgsz",
        "160",
    ]
    print("Comandă:", " ".join(cmd))
    subprocess.run(cmd, cwd=LAB2, check=True)


if __name__ == "__main__":
    main()
