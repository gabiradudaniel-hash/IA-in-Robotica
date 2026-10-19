#!/usr/bin/env python3
"""Task 5 — Număr imagini train/val din data.yaml.

    python scripts_lab/t05_split_train_val.py
"""

from __future__ import annotations

from _paths import DATA_YAML, setup_import

setup_import()
from train_yolo_simplu import SetDetectie, citeste_splituri  # noqa: E402


def main() -> None:
    train_dir, val_dir = citeste_splituri(DATA_YAML)
    set_train = SetDetectie(train_dir, 160, augmentare=False)
    set_val = SetDetectie(val_dir, 160, augmentare=False)
    print(f"data.yaml: {DATA_YAML}")
    print(f"  train: {train_dir}  →  {len(set_train)} imagini")
    print(f"  val:   {val_dir}  →  {len(set_val)} imagini")


if __name__ == "__main__":
    main()
