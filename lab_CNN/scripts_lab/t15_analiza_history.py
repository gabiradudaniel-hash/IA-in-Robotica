#!/usr/bin/env python3
"""Task 15 — Rezumat history.csv după antrenament.

    python scripts_lab/t15_analiza_history.py
    python scripts_lab/t15_analiza_history.py --csv runs/detect/yolo_simplu_person/history.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _paths import LAB2, gaseste_history_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=LAB2 / "runs" / "detect" / "yolo_simplu_person" / "history.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cale = args.csv if args.csv.is_absolute() else LAB2 / args.csv
    if not cale.is_file():
        gasit = gaseste_history_csv()
        if gasit is not None:
            cale = gasit
            print(f"Folosesc cel mai recent: {cale.relative_to(LAB2)}")
        else:
            print(f"Lipsește {cale.relative_to(LAB2)}")
            print("Rulează mai întâi: python scripts_lab/t12_antrenare_scurta.py")
            print("sau: python scripts_lab/t00_ruleaza_tot.py")
            return

    with cale.open(newline="") as f:
        randuri = list(csv.DictReader(f))
    if not randuri:
        print("Fișier gol.")
        return

    ultim = randuri[-1]
    best = min(randuri, key=lambda r: float(r["val_loss"]))
    print(f"Fișier: {cale.relative_to(LAB2)}  ({len(randuri)} epoci)\n")
    print("Ultima epocă:")
    print(
        f"  train_loss={ultim['train_loss']}  val_loss={ultim['val_loss']}  "
        f"P={ultim['precision']}  R={ultim['recall']}  scor_max={ultim['scor_max']}"
    )
    print("\nCea mai bună val_loss:")
    print(
        f"  epoca {best['epoch']}  val_loss={best['val_loss']}  "
        f"P={best['precision']}  R={best['recall']}"
    )
    gap = float(ultim["train_loss"]) - float(ultim["val_loss"])
    print(f"\nUltima epocă train_loss - val_loss = {gap:.4f} (negativ → val mai mare decât train)")


if __name__ == "__main__":
    main()
