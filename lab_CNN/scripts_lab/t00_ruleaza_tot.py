#!/usr/bin/env python3
"""Rulează toate scripturile de laborator în ordine (din lab_2/).

    python scripts_lab/t00_ruleaza_tot.py
    python scripts_lab/t00_ruleaza_tot.py --skip-train    # fără t12 (rapid)
    python scripts_lab/t00_ruleaza_tot.py --epochs 2      # antrenament scurt pentru viz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LAB2 = Path(__file__).resolve().parents[1]
LABORATOR = Path(__file__).resolve().parent
PYTHON = sys.executable

SCRIPTS_RAPIDE = [
    "t03_eticheta_si_caseta.py",
    "t04_letterbox.py",
    "t05_split_train_val.py",
    "t06_decodifica_grila.py",
    "t08_parametri_model.py",
    "t09_loss_batch.py",
    "t10_verificare_predictii.py",
    "t11_un_pas_antrenare.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-train", action="store_true", help="Sari peste t12_antrenare_scurta.py")
    parser.add_argument("--epochs", type=int, default=3, help="Epoci pentru t12 (implicit 3)")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def ruleaza(script: str, extra: list[str] | None = None) -> None:
    cale = LABORATOR / script
    cmd = [PYTHON, str(cale), *(extra or [])]
    print("\n" + "=" * 60)
    print(" ", " ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd, cwd=LAB2, check=True)


def ultim_run_detect() -> Path | None:
    detect = LAB2 / "runs" / "detect"
    if not detect.is_dir():
        return None
    candidati = sorted(detect.glob("lab_ruleaza_tot*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in candidati:
        w = d / "weights" / "best.pt"
        if w.is_file():
            return w
    return None


def main() -> None:
    args = parse_args()
    for script in SCRIPTS_RAPIDE:
        ruleaza(script)

    weights: Path | None = None
    if not args.skip_train:
        ruleaza(
            "t12_antrenare_scurta.py",
            ["--epochs", str(args.epochs), "--device", args.device, "--name", "lab_ruleaza_tot"],
        )
        weights = ultim_run_detect()

    if weights is not None:
        run_dir = weights.parent.parent
        history = run_dir / "history.csv"
        if history.is_file():
            ruleaza("t15_analiza_history.py", ["--csv", str(history.relative_to(LAB2))])
        else:
            ruleaza("t15_analiza_history.py")
        rel = weights.relative_to(LAB2)
        ruleaza("t13_potriveste_val.py", ["--weights", str(rel)])
        ruleaza("t16_vizualizeaza_detectii.py", ["--weights", str(rel), "--num", "3"])
    else:
        ruleaza("t15_analiza_history.py")
        print("\n(fără antrenament: t16 vizualizare cu best.pt omis; rulează fără --skip-train)")

    print("\nGata — toate scripturile au rulat cu succes.")


if __name__ == "__main__":
    main()
