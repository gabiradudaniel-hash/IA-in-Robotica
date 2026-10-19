#!/usr/bin/env python3
"""Task 8 — Parametri antrenabili (calcul de mână vs PyTorch).

    python scripts_lab/t08_parametri_model.py
"""

from __future__ import annotations

from _paths import setup_import

setup_import()
from train_yolo_simplu import construieste_model, numar_de_mana  # noqa: E402


def main() -> None:
    asteptat = numar_de_mana()
    print("Calcul de mână (numar_de_mana):", asteptat)
    print("\nPrimul strat Conv2d(1, 16, kernel=3):")
    print("  ponderi: 1 × 16 × 9 = 144")
    print("  bias: 16")
    print("  total: 160")
    print()
    construieste_model()
    print("\nDacă nu apare RuntimeError, numărul coincide cu PyTorch.")


if __name__ == "__main__":
    main()
