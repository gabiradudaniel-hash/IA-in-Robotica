#!/usr/bin/env python3
"""Task 3 — Citește o etichetă YOLO, calculează colțurile, desenează caseta.

    python scripts_lab/t03_eticheta_si_caseta.py
    python scripts_lab/t03_eticheta_si_caseta.py --demo
    python scripts_lab/t03_eticheta_si_caseta.py --imagine data/train/images/acquisition7_3891.jpeg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from _paths import LAB2, asigura_out, cale_eticheta_pentru_imagine, rezolva_imagine, setup_import

setup_import()
from train_yolo_simplu import citeste_casete, incadreaza  # noqa: E402

IMGSZ = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagine", type=Path, default=None, help="Cale imagine (.jpg/.jpeg/.png)")
    parser.add_argument("--demo", action="store_true", help="Folosește scripts_lab/demo/acquisition7_3891.jpeg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cale_img = rezolva_imagine(args.imagine, prefer_demo=args.demo)

    img = cv2.imread(str(cale_img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(cale_img)
    eticheta = cale_eticheta_pentru_imagine(cale_img)
    casete = citeste_casete(eticheta)
    if not casete:
        raise RuntimeError(f"Nicio casetă în {eticheta}")

    print(f"Imagine: {cale_img.relative_to(LAB2)}  shape original: {img.shape}")
    print(f"Etichetă: {eticheta.relative_to(LAB2)}")
    for i, (cx, cy, w, h) in enumerate(casete):
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        print(f"\nPersoană {i + 1} (normalizat față de imaginea originală):")
        print(f"  cx={cx:.6f}  cy={cy:.6f}  w={w:.6f}  h={h:.6f}")
        print(f"  x1={x1:.6f}  y1={y1:.6f}  x2={x2:.6f}  y2={y2:.6f}")

    panza, casete_patrat = incadreaza(img, casete, IMGSZ)
    viz = cv2.cvtColor(panza, cv2.COLOR_GRAY2BGR)
    for cx, cy, w, h in casete_patrat:
        x1 = int(round((cx - w / 2) * IMGSZ))
        y1 = int(round((cy - h / 2) * IMGSZ))
        x2 = int(round((cx + w / 2) * IMGSZ))
        y2 = int(round((cy + h / 2) * IMGSZ))
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 1)

    out = asigura_out() / f"t03_{cale_img.stem}_caseta.jpg"
    cv2.imwrite(str(out), viz)
    print(f"\nDesen salvat (după letterbox {IMGSZ}×{IMGSZ}): {out.relative_to(LAB2)}")


if __name__ == "__main__":
    main()
