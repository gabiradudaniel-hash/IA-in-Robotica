#!/usr/bin/env python3
"""Task 4 — Letterbox: panză pătrată și casete recalculate.

    python scripts_lab/t04_letterbox.py
    python scripts_lab/t04_letterbox.py --demo
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
    parser.add_argument("--imagine", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cale_img = rezolva_imagine(args.imagine, prefer_demo=args.demo)
    img = cv2.imread(str(cale_img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(cale_img)
    eticheta = cale_eticheta_pentru_imagine(cale_img)
    casete_in = citeste_casete(eticheta)
    panza, casete_out = incadreaza(img, casete_in, IMGSZ)

    print(f"Fișier: {cale_img.relative_to(LAB2)}")
    print(f"Original: {img.shape[1]}×{img.shape[0]} px")
    print(f"După incadreaza: panza {panza.shape} (așteptat ({IMGSZ}, {IMGSZ}))")
    print(f"Număr casete: {len(casete_out)}")
    for j, (cx, cy, w, h) in enumerate(casete_out):
        ok = 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
        print(f"  [{j}] cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}  în [0,1]: {ok}")

    out = asigura_out() / f"t04_{cale_img.stem}_letterbox.png"
    cv2.imwrite(str(out), panza)
    print(f"Panza salvată: {out.relative_to(LAB2)}")


if __name__ == "__main__":
    main()
