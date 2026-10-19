#!/usr/bin/env python3
"""Task 13 — TP/FP/FN și precision/recall pe val (model antrenat).

    python scripts_lab/t13_potriveste_val.py
    python scripts_lab/t13_potriveste_val.py --weights runs/detect/yolo_simplu_person/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _paths import LAB2, setup_import

setup_import()
from train_yolo_simplu import (  # noqa: E402
    SetDetectie,
    YOLOSimplu,
    construieste_incarcator,
    decodifica,
    pas_epoca,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    set_val = SetDetectie(LAB2 / "data" / "val" / "images", 160, augmentare=False)
    loader = construieste_incarcator(set_val, batch=4, workers=0, amesteca=False, seed=0)
    model = YOLOSimplu().to(device)
    if args.weights is not None:
        w = args.weights if args.weights.is_absolute() else LAB2 / args.weights
        ckpt = torch.load(w, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Ponderi: {w.relative_to(LAB2)}")
    else:
        print("Model neantrenat (fără --weights).")

    stats = pas_epoca(model, loader, device, optimizator=None)
    print(f"Pe tot val (batch=4, o trecere pas_epoca):")
    print(f"  loss={stats['loss']:.4f}  P={stats['precision']:.2f}  R={stats['recall']:.2f}  scor_max={stats['scor_max']:.2f}")


if __name__ == "__main__":
    main()
