#!/usr/bin/env python3
"""Task 6 — Grilă: tx/ty/tw/th → cx/cy/w/h (exemplu numeric + model neantrenat).

    python scripts_lab/t06_decodifica_grila.py
"""

from __future__ import annotations

import torch

from _paths import setup_import

setup_import()
from train_yolo_simplu import YOLOSimplu, decodifica  # noqa: E402


def exemplu_manual() -> None:
    gw, gx, tx = 10, 0, 0.0
    cx = (gx + torch.sigmoid(torch.tensor(tx)).item()) / gw
    print("Exemplu manual (gw=10, celula gx=0, tx=0):")
    print(f"  sigmoid(tx)=0.5  →  cx = (0 + 0.5) / 10 = {cx:.3f}")


def main() -> None:
    exemplu_manual()
    model = YOLOSimplu()
    model.eval()
    x = torch.randn(1, 1, 160, 160)
    raw = model(x)
    cx, cy, latime, inaltime, logit = decodifica(raw)
    scor = logit.sigmoid()
    print(f"\nForward aleator: raw.shape = {tuple(raw.shape)}")
    print(f"  cx  min/max: {cx.min():.3f} / {cx.max():.3f}")
    print(f"  scor obiect min/max/mean: {scor.min():.3f} / {scor.max():.3f} / {scor.mean():.3f}")
    gy, gx = 5, 3
    print(f"\nCelula (gx={gx}, gy={gy}) după inițializare cap (bias -2 pe w,h,obj):")
    print(f"  tx,ty raw: {raw[0, 0, gy, gx]:.3f}, {raw[0, 1, gy, gx]:.3f}")
    print(f"  w,h scor:  {latime[0, gy, gx]:.3f}, {inaltime[0, gy, gx]:.3f}, obj={scor[0, gy, gx]:.3f}")


if __name__ == "__main__":
    main()
