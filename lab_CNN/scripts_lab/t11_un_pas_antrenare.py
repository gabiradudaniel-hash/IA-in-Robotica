#!/usr/bin/env python3
"""Task 11 — Un pas de antrenare: loss înainte/după optimizer.step().

    python scripts_lab/t11_un_pas_antrenare.py
"""

from __future__ import annotations

import torch

from _paths import LAB2, setup_import

setup_import()
from train_yolo_simplu import SetDetectie, YOLOSimplu, calculeaza_loss, construieste_incarcator  # noqa: E402


def main() -> None:
    set_train = SetDetectie(LAB2 / "data" / "train" / "images", 160, augmentare=False)
    loader = construieste_incarcator(set_train, batch=8, workers=0, amesteca=True, seed=42)
    imagini, casete = next(iter(loader))
    model = YOLOSimplu()
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    with torch.no_grad():
        loss_inainte = calculeaza_loss(model(imagini), casete).item()
    optim.zero_grad()
    loss = calculeaza_loss(model(imagini), casete)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optim.step()
    with torch.no_grad():
        loss_dupa = calculeaza_loss(model(imagini), casete).item()

    print("Un batch, un pas AdamW (lr=1e-3):")
    print(f"  loss înainte: {loss_inainte:.4f}")
    print(f"  loss după step: {loss_dupa:.4f}")
    print(f"  diferență: {loss_inainte - loss_dupa:.4f} (ar trebui ≥ 0, de obicei)")


if __name__ == "__main__":
    main()
