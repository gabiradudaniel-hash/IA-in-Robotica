#!/usr/bin/env python3
"""README Task 9 — Loss pe un batch (model neantrenat) + cum citești cifrele.

    python scripts_lab/t09_loss_batch.py
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from _paths import LAB2, setup_import

setup_import()
from train_yolo_simplu import (  # noqa: E402
    LAMBDA_CASETA,
    PONDERE_FUNDAL,
    SetDetectie,
    YOLOSimplu,
    calculeaza_loss,
    construieste_incarcator,
    decodifica,
    iou_grile,
    tinte_din_casete,
)


def loss_detaliat(raw: torch.Tensor, casete: list[torch.Tensor]) -> dict[str, float]:
    cx, cy, latime, inaltime, logit = decodifica(raw)
    grila = raw.shape[-1]
    obiect, tinta = tinte_din_casete(casete, grila, raw.device)
    suprapunere = iou_grile(cx, cy, latime, inaltime, tinta)
    numar_obiecte = obiect.sum().clamp(min=1.0)
    loss_caseta = ((1.0 - suprapunere) * obiect).sum() / numar_obiecte

    bce = F.binary_cross_entropy_with_logits(logit, obiect, reduction="none")
    celule_persoana = bce[obiect > 0]
    loss_obiect = celule_persoana.mean() if celule_persoana.numel() else bce.sum() * 0.0
    loss_fundal = bce[obiect == 0].mean()

    iou_mediu = float((suprapunere * obiect).sum() / numar_obiecte)
    scor_mediu_pe_persoana = float(logit.sigmoid()[obiect > 0].mean())
    scor_mediu_fundal = float(logit.sigmoid()[obiect == 0].mean())

    total = LAMBDA_CASETA * loss_caseta + loss_obiect + PONDERE_FUNDAL * loss_fundal
    return {
        "loss_caseta": float(loss_caseta),
        "loss_obiect": float(loss_obiect),
        "loss_fundal": float(loss_fundal),
        "total": float(total),
        "celule_cu_persoana": int(obiect.sum()),
        "celule_fundal": int((obiect == 0).sum()),
        "iou_mediu_pe_persoana": iou_mediu,
        "scor_mediu_pe_persoana": scor_mediu_pe_persoana,
        "scor_mediu_fundal": scor_mediu_fundal,
        "grila": grila,
        "batch": raw.shape[0],
    }


def main() -> None:
    set_train = SetDetectie(LAB2 / "data" / "train" / "images", 160, augmentare=False)
    loader = construieste_incarcator(set_train, batch=4, workers=0, amesteca=False, seed=0)
    imagini, casete = next(iter(loader))
    model = YOLOSimplu()
    raw = model(imagini)
    loss_fn = calculeaza_loss(raw, casete)
    d = loss_detaliat(raw, casete)

    term_box = LAMBDA_CASETA * d["loss_caseta"]
    term_obj = d["loss_obiect"]
    term_bg = PONDERE_FUNDAL * d["loss_fundal"]

    print("=" * 60)
    print("Task 9 — Loss (model NEANTRENAT, primul batch din train)")
    print("=" * 60)
    print(f"  Batch: {d['batch']} imagini × grilă {d['grila']}×{d['grila']} = {d['batch'] * d['grila'] ** 2} celule")
    print(f"  Celule cu persoană (N₊): {d['celule_cu_persoana']}  (≈ 1 persoană/imagine)")
    print(f"  Celule fundal:            {d['celule_fundal']}")
    print()
    print("--- 1) loss_caseta = medie (1 − IoU) DOAR pe cele N₊ celule ---")
    print(f"  IoU mediu pe persoană:     {d['iou_mediu_pe_persoana']:.3f}  (aproape 0 = cutii la întâmplare)")
    print(f"  loss_caseta = 1 − IoU ≈   {d['loss_caseta']:.4f}")
    print(f"  contribuție la total:      {LAMBDA_CASETA} × {d['loss_caseta']:.4f} = {term_box:.4f}")
    print()
    print("--- 2) loss_obiect = BCE pe N₊ celule (ținta = «există persoană» = 1) ---")
    print(f"  scor mediu σ(logit) acolo: {d['scor_mediu_pe_persoana']:.3f}  (țintă: 1.0)")
    print(f"  loss_obiect:               {d['loss_obiect']:.4f}  (mare fiindcă scorul e prea mic)")
    print(f"  contribuție la total:      {term_obj:.4f}")
    print()
    print("--- 3) loss_fundal = medie BCE pe celulele goale (ținta = 0) ---")
    print(f"  scor mediu σ(logit) fundal:{d['scor_mediu_fundal']:.3f}  (țintă: 0.0)")
    print(f"  loss_fundal:               {d['loss_fundal']:.4f}")
    print(f"  contribuție la total:      {PONDERE_FUNDAL} × {d['loss_fundal']:.4f} = {term_bg:.4f}")
    print()
    print("--- TOTAL ---")
    print(f"  L = {term_box:.4f} + {term_obj:.4f} + {term_bg:.4f} = {d['total']:.4f}")
    print(f"  calculeaza_loss():         {loss_fn.item():.4f}")
    print()
    print("La antrenament, backprop coboară L: IoU ↑, scor pe persoană ↑, scor pe fundal ↓.")


if __name__ == "__main__":
    main()
