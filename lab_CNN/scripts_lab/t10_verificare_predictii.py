#!/usr/bin/env python3
"""README Task 10 — TP / FP / FN pas cu pas (casete sintetice + potriveste).

    python scripts_lab/t10_verificare_predictii.py
"""

from __future__ import annotations

import torch

from _paths import setup_import

setup_import()
from train_yolo_simplu import PRAG_IOU, PRAG_SCOR, iou_xyxy, potriveste  # noqa: E402


def iou_manual(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def afiseaza_scenariu(titlu: str, gt_xyxy: list[tuple], pred_xyxy_scor: list[tuple]) -> None:
    print("\n" + "=" * 60)
    print(titlu)
    print(f"Prag IoU pentru TP: τ = {PRAG_IOU}  (PRAG_IOU)")
    print(f"Prag scor la inferență (înainte de NMS): τ_s = {PRAG_SCOR}  (PRAG_SCOR)")
    print("=" * 60)

    gt_t = torch.tensor([list(g) for g in gt_xyxy], dtype=torch.float32)
    pred_t = torch.tensor([list(p) for p in pred_xyxy_scor], dtype=torch.float32)

    print(f"\nGround truth: K = {len(gt_xyxy)} casetă/casete (x1,y1,x2,y2)")
    for i, g in enumerate(gt_xyxy):
        print(f"  G{i+1}: {g}")

    print(f"\nPredicții (după filtru scor + NMS): M = {len(pred_xyxy_scor)}")
    for j, p in enumerate(pred_xyxy_scor):
        print(f"  P{j+1}: xyxy={p[:4]}  scor s={p[4]:.2f}")

    if gt_xyxy and pred_xyxy_scor:
        print("\nMatrice IoU(P_j, G_k):")
        for j, p in enumerate(pred_xyxy_scor):
            row = []
            for k, g in enumerate(gt_xyxy):
                iou = iou_manual(p[:4], g)
                iou_t = float(iou_xyxy(pred_t[j, :4], gt_t[k : k + 1])[0])
                row.append(f"IoU(P{j+1},G{k+1})={iou:.3f}")
            print("  " + "  ".join(row))

    tp, fp, fn = potriveste(pred_t, gt_t if gt_t.numel() else torch.zeros((0, 4)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    print(f"\nRezultat potriveste (sortare pred după scor descrescător):")
    print(f"  TP = {tp}   FP = {fp}   FN = {fn}")
    print(f"  P = TP/(TP+FP) = {p:.2f}")
    print(f"  R = TP/(TP+FN) = {r:.2f}")


def main() -> None:
    print("Task 10 — Verificare: predicție corectă vs greșită")
    print("Format casetă: (x1, y1, x2, y2, scor) normalizat pe imagine.")

    # GT: o persoană
    g1 = (0.40, 0.20, 0.60, 0.80)

    # P1: suprapunere bună cu G1 → TP
    # P2: altundeva → FP
    afiseaza_scenariu(
        "Scenariu A — 1 GT, 2 pred (una bună, una proastă)",
        gt_xyxy=[g1],
        pred_xyxy_scor=[
            (0.42, 0.22, 0.58, 0.78, 0.90),
            (0.05, 0.05, 0.15, 0.15, 0.85),
        ],
    )
    iou_bun = iou_manual((0.42, 0.22, 0.58, 0.78), g1)
    print(f"\n  → P1: IoU cu G1 = {iou_bun:.3f} ≥ {PRAG_IOU} ⇒ TP")
    print(f"  → P2: IoU cu G1 mic ⇒ FP")

    afiseaza_scenariu(
        "Scenariu B — 1 GT, 0 pred (modelul nu detectează)",
        gt_xyxy=[g1],
        pred_xyxy_scor=[],
    )
    print("  → FN = 1 (persoana ratată)")

    afiseaza_scenariu(
        "Scenariu C — 0 GT, 1 pred (alarmă falsă pe fundal)",
        gt_xyxy=[],
        pred_xyxy_scor=[(0.10, 0.10, 0.30, 0.40, 0.99)],
    )
    print("  → FP = 1")


if __name__ == "__main__":
    main()
