#!/usr/bin/env python3
"""Task 16 — Desenează GT (verde) și predicții (roșu) pe imagini val.

    python scripts_lab/t16_vizualizeaza_detectii.py
    python scripts_lab/t16_vizualizeaza_detectii.py --weights runs/detect/yolo_simplu_person/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from _paths import LAB2, asigura_out, setup_import

setup_import()
from train_yolo_simplu import (  # noqa: E402
    SetDetectie,
    YOLOSimplu,
    decodifica,
    preziceri_imagine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=LAB2 / "runs" / "detect" / "yolo_simplu_person" / "weights" / "best.pt",
    )
    parser.add_argument("--num", type=int, default=5, help="Câte imagini din val")
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def deseneaza_casete(
    panza: torch.Tensor,
    casete_norm: torch.Tensor,
    culoare: tuple[int, int, int],
    imgsz: int,
) -> None:
    """casete_norm: [N,4] cx,cy,w,h sau [N,5] cu scor."""
    for row in casete_norm.tolist():
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = int(round((cx - w / 2) * imgsz))
        y1 = int(round((cy - h / 2) * imgsz))
        x2 = int(round((cx + w / 2) * imgsz))
        y2 = int(round((cy + h / 2) * imgsz))
        cv2.rectangle(panza, (x1, y1), (x2, y2), culoare, 1)


def main() -> None:
    args = parse_args()
    weights = args.weights if args.weights.is_absolute() else LAB2 / args.weights
    if not weights.is_file():
        print(f"Lipsesc ponderile: {weights.relative_to(LAB2)}")
        print("Antrenează mai întâi (t12_antrenare_scurta.py sau train_yolo_simplu.py).")
        return

    device = torch.device(args.device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    if int(ckpt.get("imgsz", args.imgsz)) != args.imgsz:
        raise ValueError(f"Checkpoint imgsz={ckpt['imgsz']}, script imgsz={args.imgsz}")

    model = YOLOSimplu().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    set_val = SetDetectie(LAB2 / "data" / "val" / "images", args.imgsz, augmentare=False)
    out_dir = asigura_out() / "t16_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in range(min(args.num, len(set_val))):
        tensor, gt = set_val[index]
        cale = set_val.cai[index]
        with torch.no_grad():
            raw = model(tensor.unsqueeze(0).to(device))
        cx, cy, w, h, logit = decodifica(raw)
        pred = preziceri_imagine(cx[0], cy[0], w[0], h[0], logit[0]).cpu()

        img = (tensor[0].numpy() * 255).astype("uint8")
        viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if gt.numel():
            deseneaza_casete(viz, gt, (0, 255, 0), args.imgsz)
        if pred.numel():
            deseneaza_casete(viz, pred[:, :4], (0, 0, 255), args.imgsz)
            for row in pred.tolist():
                x = int(row[0] * args.imgsz)
                y = max(0, int(row[1] * args.imgsz) - 2)
                cv2.putText(viz, f"{row[4]:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        dest = out_dir / f"{cale.stem}_gt_verde_pred_rosu.jpg"
        cv2.imwrite(str(dest), viz)
        print(f"{cale.name}: GT={gt.shape[0]}  pred={pred.shape[0]}  → {dest.relative_to(LAB2)}")


if __name__ == "__main__":
    main()
