"""Căi și import pentru scripturile de laborator. Rulează comenzile din directorul lab_2/."""

from __future__ import annotations

import sys
from pathlib import Path

LAB2 = Path(__file__).resolve().parents[1]
LABORATOR = Path(__file__).resolve().parent
DATA_YAML = LAB2 / "data" / "data.yaml"
OUT = LABORATOR / "out"
DEMO_DIR = LABORATOR / "demo"
DEMO_STEM = "acquisition7_3891"
DEMO_IMAGE = DEMO_DIR / f"{DEMO_STEM}.jpeg"
DEMO_LABEL = DEMO_DIR / f"{DEMO_STEM}.txt"

SUFFIXE_IMAGINI = (".jpg", ".jpeg", ".png")


def setup_import() -> None:
    root = str(LAB2)
    if root not in sys.path:
        sys.path.insert(0, root)


def asigura_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def iter_imagini(director: Path):
    for cale in sorted(director.iterdir()):
        if cale.is_file() and cale.suffix.lower() in SUFFIXE_IMAGINI:
            yield cale


def prima_imagine(director: Path) -> Path:
    try:
        return next(iter_imagini(director))
    except StopIteration as exc:
        raise FileNotFoundError(f"Nicio imagine ({', '.join(SUFFIXE_IMAGINI)}) în {director}") from exc


def cale_eticheta_pentru_imagine(cale_img: Path) -> Path:
    """Layout YOLO train/val sau etichetă alături în demo/."""
    langa = cale_img.with_suffix(".txt")
    if langa.is_file():
        return langa
    return cale_img.parent.parent / "labels" / f"{cale_img.stem}.txt"


def gaseste_history_csv() -> Path | None:
    detect = LAB2 / "runs" / "detect"
    if not detect.is_dir():
        return None
    fisiere = sorted(detect.glob("*/history.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fisiere[0] if fisiere else None


def rezolva_imagine(cale: Path | None = None, *, prefer_demo: bool = False) -> Path:
    if cale is not None:
        return cale if cale.is_absolute() else LAB2 / cale
    if prefer_demo and DEMO_IMAGE.is_file():
        return DEMO_IMAGE
    train = LAB2 / "data" / "train" / "images"
    if train.is_dir():
        try:
            return prima_imagine(train)
        except FileNotFoundError:
            pass
    if DEMO_IMAGE.is_file():
        return DEMO_IMAGE
    raise FileNotFoundError(
        f"Nicio imagine de lucru. Pune fișiere în {train} sau folosește {DEMO_IMAGE}."
    )
