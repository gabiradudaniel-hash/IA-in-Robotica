#!/usr/bin/env python3
"""Antrenează de la zero un detector YOLO simplificat, în PyTorch.

Imaginea este împărțită într-o grilă. Fiecare celulă prezice o singură casetă
și un scor: cât de sigur este modelul că acolo se află o persoană. Aceasta
este ideea YOLO, într-o formă care se poate citi într-un laborator: o singură
grilă, o singură clasă, o singură casetă pe celulă.

Ponderile pornesc din inițializarea PyTorch. Setul este același data.yaml
ca la detectia de persoane. Cadrele grayscale sunt aduse la un pătrat.

Exemplu:
    .venv/bin/python train_yolo_simplu.py
    .venv/bin/python train_yolo_simplu.py --epochs 40 --imgsz 160 --device cpu
    .venv/bin/python train_yolo_simplu.py --versiune   # name2, name3, … fără suprascriere
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA = Path("/Users/tlala/work/research/person_pseudo_labels/data.yaml")
DEFAULT_PROJECT = Path("/Users/tlala/work/research/runs/detect")
DEFAULT_NAME = "yolo_simplu_person"

# Patru MaxPool2d reduc fiecare latură de două ori: 2^4 = 16.
STRIDE = 16
# Mediile sunt separate, deci ponderea nu depinde de câte celule are grila.
# Cu 4, un scor identic pe toată imaginea s-ar opri pe la 0.2, sub pragul de
# detecție. Celula cu persoană trebuie să urce singură peste prag.
LAMBDA_CASETA = 5.0
PONDERE_FUNDAL = 4.0
PRAG_SCOR = 0.25
PRAG_IOU = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Fișierul data.yaml al setului YOLO.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Directorul în care se scriu rulările.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Numele rulării, sub project/.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8, help="Oprire anticipată dacă loss-ul de validare stagnează.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=160,
        help="Latura pătrată a intrării. Trebuie să fie multiplu de 16. 160 dă o grilă de 10×10.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="Rata de învățare pentru AdamW.")
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda sau indexul GPU-ului.")
    parser.add_argument("--workers", type=int, default=0, help="Procese pentru DataLoader. 0 este stabil pe macOS.")
    parser.add_argument("--seed", type=int, default=0, help="Sămânța aleatoare, pentru reluarea aceleiași rulări.")
    parser.add_argument("--resume", action="store_true", help="Continuă de la weights/last.pt din project/name.")
    parser.add_argument(
        "--versiune",
        action="store_true",
        help="Nu suprascrie name: creează o rulare nouă (name2, name3, …) sub project/.",
    )
    return parser.parse_args()


def rezolva_director_rulare(project: Path, name: str, versiune_noua: bool) -> Path:
    """Implicit suprascrie project/name; cu --versiune alege primul name, name2, … liber."""
    director = project / name
    if not versiune_noua:
        return director
    if not director.exists():
        return director
    n = 2
    while (project / f"{name}{n}").exists():
        n += 1
    return project / f"{name}{n}"


def seteaza_samanta(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rezolva_dispozitiv(nume: str) -> torch.device:
    if nume == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS nu este disponibil în acest PyTorch.")
        return torch.device("mps")
    if nume == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nu este disponibilă în acest PyTorch.")
        return torch.device("cuda")
    if nume.isdigit():
        return torch.device(f"cuda:{nume}")
    return torch.device(nume)


def citeste_splituri(data_yaml: Path) -> tuple[Path, Path]:
    """Citește căile train/val din data.yaml. Căile relative sunt față de `path:`."""
    configuratie = yaml.safe_load(data_yaml.read_text())
    radacina = Path(configuratie["path"]) if configuratie.get("path") else data_yaml.parent
    if not radacina.is_absolute():
        radacina = (data_yaml.parent / radacina).resolve()
    train = Path(configuratie["train"])
    val = Path(configuratie["val"])
    if not train.is_absolute():
        train = radacina / train
    if not val.is_absolute():
        val = radacina / val
    return train, val


class YOLOSimplu(nn.Module):
    """Detector cu o grilă. Ieșirea are 5 canale pentru fiecare celulă.

    Canalele, în ordine: decalaj x, decalaj y, lățime, înălțime, obiect.
    Decalajele spun unde cade centrul în interiorul celulei. Lățimea și
    înălțimea sunt fracțiuni din imagine. Obiectul este logit-ul pentru
    „în celula aceasta este o persoană”.
    """

    def __init__(self) -> None:
        super().__init__()
        # Fiecare bloc înjumătățește imaginea. După patru blocuri, 160 devine 10.
        self.trunchi = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cap = nn.Conv2d(128, 5, kernel_size=1)
        self._initializare_cap()

    def _initializare_cap(self) -> None:
        # Decalajul 0 trece prin sigmoid în 0.5, deci centrul pornește la mijlocul celulei.
        # sigmoid(-2) ≈ 0.12: caseta pornește mică, iar scorul de obiect pornește jos,
        # fiindcă majoritatea celulelor sunt fundal.
        with torch.no_grad():
            self.cap.bias.zero_()
            self.cap.bias[2] = -2.0
            self.cap.bias[3] = -2.0
            self.cap.bias[4] = -2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cap(self.trunchi(x))


def numar_de_mana() -> int:
    """Un Conv2d(cin, cout, 3×3) are cin*cout*9 ponderi și cout bias-uri. Capul este 1×1."""
    total = 0
    intrare = 1
    for iesire in (16, 32, 64, 128):
        total += intrare * iesire * 9 + iesire
        intrare = iesire
    total += 128 * 5 + 5
    return total


def construieste_model() -> YOLOSimplu:
    """Construiește detectorul cu ponderi aleatoare. Nu se încarcă niciun checkpoint."""
    model = YOLOSimplu()
    asteptat = numar_de_mana()
    numarat = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if numarat != asteptat:
        raise RuntimeError(f"Parametrii nu coincid: PyTorch={numarat}, calcul de mână={asteptat}")
    print(f"Parametri antrenabili: {numarat}")
    print("  Conv 1→16, 16→32, 32→64, 64→128, apoi cap 128→5")
    return model


def citeste_casete(cale: Path) -> list[tuple[float, float, float, float]]:
    """Citește casete YOLO normalizate: cx, cy, lățime, înălțime."""
    if not cale.is_file():
        return []
    casete = []
    for linie in cale.read_text().splitlines():
        parti = linie.split()
        if len(parti) < 5:
            continue
        cx, cy, latime, inaltime = (float(parti[1]), float(parti[2]), float(parti[3]), float(parti[4]))
        casete.append((cx, cy, latime, inaltime))
    return casete


def incadreaza(
    imagine: np.ndarray,
    casete: list[tuple[float, float, float, float]],
    imgsz: int,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Pune cadrul grayscale într-un pătrat imgsz×imgsz și mută casetele odată cu el."""
    inaltime, latime = imagine.shape[:2]
    scara = imgsz / max(inaltime, latime)
    latime_noua = min(imgsz, int(round(latime * scara)))
    inaltime_noua = min(imgsz, int(round(inaltime * scara)))
    redimensionata = cv2.resize(imagine, (latime_noua, inaltime_noua), interpolation=cv2.INTER_AREA)
    panza = np.zeros((imgsz, imgsz), dtype=np.uint8)
    sus = (imgsz - inaltime_noua) // 2
    stanga = (imgsz - latime_noua) // 2
    panza[sus : sus + inaltime_noua, stanga : stanga + latime_noua] = redimensionata

    sx = latime_noua / latime
    sy = inaltime_noua / inaltime
    mutate = []
    for cx, cy, bw, bh in casete:
        px = cx * latime * sx + stanga
        py = cy * inaltime * sy + sus
        pw = bw * latime * sx
        ph = bh * inaltime * sy
        mutate.append((px / imgsz, py / imgsz, pw / imgsz, ph / imgsz))
    return panza, mutate


class SetDetectie(Dataset):
    """Imagini YOLO întregi, cu casetele persoanelor, aduse la pătratul rețelei."""

    def __init__(self, director_imagini: Path, imgsz: int, augmentare: bool) -> None:
        self.imgsz = imgsz
        self.augmentare = augmentare
        self.cai = sorted(
            cale
            for cale in director_imagini.iterdir()
            if cale.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.cai:
            raise RuntimeError(f"Nicio imagine în {director_imagini}")

    def __len__(self) -> int:
        return len(self.cai)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cale = self.cai[index]
        imagine = cv2.imread(str(cale), cv2.IMREAD_GRAYSCALE)
        if imagine is None:
            raise FileNotFoundError(cale)
        eticheta = cale.parent.parent / "labels" / f"{cale.stem}.txt"
        panza, casete = incadreaza(imagine, citeste_casete(eticheta), self.imgsz)
        if self.augmentare and random.random() < 0.5:
            panza = np.ascontiguousarray(panza[:, ::-1])
            casete = [(1.0 - cx, cy, bw, bh) for cx, cy, bw, bh in casete]
        tensor = torch.from_numpy(panza).float().unsqueeze(0) / 255.0
        if casete:
            tinta = torch.tensor(casete, dtype=torch.float32)
        else:
            tinta = torch.zeros((0, 4), dtype=torch.float32)
        return tensor, tinta


def colateaza(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # Casetele au lungimi diferite, deci rămân o listă. Imaginile se stivuiesc.
    imagini = torch.stack([element[0] for element in batch])
    casete = [element[1] for element in batch]
    return imagini, casete


def init_worker(worker_id: int) -> None:
    del worker_id
    samanta = torch.initial_seed() % 2**32
    random.seed(samanta)
    np.random.seed(samanta)


def construieste_incarcator(
    set_date: SetDetectie,
    batch: int,
    workers: int,
    amesteca: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        set_date,
        batch_size=batch,
        shuffle=amesteca,
        num_workers=workers,
        collate_fn=colateaza,
        generator=generator,
        worker_init_fn=init_worker if workers else None,
    )


def decodifica(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Din tensorul brut [batch, 5, grilă, grilă] face centru, mărime și logit de obiect.

    Centrul unei celule (coloană gx, rând gy) este
    ((gx + sigmoid(decalaj_x)) / grilă, (gy + sigmoid(decalaj_y)) / grilă).
    """
    _, _, gh, gw = raw.shape
    gy, gx = torch.meshgrid(
        torch.arange(gh, device=raw.device, dtype=raw.dtype),
        torch.arange(gw, device=raw.device, dtype=raw.dtype),
        indexing="ij",
    )
    tx, ty, tw, th, tobj = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
    cx = (gx + tx.sigmoid()) / gw
    cy = (gy + ty.sigmoid()) / gh
    return cx, cy, tw.sigmoid(), th.sigmoid(), tobj


def tinte_din_casete(
    casete: list[torch.Tensor],
    grila: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marchează celula în care cade centrul fiecărei persoane.

    O celulă primește o singură casetă. Dacă două persoane cad în aceeași
    celulă, rămâne caseta cu aria mai mare.
    """
    batch = len(casete)
    obiect = torch.zeros(batch, grila, grila, device=device)
    tinta = torch.zeros(batch, 4, grila, grila, device=device)
    for indice, cutii in enumerate(casete):
        arii: dict[tuple[int, int], float] = {}
        for cx, cy, latime, inaltime in cutii.tolist():
            cx = min(max(cx, 0.0), 1.0 - 1e-4)
            cy = min(max(cy, 0.0), 1.0 - 1e-4)
            gx = min(grila - 1, max(0, int(cx * grila)))
            gy = min(grila - 1, max(0, int(cy * grila)))
            arie = latime * inaltime
            if arie <= arii.get((gy, gx), -1.0):
                continue
            arii[(gy, gx)] = arie
            obiect[indice, gy, gx] = 1.0
            tinta[indice, 0, gy, gx] = cx
            tinta[indice, 1, gy, gx] = cy
            tinta[indice, 2, gy, gx] = latime
            tinta[indice, 3, gy, gx] = inaltime
    return obiect, tinta


def iou_grile(cx: torch.Tensor, cy: torch.Tensor, latime: torch.Tensor, inaltime: torch.Tensor, tinta: torch.Tensor) -> torch.Tensor:
    """IoU între caseta prezisă și cea adevărată, pentru fiecare celulă."""
    px1, py1 = cx - latime / 2, cy - inaltime / 2
    px2, py2 = cx + latime / 2, cy + inaltime / 2
    gx1 = tinta[:, 0] - tinta[:, 2] / 2
    gy1 = tinta[:, 1] - tinta[:, 3] / 2
    gx2 = tinta[:, 0] + tinta[:, 2] / 2
    gy2 = tinta[:, 1] + tinta[:, 3] / 2
    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    intersectie = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    aria_pred = latime.clamp(min=0) * inaltime.clamp(min=0)
    aria_gt = tinta[:, 2].clamp(min=0) * tinta[:, 3].clamp(min=0)
    return intersectie / (aria_pred + aria_gt - intersectie).clamp(min=1e-6)


def calculeaza_loss(
    raw: torch.Tensor,
    casete: list[torch.Tensor],
) -> torch.Tensor:
    """Loss-ul are trei părți, toate medii, ca numărul de celule să nu domine.

    Caseta: 1 - IoU, doar în celula responsabilă. Aproape 1 la început, 0 când
    caseta prezisă acoperă persoana.
    Obiect: cât de jos este scorul în celula cu persoană.
    Fundal: cât de sus este scorul în celulele goale.
    """
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
    return LAMBDA_CASETA * loss_caseta + loss_obiect + PONDERE_FUNDAL * loss_fundal


def casete_xyxy(cx: float, cy: float, latime: float, inaltime: float) -> tuple[float, float, float, float]:
    return cx - latime / 2, cy - inaltime / 2, cx + latime / 2, cy + inaltime / 2


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU între o casetă [4] și mai multe casete [N, 4], în coordonate x1,y1,x2,y2."""
    x1 = torch.maximum(a[0], b[:, 0])
    y1 = torch.maximum(a[1], b[:, 1])
    x2 = torch.minimum(a[2], b[:, 2])
    y2 = torch.minimum(a[3], b[:, 3])
    intersectie = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aria_a = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0)
    aria_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return intersectie / (aria_a + aria_b - intersectie).clamp(min=1e-6)


def eliminare_suprapuneri(casete: torch.Tensor, prag: float) -> torch.Tensor:
    """Păstrează casetele cu scor mare și le aruncă pe cele care se calcă prea tare."""
    if casete.numel() == 0:
        return casete
    ordine = casete[:, 4].argsort(descending=True)
    pastrate: list[torch.Tensor] = []
    while ordine.numel():
        curent = ordine[0]
        pastrate.append(casete[curent])
        if ordine.numel() == 1:
            break
        ramas = ordine[1:]
        suprapunere = iou_xyxy(casete[curent, :4], casete[ramas, :4])
        ordine = ramas[suprapunere < prag]
    return torch.stack(pastrate)


def preziceri_imagine(
    cx: torch.Tensor,
    cy: torch.Tensor,
    latime: torch.Tensor,
    inaltime: torch.Tensor,
    logit: torch.Tensor,
) -> torch.Tensor:
    """Toate celulele unei imagini, filtrate după scor și după suprapunere."""
    scor = logit.sigmoid()
    x1 = cx - latime / 2
    y1 = cy - inaltime / 2
    x2 = cx + latime / 2
    y2 = cy + inaltime / 2
    toate = torch.stack([x1, y1, x2, y2, scor], dim=-1).reshape(-1, 5)
    alese = toate[toate[:, 4] >= PRAG_SCOR]
    return eliminare_suprapuneri(alese, PRAG_IOU)


def potriveste(pred: torch.Tensor, adevar: torch.Tensor) -> tuple[int, int, int]:
    """Numără true positive, false positive și false negative la IoU >= PRAG_IOU."""
    if adevar.numel() == 0:
        return 0, int(pred.shape[0]), 0
    gt = torch.stack(
        [
            torch.tensor([casete_xyxy(cx, cy, latime, inaltime)], dtype=torch.float32)
            for cx, cy, latime, inaltime in adevar.tolist()
        ]
    ).squeeze(1)
    if pred.numel() == 0:
        return 0, 0, int(gt.shape[0])
    ordine = pred[:, 4].argsort(descending=True)
    folosite: set[int] = set()
    corecte = 0
    for indice in ordine.tolist():
        suprapuneri = iou_xyxy(pred[indice, :4], gt)
        for ocupat in folosite:
            suprapuneri[ocupat] = -1.0
        cel_mai_bun = int(suprapuneri.argmax())
        if float(suprapuneri[cel_mai_bun]) >= PRAG_IOU:
            folosite.add(cel_mai_bun)
            corecte += 1
    return corecte, int(pred.shape[0]) - corecte, int(gt.shape[0]) - corecte


def pas_epoca(
    model: nn.Module,
    incarcator: DataLoader,
    dispozitiv: torch.device,
    optimizator: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """O epocă. Cu optimizator se antrenează; fără el doar se evaluează."""
    antrenare = optimizator is not None
    model.train(antrenare)
    loss_total = 0.0
    vazute = 0
    tp = fp = fn = 0
    scor_max = 0.0
    for imagini, casete in incarcator:
        imagini = imagini.to(dispozitiv)
        with torch.set_grad_enabled(antrenare):
            # 1. Propagare înainte: 5 valori pentru fiecare celulă din grilă.
            raw = model(imagini)
            # 2. Loss-ul adună eroarea casetei, a celulei cu persoană și a fundalului.
            loss = calculeaza_loss(raw, casete)
            if antrenare:
                assert optimizator is not None
                # 3. Gradienții vechi nu se adună peste cei noi.
                optimizator.zero_grad()
                # 4. Backpropagation umple .grad pentru fiecare parametru.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                # 5. AdamW mută parametrii ca să coboare loss-ul.
                optimizator.step()
        loss_total += loss.item() * imagini.size(0)
        vazute += imagini.size(0)
        cx, cy, latime, inaltime, logit = decodifica(raw.detach())
        scor_max = max(scor_max, float(logit.sigmoid().max()))
        for indice in range(imagini.size(0)):
            pred = preziceri_imagine(cx[indice], cy[indice], latime[indice], inaltime[indice], logit[indice])
            corecte, false_pos, false_neg = potriveste(pred.cpu(), casete[indice])
            tp += corecte
            fp += false_pos
            fn += false_neg
    precizie = tp / (tp + fp) if tp + fp else 0.0
    rapel = tp / (tp + fn) if tp + fn else 0.0
    return {
        "loss": loss_total / vazute,
        "precision": precizie,
        "recall": rapel,
        "scor_max": scor_max,
    }


def salveaza_checkpoint(
    cale: Path,
    model: nn.Module,
    optimizator: torch.optim.Optimizer,
    epoca: int,
    cel_mai_bun: float,
    epoci_fara_progres: int,
    args: argparse.Namespace,
) -> None:
    cale.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoca,
            "model": model.state_dict(),
            "optimizer": optimizator.state_dict(),
            "best_val_loss": cel_mai_bun,
            "epoci_fara_progres": epoci_fara_progres,
            "imgsz": args.imgsz,
            "stride": STRIDE,
            "lr": args.lr,
        },
        cale,
    )


def scrie_istoric(cale: Path, randuri: list[dict[str, float | int]]) -> None:
    with cale.open("w", newline="") as fisier:
        writer = csv.DictWriter(
            fisier,
            fieldnames=["epoch", "train_loss", "val_loss", "precision", "recall", "scor_max"],
        )
        writer.writeheader()
        writer.writerows(randuri)


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Lipsește configurația setului: {args.data}")
    if args.imgsz % STRIDE != 0:
        raise ValueError(f"--imgsz trebuie să fie multiplu de {STRIDE}. Grila este imgsz/{STRIDE}.")

    if args.resume and args.versiune:
        raise ValueError("Folosește fie --resume, fie --versiune, nu pe amândouă.")

    seteaza_samanta(args.seed)
    dispozitiv = rezolva_dispozitiv(args.device)
    director = rezolva_director_rulare(args.project, args.name, args.versiune)
    ponderi = director / "weights"
    ultimul = ponderi / "last.pt"
    cel_mai_bun_fisier = ponderi / "best.pt"
    grila = args.imgsz // STRIDE

    print(
        "Antrenare YOLO simplu de la zero\n"
        f"  arhitectura : YOLOSimplu  (grilă {grila}×{grila}, o casetă pe celulă)\n"
        f"  date        : {args.data}\n"
        f"  iesire      : {director}\n"
        f"  epoci       : {args.epochs}  patience={args.patience}  imgsz={args.imgsz}  "
        f"batch={args.batch}  lr={args.lr}\n"
        f"  dispozitiv  : {dispozitiv}"
    )

    train_imagini, val_imagini = citeste_splituri(args.data)
    set_train = SetDetectie(train_imagini, args.imgsz, augmentare=True)
    set_val = SetDetectie(val_imagini, args.imgsz, augmentare=False)
    print(f"  train       : {len(set_train)} imagini")
    print(f"  val         : {len(set_val)} imagini")

    incarcator_train = construieste_incarcator(set_train, args.batch, args.workers, True, args.seed)
    incarcator_val = construieste_incarcator(set_val, args.batch, args.workers, False, args.seed + 1)

    model = construieste_model().to(dispozitiv)
    # weight_decay=0: singurul semnal este loss-ul de mai sus.
    optimizator = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    start = 1
    cel_mai_bun = float("inf")
    epoci_fara_progres = 0
    istoric: list[dict[str, float | int]] = []

    if args.resume:
        if not ultimul.is_file():
            raise FileNotFoundError(f"Nu există checkpoint de reluat: {ultimul}")
        checkpoint = torch.load(ultimul, map_location=dispozitiv, weights_only=False)
        if int(checkpoint["imgsz"]) != args.imgsz:
            raise ValueError(
                f"Checkpoint-ul a fost antrenat cu imgsz={checkpoint['imgsz']}, iar acum imgsz={args.imgsz}."
            )
        model.load_state_dict(checkpoint["model"])
        optimizator.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["epoch"]) + 1
        cel_mai_bun = float(checkpoint["best_val_loss"])
        epoci_fara_progres = int(checkpoint["epoci_fara_progres"])
        print(f"Reluare de la epoca {start}, cel mai bun loss de validare = {cel_mai_bun:.4f}")
    elif director.exists() and not args.versiune:
        print(f"Suprascriu rularea existentă: {director}")

    for epoca in range(start, args.epochs + 1):
        train = pas_epoca(model, incarcator_train, dispozitiv, optimizator)
        val = pas_epoca(model, incarcator_val, dispozitiv, None)
        istoric.append(
            {
                "epoch": epoca,
                "train_loss": train["loss"],
                "val_loss": val["loss"],
                "precision": val["precision"],
                "recall": val["recall"],
                "scor_max": val["scor_max"],
            }
        )
        print(
            f"epoca {epoca:3d}  "
            f"train loss={train['loss']:.4f}  "
            f"val loss={val['loss']:.4f}  "
            f"P={val['precision']:.2f}  R={val['recall']:.2f}  "
            f"scor_max={val['scor_max']:.2f}"
        )

        if val["loss"] < cel_mai_bun:
            cel_mai_bun = val["loss"]
            epoci_fara_progres = 0
        else:
            epoci_fara_progres += 1

        salveaza_checkpoint(ultimul, model, optimizator, epoca, cel_mai_bun, epoci_fara_progres, args)
        if epoci_fara_progres == 0:
            salveaza_checkpoint(
                cel_mai_bun_fisier, model, optimizator, epoca, cel_mai_bun, epoci_fara_progres, args
            )
        elif epoci_fara_progres >= args.patience:
            print(
                f"Oprire anticipată: loss-ul de validare nu a scăzut de {args.patience} epoci. "
                f"Cel mai bun loss = {cel_mai_bun:.4f}"
            )
            break

    director.mkdir(parents=True, exist_ok=True)
    scrie_istoric(director / "history.csv", istoric)
    print(f"Gata. Cele mai bune ponderi: {cel_mai_bun_fisier}")


if __name__ == "__main__":
    main()
