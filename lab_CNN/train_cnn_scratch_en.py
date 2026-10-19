#!/usr/bin/env python3
"""Antrenează de la zero un CNN în PyTorch pe setul de pseudo-etichete YOLO.

Același `data.yaml` ca la `train_yolo11n_scratch.py`. Detectia prezice bounding boxes;
aici rețeaua învață doar să spună dacă un b-box este o persoană sau fundal.
Ponderile pornesc din inițializarea aleatoare a straturilor. Nu se încarcă
niciun checkpoint ImageNet sau YOLO.

Decupajele pozitive vin din casetele YOLO. Pentru fiecare casetă se extrage
și un b-box de fundal, de aceeași mărime, care nu se suprapune cu persoana.
Toate cadrele păstrate în set conțin deja cel puțin o persoană, deci imaginea
întreagă nu poate fi eticheta „fără persoană”.

Exemplu:
    .venv/bin/python train_cnn_scratch_en.py
    .venv/bin/python train_cnn_scratch_en.py --epochs 40 --batch 16 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "classify"
DEFAULT_NAME = "cnn_scratch_person"

# 0 = fundal, 1 = persoană. CrossEntropyLoss primește acești indici.
CLASS_NAMES = ("fundal", "persoana")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Fișierul data.yaml al setului YOLO.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Directorul în care se scriu rulările.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Numele rulării, sub project/.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8, help="Oprire anticipată dacă loss-ul de validare stagnează.")
    parser.add_argument("--imgsz", type=int, default=64, help="Latura pătrată la care este adus fiecare decupaj.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="Rata de învățare pentru AdamW.")
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda sau indexul GPU-ului.")
    parser.add_argument("--workers", type=int, default=0, help="Procese pentru DataLoader. 0 este stabil pe macOS.")
    parser.add_argument("--seed", type=int, default=0, help="Sămânța aleatoare, pentru reluarea aceleiași rulări.")
    parser.add_argument("--resume", action="store_true", help="Continuă de la weights/last.pt din project/name.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS nu este disponibil în acest PyTorch.")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nu este disponibilă în acest PyTorch.")
        return torch.device("cuda")
    if name.isdigit():
        return torch.device(f"cuda:{name}")
    return torch.device(name)


def read_splits(data_yaml: Path) -> tuple[Path, Path]:
    """Citește căile train/val din data.yaml. Căile relative sunt față de fișierul yaml."""
    config = yaml.safe_load(data_yaml.read_text())
    root = Path(config["path"]) if config.get("path") else data_yaml.parent
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train = Path(config["train"])
    val = Path(config["val"])
    if not train.is_absolute():
        train = root / train
    if not val.is_absolute():
        val = root / val
    return train, val


class PersonCNN(nn.Module):
    """CNN mic pentru imagini grayscale: trei blocuri convoluționale, apoi un strat liniar.

    Intrarea este de dimensiune 1 x imgsz x imgsz. Ieșirea are doi logiți, câte unul
    pentru fundal și pentru persoană. Softmax-ul stă în CrossEntropyLoss.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Conv2d folosește inițializarea implicită PyTorch (Kaiming uniform).
        # Nu copiem ponderi dintr-un model antrenat anterior.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AdaptiveAvgPool2d(1) genereaza o hartă 64×1×1, pe care o turtim înainte de stratul liniar.
        hidden = self.features(x)
        hidden = hidden.flatten(1)
        return self.classifier(hidden)


def build_model() -> PersonCNN:
    """Construiește rețeaua cu ponderi aleatoare. Nu se apelează load_state_dict."""
    model = PersonCNN(num_classes=len(CLASS_NAMES))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri antrenabili: {n_params}")
    return model


@dataclass(frozen=True)
class Crop:
    path: Path
    x1: int
    y1: int
    x2: int
    y2: int
    label: int


def yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Transformă o detectie YOLO (clasă cx cy w h, normalizate) în pixeli x1,y1,x2,y2."""
    if not label_path.is_file():
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = int(round((cx - bw / 2) * width))
        y1 = int(round((cy - bh / 2) * height))
        x2 = int(round((cx + bw / 2) * width))
        y2 = int(round((cy + bh / 2) * height))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            boxes.append((x1, y1, x2, y2))
    return boxes


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def background_crop(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    bw: int,
    bh: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    """Alege o fereastră de aceeași mărime cu caseta curentă, cu IoU mic față de toate casetele."""
    if bw >= width or bh >= height:
        return None
    for _ in range(40):
        nx1 = rng.randint(0, width - bw)
        ny1 = rng.randint(0, height - bh)
        window = (nx1, ny1, nx1 + bw, ny1 + bh)
        if all(iou(window, box) < 0.05 for box in boxes):
            return window
    return None


class CropDataset(Dataset):
    """Decupaje persoană / fundal citite din imaginile și etichetele unui split YOLO."""

    def __init__(self, image_dir: Path, imgsz: int, augment: bool, seed: int) -> None:
        self.imgsz = imgsz
        self.augment = augment
        self.samples = self._build(image_dir, random.Random(seed))
        if not self.samples:
            raise RuntimeError(f"Niciun decupaj în {image_dir}")

    def _build(self, image_dir: Path, rng: random.Random) -> list[Crop]:
        samples: list[Crop] = []
        paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            height, width = image.shape[:2]
            label_path = image_dir.parent / "labels" / f"{path.stem}.txt"
            boxes = yolo_boxes(label_path, width, height)
            for box in boxes:
                x1, y1, x2, y2 = box
                samples.append(Crop(path, x1, y1, x2, y2, 1))
                background = background_crop(boxes, width, height, x2 - x1, y2 - y1, rng)
                if background is not None:
                    fx1, fy1, fx2, fy2 = background
                    samples.append(Crop(path, fx1, fy1, fx2, fy2, 0))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(sample.path)
        crop = image[sample.y1 : sample.y2, sample.x1 : sample.x2]
        if crop.size == 0:
            crop = np.zeros((8, 8), dtype=np.uint8)
        # Persoanele sunt înalte și înguste. Umplem până la pătrat ca să nu le turtim la redimensionare.
        crop = pad_to_square(crop)
        crop = cv2.resize(crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        if self.augment and random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
        tensor = torch.from_numpy(crop).float().unsqueeze(0) / 255.0
        return tensor, sample.label


def pad_to_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side), dtype=image.dtype)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = image
    return canvas


def count_classes(dataset: CropDataset) -> str:
    counts = [0, 0]
    for sample in dataset.samples:
        counts[sample.label] += 1
    return f"fundal={counts[0]}  persoana={counts[1]}"


def init_worker(worker_id: int) -> None:
    # Fiecare proces DataLoader primește un seed propriu, derivată din seed-ul principal.
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_loader(dataset: CropDataset, batch: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        worker_init_fn=init_worker if workers else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """O epocă completă. Cu optimizator se antrenează; fără el doar se evaluează."""
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            # 1. Propagare înainte: doi logiți pentru fiecare decupaj din batch.
            logits = model(images)
            # 2. Loss-ul scalar măsoară cât de departe sunt logiții de etichete.
            loss = criterion(logits, labels)
            if training:
                assert optimizer is not None
                # 3. Gradienții vechi nu se adună peste cei noi.
                optimizer.zero_grad()
                # 4. Backpropagation umple .grad pentru fiecare parametru.
                loss.backward()
                # 5. AdamW mută parametrii ca să coboare loss-ul.
                optimizer.step()
        loss_total += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return loss_total / seen, correct / seen


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best: float,
    epochs_without_improvement: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best,
            "epochs_without_improvement": epochs_without_improvement,
            "class_names": CLASS_NAMES,
            "imgsz": args.imgsz,
            "lr": args.lr,
        },
        path,
    )


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Lipsește configurația setului: {args.data}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = args.project / args.name
    weights_dir = run_dir / "weights"
    last = weights_dir / "last.pt"
    best_path = weights_dir / "best.pt"

    print(
        "Antrenare CNN de la zero\n"
        f"  arhitectura : PersonCNN  (inițializare aleatoare, fără ponderi preantrenate)\n"
        f"  date        : {args.data}\n"
        f"  iesire      : {run_dir}\n"
        f"  epoci       : {args.epochs}  patience={args.patience}  imgsz={args.imgsz}  "
        f"batch={args.batch}  lr={args.lr}\n"
        f"  dispozitiv  : {device}"
    )

    train_images, val_images = read_splits(args.data)
    train_set = CropDataset(train_images, args.imgsz, augment=True, seed=args.seed)
    val_set = CropDataset(val_images, args.imgsz, augment=False, seed=args.seed + 1)
    print(f"  train       : {len(train_set)} decupaje  ({count_classes(train_set)})")
    print(f"  val         : {len(val_set)} decupaje  ({count_classes(val_set)})")

    train_loader = build_loader(train_set, args.batch, args.workers, True, args.seed)
    val_loader = build_loader(val_set, args.batch, args.workers, False, args.seed + 1)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start = 1
    best = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    if args.resume:
        if not last.is_file():
            raise FileNotFoundError(f"Nu există checkpoint de reluat: {last}")
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        print(f"Reluare de la epoca {start}, cel mai bun loss de validare = {best:.4f}")
    elif run_dir.exists():
        raise FileExistsError(
            f"Rularea există deja: {run_dir}. Alege alt --name sau folosește --resume."
        )

    for epoch in range(start, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoca {epoch:3d}  "
            f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_loss < best:
            best = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(last, model, optimizer, epoch, best, epochs_without_improvement, args)
        if epochs_without_improvement == 0:
            save_checkpoint(
                best_path, model, optimizer, epoch, best, epochs_without_improvement, args
            )
        elif epochs_without_improvement >= args.patience:
            print(
                f"Oprire anticipată: loss-ul de validare nu a scăzut de {args.patience} epoci. "
                f"Cel mai bun loss = {best:.4f}"
            )
            break

    run_dir.mkdir(parents=True, exist_ok=True)
    write_history(run_dir / "history.csv", history)
    print(f"Gata. Cele mai bune ponderi: {best_path}")


if __name__ == "__main__":
    main()
