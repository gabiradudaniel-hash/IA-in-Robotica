# Person CNN, antrenat de la zero

Clasifică un decupaj grayscale în **persoană** sau **fundal**. Rețeaua este un CNN mic în PyTorch, cu ponderi inițializate aleatoriu. Nu se încarcă un checkpoint ImageNet sau YOLO.

Setul din `data/` are cadre de dronă și casete YOLO pentru clasa `person`. La antrenare, fiecare casetă devine un decupaj pozitiv. Pentru fiecare casetă se alege și un decupaj de fundal, de aceeași mărime, care nu se suprapune cu persoana.

## Instalare

Rulează comenzile din **rădăcina laboratorului** (`lab_CNN/`), acolo unde se află `data.zip` și `requirements.txt`.

### 1. Set de date (`data.zip`)

Dacă nu ai deja folderul `data/` (cu `data.yaml`, `train/`, `val/`), dezarhivează arhiva. După extragere trebuie să existe `data/data.yaml`.

**macOS / Linux**

```bash
cd lab_CNN
unzip -o data.zip
```

**Windows — PowerShell**

```powershell
cd lab_CNN
Expand-Archive -LiteralPath data.zip -DestinationPath . -Force
```

**Windows — Command Prompt** (Windows 10 sau mai nou; `tar` este inclus)

```bat
cd lab_CNN
tar -xf data.zip
```

**Alternativă** (orice sistem, dacă ai Python instalat și lipsește `unzip`):

```bash
cd lab_CNN
python3 -m zipfile -e data.zip .
```

Pe Windows, înlocuiește `python3` cu `py -3` sau `.\.venv\Scripts\python.exe` după ce ai creat mediul virtual.

### 2. Mediu Python

**macOS sau Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows, PowerShell.** Nu folosi `Activate.ps1`: politica sistemului îl blochează. Python-ul din mediul virtual se apelează direct:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_cnn_scratch_en.py
```

**Windows, Command Prompt**, dacă preferi activarea:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python train_cnn_scratch_en.py
```

## Laborator YOLO simplu

Manual (task-uri 1–16): **`README_YOLO_SIMPLU.md`**. Activitățile se rulează din `lab_2/`:

```bash
python scripts_lab/t00_ruleaza_tot.py              # toate scripturile
python scripts_lab/t00_ruleaza_tot.py --skip-train # fără antrenament
python scripts_lab/t03_eticheta_si_caseta.py
```

Lista completă: `scripts_lab/README.md`. Demo: `scripts_lab/demo/`.

## Antrenare

Din rădăcina acestui repo, după activarea mediului pe macOS sau Linux:

```bash
python train_cnn_scratch_en.py
```

Implicit: 40 de epoci, imagini de 64×64, batch 16, dispozitiv `cpu`.

```bash
python train_cnn_scratch_en.py --epochs 40 --batch 16 --device cpu
```

Pe un Mac cu GPU Apple se poate folosi `--device mps`. Pe NVIDIA, `--device cuda`.

Ponderile și istoricul ajung în `runs/classify/cnn_scratch_person/`. Fișierul cel mai bun este `weights/best.pt`. La fiecare epocă se afișează loss-ul și acuratețea pe antrenare și pe validare.

Reluarea unei rulări întrerupte:

```bash
python train_cnn_scratch_en.py --resume
```

## Date

| Split | Imagini |
| --- | ---: |
| train | 227 |
| val | 9 |

Imaginile sunt un subset din [PULP-DroNet v3](https://github.com/pulp-platform/pulp-dronet), publicat sub [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Casetele de persoană sunt pseudo-etichete YOLO. Setul inițial a folosit YOLO11s. Cadrele adăugate din `acquisition7`, `acquisition11`, `acquisition13`, `acquisition35`, `acquisition37` și `acquisition39` sunt etichetate cu YOLO26x. `acquisition4` nu conține persoane și nu a intrat în set. Folosirea este necomercială, cu atribuire, iar lucrările derivate păstrează aceeași licență.
