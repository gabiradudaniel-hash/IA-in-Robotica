# Laborator: detector YOLO simplu (`train_yolo_simplu.py`)

**Script principal:** `train_yolo_simplu.py`  
**Set de date:** `data/data.yaml` (detectie persoană, format YOLO)  
**Obiectiv:** să înțelegeți un detector tip YOLO — etichete, grilă, rețea, loss, antrenare, metrici — în PyTorch lizibil, fără Ultralytics.

Parcurgeți **task-urile în ordine**. Fiecare task cu script are comanda în `scripts_lab/` (din directorul `lab_2/`):

```bash
python scripts_lab/t00_ruleaza_tot.py              # toate scripturile (+ antrenament scurt)
python scripts_lab/t00_ruleaza_tot.py --skip-train # fără antrenament
```

Imaginile din set sunt **`.jpeg`**. Demo fix: `scripts_lab/demo/`. Ieșiri vizuale: `scripts_lab/out/`.

**Formule:** README-ul folosește LaTeX — inline `$...$`, pe rând `$$...$$`. Pe **GitHub** se randează în preview-ul fișierului `.md`. În **Cursor/VS Code**: `⌘ Shift V` (preview); dacă nu vezi formule, instalează extensia *Markdown+Math* sau deschide pe GitHub.

---

## Cuprins

- [Pregătire](#pregătire)
- [Task 1 — Detectie vs clasificare, ideea YOLO](#task-1--detectie-vs-clasificare-ideea-yolo)
- [Task 2 — Configurare și încărcarea datelor în cod](#task-2--configurare-și-încărcarea-datelor-în-cod)
- [Task 3 — Etichetă YOLO și casetă](#task-3--etichetă-yolo-și-casetă)
- [Task 4 — Letterbox](#task-4--letterbox)
- [Task 5 — Split train/val și docstring rețea](#task-5--split-trainval-și-docstring-rețea)
- [Task 6 — Grilă, `tx ty tw th` → `cx cy w h`](#task-6--grilă-tx-ty-tw-th--cx-cy-w-h)
- [Task 7 — Arhitectura `YOLOSimplu`](#task-7--arhitectura-yolosimplu)
- [Task 8 — Număr de parametri](#task-8--număr-de-parametri)
- [Task 9 — IoU și loss de antrenare](#task-9--iou-și-loss-de-antrenare)
- [Task 10 — Verificare predicție (TP, FP, FN)](#task-10--verificare-predicție-tp-fp-fn)
- [Task 11 — O epocă: antrenare vs validare](#task-11--o-epocă-antrenare-vs-validare)
- [Task 12 — Antrenament scurt](#task-12--antrenament-scurt)
- [Task 13 — P/R pe setul val (model antrenat)](#task-13--pr-pe-setul-val-model-antrenat)
- [Task 14 — Pipeline complet `main`](#task-14--pipeline-complet-main)
- [Task 15 — Analiza `history.csv`](#task-15--analiza-historycsv)
- [Task 16 — Vizualizare și proiect final](#task-16--vizualizare-și-proiect-final)
- [Anexe](#anexe)

---

## Pregătire

### Ce veți învăța

- diferența **clasificare** vs **detectie obiecte**;
- format **etichete YOLO** (`class cx cy w h`, normalizat);
- ideea **YOLO**: imagine → hartă → **grilă** → casetă + scor;
- `Dataset`, `DataLoader`, `forward`, `loss`, `backward`, checkpoint;
- **precision**, **recall**, `history.csv`.

### Prerequisite

Python; PyTorch (`nn.Module`, `Dataset`, `DataLoader`); Conv2D + MaxPool (lab CNN anterior ajută). Opțional: IoU.

### Instalare

```bash
cd lab_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Structura setului

`data/data.yaml`:

```yaml
path: .
train: train/images
val: val/images
nc: 1
names:
  0: person
```

- Imagini: `data/train/images/`, `data/val/images/` (grayscale).
- Etichete: `train/labels/<stem>.txt` pentru `train/images/<stem>.jpeg`.
- Linie etichetă: `class_id cx cy width height` — normalizat la [0,1] față de imaginea **originală**.

Exemplu:

```text
0 0.561125 0.621504 0.179265 0.737622
```

Clasă `0` (person), centru (56,1%, 62,2%), lățime 17,9%, înălțime 73,8%.

### Antrenament (comandă de referință)

```bash
python train_yolo_simplu.py \
  --data data/data.yaml \
  --project runs/detect \
  --name yolo_simplu_person \
  --epochs 40 \
  --imgsz 160 \
  --batch 16 \
  --device cpu
```

### Constante în cod

| Simbol | Valoare | Rol |
|--------|---------|-----|
| `STRIDE` | 16 | 4× MaxPool(2) → grila = imgsz/16 |
| `LAMBDA_CASETA` | 5.0 | Pondere loss casetă (IoU) |
| `PONDERE_FUNDAL` | 4.0 | Pondere BCE celule goale |
| `PRAG_SCOR` | 0.25 | Prag objectness la inferență |
| `PRAG_IOU` | 0.5 | NMS și TP la evaluare |

La `imgsz=160` → grilă **10×10**.

### După antrenament

Sub `runs/detect/yolo_simplu_person/` (sau `…2` cu `--versiune`): `weights/best.pt`, `weights/last.pt`, `history.csv`.

---

## Task 1 — Detectie vs clasificare, ideea YOLO

**Focus:** problemă, intuiție YOLO (înainte de cod).

### Clasificare vs detectie

| | Clasificare (`train_cnn_scratch_en.py`) | Detectie (acest lab) |
|---|----------------------------------------|----------------------|
| Intrare | Decupaj mic (ex. 64×64) | Cadru întreg (ex. 160×160) |
| Ieșire | Probabilitate „persoană” | **Casete** (x, y, w, h) + scor |
| Etichetă | 0/1 per decupaj | Coordonate per persoană |

Lab-ul CNN folosea YOLO doar ca **sursă de decupaje**. Aici: **imagine întreagă** + casete.

### Ideea YOLO (versiunea din script)

1. Imaginea devine o **grilă** (ex. 10×10).
2. Fiecare celulă emite **5 valori**: offset centru, lățime, înălțime, **scor obiect**.
3. La antrenare: **celula unde cade centrul** persoanei regresează caseta (assign simplu).
4. La test: prag pe scor + **NMS** (elimină duplicate).

Nu e YOLOv8 complet (o grilă, o clasă, o casetă/celulă), dar **logica e aceeași**.

---

## Task 2 — Configurare și încărcarea datelor în cod

**Focus cod:** `parse_args`, `citeste_splituri`, `citeste_casete`, `incadreaza`.

Deschideți `train_yolo_simplu.py`:

1. **`parse_args()`** — `epochs`, `patience`, `imgsz`, `batch`, `lr`, `device`, `seed`, `resume`, `versiune`.
2. **`citeste_splituri(data_yaml)`** — căi train/val relativ la `path:`.
3. **`citeste_casete(cale)`** — `(cx, cy, w, h)` din `.txt`.
4. **`incadreaza(imagine, casete, imgsz)`** — scalare la pătrat, padding, recalcul casete normalizate.


---

## Task 3 — Etichetă YOLO și casetă

**Script:**

```bash
python scripts_lab/t03_eticheta_si_caseta.py
python scripts_lab/t03_eticheta_si_caseta.py --demo
python scripts_lab/t03_eticheta_si_caseta.py --imagine data/train/images/acquisition7_3891.jpeg
```

Scriptul afișează `cx, cy, w, h` și `x1,y1,x2,y2`, salvează caseta desenată după letterbox.

`cx` este Centrul pe orizontală, fracție din lățime.

---

## Task 4 — Letterbox

**Script:**

```bash
python scripts_lab/t04_letterbox.py
```


---

## Task 5 — Split train/val și docstring rețea

Citiți docstring-ul clasei **`YOLOSimplu`** (5 canale de ieșire).

**Script:**

```bash
python scripts_lab/t05_split_train_val.py
```

---

## Task 6 — Grilă, `tx ty tw th` → `cx cy w h`

**Focus:** forma ieșirii rețelei și funcția **`decodifica`** — singurul loc unde `tx, ty, tw, th` devin casetă pe imagine (normalizat 0…1, ca etichetele YOLO după letterbox).

### 6.1 De la imagine la grilă

`G = imgsz / 16` (ex. `imgsz=160` → **`G=10`**).

```text
Intrare:      [B, 1, 160, 160]
După trunchi: [B, 128, 10, 10]
După cap:     [B, 5, 10, 10]     ← 5 numere per celulă
```

Fiecare celulă are indici întregi **`gx`** (coloană, 0…G−1) și **`gy`** (rând). Rețeaua nu scrie direct `cx, cy, w, h`; scrie **`tx, ty, tw, th, tobj`** (logit-uri — numere reale oarecare).

| Canal | Simbol | Rol |
|-------|--------|-----|
| 0 | `tx` | offset centru X **în celulă** |
| 1 | `ty` | offset centru Y **în celulă** |
| 2 | `tw` | lățime casetă (devine `w`) |
| 3 | `th` | înălțime casetă (devine `h`) |
| 4 | `tobj` | scor „e persoană aici?” (nu e coordonată) |

### 6.2 Transformarea (tot ce face `decodifica`)

Fie $\sigma(x) = \frac{1}{1 + e^{-x}}$. Pentru celula $(g_x, g_y)$:

$$
c_x = \frac{g_x + \sigma(t_x)}{G},\quad
c_y = \frac{g_y + \sigma(t_y)}{G},\quad
w = \sigma(t_w),\quad h = \sigma(t_h)
$$

Rezultatul **`(cx, cy, w, h)`** e în același format ca eticheta YOLO pe pătratul letterbox. Colțuri: `x1 = cx − w/2`, `y1 = cy − h/2`, etc. Pixeli pe 160×160: înmulțești cu `imgsz`.

**De ce σ?** Convolutia + bias poate da `tx = −100` sau `+50`. Tu ai nevoie de offset în **(0, 1)** și de **`w, h` pozitive, ≤ 1**. Sigmoid e doar „împinge numărul în intervalul 0…1” — nu e magie YOLO, e legătura ieșire liberă → coordonate valide.

**Exemplu** (`G=10`, celula `gx=5`, `gy=7`, `tx=0`, `ty=0`, `tw=−2`, `th=−2`):

```text
cx = (5 + 0.5) / 10 = 0.55
cy = (7 + 0.5) / 10 = 0.75
w  = h ≈ 0.12
```

**Exemplu 2** (`gx=0`, `tx=0` → `cx = 0.05`).

La inferență, $s = \sigma(t_{\mathrm{obj}})$ se compară cu $\tau_s = 0{,}25$ (`PRAG_SCOR`). La loss, PyTorch folosește logit-ul direct (`binary_cross_entropy_with_logits`).

### 6.3 Rulează

```bash
python scripts_lab/t06_decodifica_grila.py
```

Citește în cod **`decodifica`** din `train_yolo_simplu.py` — trebuie să recunoști formulele de mai sus.

**Verificare:**

1. Câte celule la `imgsz=128`? (**8×8.**) De ce `imgsz` multiplu de 16?

---

## Task 7 — Arhitectura `YOLOSimplu`

**Trunchi:** Conv3×3 → ReLU → MaxPool ×4; canale 1→16→32→64→128.

**Cap:** `Conv2d(128, 5, 1×1)`.



---

## Task 8 — Număr de parametri

Calcul Conv2d: `cin * cout * k² + cout`. Primul strat 1→16, k=3: **160** parametri.

**Script:**

```bash
python scripts_lab/t08_parametri_model.py
```

---

## Task 9 — IoU și loss de antrenare

**Antrenare** (nu e aceeași regulă ca TP/FP la validare). Grilă $G \times G$, ~99 celule goale, 1 cu persoană.

**Assign:** centrul GT → celula $(g_x, g_y)$, acolo `obiect = 1`.

**IoU** (casete în coordonate normalizate):

$$
\mathrm{IoU} = \frac{A_{\cap}}{A_{\cup}} \in [0,1]
$$

**Loss** (3 termeni, apoi sumă):

$$
\mathcal{L}_{\mathrm{box}} = \frac{1}{N_{+}} \sum_{\text{celule cu persoană}} (1 - \mathrm{IoU})
$$

$$
\mathcal{L}_{\mathrm{obj}} = \mathrm{BCE}(t_{\mathrm{obj}}, 1) \quad \text{(medie pe cele } N_{+} \text{ celule)}
$$

$$
\mathcal{L}_{\mathrm{bg}} = \mathrm{mean}\big(\mathrm{BCE}(t_{\mathrm{obj}}, 0)\big) \quad \text{(pe cele } G^2 - N_{+} \text{ celule goale)}
$$

$$
\mathcal{L} = 5\,\mathcal{L}_{\mathrm{box}} + \mathcal{L}_{\mathrm{obj}} + 4\,\mathcal{L}_{\mathrm{bg}}
$$

În cod: `loss_caseta` = $\mathcal{L}_{\mathrm{box}}`, `loss_obiect` = $\mathcal{L}_{\mathrm{obj}}`, `loss_fundal` = $\mathcal{L}_{\mathrm{bg}}$.

### Ce face fiecare termen

Pe fiecare celulă rețeaua dă o casetă $(c_x,c_y,w,h)$ și logit-ul $t_{\mathrm{obj}}$. Loss-ul antrenează **doar la antrenare** (nu e aceeași regulă ca TP/FP din Task 10).

#### $\mathcal{L}_{\mathrm{box}}$ — forma cutiei (`loss_caseta`)

| | |
|---|---|
| **Unde** | Doar pe celulele cu `obiect = 1` ($N_+$ celule; de obicei 1 per imagine). |
| **Ce compară** | Caseta **prezisă** vs caseta **din etichetă** (IoU). |
| **Ce învață** | Să **mute și redimensioneze** dreptunghiul peste persoană. |
| **Ce nu face** | Nu decide „e persoană aici?” — doar „dacă știm că e aici, cutia e corectă?”. |
| **× 5** | Fără pondere, modelul ar putea învăța doar scorul, cu cutii slabe. |

#### $\mathcal{L}_{\mathrm{obj}}$ — scor pe celula cu persoană (`loss_obiect`)

| | |
|---|---|
| **Unde** | Aceleași $N_+$ celule. |
| **Țintă** | $t_{\mathrm{obj}}$ trebuie să corespundă clasei **1** (există persoană). |
| **Ce învață** | Să ridice $s = \sigma(t_{\mathrm{obj}})$ **doar** pe celula unde cade centrul persoanei. |
| **Intuiție** | „Aici știm că e ceva — creste increderea in aceasta predictie.” |

#### $\mathcal{L}_{\mathrm{bg}}$ — scor pe fundal (`loss_fundal`)

| | |
|---|---|
| **Unde** | Cele $G^2 - N_+$ celule goale (~99/imagine). |
| **Țintă** | $t_{\mathrm{obj}}$ → **0** (nu e persoană). |
| **Ce învață** | Să **coboare** scorul peste aproape toată grila. |
| **× 4** | Sunt multe celule fundal; altfel scor mediu ~0.2 peste tot poate părea „acceptabil”. |

#### Rezumat

| Termen | O propoziție |
|--------|----------------|
| $\mathcal{L}_{\mathrm{box}}$ | Potrivește **dreptunghiul**. |
| $\mathcal{L}_{\mathrm{obj}}$ | **Aprinde** scorul pe celula corectă. |
| $\mathcal{L}_{\mathrm{bg}}$ | **Stinge** scorul pe restul grilei. |

**BCE** (binary cross-entropy) = penalizare când scorul (din logit) nu e 0/1 cum cere ținta; PyTorch: `binary_cross_entropy_with_logits`.

**Script:**

```bash
python scripts_lab/t09_loss_batch.py
```

### Cum citești output-ul (exemplu model neantrenat)

Batch **4 imagini** → de obicei **4 celule cu persoană** ($N_+ = 4$), restul **396** celule fundal.

| Ce vezi | Înseamnă |
|---------|----------|
| `loss_caseta ≈ 0.88` | IoU mediu $\approx 1 - 0.88 = 0.12$ — cutiile prezise **nu** acoperă persoana (normal la început). |
| `× 5` → **~4.4** | Cea mai mare parte din total — modelul trebuie să învețe **forma** casetei. |
| `loss_obiect ≈ 2.15` | Pe cele 4 celule vrem scor **1**; modelul dă ~**0.12** → penalizare mare. Contribuție **~2.15**. |
| `loss_fundal ≈ 0.12` | Pe ~396 celule vrem scor **0**; eroarea medie e mică, dar `× 4` → **~0.5** tot adună. |
| **total ≈ 7.05** | $5 \cdot 0.88 + 2.15 + 4 \cdot 0.12 \approx 7$ — **nu** e „7 greșeli”, e o singură cifră de optimizat. |

După multe epoci: `loss_caseta` și `loss_obiect` scad; `scor` pe persoană urcă peste 0.25 (Task 10).

---

## Task 10 — Verificare predicție (TP, FP, FN)

**Validare:** ce e detectie **corectă** / **greșită**. Constante: $\tau_s = 0{,}25$, $\tau = 0{,}5$.

### 1. Decodificare (per celulă)

$$
c_x = \frac{g_x + \sigma(t_x)}{G},\quad c_y = \frac{g_y + \sigma(t_y)}{G},\quad
w = \sigma(t_w),\quad h = \sigma(t_h),\quad s = \sigma(t_{\mathrm{obj}})
$$

$$
x_1 = c_x - \tfrac{w}{2},\ y_1 = c_y - \tfrac{h}{2},\ x_2 = c_x + \tfrac{w}{2},\ y_2 = c_y + \tfrac{h}{2}
$$

### 2. Filtru scor

Păstrezi doar celulele cu $s \ge \tau_s$.

### 3. NMS

Sortare după $s$ descrescător; elimini perechi cu $\mathrm{IoU} \ge \tau$. Rezultat: predicții $\mathcal{P}$, GT $\mathcal{G}$.

### 4. Potrivire (`potriveste`)

Pentru fiecare $P_j$ (în ordinea scorului):

$$
k^* = \arg\max_k \mathrm{IoU}(P_j, G_k)
$$

- Dacă $G_{k^*}$ liber și $\mathrm{IoU}(P_j, G_{k^*}) \ge \tau$ → **TP**
- Altfel → **FP**

$$
\mathrm{FN} = |\mathcal{G}| - \#\{\text{GT potrivite}\}
$$

### 5. Metrici (pe epocă)

$$
P = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad
R = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
$$

| | Condiție |
|---|----------|
| **TP** | pred trece filtre + $\mathrm{IoU} \ge \tau$ cu GT nefolosit |
| **FP** | pred trece filtre, fără GT potrivit |
| **FN** | GT fără pred potrivit |

**Script (3 scenarii sintetice + `potriveste`):**

```bash
python scripts_lab/t10_verificare_predictii.py
```

---

## Task 11 — O epocă: antrenare vs validare

| Pas | Antrenare | Validare |
|-----|-----------|----------|
| `forward` | da | da |
| `backward` / `step` | da | nu |

Pași în `pas_epoca`: forward → loss → `zero_grad` → backward → clip grad → AdamW.

**Script:**

```bash
python scripts_lab/t11_un_pas_antrenare.py
```

Metricile P/R = Task 10 (`preziceri_imagine` + `potriveste` în cod).

**Verificare:** Ce face `patience` în `main`? (Early stopping când val loss stagnează.)

---

## Task 12 — Antrenament scurt

**Script:**

```bash
python scripts_lab/t12_antrenare_scurta.py
```

Sau:

```bash
python train_yolo_simplu.py --data data/data.yaml --project runs/detect \
  --name lab_test --versiune --epochs 5 --batch 8 --device cpu
```

Urmăriți: `train loss`, `val loss`, `scor_max`, `P`, `R`.

---

## Task 13 — P/R pe setul val (model antrenat)

```bash
python scripts_lab/t13_potriveste_val.py
python scripts_lab/t13_potriveste_val.py --weights runs/detect/yolo_simplu_person/weights/best.pt
```

---

## Task 14 — Pipeline complet `main`

1. Validare args (`imgsz % 16`, `data.yaml`).
2. Seed, device, director rulare.
3. Train (augment) + val.
4. Model + AdamW; opțional `--resume`.
5. Buclă epoci → checkpoint; **early stopping**; `best.pt` la val loss minim.
6. `history.csv`.

**Antrenament recomandat:**

```bash
python train_yolo_simplu.py \
  --data data/data.yaml \
  --project runs/detect \
  --name yolo_simplu_person \
  --epochs 40 --patience 8 --imgsz 160 --batch 16 --lr 1e-3 --device cpu --seed 0
```

Reluare: `--resume`. Rulare nouă fără suprascriere: `--versiune`.

**Verificare:** Ce conține `best.pt`? (`state_dict` model, optimizer, epocă, metadate.)

---

## Task 15 — Analiza `history.csv`

**Script:**

```bash
python scripts_lab/t15_analiza_history.py
```

Comparați `train_loss` vs `val_loss`, `precision`, `recall`, `scor_max`.

---

## Task 16 — Vizualizare și proiect final

**Vizualizare:**

```bash
python scripts_lab/t16_vizualizeaza_detectii.py \
  --weights runs/detect/yolo_simplu_person/weights/best.pt
```

Verde = GT, roșu = predicții în `scripts_lab/out/t16_viz/`.

**La alegere:**

- Raport 1 pagină: FP/FN explicat cu assign pe grilă.
- Ablation: `PONDERE_FUNDAL=1` vs 4 timp de 10 epoci.

---

## Anexe

### Hartă funcții → rol

| Funcție | Rol |
|---------|-----|
| `parse_args` | CLI |
| `citeste_splituri` | Căi train/val |
| `citeste_casete` | Parse `.txt` YOLO |
| `incadreaza` | Letterbox + casete |
| `SetDetectie` | PyTorch Dataset |
| `colateaza` | Batch imagini + listă casete |
| `YOLOSimplu` | Rețea |
| `decodifica` | Raw → cx,cy,w,h,logit |
| `tinte_din_casete` | GT pe grilă |
| `iou_grile` | IoU per celulă |
| `calculeaza_loss` | Antrenare |
| `preziceri_imagine` | Detecții per imagine |
| `eliminare_suprapuneri` | NMS |
| `potriveste` | TP/FP/FN |
| `pas_epoca` | O epocă train/val |
| `main` | Orchestrare |

### Glosar

| Termen | Definiție |
|--------|-----------|
| **Bounding box** | (cx, cy, w, h) |
| **Stride** | imgsz / dimensiune grilă |
| **Objectness** | Scor „obiect în celulă” |
| **Responsible cell** | Celula care învață caseta pentru un GT |
| **Letterbox** | Pătrat + padding |
| **NMS** | Elimină detecții duplicate |
| **IoU** | Suprapunere casete |

### YOLO simplu vs YOLO industrial

| Aspect | Acest lab | YOLO tipic (v8+) |
|--------|-----------|------------------|
| Clase | 1 | multe |
| Scări | o grilă | FPN |
| Assign | centru în celulă | TAL / etc. |
| Loss | IoU + BCE | DFL, CIoU, … |

### Depanare

| Problemă | Soluție |
|----------|---------|
| `--data` greșit | `--data data/data.yaml` |
| `imgsz` nu e multiplu de 16 | 128, 160, 192 |
| P=R=0 mult timp | mai multe epoci; urmăriți `scor_max` |
| macOS multiprocessing | `--workers 0` |

### Rezumat (7 propoziții)

1. Cadru pătrat grayscale.  
2. CNN → grilă 10×10, 5 numere/celulă.  
3. Celula cu centrul persoanei e responsabilă.  
4. Loss: IoU + obiect + fundal.  
5. AdamW + backprop.  
6. Inferență: prag + NMS.  
7. P/R pe validare.

---

*Aliniat la `train_yolo_simplu.py` și `scripts_lab/` — 2026.*
