# Laborator YOLO simplu

Rulezi din `lab_2/` (cu venv activ și `--data data/data.yaml` la antrenament).

```bash
python scripts_lab/t00_ruleaza_tot.py
python scripts_lab/t00_ruleaza_tot.py --skip-train
```

| Script | Task manual |
|--------|-------------|
| `t00_ruleaza_tot.py` | toate scripturile din manual (orchestrator) |
| `t03_eticheta_si_caseta.py` | 3 |
| `t04_letterbox.py` | 4 |
| `t05_split_train_val.py` | 5 |
| `t06_decodifica_grila.py` | 6 |
| `t08_parametri_model.py` | 8 |
| `t09_loss_batch.py` | 9 |
| `t10_verificare_predictii.py` | 10 |
| `t11_un_pas_antrenare.py` | 11 |
| `t12_antrenare_scurta.py` | 12 |
| `t13_potriveste_val.py` | 13 |
| `t15_analiza_history.py` | 15 |
| `t16_vizualizeaza_detectii.py` | 16 |

Task **1**, **2**, **7** și **14** nu au script dedicat (doar text în manual / CLI `train_yolo_simplu.py`).
