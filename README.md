# Vers une chaîne ECG multimodale déployable sur edge (fondations SMHFL)
## Volet Edge — Taher KHALLAF

### 📌 Vue d'ensemble
Ce projet constitue le **volet Edge** de la chaîne de classification ECG. L'objectif est d'optimiser et de déployer un modèle de classification d'ECG 12 dérivations (CNN-1D + BiLSTM) sur une cible embarquée (Raspberry Pi), puis d'intégrer l'inférence temps réel via MQTT.

- **Dataset :** Chapman-Shaoxing 12-lead ECG Database (format WFDB, 500 Hz, 10s par enregistrement)
- **Modèle de référence :** CNN-1D (3 blocs) + BiLSTM (2 couches, bidirectionnel)
- **Interface inter-volets :** Modèle exporté au format **ONNX** (`edge/models/ecg_model_fp32.onnx` et `edge/models/ecg_model_int8_dynamic.onnx`)

---

### 📂 Structure du dépôt

```
.
├── edge/                             # Dossier principal du volet Edge
│   ├── configs/
│   │   └── config.yaml              # Hyperparamètres, chemins & mapping SNOMED-CT
│   ├── models/
│   │   ├── best_model.pth           # Poids du modèle PyTorch entraîné
│   │   ├── ecg_model_fp32.onnx      # Modèle ONNX FP32 exporté
│   │   └── ecg_model_int8_dynamic.onnx # Modèle ONNX quantifié INT8
│   ├── notebooks/
│   │   └── 01_ecg_onboarding.py    # Script d'exploration et onboarding ECG
│   ├── reports/
│   │   ├── accuracy_comparison.md   # Évaluation précision/F1 (FP32 vs INT8)
│   │   ├── benchmark_ecg_model_fp32.md # Benchmarks latence & taille (FP32)
│   │   ├── benchmark_ecg_model_int8_dynamic.md # Benchmarks latence & taille (INT8)
│   │   └── fiche_onnx.md            # Fiche technique du modèle ONNX pour Iman
│   ├── scripts/
│   │   └── extract_dataset.py       # Script d'extraction du dataset WFDB
│   └── src/
│       ├── data_loader.py           # Pipeline de chargement WFDB & parsing SNOMED-CT
│       ├── preprocessing.py         # Filtre passe-bande 0.5-45Hz, z-score, resampling
│       ├── model.py                 # Architecture PyTorch CNN-1D + BiLSTM
│       ├── train.py                 # Script d'entraînement
│       ├── export_onnx.py           # Conversion PyTorch -> ONNX
│       ├── quantize_onnx.py         # Quantification dynamique INT8
│       ├── evaluate_onnx_accuracy.py# Validation précision & Macro F1 ONNX
│       └── benchmark_onnx.py        # Benchmark de latence et de taille
├── requirements.txt                 # Dépendances Python
└── README.md                        # Documentation globale
```

---

### 🚀 Instructions d'exécution

#### 1. Configuration de l'environnement
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Extraction des données Chapman-Shaoxing
```powershell
python edge/scripts/extract_dataset.py
```

#### 3. Onboarding & Exploration ECG
```powershell
python edge/notebooks/01_ecg_onboarding.py
```

#### 4. Export ONNX & Quantification INT8
```powershell
# Conversion PyTorch -> ONNX (FP32)
python -m edge.src.export_onnx --model-path edge/models/best_model.pth

# Quantification dynamique (INT8)
python -m edge.src.quantize_onnx --input-onnx edge/models/ecg_model_fp32.onnx

# Évaluation de la précision (FP32 vs INT8)
python -m edge.src.evaluate_onnx_accuracy

# Benchmarking latence & taille sur PC
python -m edge.src.benchmark_onnx --model-path edge/models/ecg_model_fp32.onnx
python -m edge.src.benchmark_onnx --model-path edge/models/ecg_model_int8_dynamic.onnx
```

---

### 📊 Synthèse des résultats actuels (PC)

| Métrique | FP32 | INT8 Dynamique | Gain / Δ |
|---|---|---|---|
| **Taille du modèle** | 3.64 MB | **0.93 MB** | **-74.4%** |
| **Accuracy** | 98.93% | **99.01%** | +0.08% |
| **Macro F1** | 0.9885 | **0.9892** | +0.0007 |
| **Latence moy. (PC CPU)** | 18.01 ms | 41.88 ms | *(Overhead SIMD CPU)* |

---

### 📅 Planning du stage (6 semaines)

- [x] **Semaine 1 :** Onboarding ECG, environnement, extraction dataset & modèle de remplacement.
- [x] **Semaine 2 :** Export ONNX (FP32) & premier benchmark sur PC.
- [x] **Semaine 3 :** Quantification INT8 dynamique, validation précision & benchmarks de comparaison.
- [ ] **Semaine 4 :** **[JALON CLÉ]** Réception/Bascule sur le modèle d'Iman, déploiement & benchmark réel sur **Raspberry Pi**.
- [ ] **Semaine 5 :** Implémentation du pipeline temps réel ECG → Inférence Edge → **MQTT**.
- [ ] **Semaine 6 :** Consolidation, rapport final, fiche de déploiement et présentation.
