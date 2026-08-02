# Vers une chaîne ECG multimodale déployable sur edge (fondations SMHFL)
## Volet Edge — Taher KHALLAF

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white" alt="ONNX" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-6B47FF?logo=onnx&logoColor=white" alt="ONNX Runtime" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Raspberry%20Pi%204-A22846?logo=raspberrypi&logoColor=white" alt="Raspberry Pi 4" />
  <img src="https://img.shields.io/badge/MQTT-660066?logo=mqtt&logoColor=white" alt="MQTT" />
  <img src="https://img.shields.io/badge/Eclipse%20Mosquitto-3C5280?logo=eclipsemosquitto&logoColor=white" alt="Mosquitto" />
  <img src="https://img.shields.io/badge/WFDB-ECG%20Toolkit-2E8B57" alt="WFDB" />
</p>

### 📌 Vue d'ensemble
Ce projet constitue le **volet Edge** de la chaîne de classification ECG. L'objectif est d'optimiser et de déployer un modèle de classification d'ECG 12 dérivations (CNN-1D + BiLSTM) sur une cible embarquée (Raspberry Pi), puis d'intégrer l'inférence temps réel via MQTT.

- **Dataset :** Chapman-Shaoxing 12-lead ECG Database (format WFDB, 500 Hz, 10s par enregistrement)
- **Modèle de référence :** CNN-1D (3 blocs) + BiLSTM (2 couches, bidirectionnel)
- **Interface inter-volets :** Modèle exporté au format **ONNX** — 3 variantes disponibles (`fp32`, `int8_dynamic`, `int8_static`)
- **Cible matérielle :** Raspberry Pi 4 (aarch64) — inférence + publication MQTT validées de bout en bout

---

### 📂 Structure du dépôt

```
.
├── edge/                                    # Dossier principal du volet Edge
│   ├── configs/
│   │   └── config.yaml                     # Hyperparamètres, chemins, mapping SNOMED-CT, config MQTT
│   ├── models/
│   │   ├── best_model.pth                  # Poids du modèle PyTorch entraîné
│   │   ├── ecg_model_fp32.onnx             # Modèle ONNX FP32 exporté
│   │   ├── ecg_model_int8_dynamic.onnx     # Modèle ONNX quantifié INT8 (dynamique)
│   │   ├── ecg_model_int8_static.onnx      # Modèle ONNX quantifié INT8 (statique) — recommandé
│   │   └── demo_signals.npz                # Signaux ECG réels pour la démo temps réel (Pi)
│   ├── notebooks/
│   │   └── 01_ecg_onboarding.py            # Script d'exploration et onboarding ECG
│   ├── reports/
│   │   ├── accuracy_comparison_dynamique.md          # Précision/F1 — FP32 vs INT8 dynamique
│   │   ├── accuracy_comparison_static.md   # Précision/F1 — FP32 vs INT8 statique
│   │   ├── benchmark_ecg_model_fp32.md     # Benchmark PC — FP32
│   │   ├── benchmark_ecg_model_int8_dynamic.md  # Benchmark PC — INT8 dynamique
│   │   ├── benchmark_ecg_model_int8_static.md   # Benchmark PC — INT8 statique
│   │   ├── raspberry_pi_benchmark_dynamique.md       # Benchmark matériel Pi 4 — FP32 vs INT8 dynamique
│   │   ├── raspberry_pi_benchmark_static.md# Benchmark matériel Pi 4 — FP32 vs INT8 statique
│   │   └── fiche_onnx.md                   # Fiche technique du modèle ONNX (interface avec Iman)
│   ├── scripts/
│   │   └── extract_dataset.py              # Extraction du dataset WFDB
│   ├── edge_deploy/
│   │   ├── inference_pi.py                 # Benchmark matériel dédié Raspberry Pi (latence, RAM, CPU)
│   │   └── run_pi_benchmark.sh             # Script one-click de benchmark sur le Pi
│   ├── pi_setup_guide.md                   # Guide pas-à-pas : configuration du Raspberry Pi (premier démarrage)
│   ├── requirements_pi.txt                 # Dépendances légères pour le Pi (inférence uniquement)
│   └── src/
│       ├── data_loader.py                  # Pipeline WFDB & parsing SNOMED-CT
│       ├── preprocessing.py                # Filtre passe-bande 0.5–45 Hz, z-score, resampling
│       ├── model.py                        # Architecture PyTorch CNN-1D + BiLSTM
│       ├── train.py                        # Script d'entraînement (avec reprise sur checkpoint)
│       ├── export_onnx.py                  # Conversion PyTorch → ONNX
│       ├── quantize_onnx.py                # Quantification dynamique INT8
│       ├── quantize_onnx_static.py         # Quantification statique INT8 (Conv/Gemm, LSTM exclu)
│       ├── evaluate_onnx_accuracy.py       # Validation précision & Macro F1 (ONNX)
│       ├── benchmark_onnx.py               # Benchmark latence & taille (PC)
│       ├── export_demo_signals.py          # Export de signaux ECG réels pour la démo Pi (léger, numpy only)
│       ├── realtime_inference_publisher.py # Inférence + publication MQTT temps réel (Pi)
│       └── mqtt_subscriber_test.py         # Abonné MQTT de test (affiche les prédictions en direct)
├── requirements.txt                        # Dépendances Python (PC — entraînement/export/quantification)
└── README.md                               # Documentation globale
```

---

### 🚀 Instructions d'exécution

#### 1. Configuration de l'environnement (PC)
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

#### 4. Export ONNX & Quantification
```powershell
# Conversion PyTorch -> ONNX (FP32)
python -m edge.src.export_onnx --model-path edge/models/best_model.pth

# Quantification dynamique (INT8)
python -m edge.src.quantize_onnx --model-path edge/models/ecg_model_fp32.onnx

# Quantification statique (INT8, Conv/Gemm uniquement — recommandée)
python -m edge.src.quantize_onnx_static --model-path edge/models/ecg_model_fp32.onnx --num-calibration-samples 200

# Évaluation de la précision (FP32 vs INT8)
python -m edge.src.evaluate_onnx_accuracy --fp32 edge/models/ecg_model_fp32.onnx --int8 edge/models/ecg_model_int8_static.onnx

# Benchmarking latence & taille sur PC
python -m edge.src.benchmark_onnx --model-path edge/models/ecg_model_fp32.onnx
python -m edge.src.benchmark_onnx --model-path edge/models/ecg_model_int8_static.onnx
```

#### 5. Déploiement Raspberry Pi
Voir [`pi_setup_guide.md`](edge/pi_setup_guide.md) pour la configuration complète (flash OS, SSH, environnement Python).
```bash
# Sur le Pi, dans le venv
python3 -m edge.edge_deploy.inference_pi \
    --fp32-model edge/models/ecg_model_fp32.onnx \
    --int8-model edge/models/ecg_model_int8_static.onnx
```

#### 6. Chaîne temps réel MQTT (Pi → PC)
```powershell
# Sur le PC : lancer le broker Mosquitto, puis l'abonné
python edge/src/mqtt_subscriber_test.py
```
```bash
# Sur le PC : exporter des signaux réels de démo
python edge/src/export_demo_signals.py --num-samples 30

# Sur le Pi : publier des prédictions en temps réel sur des signaux réels
python3 edge/src/realtime_inference_publisher.py \
    --model-path edge/models/ecg_model_int8_static.onnx \
    --signals-path edge/models/demo_signals.npz \
    --broker <IP_DU_PC> --cycles 30 --interval 2
```

---

### 📊 Synthèse des résultats actuels

#### PC (Windows, CPU)

| Métrique | FP32 | INT8 Dynamique | INT8 Statique |
|---|---|---|---|
| **Taille du modèle** | 3.64 MB | 0.93 MB (-74.3%) | 3.19 MB (-12.3%) |
| **Accuracy** | 98.93% | 99.01% | 98.93% |
| **Macro F1** | 0.9885 | 0.9892 | 0.9873 |
| **Latence moy.** | 18.01 ms | 41.88 ms (2.3x plus lent) | **15.57 ms (14% plus rapide)** |

#### Raspberry Pi 4 (aarch64)

| Métrique | FP32 | INT8 Dynamique | INT8 Statique |
|---|---|---|---|
| **Taille du modèle** | 3.64 MB | 0.93 MB | 3.19 MB |
| **Latence moy.** | 93.17 ms | 121.06 ms (1.3x plus lent) | **86.31 ms (8% plus rapide)** |
| **Throughput** | 10.70 s/sec | 8.24 s/sec | 11.55 s/sec |

➡️ **Modèle recommandé pour le déploiement : `ecg_model_int8_static.onnx`** — seule variante qui améliore à la fois la taille ET la latence, sur PC comme sur Pi 4, sans perte d'accuracy significative. Détails et justification dans [`fiche_onnx.md`](edge/reports/fiche_onnx.md).

---

### 📅 Planning du stage (6 semaines)

- [x] **Semaine 1 :** Onboarding ECG, environnement, extraction dataset & modèle de remplacement.
- [x] **Semaine 2 :** Export ONNX (FP32) & premier benchmark sur PC.
- [x] **Semaine 3 :** Quantification INT8 dynamique, validation précision & benchmarks de comparaison.
- [x] **Semaine 4 :** Déploiement & benchmark réel sur **Raspberry Pi 4** (FP32, INT8 dynamique, INT8 statique).
  - [ ] **[JALON CLÉ]** Réception/Bascule sur le modèle d'Iman — interface ONNX déjà alignée sur les 4 classes.
- [x] **Semaine 5 :** Pipeline temps réel ECG → Inférence Edge → **MQTT**, validé Pi → PC sur signaux réels.
  - [ ] Mesure de la consommation électrique sur Raspberry Pi.
- [ ] **Semaine 6 :** Consolidation, rapport final, fiche de déploiement (L3) et présentation.