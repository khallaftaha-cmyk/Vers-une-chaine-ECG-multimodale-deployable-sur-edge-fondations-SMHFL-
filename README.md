# Vers une chaîne ECG multimodale déployable sur edge (fondations SMHFL)
## Volet Edge — Taher KHALLAF

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/TensorFlow%2FKeras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow/Keras" />
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
Volet Edge de la chaîne de classification ECG : optimiser et déployer un modèle de classification ECG 12 dérivations sur Raspberry Pi, avec inférence temps réel via MQTT. Deux modèles ont été intégrés au cours du stage :

| | Modèle placeholder (Taher) | Modèle final (Iman) |
|---|---|---|
| Framework | PyTorch | Keras/TensorFlow → ONNX (tf2onnx) |
| Architecture | CNN-1D + BiLSTM | CNN-1D + BiLSTM |
| Entrée ONNX | channels-first `(batch,12,5000)` | channels-last `(batch,5000,12)` |
| Recommandé pour déploiement | **INT8 statique** | **FP32** |
| Accuracy | 98.93% | 96.04% |

Les deux conventions d'entrée différentes sont gérées automatiquement (voir `src/onnx_io_utils.py`) — aucun script n'a besoin de savoir quel modèle est chargé.

---

### 📂 Structure du dépôt

```
.
├── edge/
│   ├── configs/
│   │   └── config.yaml                     # Hyperparamètres, mapping SNOMED-CT, classes, config MQTT
│   ├── models/
│   │   ├── best_model.pth                  # Poids PyTorch (modèle placeholder)
│   │   ├── ecg_model_*.onnx                # Modèle placeholder : fp32 / int8_dynamic / int8_static
│   │   ├── chapman_ecg_model_*.onnx        # Modèle final (Iman) : fp32 / int8_dynamic / int8_static
│   │   ├── demo_signals.npz                # Signaux réels pour démo MQTT (modèle placeholder, 45 Hz)
│   │   ├── demo_signals_iman.npz           # Signaux réels pour démo MQTT (modèle Iman, 40 Hz)
│   │   └── mbd_reference*.npz              # Références MIL/SIL pour vérification de fidélité
│   ├── reports/
│   │   ├── accuracy_comparison*.md         # Précision/F1 + matrices de confusion
│   │   ├── benchmark_*.md                  # Benchmarks PC (taille, latence)
│   │   ├── raspberry_pi_benchmark*.md      # Benchmarks matériel Pi 4
│   │   ├── mbd_fidelity_report*.md         # Fidélité MIL/SIL/PIL
│   │   └── fiche_onnx.md                   # Fiche technique des deux modèles
│   ├── edge_deploy/
│   │   ├── inference_pi.py                 # Benchmark matériel Pi (2 ou 3 modèles en un passage)
│   │   └── run_pi_benchmark.sh
│   ├── pi_setup_guide.md                   # Guide de configuration Raspberry Pi (premier démarrage)
│   ├── requirements_pi.txt                 # Dépendances légères pour le Pi (inférence uniquement)
│   └── src/
│       ├── data_loader.py, preprocessing.py, model.py, train.py
│       ├── export_onnx.py                  # Conversion PyTorch → ONNX
│       ├── quantize_onnx.py                # Quantification dynamique INT8
│       ├── quantize_onnx_static.py         # Quantification statique INT8 (Conv/Gemm, LSTM exclu)
│       ├── onnx_io_utils.py                # Détection auto d'orientation (channels-first/last)
│       ├── validate_onnx_interface.py      # Validation de contrat d'interface avant intégration
│       ├── evaluate_onnx_accuracy.py       # Précision + matrice de confusion (--highcut, --pred-permutation)
│       ├── benchmark_onnx.py               # Benchmark latence & taille (PC)
│       ├── export_demo_signals.py          # Export de signaux réels pour démo (--highcut configurable)
│       ├── realtime_inference_publisher.py # Inférence + publication MQTT (--pred-permutation, auto-reconnect)
│       ├── mqtt_subscriber_test.py         # Abonné MQTT de test
│       ├── mbd_fidelity_export.py          # Export référence MIL/SIL (--skip-mil pour modèles non-PyTorch)
│       └── mbd_fidelity_check.py           # Comparaison MIL/SIL/PIL
├── requirements.txt
└── README.md
```

---

### 🚀 Instructions d'exécution

#### 1. Environnement (PC)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Export, quantification, évaluation
```powershell
# Modèle placeholder (Taher)
python -m edge.src.export_onnx --model-path edge/models/best_model.pth
python -m edge.src.quantize_onnx_static --model-path edge/models/ecg_model_fp32.onnx
python -m edge.src.evaluate_onnx_accuracy --fp32 edge/models/ecg_model_fp32.onnx --int8 edge/models/ecg_model_int8_static.onnx

# Modèle final (Iman) -- filtre 40 Hz + correction d'ordre des classes obligatoires
python -m edge.src.validate_onnx_interface --model-path edge/models/chapman_ecg_model_fp32.onnx
python -m edge.src.quantize_onnx_static --model-path edge/models/chapman_ecg_model_fp32.onnx --highcut 40.0
python -m edge.src.evaluate_onnx_accuracy --fp32 edge/models/chapman_ecg_model_fp32.onnx --int8 edge/models/chapman_ecg_model_int8_static.onnx --highcut 40.0 --pred-permutation 2,3,0,1
```

#### 3. Déploiement Raspberry Pi
Voir [`pi_setup_guide.md`](edge/pi_setup_guide.md).
```bash
python3 -m edge.edge_deploy.inference_pi \
    --fp32-model edge/models/chapman_ecg_model_fp32.onnx \
    --int8-model edge/models/chapman_ecg_model_int8_dynamic.onnx \
    --static-model edge/models/chapman_ecg_model_int8_static.onnx
```

#### 4. Chaîne temps réel MQTT
```powershell
# PC : broker Mosquitto + abonné
python edge/src/mqtt_subscriber_test.py
```
```bash
# PC : export des signaux (filtre adapté au modèle utilisé)
python edge/src/export_demo_signals.py --highcut 40.0 --output edge/models/demo_signals_iman.npz

# Pi : publication temps réel (modèle Iman -- filtre + permutation)
python3 edge/src/realtime_inference_publisher.py \
    --model-path edge/models/chapman_ecg_model_int8_static.onnx \
    --signals-path edge/models/demo_signals_iman.npz \
    --pred-permutation 2,3,0,1 --broker <IP_DU_PC> --cycles 30 --interval 2
```

#### 5. Vérification de fidélité MBD
```bash
# Modèle Iman (SIL uniquement -- pas d'équivalent PyTorch)
python edge/src/mbd_fidelity_export.py --skip-mil \
    --signals-path edge/models/demo_signals_iman.npz \
    --fp32-onnx edge/models/chapman_ecg_model_fp32.onnx \
    --static-onnx edge/models/chapman_ecg_model_int8_static.onnx \
    --output edge/models/mbd_reference_iman.npz
python edge/src/mbd_fidelity_check.py --reference edge/models/mbd_reference_iman.npz --output-report mbd_fidelity_report_iman.md
```

---

### 📊 Synthèse des résultats finaux

#### Modèle placeholder (Taher) — PC / Raspberry Pi 4

| Métrique | FP32 | INT8 Dynamique | INT8 Statique |
|---|---|---|---|
| Taille | 3.64 MB | 0.93 MB | 3.19 MB |
| Latence PC | 18.01 ms | 41.88 ms | **15.57 ms** |
| Latence Pi 4 | 93.17 ms | 121.06 ms | **86.31 ms** |
| Accuracy | 98.93% | 99.01% | 98.93% |

➡️ **Modèle recommandé : `ecg_model_int8_static.onnx`**

#### Modèle final (Iman) — PC / Raspberry Pi 4

| Métrique | FP32 | INT8 Dynamique | INT8 Statique |
|---|---|---|---|
| Taille | 0.75 MB | 0.21 MB | 0.49 MB |
| Latence PC | **3.75 ms** | 23.76 ms | 6.24 ms |
| Latence Pi 4 | **26.15 ms** | 33.82 ms | 36.03 ms |
| Accuracy | 96.04% | 95.96% | 96.04% |

➡️ **Modèle recommandé : `chapman_ecg_model_fp32.onnx`** — la quantification n'apporte aucun gain de latence sur cette architecture plus légère (contrairement au modèle placeholder).

⚠️ Ce modèle nécessite un filtre passe-bande **0.5–40 Hz** (et non 45 Hz) et une **correction de l'ordre des classes en sortie** (`--pred-permutation 2,3,0,1`) — voir [`fiche_onnx.md`](edge/reports/fiche_onnx.md) pour le détail complet de ces deux problèmes et leur résolution.

---

### 📅 Planning du stage (6 semaines)

- [x] **Semaine 1 :** Onboarding ECG, environnement, modèle placeholder.
- [x] **Semaine 2 :** Export ONNX (FP32) & premier benchmark sur PC.
- [x] **Semaine 3 :** Quantification INT8 dynamique + statique, validation précision.
- [x] **Semaine 4 :** Déploiement & benchmark réel sur Raspberry Pi 4.
- [x] **Semaine 5 :** Chaîne temps réel MQTT, validée sur signaux réels.
- [x] **Semaine 6 :** Intégration du modèle final d'Iman (résolution des écarts de convention d'entrée, prétraitement, et ordre des classes), consolidation, note de déploiement (L3).

**Non réalisé :** mesure de consommation électrique sur Raspberry Pi (équipement non disponible — voir `fiche_onnx.md`).