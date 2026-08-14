# Fiche ONNX — Modèle CNN-1D + BiLSTM (volet Edge, Taher KHALLAF)

## Identité

| Champ | Valeur |
|---|---|
| Modèles disponibles (Taher) | `ecg_model_fp32.onnx`, `ecg_model_int8_dynamic.onnx`, `ecg_model_int8_static.onnx` |
| Modèle disponible (Iman) | `chapman_ecg_model_fp32.onnx`, `chapman_ecg_model_int8_dynamic.onnx`, `chapman_ecg_model_int8_static.onnx` |
| Modèle recommandé pour déploiement Pi | **`ecg_model_int8_static.onnx`** (Taher) / **`chapman_ecg_model_fp32.onnx`** (Iman — déjà très léger) |
| Dataset d'entraînement | Chapman-Shaoxing (WFDB) — les deux modèles |
| Opset ONNX | 17 (Taher) / à confirmer (Iman) |
| Exporteur | `torch.onnx.export` (Taher) / framework Iman à confirmer |

## Architecture

### Modèle Taher (CNN-1D + BiLSTM)
- **Type :** CNN-1D + BiLSTM
- **Canaux CNN :** 64 → 128 → 256 (~1.3M paramètres)
- **Kernels CNN :** 7, 5, 3
- **LSTM :** bidirectionnel, hidden_size=128, num_layers=2
- **Tête de classification :** Linear(256→64) → ReLU → Dropout → Linear(64→4)
- **Input :** channels-first `(batch, 12, 5000)`

### Modèle Iman (architecture légère)
- **Taille FP32 :** 0.75 MB → architecture significativement plus petite
- **Input :** channels-last `(batch, 5000, 12)` — géré automatiquement par `onnx_io_utils.prepare_input()`

## Prétraitement

- Filtre passe-bande **0.5–45 Hz** (Butterworth, ordre 4, `filtfilt`)
- Normalisation **z-score** par dérivation
- Padding/troncature à **5000 échantillons** (10s @ 500 Hz)
- ⚠️ Le même prétraitement doit être appliqué aux deux modèles pour que les résultats soient comparables.

## Entrée / Sortie (contrat d'interface partagé)

| | Nom | Shape Taher | Shape Iman | Type |
|---|---|---|---|---|
| Entrée | `ecg_input` | `(batch, 12, 5000)` | `(batch, 5000, 12)` | float32 |
| Sortie | `classification` | `(batch, 4)` | `(batch, 4)` | float32 (logits) |

- 12 dérivations, ordre : I, II, III, aVR, aVL, aVF, V1–V6
- Axe batch dynamique (`dynamic_axes` activé)
- La différence channels-first/last est gérée automatiquement par `onnx_io_utils.detect_orientation()` et `prepare_input()`

## Classes (4) — partagées

| Classe | Code SNOMED | Index |
|---|---|---|
| Sinus Bradycardia (SB) | 426177001 | 0 |
| Sinus Rhythm (SR) | 426783006 | 1 |
| Atrial Fibrillation (AFIB) | 164889003 | 2 |
| GSVT | 426761007 | 3 |

⚠️ **Point critique :** l'ordre des classes doit être identique entre les deux modèles. L'évaluation de précision d'Iman sur le test set Chapman montre une accuracy ~0.66% (aléatoire) — cela indique un **désalignement du class_to_idx** entre son entraînement et notre test set. À confirmer avec Iman avant toute comparaison d'accuracy.

## Quantification

Deux approches testées sur le modèle Taher :

| | Dynamique | Statique |
|---|---|---|
| Ops quantifiés | MatMul, LSTM | Conv, Gemm, MatMul (**LSTM exclu**) |
| Calibration requise | Non | Oui (200 échantillons réels) |
| Réduction de taille | -74.3% | -12.3% |
| Effet sur la latence | **Dégrade** | **Améliore** |

**Recommandation : quantification statique.** La quantification dynamique réduit la taille mais dégrade la latence car le Cortex-A72 (Pi 4, ARMv8.0) n'a pas d'accélération INT8 matérielle (SDOT/UDOT, disponible à partir d'ARMv8.2). La quantification statique cible les Conv (goulot de calcul) et donne un gain réel de latence avec une perte d'accuracy négligeable.

**Pour le modèle Iman :** quantification recommandée après confirmation du class_to_idx. Son modèle est déjà très petit (0.75 MB) — la statique resterait préférable pour la latence.

## Performance (PC, Windows CPU — Intel i5-8250U)

### Modèle Taher (CNN-1D + BiLSTM)

| Métrique | FP32 | INT8 dynamique | INT8 statique |
|---|---|---|---|
| Taille modèle | 3.64 MB | 0.93 MB (-74.3%) | 3.19 MB (-12.3%) |
| Latence moyenne | 18.01 ms | 41.88 ms (+2.3×) | **15.57 ms (-14%)** |
| Latence p95 | 20.68 ms | 45.29 ms | 18.42 ms |
| Accuracy | 0.9893 | 0.9901 | 0.9893 |
| Macro F1 | 0.9885 | 0.9892 (+0.0007) | 0.9873 (-0.0012) |

### Modèle Iman (architecture légère, channels-last)

| Métrique | FP32 | INT8 dynamique | INT8 statique |
|---|---|---|---|
| Taille modèle | 0.75 MB | 0.21 MB (-72.0%) | 0.49 MB (-34.7%) |
| Latence moyenne | **3.75 ms** | 23.76 ms (+6.3×) | **6.96 ms** |
| Latence p95 | 4.09 ms | 33.49 ms | 8.27 ms |
| Accuracy | ⚠️ *invalide* | ⚠️ *invalide* | ⚠️ *invalide* |
| Macro F1 | ⚠️ *invalide* | ⚠️ *invalide* | ⚠️ *invalide* |

*L'accuracy d'Iman n'est pas mesurable avec notre test set actuel (désalignement class_to_idx présumé).*

## Performance (Raspberry Pi 4, aarch64 — Cortex-A72 @ 1.8GHz, 4 Go RAM)

### Modèle Taher

| Métrique | FP32 | INT8 dynamique | INT8 statique |
|---|---|---|---|
| Taille modèle | 3.64 MB | 0.93 MB | 3.19 MB |
| Latence moyenne | 93.17 ms | 121.06 ms (+1.3×) | **86.31 ms (-7%)** |
| Latence p95 | 93.44 ms | 127.10 ms | 86.57 ms |
| Throughput | 10.70 s/sec | 8.24 s/sec | **11.55 s/sec** |
| RAM | 71.24 MB | 74.16 MB | 75.95 MB |
| Charge CPU | 99.5% | 100.0% | 100.0% |

### Modèle Iman (mesuré sur Pi 4)

| Métrique | FP32 | INT8 dynamique | INT8 statique |
|---|---|---|---|
| Taille modèle | 0.75 MB | 0.21 MB | 0.49 MB |
| Latence moyenne | **26.19 ms** | 33.67 ms | 35.65 ms |
| Latence p95 | 26.35 ms | 33.79 ms | 36.03 ms |
| Throughput | 37.70 s/sec | 29.41 s/sec | 27.79 s/sec |
| RAM | 72.68 MB | 74.78 MB | 75.45 MB |

**Contrainte temps réel :** toutes les variantes traitent 10s d'ECG bien en-deçà de 10 000 ms. La latence max observée est 121 ms (Taher, INT8 dynamique sur Pi 4).

## Chaîne temps réel (MQTT)

- Chaîne validée de bout en bout : Raspberry Pi 4 (inférence ONNX + publication) → broker Mosquitto (PC) → subscriber (PC).
- Modèle utilisé en démo : `ecg_model_int8_static.onnx` (Taher).
- Prédictions publiées : classe prédite, confiance, classe réelle, indicateur correct/incorrect — à partir de signaux ECG réels du jeu de test.
- Reconnexion MQTT automatique avec backoff implémentée.

## Fidélité MBD (PyTorch → ONNX → Pi)

| Étape | Accord prédictions | Max |Δlogit| | Moy |Δlogit| |
|---|---|---|---|
| PyTorch (MIL) → ONNX FP32 (SIL) | **100%** | 0.000002 | 0.000001 |
| ONNX FP32 (SIL) → ONNX INT8 statique (SIL) | **100%** | 0.189241 | 0.039212 |
| ONNX INT8 statique (SIL) → Pi 4 (PIL) | **100%** | 0.088106 | 0.014684 |

100% d'accord de bout en bout sur 30 signaux réels — la chaîne de déploiement préserve le comportement du modèle.

## Historique des changements

- Correction du mapping SNOMED `426761007` : `Sinus Irregularity` → `GSVT`, pour aligner les classes avec le modèle d'Iman.
- Export ONNX : bascule vers l'exporteur legacy (`dynamo=False`) suite à une erreur de shape inference sur le BiLSTM.
- Ajout de la quantification statique (Conv/Gemm/MatMul, LSTM exclu) suite à la dégradation de latence de la quantification dynamique.
- Ajout de `onnx_io_utils.py` : détection automatique channels-first/last pour la compatibilité avec le modèle d'Iman.
- Ajout de `validate_onnx_interface.py` : validation du contrat d'interface en quelques secondes.

## Reste à faire

- Confirmer le class_to_idx d'Iman et re-évaluer la précision de son modèle quantifié.
- Mesure de la consommation électrique sur Raspberry Pi (nécessite un wattmètre USB-C).
- Documentation finale (rapport de stage).