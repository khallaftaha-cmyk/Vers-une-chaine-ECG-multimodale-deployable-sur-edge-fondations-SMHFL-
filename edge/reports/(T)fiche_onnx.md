# Fiche ONNX — Modèle CNN-1D + BiLSTM (volet Edge, Taher KHALLAF)

## Identité

| Champ | Valeur |
|---|---|
| Modèles disponibles | `ecg_model_fp32.onnx`, `ecg_model_int8_dynamic.onnx`, `ecg_model_int8_static.onnx` |
| Modèle recommandé pour déploiement | **`ecg_model_int8_static.onnx`** (voir §Quantification) |
| Dataset d'entraînement | Chapman-Shaoxing (WFDB) — modèle placeholder en attendant le modèle d'Iman |
| Opset ONNX | 17 |
| Exporteur | `torch.onnx.export` (legacy TorchScript, `dynamo=False` — voir Historique) |

## Architecture

- **Type :** CNN-1D + BiLSTM
- **Canaux CNN :** 64 → 128 → 256 (~1.3M paramètres)
- **Kernels CNN :** 7, 5, 3
- **LSTM :** bidirectionnel, hidden_size=128, num_layers=2
- **Tête de classification :** Linear(256→128) → ReLU → Dropout → Linear(128→num_classes)

## Prétraitement

- Filtre passe-bande **0.5–45 Hz** (Butterworth, ordre 4, `filtfilt`)
- Normalisation **z-score** par dérivation
- Padding/troncature à **5000 échantillons** (10s @ 500 Hz)
- ⚠️ Iman doit appliquer le même prétraitement pour que les modèles soient comparables.

## Entrée / Sortie

| | Nom | Shape | Type |
|---|---|---|---|
| Entrée | `ecg_input` | `(batch, 12, 5000)` | float32 |
| Sortie | `classification` | `(batch, num_classes)` | float32 (logits) |

- 12 dérivations, ordre : I, II, III, aVR, aVL, aVF, V1–V6
- Axe batch dynamique (`dynamic_axes` activé)

## Classes (4)

Alignées avec le modèle d'Iman via le SNOMED-CT :

| Classe | Code SNOMED |
|---|---|
| Sinus Bradycardia (SB) | 426177001 |
| Sinus Rhythm (SR) | 426783006 |
| Atrial Fibrillation (AFIB) | 164889003 |
| GSVT | 426761007 |

## Quantification

Deux approches testées :

| | Dynamique | Statique |
|---|---|---|
| Ops quantifiés | MatMul, LSTM | Conv, Gemm, MatMul (**LSTM exclu**, support limité en statique) |
| Calibration requise | Non | Oui (200 échantillons réels du train set) |
| Réduction de taille | -74.3% | -12.3% |
| Effet sur la latence | **Dégrade** (voir §Performance) | **Améliore** |

**Recommandation : quantification statique.** La quantification dynamique réduit fortement la taille mais dégrade la latence sur PC ET sur Raspberry Pi 4, car (1) elle ne touche pas les couches Conv1d qui concentrent l'essentiel du calcul, et (2) le Cortex-A72 du Pi 4 (ARMv8.0) n'a pas d'accélération matérielle INT8 (SDOT/UDOT, disponible seulement à partir d'ARMv8.2/Cortex-A76), donc le surcoût de quantification/déquantification par appel n'est jamais compensé. La quantification statique cible les Conv (le vrai goulot de calcul) et exclut le BiLSTM, ce qui donne un gain réel de latence sur les deux plateformes, avec une perte d'accuracy négligeable.

Pré-traitement ONNX Runtime (`quant_pre_process`) appliqué avant les deux méthodes.

## Performance (PC, Windows CPU)

| Métrique | FP32 | INT8 dynamique | INT8 statique |
|---|---|---|---|
| Taille modèle | 3.64 MB | 0.93 MB (-74.3%) | 3.19 MB (-12.3%) |
| Latence moyenne | 18.01 ms | 41.88 ms (2.3x plus lent) | **15.57 ms (14% plus rapide)** |
| Latence p95 | 20.68 ms | 45.29 ms | 18.42 ms |
| Accuracy | 0.9893 | 0.9901 | 0.9893 |
| Macro F1 | 0.9885 | 0.9892 (+0.0007) | 0.9873 (-0.0012) |

## Performance (Raspberry Pi 4, aarch64)

| Métrique | FP32 | INT8 dynamique | INT8 statique |
|---|---|---|---|
| Taille modèle | 3.64 MB | 0.93 MB | 3.19 MB |
| Latence moyenne | 93.17 ms | 121.06 ms (1.3x plus lent) | **86.31 ms (1.08x plus rapide)** |
| Latence p95 | 93.44 ms | 127.10 ms | 86.57 ms |
| Throughput | 10.70 s/sec | 8.24 s/sec | 11.55 s/sec |
| RAM | 71.24 MB | 74.16 MB | 75.95 MB |
| Charge CPU moyenne | 99.5% | 100.0% | 100.0% |
| Consommation électrique | *non mesurée — à faire* | | |

**Contrainte temps réel :** 10s de signal ECG doivent être traités bien en-deçà de 10 000 ms. Les trois versions la respectent largement (latence max observée : 121 ms sur Pi 4).

## Chaîne temps réel (MQTT)

- Chaîne validée de bout en bout : Raspberry Pi 4 (inférence + publication) → broker Mosquitto (PC) → subscriber (PC), sur le réseau local.
- Modèle utilisé : `ecg_model_int8_static.onnx`.
- Chaque prédiction publiée inclut la classe prédite, la confiance, et (en mode démo) la classe réelle + indicateur correct/incorrect, à partir de signaux ECG réels du jeu de test (export via `export_demo_signals.py`, sans dépendance lourde côté Pi).

## Historique des changements

- Correction du mapping SNOMED `426761007` : `Sinus Irregularity` → `GSVT`, pour aligner les classes avec le modèle d'Iman (voir échange avec Mme Bouayad).
- Export ONNX : bascule vers l'exporteur legacy (`dynamo=False`) suite à une erreur de shape inference sur le BiLSTM (`Inferred shape and existing shape differ in dimension 0: (256) vs (128)`) avec l'exporteur dynamo par défaut.
- Ajout de la quantification statique (Conv/Gemm/MatMul, LSTM exclu) suite à la dégradation de latence observée avec la quantification dynamique sur PC et Raspberry Pi 4.

## Reste à faire

- Mesure de la consommation électrique sur Raspberry Pi.
- Bascule sur le modèle entraîné d'Iman dès disponibilité (interface déjà alignée sur les 4 classes).
- Documentation de fidélité MBD (PyTorch → ONNX FP32 → ONNX INT8 statique → Pi réel).