# Fiche ONNX — Modèle CNN-1D + BiLSTM (volet Edge, Taher KHALLAF)

## Identité

| Champ | Valeur |
|---|---|
| Nom du modèle | `ecg_model_fp32.onnx` / `ecg_model_int8_dynamic.onnx` |
| Dataset d'entraînement | Chapman-Shaoxing (WFDB) |
| Opset ONNX | 17 |
| Exporteur | `torch.onnx.export` (legacy TorchScript, `dynamo=False`) |

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

- Méthode : **dynamique** (poids en INT8, activations en float, calculées à la volée)
- Justification : pas de dataset de calibration nécessaire ; le BiLSTM est mieux géré en dynamique qu'en statique
- Pré-traitement ONNX Runtime (`quant_pre_process`) appliqué avant quantification

## Performance (PC, CPU)

*À remplir avec les résultats de `benchmark_onnx.py` et `evaluate_onnx_accuracy.py` :*

| Métrique | FP32 | INT8 |
|---|---|---|
| Taille modèle | 3.64 MB | 0.93 MB |
| Latence moyenne | 18.01 ms | 41.88 ms |
| Latence p95 | 20.68 ms | 45.29 ms |
| Accuracy | 98.93% | 99.01% |
| Macro F1 | 0.9885 | 0.9892 |

## Performance (Raspberry Pi)

*À remplir en semaine 4, après déploiement sur cible.*

| Métrique | Valeur |
|---|---|
| Latence moyenne | ___ ms |
| Consommation | ___ |

## Historique des changements

- Correction du mapping SNOMED `426761007` : `Sinus Irregularity` → `GSVT`, pour aligner les classes avec le modèle d'Iman (voir échange avec Mme Bouayad).
