# Fiche ONNX — Volet Edge (Taher KHALLAF)

Ce document couvre les deux modèles utilisés au cours du stage : le modèle
placeholder entraîné par Taher (PyTorch), et le modèle final entraîné par
Iman sur Chapman-Shaoxing (Keras/TensorFlow).

---

## Modèle 1 — Placeholder (Taher, PyTorch)

### Identité

| Champ | Valeur |
|---|---|
| Modèles disponibles | `ecg_model_fp32.onnx`, `ecg_model_int8_dynamic.onnx`, `ecg_model_int8_static.onnx` |
| Modèle recommandé | **`ecg_model_int8_static.onnx`** |
| Framework d'origine | PyTorch → ONNX (exporteur legacy, `dynamo=False`) |
| Opset ONNX | 17 |

### Architecture

- CNN-1D (canaux 64→128→256, kernels 7/5/3) + BiLSTM (hidden=128, 2 couches)
- Tête : Linear(256→128) → ReLU → Dropout → Linear(128→4)

### Entrée / Sortie

| | Nom | Shape | Type |
|---|---|---|---|
| Entrée | `ecg_input` | `(batch, 12, 5000)` — **channels-first** | float32 |
| Sortie | `classification` | `(batch, 4)` — logits bruts (pas de softmax dans le graphe) | float32 |

### Prétraitement

- Filtre passe-bande 0.5–45 Hz, normalisation z-score par enregistrement.

### Classes (ordre confirmé)

| Index | Classe | Code SNOMED |
|---|---|---|
| 0 | Sinus Bradycardia (SB) | 426177001 |
| 1 | Sinus Rhythm (SR) | 426783006 |
| 2 | Atrial Fibrillation (AFIB) | 164889003 |
| 3 | GSVT | 426761007 |

### Performance (PC / Raspberry Pi 4)

| Métrique | FP32 | INT8 Dynamique | INT8 Statique |
|---|---|---|---|
| Taille | 3.64 MB | 0.93 MB | 3.19 MB |
| Latence PC | 18.01 ms | 41.88 ms | **15.57 ms** |
| Latence Pi 4 | 93.17 ms | 121.06 ms | **86.31 ms** |
| Accuracy | 98.93% | 99.01% | 98.93% |
| Macro F1 | 0.9885 | 0.9892 | 0.9873 |

**Recommandation : INT8 statique** — seule variante améliorant taille ET latence, sur PC comme sur Pi 4.

### Fidélité MBD (MIL / SIL / PIL)

| Comparaison | Accord prédiction | Écart max | Écart moyen |
|---|---|---|---|
| PyTorch (MIL) vs ONNX FP32 (SIL, PC) | 100.00% | 0.000002 | 0.000001 |
| ONNX FP32 (SIL) vs ONNX INT8 statique (SIL) | 100.00% | 0.189241 | 0.039212 |
| ONNX INT8 statique (SIL) vs Raspberry Pi 4 (PIL) | 100.00% | 0.088106 | 0.014684 |

---

## Modèle 2 — Final (Iman, Chapman-Shaoxing, Keras/TensorFlow)

### Identité

| Champ | Valeur |
|---|---|
| Modèles disponibles | `chapman_ecg_model_fp32.onnx`, `chapman_ecg_model_int8_dynamic.onnx`, `chapman_ecg_model_int8_static.onnx` |
| Modèle recommandé | **`chapman_ecg_model_fp32.onnx`** (voir Performance) |
| Framework d'origine | Keras/TensorFlow → ONNX via tf2onnx |
| Opset ONNX | 13 (fonctionnel malgré < 17 requis pour le modèle Taher) |

### Architecture

Conv1D(128, k=7) + BN + ReLU → MaxPool(4) → Conv1D(128, k=5) + BN + ReLU →
MaxPool(4) → BiLSTM(64) → Dropout(0.5) → Dense(4, softmax) — 193 284 paramètres.

### Entrée / Sortie

| | Nom | Shape | Type |
|---|---|---|---|
| Entrée | `ecg_input` | `(batch, 5000, 12)` — **channels-last** (convention Keras) | float32 |
| Sortie | `dense_1` | `(batch, 4)` — **probabilités déjà softmax dans le graphe** | float32 |

⚠️ Gérées automatiquement côté Edge via `onnx_io_utils.py` (détection d'orientation) — aucune action manuelle requise à l'usage.

### Prétraitement obligatoire

- Filtre passe-bande Butterworth ordre 4, **0.5–40 Hz** (différent du modèle 1 : 45 Hz), `filtfilt`.
- Normalisation z-score **par enregistrement**.

### ⚠️ Correction requise : ordre des classes en sortie

L'ordre documenté dans la fiche d'origine d'Iman ([SB, SR, AFIB, GSVT]) ne
correspond **pas** à l'ordre réel des neurones de sortie du modèle exporté.
Confirmé par matrice de confusion (schéma de permutation cohérent à >95%
sur toutes les classes) :

| Index réel du modèle | Classe réelle |
|---|---|
| 0 | AFIB |
| 1 | GSVT |
| 2 | SB |
| 3 | SR |

Cause probable : encodage alphabétique (LabelEncoder ou équivalent) côté
entraînement, non reflété dans la documentation d'origine.

**Tout script consommant la sortie brute de ce modèle doit appliquer la
permutation `[2, 3, 0, 1]`** (déjà intégrée via `--pred-permutation 2,3,0,1`
dans `evaluate_onnx_accuracy.py` et `realtime_inference_publisher.py`).

### Performance (PC / Raspberry Pi 4)

| Métrique | FP32 | INT8 Dynamique | INT8 Statique |
|---|---|---|---|
| Taille | 0.75 MB | 0.21 MB | 0.49 MB |
| Latence PC | **3.75 ms** | 23.76 ms | 6.24 ms |
| Latence Pi 4 | **26.15 ms** | 33.82 ms | 36.03 ms |
| Accuracy (corrigée) | 96.04% | 95.96% | 96.04% |
| Macro F1 | 0.9385 | 0.9378 | 0.9384 |

**Recommandation : FP32** — contrairement au modèle 1, la quantification
dégrade la latence sur cette architecture plus légère (aucun gain,
quelle que soit la méthode), sur PC comme sur Pi 4.

### Fidélité MBD (SIL / PIL — pas de comparaison MIL, modèle Keras)

| Comparaison | Accord prédiction | Écart max | Écart moyen |
|---|---|---|---|
| ONNX FP32 (SIL, PC) vs ONNX INT8 statique (SIL, PC) | 100.00% | 0.057191 | 0.003256 |
| ONNX INT8 statique (SIL, PC) vs Raspberry Pi 4 (PIL) | 100.00% | 0.020382 | 0.000374 |

### Chaîne temps réel MQTT

Validée de bout en bout avec ce modèle (INT8 statique, signaux réels,
filtre 40 Hz, permutation de classes appliquée) : Raspberry Pi 4 → broker
Mosquitto → PC, prédictions correctes en direct.

---

## Historique des corrections (chronologique)

1. Mapping SNOMED `426761007` : `Sinus Irregularity` → `GSVT` (alignement classes avec Iman).
2. Export ONNX (modèle 1) : bascule vers l'exporteur legacy (`dynamo=False`) suite à une erreur de shape sur le BiLSTM.
3. Quantification statique ajoutée suite à la dégradation de latence observée en dynamique.
4. Intégration modèle Iman : détection automatique de l'orientation channels-last (`onnx_io_utils.py`).
5. Correction du filtre passe-bande (45→40 Hz) pour le modèle Iman.
6. Correction de l'ordre des classes en sortie du modèle Iman (permutation `[2,3,0,1]`).

## Reste à faire

- Mesure de la consommation électrique sur Raspberry Pi — **non réalisée, équipement non disponible**.