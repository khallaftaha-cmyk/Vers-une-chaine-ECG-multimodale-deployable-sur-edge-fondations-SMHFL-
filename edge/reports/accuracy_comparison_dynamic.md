# FP32 vs INT8 Accuracy Comparison

Class mapping: `{'Sinus Bradycardia': 0, 'Sinus Rhythm': 1, 'Atrial Fibrillation': 2, 'GSVT': 3}`

| Model | Accuracy | Macro F1 |
|---|---|---|
| FP32 | 0.0066 | 0.0063 |
| INT8 | 0.0066 | 0.0063 |

**Macro F1 delta (INT8 - FP32):** +0.0000

## FP32 classification report
```
              precision    recall  f1-score   support

           0       0.02      0.01      0.01       584
           1       0.03      0.01      0.01       274
           2       0.00      0.00      0.00       267
           3       0.00      0.00      0.00        88

    accuracy                           0.01      1213
   macro avg       0.01      0.00      0.01      1213
weighted avg       0.01      0.01      0.01      1213

```

## INT8 classification report
```
              precision    recall  f1-score   support

           0       0.02      0.01      0.01       584
           1       0.03      0.01      0.01       274
           2       0.00      0.00      0.00       267
           3       0.00      0.00      0.00        88

    accuracy                           0.01      1213
   macro avg       0.01      0.00      0.01      1213
weighted avg       0.01      0.01      0.01      1213

```
