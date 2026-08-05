# FP32 vs INT8 Accuracy Comparison

Class mapping: `{'Sinus Bradycardia': 0, 'Sinus Rhythm': 1, 'Atrial Fibrillation': 2, 'GSVT': 3}`

| Model | Accuracy | Macro F1 |
|---|---|---|
| FP32 | 0.9893 | 0.9885 |
| INT8 | 0.9893 | 0.9873 |

**Macro F1 delta (INT8 - FP32):** -0.0012

## FP32 classification report
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       584
           1       1.00      0.97      0.99       274
           2       0.99      0.99      0.99       267
           3       0.99      0.99      0.99        88

    accuracy                           0.99      1213
   macro avg       0.99      0.99      0.99      1213
weighted avg       0.99      0.99      0.99      1213

```

## INT8 classification report
```
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       584
           1       1.00      0.97      0.99       274
           2       0.99      0.99      0.99       267
           3       0.99      0.98      0.98        88

    accuracy                           0.99      1213
   macro avg       0.99      0.98      0.99      1213
weighted avg       0.99      0.99      0.99      1213

```
