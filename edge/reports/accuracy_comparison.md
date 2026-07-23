# FP32 vs INT8 Accuracy Comparison

Class mapping: `{'Sinus Bradycardia': 0, 'Sinus Rhythm': 1, 'Atrial Fibrillation': 2, 'Sinus Tachycardia': 3}`

| Model | Accuracy | Macro F1 |
|---|---|---|
| FP32 | 0.9772 | 0.9740 |
| INT8 | 0.9757 | 0.9720 |

**Macro F1 delta (INT8 - FP32):** -0.0020

## FP32 classification report
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       584
           1       0.96      0.96      0.96       274
           2       0.99      0.95      0.97       267
           3       0.98      0.97      0.98       235

    accuracy                           0.98      1360
   macro avg       0.98      0.97      0.97      1360
weighted avg       0.98      0.98      0.98      1360

```

## INT8 classification report
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       584
           1       0.95      0.96      0.96       274
           2       0.99      0.95      0.97       267
           3       0.98      0.96      0.97       235

    accuracy                           0.98      1360
   macro avg       0.98      0.97      0.97      1360
weighted avg       0.98      0.98      0.98      1360

```
