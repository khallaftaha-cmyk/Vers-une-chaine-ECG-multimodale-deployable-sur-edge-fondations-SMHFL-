# FP32 vs INT8 Accuracy Comparison

Class mapping: `{'Sinus Bradycardia': 0, 'Sinus Rhythm': 1, 'Atrial Fibrillation': 2, 'GSVT': 3}`

| Model | Accuracy | Macro F1 |
|---|---|---|
| FP32 | 0.9604 | 0.9385 |
| INT8 | 0.9604 | 0.9384 |

**Macro F1 delta (INT8 - FP32):** -0.0001

## Class order
`['Sinus Bradycardia', 'Sinus Rhythm', 'Atrial Fibrillation', 'GSVT']`

## FP32 confusion matrix (rows=true, cols=predicted)
```
[[572   7   5   0]
 [ 11 260   1   2]
 [  1   1 262   3]
 [  0   0  17  71]]
```

## INT8 confusion matrix (rows=true, cols=predicted)
```
[[574   5   5   0]
 [ 13 258   1   2]
 [  1   1 262   3]
 [  0   0  17  71]]
```

## FP32 classification report
```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       584
           1       0.97      0.95      0.96       274
           2       0.92      0.98      0.95       267
           3       0.93      0.81      0.87        88

    accuracy                           0.96      1213
   macro avg       0.95      0.93      0.94      1213
weighted avg       0.96      0.96      0.96      1213

```

## INT8 classification report
```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       584
           1       0.98      0.94      0.96       274
           2       0.92      0.98      0.95       267
           3       0.93      0.81      0.87        88

    accuracy                           0.96      1213
   macro avg       0.95      0.93      0.94      1213
weighted avg       0.96      0.96      0.96      1213

```
