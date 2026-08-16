# MBD Fidelity Report (MIL / SIL / PIL)

**Date:** 2026-08-15 19:00:29

**Signals used:** 30 real ECG test samples

| Comparison | Prediction agreement | Max |logit diff| | Mean |logit diff| |
|---|---|---|---|
| ONNX FP32 (SIL, PC) vs ONNX INT8 static (SIL, PC) | 100.00% | 0.057191 | 0.003256 |
| ONNX INT8 static (SIL, PC) vs Pi 4 (PIL) | 100.00% | 0.020382 | 0.000374 |

**Interpretation:** prediction agreement close to 100% and small logit differences confirm each stage of the deployment pipeline (training -> ONNX export -> quantization -> target hardware) preserves the model's behavior. Large drops in agreement at any one stage pinpoint exactly where fidelity was lost.
