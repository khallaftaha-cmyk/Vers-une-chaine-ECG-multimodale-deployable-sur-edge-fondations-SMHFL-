# MBD Fidelity Report (MIL / SIL / PIL)

**Date:** 2026-08-06 11:07:38

**Signals used:** 30 real ECG test samples

| Comparison | Prediction agreement | Max |logit diff| | Mean |logit diff| |
|---|---|---|---|
| PyTorch (MIL) vs ONNX FP32 (SIL, PC) | 100.00% | 0.000002 | 0.000001 |
| ONNX FP32 (SIL, PC) vs ONNX INT8 static (SIL, PC) | 100.00% | 0.189241 | 0.039212 |

**Interpretation:** prediction agreement close to 100% and small logit differences confirm each stage of the deployment pipeline (training -> ONNX export -> quantization -> target hardware) preserves the model's behavior. Large drops in agreement at any one stage pinpoint exactly where fidelity was lost.
