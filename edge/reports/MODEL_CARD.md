# Model Card — Automated CI/CD Release

**Release Version:** `dev-test`  
**Pipeline Run Date:** 2026-10-11 11:53:49 UTC  
**Pipeline Status:** ✅ Quality Gates & Interface Validation Passed  

---

## Model Architecture & Deployment Profile
- **Target Application:** Real-Time 12-Lead ECG Arrhythmia Classification
- **Classes:** 4 Target Rhythms (Sinus Bradycardia, Sinus Rhythm, Atrial Fibrillation, GSVT)
- **Primary Model:** CNN-1D + BiLSTM
- **Execution Target:** Raspberry Pi 4 (Broadcom BCM2711, 4x Cortex-A72 @ 1.8GHz, ARMv8-A 64-bit)
- **Deployment Strategy:** Selective Static INT8 Quantization (Conv1D + Gemm calibrated)
- **Inference Runtime:** ONNX Runtime (CPUExecutionProvider with multi-thread parallelism)
- **Telemetry Protocol:** MQTT over TCP (payload: predicted class, confidence, latency)

---

## Automated Quality Gate & Benchmark Results

| Evaluation Metric | Baseline FP32 | Optimized INT8 Static | Edge Operational Requirement | Quality Gate Status |
|---|---|---|---|---|
| **Model Size** | `3.64 MB (3726.27 KB)` | `3.19 MB (3269.60 KB)` | `< 5.0 MB` (Flash footprint) | ✅ PASS |
| **SIL Latency (Mean)** | `18.01 ms` | `15.57 ms` | `< 50.0 ms` | ✅ PASS |
| **SIL Latency (P95)** | `20.68 ms` | `18.42 ms` | `< 80.0 ms` | ✅ PASS |
| **Throughput** | `54.17 samples/sec` | `33.11 samples/sec` | `> 10.0 samples/sec` | ✅ PASS |
| **Target Pi Latency** | `~93.17 ms` (Taher) / `~26.15 ms` (Iman) | `~86.31 ms` (Taher) / `~36.03 ms` (Iman) | `< 10,000 ms` (Real-Time Window) | ✅ PASS |
| **Interface Contract** | `[1, 12, 5000]` / `[1, 5000, 12]` | Autodetected & Transposed | Strict I/O Match | ✅ PASS |

---

## Verification & Fidelity Assurance (MBD)
- **MIL ➔ SIL Agreement:** 100% decision match between native training framework and ONNX export.
- **SIL ➔ PIL Agreement:** 100% prediction match on target Raspberry Pi hardware.
- **Max Absolute Logit Delta:** `< 0.09` across INT8 quantization stages.

---

## Edge Container Deployment Command
To run this exact model release inside a verified multi-arch container on the Raspberry Pi:

```bash
docker pull ghcr.io/user/ecg-edge:latest

docker run -d \
  --name ecg-inference-engine \
  --restart always \
  --net=host \
  ghcr.io/user/ecg-edge:latest \
  --broker 192.168.1.100 \
  --model-path models/ecg_model_int8_static.onnx
```
