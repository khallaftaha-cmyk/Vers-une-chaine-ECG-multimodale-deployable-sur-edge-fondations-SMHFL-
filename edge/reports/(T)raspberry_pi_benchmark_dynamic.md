# 🍓 Raspberry Pi / Edge Benchmark Report

**Date of Execution:** 2026-07-28 15:49:32  
**Target Device:** `Linux (aarch64)` — `ARM Processor`  
**CPU Cores:** Physical: `4`, Logical: `4`  
**Total System RAM:** `3.71 GB`  
**ONNX Runtime Version:** `1.28.0`  

---

## 📊 Performance Comparison Summary

| Metric | FP32 Model | INT8 Quantized Model | Difference / Gain |
|---|---|---|---|
| **Model Size** | `3.64 MB` | `0.93 MB` | **-74.3%** |
| **Mean Latency** | `93.17 ms` | `121.06 ms` | **0.77x speedup** |
| **P95 Latency** | `93.44 ms` | `127.10 ms` | — |
| **P99 Latency** | `93.51 ms` | `140.62 ms` | — |
| **Throughput** | `10.70 s/sec` | `8.24 s/sec` | — |
| **RAM Footprint** | `71.24 MB` | `74.16 MB` | — |
| **Avg CPU Load** | `99.9%` | `100.0%` | — |

---

## 🔍 Detailed Model Breakdown

### 1. FP32 Model (`edge/models/ecg_model_fp32.onnx`)
- **Input Shape:** `[1, 12, 5000]` (12 leads, 5000 samples @ 500Hz)
- **Output Shape:** `[1, 4]` (4 classes)
- **Min / Max Latency:** `92.77 ms` / `93.53 ms`

### 2. INT8 Dynamic Quantized Model (`edge/models/ecg_model_int8_dynamic.onnx`)
- **Input Shape:** `[1, 12, 5000]`
- **Output Shape:** `[1, 4]`
- **Min / Max Latency:** `117.02 ms` / `147.38 ms`

---

## 🎯 Key Findings & Edge Suitability
- **Real-Time Requirement:** 10 seconds of 12-lead ECG data is processed in **`121.06 ms`** (well below the 10,000 ms real-time window constraint).
- **Storage & Memory Efficiency:** Model size reduced from `3.64 MB` to `0.93 MB`.
