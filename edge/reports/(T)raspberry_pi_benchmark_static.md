# 🍓 Raspberry Pi / Edge Benchmark Report

**Date of Execution:** 2026-07-29 13:45:50  
**Target Device:** `Linux (aarch64)` — `ARM Processor`  
**CPU Cores:** Physical: `4`, Logical: `4`  
**Total System RAM:** `3.71 GB`  
**ONNX Runtime Version:** `1.28.0`  

---

## 📊 Performance Comparison Summary

| Metric | FP32 Model | INT8 Quantized Model | Difference / Gain |
|---|---|---|---|
| **Model Size** | `3.64 MB` | `3.19 MB` | **-12.3%** |
| **Mean Latency** | `92.97 ms` | `86.31 ms` | **1.08x speedup** |
| **P95 Latency** | `93.34 ms` | `86.57 ms` | — |
| **P99 Latency** | `93.57 ms` | `86.79 ms` | — |
| **Throughput** | `10.72 s/sec` | `11.55 s/sec` | — |
| **RAM Footprint** | `71.09 MB` | `75.95 MB` | — |
| **Avg CPU Load** | `99.5%` | `100.0%` | — |

---

## 🔍 Detailed Model Breakdown

### 1. FP32 Model (`edge/models/ecg_model_fp32.onnx`)
- **Input Shape:** `[1, 12, 5000]` (12 leads, 5000 samples @ 500Hz)
- **Output Shape:** `[1, 4]` (4 classes)
- **Min / Max Latency:** `92.54 ms` / `93.64 ms`

### 2. INT8 Dynamic Quantized Model (`edge/models/ecg_model_int8_static.onnx`)
- **Input Shape:** `[1, 12, 5000]`
- **Output Shape:** `[1, 4]`
- **Min / Max Latency:** `85.62 ms` / `86.96 ms`

---

## 🎯 Key Findings & Edge Suitability
- **Real-Time Requirement:** 10 seconds of 12-lead ECG data is processed in **`86.31 ms`** (well below the 10,000 ms real-time window constraint).
- **Storage & Memory Efficiency:** Model size reduced from `3.64 MB` to `3.19 MB`.
