# Raspberry Pi / Edge Benchmark Report

**Date of Execution:** 2026-08-07 21:08:07
**Target Device:** `Linux (aarch64)` -- `ARM Processor`
**CPU Cores:** Physical: `4`, Logical: `4`
**Total System RAM:** `3.71 GB`
**ONNX Runtime Version:** `1.28.0`

---

## Performance Comparison Summary

| Metric | FP32 Model | INT8 Dynamic Model | INT8 Static Model |
|---|---|---|---|
| **Model Size** | 0.75 MB | 0.21 MB | 0.49 MB |
| **Mean Latency** | 26.19 ms | 33.67 ms | 35.65 ms |
| **P95 Latency** | 26.35 ms | 33.79 ms | 36.03 ms |
| **P99 Latency** | 26.42 ms | 34.08 ms | 36.18 ms |
| **Throughput** | 37.70 s/sec | 29.41 s/sec | 27.79 s/sec |
| **RAM Footprint** | 72.68 MB | 74.78 MB | 75.45 MB |
| **Avg CPU Load** | 99.3% | 100.0% | 100.0% |

---

## Detailed Model Breakdown

### FP32 Model (`edge/models/chapman_ecg_model_fp32.onnx`)
- **Input Shape:** `[1, 5000, 12]`
- **Output Shape:** `[1, 4]`
- **Min / Max Latency:** `25.89 ms` / `26.47 ms`
### INT8 Dynamic Model (`edge/models/chapman_ecg_model_int8_dynamic.onnx`)
- **Input Shape:** `[1, 5000, 12]`
- **Output Shape:** `[1, 4]`
- **Min / Max Latency:** `33.54 ms` / `34.18 ms`
### INT8 Static Model (`edge/models/chapman_ecg_model_int8_static.onnx`)
- **Input Shape:** `[1, 5000, 12]`
- **Output Shape:** `[1, 4]`
- **Min / Max Latency:** `35.02 ms` / `36.25 ms`

---

## Key Findings & Edge Suitability
- **Real-Time Requirement:** all variants processed 10s of 12-lead ECG data well below the 10,000 ms real-time window constraint.
- **Fastest variant:** FP32 (26.19 ms mean latency)
- **Smallest variant:** INT8 Dynamic (0.21 MB)
- **INT8 Dynamic vs FP32 size:** +72.5%
- **INT8 Static vs FP32 size:** +34.3%
- **INT8 Dynamic vs FP32 latency:** 0.78x slowdown
- **INT8 Static vs FP32 latency:** 0.73x slowdown
