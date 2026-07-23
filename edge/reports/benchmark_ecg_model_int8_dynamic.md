# ONNX Benchmark Report

**Date:** 2026-07-23 15:10:15  
**ONNX Runtime Version:** 1.27.0  
**Model Path:** `models/ecg_model_int8_dynamic.onnx`  

## Model Info
| Metric | Value |
|---|---|
| Size | 0.95 MB (975.80 KB) |
| Input Shape | `[1, 12, 5000]` |
| Output Shape | `[1, 4]` |

## Performance
| Metric | Value |
|---|---|
| Throughput | 23.58 samples/sec |
| Latency (Mean) | 46.12 ms |
| Latency (Std) | 9.30 ms |
| Latency (Min) | 40.21 ms |
| Latency (Max) | 100.00 ms |
| Latency (p95) | 62.08 ms |
| Latency (p99) | 67.66 ms |
