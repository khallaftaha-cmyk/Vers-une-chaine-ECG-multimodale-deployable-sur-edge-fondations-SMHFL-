# ONNX Benchmark Report

**Date:** 2026-07-27 10:23:53  
**ONNX Runtime Version:** 1.27.0  
**Model Path:** `models/ecg_model_int8_dynamic.onnx`  

## Model Info
| Metric | Value |
|---|---|
| Size | 0.93 MB (957.23 KB) |
| Input Shape | `[1, 12, 5000]` |
| Output Shape | `[1, 4]` |

## Performance
| Metric | Value |
|---|---|
| Throughput | 21.57 samples/sec |
| Latency (Mean) | 41.88 ms |
| Latency (Std) | 5.08 ms |
| Latency (Min) | 40.41 ms |
| Latency (Max) | 86.57 ms |
| Latency (p95) | 45.29 ms |
| Latency (p99) | 58.10 ms |
