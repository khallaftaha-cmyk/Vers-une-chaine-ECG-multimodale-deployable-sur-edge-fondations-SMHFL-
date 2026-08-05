# ONNX Benchmark Report

**Date:** 2026-07-28 21:14:17  
**ONNX Runtime Version:** 1.27.0  
**Model Path:** `models/ecg_model_int8_static.onnx`  

## Model Info
| Metric | Value |
|---|---|
| Size | 3.19 MB (3269.60 KB) |
| Input Shape | `[1, 12, 5000]` |
| Output Shape | `[1, 4]` |

## Performance
| Metric | Value |
|---|---|
| Throughput | 33.11 samples/sec |
| Latency (Mean) | 15.57 ms |
| Latency (Std) | 1.63 ms |
| Latency (Min) | 13.55 ms |
| Latency (Max) | 19.70 ms |
| Latency (p95) | 18.42 ms |
| Latency (p99) | 19.61 ms |
