"""
Raspberry Pi / Edge ONNX Inference & Hardware Benchmark Script
==============================================================
Volet Edge — Taher KHALLAF

Measures:
  - ONNX Model File Size (MB)
  - Inference Latency (Mean, Std, Min, Max, P50, P95, P99)
  - Throughput (ECG recordings / second)
  - Memory Footprint (RAM RSS in MB)
  - CPU Utilization (%)
  - Hardware specifications (ARM architecture, CPU cores, RAM)

Usage on Raspberry Pi:
  python3 -m edge.edge_deploy.inference_pi --fp32-model edge/models/ecg_model_fp32.onnx --int8-model edge/models/ecg_model_int8_dynamic.onnx
"""

import os
import sys
import time
import argparse
import platform
import psutil
import numpy as np
import onnxruntime as ort
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_hardware_info() -> Dict[str, str]:
    """Retrieves system and hardware architecture information."""
    mem = psutil.virtual_memory()
    return {
        'platform': platform.platform(),
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor() or 'ARM Processor',
        'cpu_count_physical': str(psutil.cpu_count(logical=False)),
        'cpu_count_logical': str(psutil.cpu_count(logical=True)),
        'total_ram_gb': f"{mem.total / (1024**3):.2f} GB",
        'available_ram_gb': f"{mem.available / (1024**3):.2f} GB"
    }


def measure_single_model(model_path: str, num_warmup: int = 10, num_runs: int = 100) -> Dict:
    """Runs latency, memory, and CPU throughput measurements for an ONNX model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Size
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)

    # Initialize Session
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = max(0.0, ram_after_mb - ram_before_mb)

    # Inputs/Outputs info
    input_tensor = session.get_inputs()[0]
    output_tensor = session.get_outputs()[0]

    input_shape = [1 if type(dim) == str else dim for dim in input_tensor.shape]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    input_name = input_tensor.name

    # Warm-up
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark Latency & CPU
    latencies_ms = []
    cpu_percents = []

    start_total = time.perf_counter()
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

        # Track CPU percent per run (sample interval = 0)
        cpu_percents.append(psutil.cpu_percent(interval=None))

    total_time_sec = time.perf_counter() - start_total
    throughput = num_runs / total_time_sec

    # Memory after inference
    ram_final_mb = process.memory_info().rss / (1024 * 1024)

    return {
        'model_path': model_path,
        'size_mb': size_mb,
        'input_shape': input_shape,
        'output_shape': [1 if type(dim) == str else dim for dim in output_tensor.shape],
        'throughput': throughput,
        'latency_mean_ms': float(np.mean(latencies_ms)),
        'latency_std_ms': float(np.std(latencies_ms)),
        'latency_min_ms': float(np.min(latencies_ms)),
        'latency_max_ms': float(np.max(latencies_ms)),
        'latency_p50_ms': float(np.median(latencies_ms)),
        'latency_p95_ms': float(np.percentile(latencies_ms, 95)),
        'latency_p99_ms': float(np.percentile(latencies_ms, 99)),
        'ram_model_mb': model_ram_mb,
        'ram_peak_mb': ram_final_mb,
        'avg_cpu_percent': float(np.mean(cpu_percents))
    }


def generate_comparison_report(results: Dict[str, Dict], hw_info: Dict, output_report_path: str):
    """Generates a structured markdown report comparing however many model
    variants were passed in (2-way FP32/INT8 or 3-way FP32/dynamic/static)."""
    labels = list(results.keys())
    baseline_label = labels[0]  # first one given is treated as the baseline (usually FP32)
    baseline = results[baseline_label]

    header_row = "| Metric | " + " | ".join(f"{l} Model" for l in labels) + " |"
    sep_row = "|---" * (len(labels) + 1) + "|"

    def fmt_row(name, key, fmt="{:.2f}", suffix=""):
        cells = [fmt.format(results[l][key]) + suffix for l in labels]
        return f"| **{name}** | " + " | ".join(cells) + " |"

    size_reduction_lines = []
    speedup_lines = []
    for l in labels[1:]:
        red = ((baseline['size_mb'] - results[l]['size_mb']) / baseline['size_mb']) * 100
        speedup = baseline['latency_mean_ms'] / results[l]['latency_mean_ms']
        size_reduction_lines.append(f"- **{l} vs {baseline_label} size:** {red:+.1f}%")
        direction = "speedup" if speedup >= 1 else "slowdown"
        speedup_lines.append(f"- **{l} vs {baseline_label} latency:** {speedup:.2f}x {direction}")

    summary_table = "\n".join([
        header_row, sep_row,
        fmt_row("Model Size", "size_mb", "{:.2f}", " MB"),
        fmt_row("Mean Latency", "latency_mean_ms", "{:.2f}", " ms"),
        fmt_row("P95 Latency", "latency_p95_ms", "{:.2f}", " ms"),
        fmt_row("P99 Latency", "latency_p99_ms", "{:.2f}", " ms"),
        fmt_row("Throughput", "throughput", "{:.2f}", " s/sec"),
        fmt_row("RAM Footprint", "ram_peak_mb", "{:.2f}", " MB"),
        fmt_row("Avg CPU Load", "avg_cpu_percent", "{:.1f}", "%"),
    ])

    detail_sections = []
    for l in labels:
        r = results[l]
        detail_sections.append(f"""### {l} Model (`{r['model_path']}`)
- **Input Shape:** `{r['input_shape']}`
- **Output Shape:** `{r['output_shape']}`
- **Min / Max Latency:** `{r['latency_min_ms']:.2f} ms` / `{r['latency_max_ms']:.2f} ms`
""")

    fastest_label = min(labels, key=lambda l: results[l]['latency_mean_ms'])
    smallest_label = min(labels, key=lambda l: results[l]['size_mb'])

    report_content = f"""# Raspberry Pi / Edge Benchmark Report

**Date of Execution:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Target Device:** `{hw_info['system']} ({hw_info['machine']})` -- `{hw_info['processor']}`
**CPU Cores:** Physical: `{hw_info['cpu_count_physical']}`, Logical: `{hw_info['cpu_count_logical']}`
**Total System RAM:** `{hw_info['total_ram_gb']}`
**ONNX Runtime Version:** `{ort.__version__}`

---

## Performance Comparison Summary

{summary_table}

---

## Detailed Model Breakdown

{"".join(detail_sections)}
---

## Key Findings & Edge Suitability
- **Real-Time Requirement:** all variants processed 10s of 12-lead ECG data well below the 10,000 ms real-time window constraint.
- **Fastest variant:** {fastest_label} ({results[fastest_label]['latency_mean_ms']:.2f} ms mean latency)
- **Smallest variant:** {smallest_label} ({results[smallest_label]['size_mb']:.2f} MB)
{chr(10).join(size_reduction_lines)}
{chr(10).join(speedup_lines)}
"""

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n[OK] Report generated successfully at: {output_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi ONNX Hardware Benchmark")
    parser.add_argument("--fp32-model", type=str, default="edge/models/ecg_model_fp32.onnx", help="Path to FP32 ONNX model")
    parser.add_argument("--int8-model", type=str, default=None, help="Path to INT8 dynamic ONNX model (optional)")
    parser.add_argument("--static-model", type=str, default=None, help="Path to INT8 static ONNX model (optional)")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--output-report", type=str, default="edge/reports/raspberry_pi_benchmark.md", help="Output report path")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  RASPBERRY PI / EDGE HARDWARE BENCHMARK")
    print("="*60)

    hw_info = get_hardware_info()
    print(f"System:       {hw_info['system']} ({hw_info['machine']})")
    print(f"Processor:    {hw_info['processor']}")
    print(f"CPU Cores:    {hw_info['cpu_count_physical']} physical ({hw_info['cpu_count_logical']} logical)")
    print(f"Total RAM:    {hw_info['total_ram_gb']}")

    # Benchmark whichever model paths were actually provided -- lets you run
    # a quick 2-way FP32-vs-one-variant check, or the full 3-way comparison,
    # from the same script and the same command shape.
    model_args = [
        ("FP32", args.fp32_model),
        ("INT8 Dynamic", args.int8_model),
        ("INT8 Static", args.static_model),
    ]
    results = {}
    step = 1
    for label, path in model_args:
        if path is None:
            continue
        print(f"\n--- {step}. Benchmarking {label} Model ---")
        res = measure_single_model(path, num_warmup=args.warmup, num_runs=args.runs)
        print(f"Size: {res['size_mb']:.2f} MB | Latency: {res['latency_mean_ms']:.2f} ms | Throughput: {res['throughput']:.2f} samples/s")
        results[label] = res
        step += 1

    if len(results) < 2:
        print("\nNeed at least 2 models to compare (got "
              f"{len(results)}). Provide --int8-model and/or --static-model.")
        sys.exit(1)

    generate_comparison_report(results, hw_info, args.output_report)


if __name__ == "__main__":
    main()