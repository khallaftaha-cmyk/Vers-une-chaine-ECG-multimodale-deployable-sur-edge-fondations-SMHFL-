#!/bin/bash
# ============================================================
# One-click Raspberry Pi Hardware Benchmark & Report Generator
# Volet Edge — Taher KHALLAF
# Now runs all 3 model variants (FP32, INT8 dynamic, INT8 static)
# in one pass, instead of needing two separate manual runs.
# ============================================================

set -e

echo "Starting Raspberry Pi ECG Model Benchmark (3-way: FP32 / INT8 dynamic / INT8 static)..."

# Activate virtualenv if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run python benchmark script
python3 -m edge.edge_deploy.inference_pi \
    --fp32-model edge/models/ecg_model_fp32.onnx \
    --int8-model edge/models/ecg_model_int8_dynamic.onnx \
    --static-model edge/models/ecg_model_int8_static.onnx \
    --output-report edge/reports/raspberry_pi_benchmark.md

echo ""
echo "Benchmark complete!"
echo "Report generated at: edge/reports/raspberry_pi_benchmark.md"