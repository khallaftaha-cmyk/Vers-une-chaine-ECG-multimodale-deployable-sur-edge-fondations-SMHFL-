#!/bin/bash
# ============================================================
# One-click Raspberry Pi Hardware Benchmark & Report Generator
# Volet Edge — Taher KHALLAF
# ============================================================

set -e

echo "🍓 Starting Raspberry Pi ECG Model Benchmark..."

# Activate virtualenv if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run python benchmark script
python3 -m edge.edge_deploy.inference_pi \
    --fp32-model edge/models/ecg_model_fp32.onnx \
    --int8-model edge/models/ecg_model_int8_static.onnx \
    --output-report edge/reports/raspberry_pi_benchmark_static.md

echo ""
echo "✅ Benchmark complete!"
echo "📄 Report generated at: edge/reports/raspberry_pi_benchmark_static.md"
