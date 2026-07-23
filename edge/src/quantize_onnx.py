"""Quantize the FP32 ONNX model to INT8 (dynamic quantization).

Dynamic quantization is the right starting point here specifically because
the model includes a BiLSTM -- static quantization handles recurrent ops
less cleanly, and dynamic doesn't need a calibration dataset.

Handles the same external-data detail as benchmark_onnx.py: your FP32
model is really two files (ecg_model_fp32.onnx + ecg_model_fp32.onnx.data),
so size comparisons sum both.

Also runs ONNX Runtime's official pre-processing step (quant_pre_process)
before quantizing. This is NOT optional busywork -- models exported via
PyTorch's newer dynamo-based exporter can carry shape metadata around
LSTM weights that the quantizer's basic shape-inference check chokes on
(you'll see "Inferred shape and existing shape differ" if you skip this).
quant_pre_process runs a more robust, symbolic shape-inference pass that
resolves it before quantize_dynamic ever gets involved.

Usage:
    python src/quantize_onnx.py --model-path models/ecg_model_fp32.onnx
"""

import sys
import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_total_size(onnx_path: Path) -> int:
    """External-data models need both files' sizes summed -- the .onnx
    file alone (just the graph) is a tiny fraction of the real size."""
    total = onnx_path.stat().st_size
    data_path = Path(str(onnx_path) + '.data')
    if data_path.exists():
        total += data_path.stat().st_size
    return total


def quantize(model_path: str, output_path: str = None) -> str:
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if output_path is None:
        # ecg_model_fp32.onnx -> ecg_model_int8_dynamic.onnx
        output_name = model_path.name.replace('fp32', 'int8_dynamic')
        output_path = model_path.parent / output_name
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # Pre-process: robust symbolic shape inference before quantizing.
    # Writes an intermediate file, e.g. ecg_model_fp32.preprocessed.onnx.
    preprocessed_path = model_path.parent / (model_path.stem + '.preprocessed.onnx')
    print(f"Pre-processing (symbolic shape inference): {preprocessed_path.name}")

    try:
        quant_pre_process(
            input_model=str(model_path),
            output_model_path=str(preprocessed_path),
        )
    except Exception as e:
        print(f"Symbolic shape inference failed ({e}); retrying with skip_symbolic_shape=True")
        quant_pre_process(
            input_model=str(model_path),
            output_model_path=str(preprocessed_path),
            skip_symbolic_shape=True,
        )

    print(f"Quantizing {preprocessed_path.name}")
    print(f"       -> {output_path}  (dynamic, INT8)")
    quantize_dynamic(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    fp32_size = get_total_size(model_path)
    int8_size = get_total_size(output_path)
    reduction = 100 * (1 - int8_size / fp32_size)

    print(f"\nFP32 size: {fp32_size / 1024 / 1024:.2f} MB")
    print(f"INT8 size: {int8_size / 1024 / 1024:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")
    print(f"\nSaved quantized model to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--model-path", required=True, help="Path to FP32 .onnx model")
    parser.add_argument("--output", default=None, help="Output path for INT8 model")
    args = parser.parse_args()
    quantize(args.model_path, args.output)
