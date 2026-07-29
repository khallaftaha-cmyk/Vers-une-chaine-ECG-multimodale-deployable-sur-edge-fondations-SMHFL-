"""Static quantization of the FP32 ONNX model -- targets Conv1d layers
(where the bulk of this model's compute lives) for a real latency win,
unlike dynamic quantization which only touches MatMul/LSTM and left the
CNN stack untouched.

Static quantization needs real calibration data to measure activation
ranges ahead of time (that's the trade-off vs dynamic: no per-call
quantize/dequantize overhead at inference, but you have to calibrate
once, up front, with representative signals).

The BiLSTM is deliberately EXCLUDED from quantization here
(op_types_to_quantize=['Conv', 'Gemm', 'MatMul'] naturally skips LSTM
nodes) -- static quantization support for recurrent ops is poor, and
your dynamic-quantization script already documented this same caveat.
So: Conv layers get quantized (the actual latency driver), LSTM stays
FP32 (the accuracy-sensitive, poorly-supported part).

Usage:
    python src/quantize_onnx_static.py --model-path models/ecg_model_fp32.onnx --num-calibration-samples 200
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
    CalibrationDataReader,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config, create_dataloaders


class ECGCalibrationDataReader(CalibrationDataReader):
    """Feeds real training-set ECG signals to the quantizer for
    calibration. Uses the TRAIN split (not val/test) so we don't leak
    any test-set information into the quantization parameters --
    calibration is model prep, not evaluation."""

    def __init__(self, train_loader, input_name: str, num_samples: int):
        self.input_name = input_name
        self.num_samples = num_samples
        self._iterator = self._build_iterator(train_loader)

    def _build_iterator(self, train_loader):
        count = 0
        for signals, _labels in train_loader:
            x = signals.numpy().astype(np.float32)
            for i in range(x.shape[0]):
                if count >= self.num_samples:
                    return
                yield {self.input_name: x[i:i+1]}
                count += 1

    def get_next(self):
        return next(self._iterator, None)


def quantize_static_model(model_path: str, num_calibration_samples: int, output_path: str = None) -> str:
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if output_path is None:
        output_name = model_path.name.replace('fp32', 'int8_static')
        output_path = model_path.parent / output_name
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # Pre-process: same robust shape inference step as dynamic quantization.
    preprocessed_path = model_path.parent / (model_path.stem + '.static_preprocessed.onnx')
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

    # Get real calibration data from the training set.
    print(f"Loading training data for calibration ({num_calibration_samples} samples)...")
    config = load_config()
    train_loader, _val_loader, _test_loader, _class_to_idx = create_dataloaders(config)

    import onnxruntime as ort
    probe_session = ort.InferenceSession(str(preprocessed_path), providers=['CPUExecutionProvider'])
    input_name = probe_session.get_inputs()[0].name

    calibration_reader = ECGCalibrationDataReader(train_loader, input_name, num_calibration_samples)

    print(f"Running static quantization -> {output_path}")
    print("Quantizing Conv/Gemm/MatMul only -- LSTM stays FP32.")
    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,          # per-channel weight scales -- important for Conv accuracy
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],  # deliberately excludes LSTM
    )

    fp32_size = model_path.stat().st_size
    int8_size = output_path.stat().st_size
    reduction = 100 * (1 - int8_size / fp32_size)

    print(f"\nFP32 size: {fp32_size / 1024 / 1024:.2f} MB")
    print(f"INT8 (static) size: {int8_size / 1024 / 1024:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")
    print(f"\nSaved statically quantized model to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static-quantize ONNX model to INT8 (Conv/Gemm/MatMul only)")
    parser.add_argument("--model-path", required=True, help="Path to FP32 .onnx model")
    parser.add_argument("--num-calibration-samples", type=int, default=200,
                         help="Number of training samples to use for calibration")
    parser.add_argument("--output", default=None, help="Output path for statically quantized model")
    args = parser.parse_args()
    quantize_static_model(args.model_path, args.num_calibration_samples, args.output)