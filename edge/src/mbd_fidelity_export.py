"""Model-Based Design fidelity export (MIL / SIL reference).

Runs the SAME real ECG signals through three stages and saves every
stage's output for later comparison:
  - MIL (Model-in-the-Loop):  PyTorch model, best_model.pth
  - SIL (Software-in-the-Loop): ONNX FP32, on this PC
  - SIL (Software-in-the-Loop): ONNX INT8 static, on this PC

Run mbd_fidelity_check.py afterwards -- on the PC to compare stages
against each other, or on the Raspberry Pi (PIL: Processor-in-the-Loop)
to compare its real hardware output against this PC-computed reference.

Usage:
    python src/mbd_fidelity_export.py --signals-path models/demo_signals.npz
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model
from src.data_loader import load_config

DEFAULT_SIGNALS = PROJECT_ROOT / 'models' / 'demo_signals.npz'
DEFAULT_OUTPUT = PROJECT_ROOT / 'models' / 'mbd_reference.npz'


def run_pytorch(signals: np.ndarray, weights_path: Path, config: dict) -> np.ndarray:
    model = build_model(config)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(signals).float()
        logits = model(x).numpy()
    return logits


def run_onnx(signals: np.ndarray, onnx_path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = []
    for i in range(signals.shape[0]):
        out = session.run(None, {input_name: signals[i:i+1]})[0]
        outputs.append(out[0])
    return np.stack(outputs, axis=0)


def main(signals_path: str, weights_path: str, fp32_onnx: str, static_onnx: str, output_path: str):
    signals_path = Path(signals_path)
    data = np.load(signals_path, allow_pickle=False)
    signals = data['signals']
    labels = data['labels']
    class_names = [str(c) for c in data['class_names']]

    config = load_config()

    print(f"Running PyTorch (MIL reference) on {len(signals)} signals...")
    pytorch_logits = run_pytorch(signals, PROJECT_ROOT / weights_path, config)

    print("Running ONNX FP32 (SIL) on PC...")
    fp32_logits = run_onnx(signals, PROJECT_ROOT / fp32_onnx)

    print("Running ONNX INT8 static (SIL) on PC...")
    static_logits = run_onnx(signals, PROJECT_ROOT / static_onnx)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        signals=signals,
        labels=labels,
        class_names=np.array(class_names),
        pytorch_logits=pytorch_logits,
        fp32_logits=fp32_logits,
        static_logits=static_logits,
    )
    print(f"\nSaved MBD reference (PyTorch + ONNX FP32 + ONNX INT8 static) -> {output_path}")
    print("Copy this file to the Raspberry Pi and run mbd_fidelity_check.py there "
          "to complete the MIL/SIL/PIL comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MIL/SIL reference outputs for MBD fidelity check")
    parser.add_argument("--signals-path", default=str(DEFAULT_SIGNALS))
    parser.add_argument("--weights-path", default="models/best_model.pth")
    parser.add_argument("--fp32-onnx", default="models/ecg_model_fp32.onnx")
    parser.add_argument("--static-onnx", default="models/ecg_model_int8_static.onnx")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    main(args.signals_path, args.weights_path, args.fp32_onnx, args.static_onnx, args.output)