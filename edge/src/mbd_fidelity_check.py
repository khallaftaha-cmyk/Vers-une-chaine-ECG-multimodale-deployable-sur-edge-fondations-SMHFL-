"""Model-Based Design fidelity check (completes the MIL/SIL/PIL comparison).

Run this on the SAME machine as mbd_fidelity_export.py to compare
PyTorch vs ONNX FP32 vs ONNX INT8 static on the PC alone (SIL-only
check) -- or copy mbd_reference.npz to the Raspberry Pi and run it
there to add PIL (Processor-in-the-Loop): does the real target hardware
reproduce the same predictions as the PC-computed reference?

Only needs numpy + onnxruntime -- safe to run on the Pi.

Usage (on PC, SIL-only):
    python src/mbd_fidelity_check.py --reference models/mbd_reference.npz

Usage (on Raspberry Pi, adds PIL comparison):
    python3 src/mbd_fidelity_check.py --reference models/mbd_reference.npz \
        --target-model models/ecg_model_int8_static.onnx --target-label "Pi (PIL)"
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / 'reports'


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def compare_stage(ref_logits: np.ndarray, other_logits: np.ndarray, ref_name: str, other_name: str) -> dict:
    ref_pred = np.argmax(ref_logits, axis=1)
    other_pred = np.argmax(other_logits, axis=1)
    agreement = float((ref_pred == other_pred).mean())
    max_abs_diff = float(np.max(np.abs(ref_logits - other_logits)))
    mean_abs_diff = float(np.mean(np.abs(ref_logits - other_logits)))
    return {
        'comparison': f"{ref_name} vs {other_name}",
        'prediction_agreement': agreement,
        'max_abs_logit_diff': max_abs_diff,
        'mean_abs_logit_diff': mean_abs_diff,
    }


def run_onnx_on_signals(signals: np.ndarray, onnx_path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = []
    for i in range(signals.shape[0]):
        out = session.run(None, {input_name: signals[i:i+1]})[0]
        outputs.append(out[0])
    return np.stack(outputs, axis=0)


def main(reference_path: str, target_model: str, target_label: str):
    ref_path = Path(reference_path)
    if not ref_path.is_absolute():
        ref_path = PROJECT_ROOT / ref_path
    data = np.load(ref_path, allow_pickle=False)
    signals = data['signals']
    labels = data['labels']
    class_names = [str(c) for c in data['class_names']]
    pytorch_logits = data['pytorch_logits']
    fp32_logits = data['fp32_logits']
    static_logits = data['static_logits']

    comparisons = []

    # Always compare the reference stages against each other (MIL vs SIL FP32, SIL FP32 vs SIL INT8 static)
    comparisons.append(compare_stage(pytorch_logits, fp32_logits, "PyTorch (MIL)", "ONNX FP32 (SIL, PC)"))
    comparisons.append(compare_stage(fp32_logits, static_logits, "ONNX FP32 (SIL, PC)", "ONNX INT8 static (SIL, PC)"))

    # Optionally add a target platform (e.g. the Pi) actually running a model now
    if target_model:
        target_path = Path(target_model)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
        print(f"Running {target_label} inference on {len(signals)} signals...")
        target_logits = run_onnx_on_signals(signals, target_path)
        comparisons.append(compare_stage(static_logits, target_logits, "ONNX INT8 static (SIL, PC)", target_label))

    print("\n" + "=" * 60)
    print("MBD FIDELITY REPORT (MIL / SIL / PIL)")
    print("=" * 60)
    for c in comparisons:
        print(f"\n{c['comparison']}")
        print(f"  Prediction agreement: {c['prediction_agreement']*100:.2f}%")
        print(f"  Max |logit diff|:     {c['max_abs_logit_diff']:.6f}")
        print(f"  Mean |logit diff|:    {c['mean_abs_logit_diff']:.6f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / 'mbd_fidelity_report.md'
    with open(report_path, 'w') as f:
        f.write("# MBD Fidelity Report (MIL / SIL / PIL)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Signals used:** {len(signals)} real ECG test samples\n\n")
        f.write("| Comparison | Prediction agreement | Max |logit diff| | Mean |logit diff| |\n")
        f.write("|---|---|---|---|\n")
        for c in comparisons:
            f.write(f"| {c['comparison']} | {c['prediction_agreement']*100:.2f}% "
                     f"| {c['max_abs_logit_diff']:.6f} | {c['mean_abs_logit_diff']:.6f} |\n")
        f.write("\n**Interpretation:** prediction agreement close to 100% and small logit "
                "differences confirm each stage of the deployment pipeline (training -> "
                "ONNX export -> quantization -> target hardware) preserves the model's "
                "behavior. Large drops in agreement at any one stage pinpoint exactly "
                "where fidelity was lost.\n")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBD fidelity check (MIL/SIL/PIL)")
    parser.add_argument("--reference", default="models/mbd_reference.npz",
                         help="Path to the .npz from mbd_fidelity_export.py")
    parser.add_argument("--target-model", default=None,
                         help="Optional: path to an ONNX model to run HERE and compare "
                              "against the PC reference (e.g. on the Pi, for PIL)")
    parser.add_argument("--target-label", default="Target platform (PIL)",
                         help="Label for the target platform in the report")
    args = parser.parse_args()
    main(args.reference, args.target_model, args.target_label)