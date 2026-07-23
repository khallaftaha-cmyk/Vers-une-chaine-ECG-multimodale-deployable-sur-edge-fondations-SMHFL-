"""Compare real classification accuracy (macro F1) between the FP32 and
INT8 ONNX models on your actual held-out test set. This is the piece
the placeholder model could never give you -- it had no real labels, so
"accuracy" wasn't measurable. Now it is.

Usage:
    python src/evaluate_onnx_accuracy.py --fp32 models/ecg_model_fp32.onnx --int8 models/ecg_model_int8_dynamic.onnx
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score, accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config, create_dataloaders

REPORTS_DIR = PROJECT_ROOT / 'reports'


def evaluate_onnx_model(onnx_path: str, test_loader):
    onnx_path = Path(onnx_path)
    if not onnx_path.is_absolute():
        onnx_path = PROJECT_ROOT / onnx_path

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    all_preds, all_labels = [], []
    for signals, labels in test_loader:
        x = signals.numpy().astype(np.float32)
        logits = sess.run(None, {input_name: x})[0]
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds)
    return acc, macro_f1, report


def main(fp32_path: str, int8_path: str):
    config = load_config()
    _, _, test_loader, class_to_idx = create_dataloaders(config)

    print("Evaluating FP32 model on test set...")
    fp32_acc, fp32_f1, fp32_report = evaluate_onnx_model(fp32_path, test_loader)

    print("Evaluating INT8 model on test set...")
    int8_acc, int8_f1, int8_report = evaluate_onnx_model(int8_path, test_loader)

    print(f"\n{'Model':<10}{'Accuracy':>12}{'Macro F1':>12}")
    print(f"{'FP32':<10}{fp32_acc:>12.4f}{fp32_f1:>12.4f}")
    print(f"{'INT8':<10}{int8_acc:>12.4f}{int8_f1:>12.4f}")
    print(f"\nMacro F1 delta (INT8 - FP32): {int8_f1 - fp32_f1:+.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / 'accuracy_comparison.md'
    with open(report_path, 'w') as f:
        f.write("# FP32 vs INT8 Accuracy Comparison\n\n")
        f.write(f"Class mapping: `{class_to_idx}`\n\n")
        f.write("| Model | Accuracy | Macro F1 |\n|---|---|---|\n")
        f.write(f"| FP32 | {fp32_acc:.4f} | {fp32_f1:.4f} |\n")
        f.write(f"| INT8 | {int8_acc:.4f} | {int8_f1:.4f} |\n\n")
        f.write(f"**Macro F1 delta (INT8 - FP32):** {int8_f1 - fp32_f1:+.4f}\n\n")
        f.write("## FP32 classification report\n```\n" + fp32_report + "\n```\n\n")
        f.write("## INT8 classification report\n```\n" + int8_report + "\n```\n")

    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FP32 vs INT8 ONNX accuracy")
    parser.add_argument("--fp32", required=True, help="Path to FP32 .onnx model")
    parser.add_argument("--int8", required=True, help="Path to INT8 .onnx model")
    args = parser.parse_args()
    main(args.fp32, args.int8)
