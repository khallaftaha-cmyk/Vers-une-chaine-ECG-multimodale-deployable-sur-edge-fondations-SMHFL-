"""Compare real classification accuracy (macro F1) between the FP32 and
INT8 ONNX models on your actual held-out test set. This is the piece
the placeholder model could never give you -- it had no real labels, so
"accuracy" wasn't measurable. Now it is.

Works with models from either framework/orientation (your own PyTorch
channels-first models, or Iman's Keras channels-last model) via
onnx_io_utils' automatic orientation detection.

Usage:
    python src/evaluate_onnx_accuracy.py --fp32 models/ecg_model_fp32.onnx --int8 models/ecg_model_int8_dynamic.onnx
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config, create_dataloaders
from src.onnx_io_utils import load_adaptive_session, prepare_input

REPORTS_DIR = PROJECT_ROOT / 'reports'


def evaluate_onnx_model(onnx_path: str, test_loader, pred_permutation=None):
    onnx_path = Path(onnx_path)
    if not onnx_path.is_absolute():
        onnx_path = PROJECT_ROOT / onnx_path

    session, input_name, orientation = load_adaptive_session(onnx_path)
    print(f"  ({onnx_path.name}: detected {orientation} input)")

    all_preds, all_labels = [], []
    for signals, labels in test_loader:
        x = signals.numpy().astype(np.float32)   # (batch, 12, 5000) -- our canonical format
        x = prepare_input(x, orientation)          # transposed only if this model needs it
        logits = session.run(None, {input_name: x})[0]
        preds = np.argmax(logits, axis=1)
        if pred_permutation is not None:
            # Remaps the model's raw output index to THIS pipeline's class_to_idx
            # order -- use when a model's real output index order doesn't match
            # what its documentation/fiche claims (see confusion matrix evidence).
            preds = np.array([pred_permutation[p] for p in preds])
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, macro_f1, report, cm


def main(fp32_path: str, int8_path: str, lowcut: float, highcut: float, pred_permutation=None):
    config = load_config()
    # Override bandpass cutoffs -- different models can require different
    # preprocessing. Iman's fiche specifies 0.5-40 Hz; our own model used
    # the config/preprocessing.py default of 0.5-45 Hz. Mismatching this
    # produces near-zero accuracy even on a genuinely good model.
    config.setdefault('preprocessing', {})
    config['preprocessing']['lowcut'] = lowcut
    config['preprocessing']['highcut'] = highcut
    print(f"Using bandpass filter: {lowcut}-{highcut} Hz")
    _, _, test_loader, class_to_idx = create_dataloaders(config)

    print("Evaluating FP32 model on test set...")
    fp32_acc, fp32_f1, fp32_report, fp32_cm = evaluate_onnx_model(fp32_path, test_loader, pred_permutation)

    print("Evaluating INT8 model on test set...")
    int8_acc, int8_f1, int8_report, int8_cm = evaluate_onnx_model(int8_path, test_loader, pred_permutation)

    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    print(f"\nClass order (index -> name): {class_names}")
    print(f"\nFP32 confusion matrix (rows=true, cols=predicted):\n{fp32_cm}")
    print(f"\nINT8 confusion matrix (rows=true, cols=predicted):\n{int8_cm}")

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
        f.write(f"## Class order\n`{class_names}`\n\n")
        f.write(f"## FP32 confusion matrix (rows=true, cols=predicted)\n```\n{fp32_cm}\n```\n\n")
        f.write(f"## INT8 confusion matrix (rows=true, cols=predicted)\n```\n{int8_cm}\n```\n\n")
        f.write("## FP32 classification report\n```\n" + fp32_report + "\n```\n\n")
        f.write("## INT8 classification report\n```\n" + int8_report + "\n```\n")

    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FP32 vs INT8 ONNX accuracy")
    parser.add_argument("--fp32", required=True, help="Path to FP32 .onnx model")
    parser.add_argument("--int8", required=True, help="Path to INT8 .onnx model")
    parser.add_argument("--lowcut", type=float, default=0.5, help="Bandpass lowcut Hz")
    parser.add_argument("--highcut", type=float, default=45.0,
                         help="Bandpass highcut Hz (our own model: 45.0 default; Iman's model requires 40.0 per her fiche)")
    parser.add_argument("--pred-permutation", type=str, default=None,
                         help="Comma-separated remap for a model whose real output index order doesn't "
                              "match its documented order (e.g. Iman's model: '2,3,0,1' -- confirmed via "
                              "confusion matrix, likely caused by alphabetical LabelEncoder ordering "
                              "[AFIB,GSVT,SB,SR] vs the semantic order [SB,SR,AFIB,GSVT] in her fiche)")
    args = parser.parse_args()
    pred_permutation = [int(x) for x in args.pred_permutation.split(',')] if args.pred_permutation else None
    main(args.fp32, args.int8, args.lowcut, args.highcut, pred_permutation)