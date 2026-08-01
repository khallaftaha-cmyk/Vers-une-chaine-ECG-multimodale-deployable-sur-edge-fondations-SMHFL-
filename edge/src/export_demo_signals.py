"""Export a batch of real, already-preprocessed ECG test signals (+ true
labels) to a single compact .npz file, for use by the Pi's real-time
publisher demo.

Why this exists: the Pi deliberately doesn't have pandas/wfdb/scipy
installed (see requirements_pi.txt -- inference-only, lightweight). This
script does the heavy lifting HERE, on the PC, using your full data
pipeline (same preprocessing as training: bandpass filter + z-score),
then saves the result as plain numpy arrays. The Pi only needs `numpy`
to load and loop through them -- no dataset, no wfdb, no scipy required
on-device.

Usage:
    python src/export_demo_signals.py --num-samples 30
"""

import sys
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config, create_dataloaders

DEFAULT_OUTPUT = PROJECT_ROOT / 'models' / 'demo_signals.npz'


def export_demo_signals(num_samples: int, output_path: Path = DEFAULT_OUTPUT):
    config = load_config()
    _train_loader, _val_loader, test_loader, class_to_idx = create_dataloaders(config)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    signals_out = []
    labels_out = []

    for signals, labels in test_loader:
        for i in range(signals.shape[0]):
            if len(signals_out) >= num_samples:
                break
            signals_out.append(signals[i].numpy().astype(np.float32))
            labels_out.append(int(labels[i]))
        if len(signals_out) >= num_samples:
            break

    signals_arr = np.stack(signals_out, axis=0)  # (N, 12, 5000)
    labels_arr = np.array(labels_out, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        signals=signals_arr,
        labels=labels_arr,
        class_names=np.array(class_names),
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"Exported {len(signals_out)} real test signals -> {output_path} ({size_kb:.1f} KB)")
    print(f"Class names (label order): {class_names}")
    print(f"Label distribution: {np.bincount(labels_arr, minlength=len(class_names))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export real ECG test signals for the Pi real-time demo")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of signals to export")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path")
    args = parser.parse_args()
    output = Path(args.output) if args.output else DEFAULT_OUTPUT
    export_demo_signals(args.num_samples, output)