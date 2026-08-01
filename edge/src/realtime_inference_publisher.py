"""Run ONNX inference repeatedly and publish each result over MQTT.

Loops through REAL, already-preprocessed ECG test signals (exported by
src/export_demo_signals.py on the PC, since the Pi doesn't have the full
data pipeline installed). Each cycle publishes the predicted class,
confidence, AND the true label if available -- so the subscriber can show
whether each prediction was actually correct, live.

Falls back to random dummy input (the old behavior) if no --signals-path
is given, e.g. for a quick connectivity smoke test.

Run src/mqtt_subscriber_test.py FIRST in another terminal, then run this.

Usage (real signals, recommended):
    python src/realtime_inference_publisher.py --model-path models/ecg_model_int8_static.onnx --signals-path models/demo_signals.npz

Usage (dummy input, connectivity test only):
    python src/realtime_inference_publisher.py --model-path models/ecg_model_fp32.onnx
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Deliberately NOT importing from src.data_loader here -- that module pulls
# in pandas, wfdb, sklearn (training-only dependencies). This publisher only
# needs the tiny bit of config.yaml under 'edge:', so we read it directly
# with PyYAML. Keeps the Pi's install lightweight (matches requirements_pi.txt).
def load_edge_config() -> dict:
    config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_demo_signals(signals_path: Path):
    """Loads the .npz produced by export_demo_signals.py. Only needs
    numpy -- no pandas/wfdb required on the Pi."""
    data = np.load(signals_path, allow_pickle=False)
    signals = data['signals']            # (N, 12, 5000) float32
    labels = data['labels']              # (N,) int64
    class_names = [str(c) for c in data['class_names']]
    return signals, labels, class_names


def generate_dummy_input(shape) -> np.ndarray:
    """Fallback for a quick connectivity test when no real signals are
    supplied. Not meaningful for demo purposes -- prefer --signals-path."""
    return np.random.randn(*shape).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def run_inference(session: ort.InferenceSession, input_name: str, x: np.ndarray):
    logits = session.run(None, {input_name: x})[0][0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence


def main(model_path: str, num_cycles: int, interval_seconds: float,
         broker_override: str = None, signals_path: str = None):
    config = load_edge_config()
    edge_cfg = config.get('edge', {})
    broker = broker_override or edge_cfg.get('mqtt_broker', 'localhost')
    port = edge_cfg.get('mqtt_port', 1883)
    topic = edge_cfg.get('mqtt_topic', 'ecg/inference')

    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    shape = [1 if isinstance(dim, str) else dim for dim in input_shape]

    # Real signals if provided, otherwise dummy noise (connectivity test only).
    signals = labels = class_names = None
    if signals_path:
        signals_path = Path(signals_path)
        if not signals_path.is_absolute():
            signals_path = PROJECT_ROOT / signals_path
        print(f"Loading real demo signals: {signals_path}")
        signals, labels, class_names = load_demo_signals(signals_path)
        print(f"Loaded {len(signals)} real signals. Classes: {class_names}")
        if num_cycles > len(signals):
            print(f"Note: --cycles ({num_cycles}) > available signals ({len(signals)}); looping back to the start.")
    else:
        print("WARNING: no --signals-path given -- publishing random dummy input, "
              "not real predictions. Use --signals-path for a meaningful demo.")
        class_names = ["Sinus Bradycardia", "Sinus Rhythm", "Atrial Fibrillation", "GSVT"]

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    print(f"Connecting to broker at {broker}:{port}...")
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    print(f"Publishing to topic '{topic}'. Running {num_cycles} cycle(s), "
          f"{interval_seconds}s apart. Press Ctrl+C to stop early.\n")

    try:
        for i in range(num_cycles):
            true_class = None
            if signals is not None:
                idx = i % len(signals)
                x = signals[idx:idx+1]  # (1, 12, 5000), already the right shape
                true_class = class_names[labels[idx]]
            else:
                x = generate_dummy_input(shape)

            pred_idx, confidence = run_inference(session, input_name, x)
            pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

            payload = {
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "predicted_class": pred_class,
                "confidence": confidence,
            }
            if true_class is not None:
                payload["true_class"] = true_class
                payload["correct"] = (pred_class == true_class)

            client.publish(topic, json.dumps(payload))
            print(f"Published: {payload}")

            if i < num_cycles - 1:
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish ONNX inference results over MQTT")
    parser.add_argument("--model-path", required=True, help="Path to .onnx model")
    parser.add_argument("--signals-path", default=None,
                         help="Path to .npz of real demo signals (from export_demo_signals.py). "
                              "If omitted, falls back to random dummy input.")
    parser.add_argument("--cycles", type=int, default=10, help="Number of inference cycles")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between cycles")
    parser.add_argument("--broker", default=None,
                         help="Override the broker host from config.yaml (e.g. your PC's IP when running from the Pi)")
    args = parser.parse_args()
    main(args.model_path, args.cycles, args.interval, args.broker, args.signals_path)