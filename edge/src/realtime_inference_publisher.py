"""Run ONNX inference repeatedly and publish each result over MQTT.

For now this uses random dummy input on each cycle (same idea as
benchmark_onnx.py) just to prove the real-time chain works: inference
-> MQTT publish -> subscriber receives it. Once you're deploying on
the actual Raspberry Pi with a real ECG source, swap out
`generate_dummy_input()` for your real acquisition step.

Run src/mqtt_subscriber_test.py FIRST in another terminal, then run
this.

Usage:
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

# Placeholder until class_to_idx is wired in from a real dataset scan --
# order must match your config.yaml label_mapping's top-4 classes.
CLASS_NAMES = ["Sinus Bradycardia", "Sinus Rhythm", "Atrial Fibrillation", "GSVT"]


def generate_dummy_input(shape) -> np.ndarray:
    """Stand-in for real ECG acquisition. Replace this with actual
    signal capture once you're on the Pi with a real input source."""
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


def main(model_path: str, num_cycles: int, interval_seconds: float, broker_override: str = None):
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

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    print(f"Connecting to broker at {broker}:{port}...")
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    print(f"Publishing to topic '{topic}'. Running {num_cycles} cycle(s), "
          f"{interval_seconds}s apart. Press Ctrl+C to stop early.\n")

    try:
        for i in range(num_cycles):
            x = generate_dummy_input(shape)
            pred_idx, confidence = run_inference(session, input_name, x)
            pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

            payload = {
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "predicted_class": pred_class,
                "confidence": confidence,
            }
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
    parser.add_argument("--cycles", type=int, default=10, help="Number of inference cycles")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between cycles")
    parser.add_argument("--broker", default=None,
                         help="Override the broker host from config.yaml (e.g. your PC's IP when running from the Pi)")
    args = parser.parse_args()
    main(args.model_path, args.cycles, args.interval, args.broker)