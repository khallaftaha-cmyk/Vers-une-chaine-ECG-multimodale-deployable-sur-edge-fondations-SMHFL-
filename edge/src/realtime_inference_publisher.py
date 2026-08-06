"""Run ONNX inference repeatedly and publish each result over MQTT.

Loops through REAL, already-preprocessed ECG test signals (exported by
src/export_demo_signals.py on the PC, since the Pi doesn't have the full
data pipeline installed). Each cycle publishes the predicted class,
confidence, AND the true label if available -- so the subscriber can show
whether each prediction was actually correct, live.

Falls back to random dummy input (the old behavior) if no --signals-path
is given, e.g. for a quick connectivity smoke test.

Class names come from configs/config.yaml's model.classes (single source
of truth) when --signals-path isn't given. When real signals ARE given,
export_demo_signals.py's own class_names (baked into the .npz from the
real class_to_idx at export time) take priority, since that's the most
authoritative ordering available.

Includes automatic MQTT reconnection: if the broker connection drops
mid-run (Wi-Fi hiccup, broker restart, etc.), the publisher retries with
backoff instead of crashing the whole demo.

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# onnx_io_utils only needs numpy/onnxruntime -- safe to import on the Pi,
# unlike src.data_loader (pandas/wfdb/sklearn).
from src.onnx_io_utils import detect_orientation, prepare_input


# Deliberately NOT importing from src.data_loader here -- that module pulls
# in pandas, wfdb, sklearn (training-only dependencies). This publisher only
# needs config.yaml itself, so we read it directly with PyYAML. Keeps the
# Pi's install lightweight (matches requirements_pi.txt).
def load_config() -> dict:
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


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print("MQTT connected.")
    else:
        print(f"MQTT connect failed, reason code: {reason_code}")


def on_disconnect(client, userdata, flags, reason_code, properties=None):
    if reason_code != 0:
        print(f"MQTT disconnected unexpectedly (reason code: {reason_code}). "
              f"paho will auto-reconnect with backoff...")


def main(model_path: str, num_cycles: int, interval_seconds: float,
         broker_override: str = None, signals_path: str = None):
    config = load_config()
    edge_cfg = config.get('edge', {})
    broker = broker_override or edge_cfg.get('mqtt_broker', 'localhost')
    port = edge_cfg.get('mqtt_port', 1883)
    topic = edge_cfg.get('mqtt_topic', 'ecg/inference')
    config_classes = config.get('model', {}).get('classes', [])

    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    shape = [1 if isinstance(dim, str) else dim for dim in input_shape]
    orientation = detect_orientation(session)
    print(f"Detected model input orientation: {orientation}")

    # Real signals if provided, otherwise dummy noise (connectivity test only).
    signals = labels = None
    if signals_path:
        signals_path = Path(signals_path)
        if not signals_path.is_absolute():
            signals_path = PROJECT_ROOT / signals_path
        print(f"Loading real demo signals: {signals_path}")
        signals, labels, npz_class_names = load_demo_signals(signals_path)
        class_names = npz_class_names  # most authoritative: baked in at export time
        print(f"Loaded {len(signals)} real signals. Classes: {class_names}")
        if config_classes and config_classes != npz_class_names:
            print(f"WARNING: config.yaml's model.classes {config_classes} does not match "
                  f"the .npz's class_names {npz_class_names} -- config.yaml may be stale.")
        if num_cycles > len(signals):
            print(f"Note: --cycles ({num_cycles}) > available signals ({len(signals)}); looping back to the start.")
    else:
        if not config_classes:
            print("ERROR: no --signals-path given AND configs/config.yaml has no model.classes defined. "
                  "Can't determine class names. Add model.classes to config.yaml or pass --signals-path.")
            sys.exit(1)
        print("WARNING: no --signals-path given -- publishing random dummy input, "
              "not real predictions. Use --signals-path for a meaningful demo.")
        class_names = config_classes

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)  # backoff on dropped connections

    print(f"Connecting to broker at {broker}:{port}...")
    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        print(f"Initial connection failed: {e}")
        sys.exit(1)
    client.loop_start()  # background thread also handles auto-reconnect

    print(f"Publishing to topic '{topic}'. Running {num_cycles} cycle(s), "
          f"{interval_seconds}s apart. Press Ctrl+C to stop early.\n")

    try:
        for i in range(num_cycles):
            true_class = None
            if signals is not None:
                idx = i % len(signals)
                x = signals[idx:idx+1]           # (1, 12, 5000) -- canonical
                x = prepare_input(x, orientation)  # transposed only if this model needs it
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

            result = client.publish(topic, json.dumps(payload))
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"WARNING: publish failed (rc={result.rc}) -- message may be lost, "
                      f"will keep trying on subsequent cycles.")
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