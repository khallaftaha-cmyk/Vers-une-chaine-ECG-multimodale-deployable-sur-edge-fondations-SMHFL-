"""Real-time listener for ECG inference results published over MQTT.

Run this FIRST, in its own terminal, and leave it running. Every time
the publisher (src/realtime_inference_publisher.py) sends a new
prediction, it'll show up here immediately -- this is your visible
proof that the "ECG input -> edge inference -> result via MQTT" chain
actually works end-to-end.

Usage:
    python src/mqtt_subscriber_test.py
"""

import sys
import json
from pathlib import Path
import paho.mqtt.client as mqtt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"Connected to broker. Subscribing to '{userdata['topic']}'...")
        client.subscribe(userdata['topic'])
    else:
        print(f"Connection failed with reason code: {reason_code}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        print(f"\n[{payload.get('timestamp', '?')}] "
              f"Prediction: {payload.get('predicted_class', '?')} "
              f"(confidence: {payload.get('confidence', 0):.3f})")
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"\nReceived non-JSON message: {msg.payload}")


def main():
    config = load_config()
    edge_cfg = config.get('edge', {})
    broker = edge_cfg.get('mqtt_broker', 'localhost')
    port = edge_cfg.get('mqtt_port', 1883)
    topic = edge_cfg.get('mqtt_topic', 'ecg/inference')

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata={'topic': topic})
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to broker at {broker}:{port}...")
    client.connect(broker, port, keepalive=60)

    print("Listening for predictions. Press Ctrl+C to stop.\n")
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")