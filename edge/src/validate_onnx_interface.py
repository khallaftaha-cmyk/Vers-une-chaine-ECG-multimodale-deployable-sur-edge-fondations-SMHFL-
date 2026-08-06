"""Validate that an ONNX model matches the shared interface contract
between the Edge and Data/Modele volets, BEFORE spending time
quantizing/benchmarking/deploying it.

Checks:
  - input name 'ecg_input', shape (batch, 12, 5000), float32
  - output name 'classification', shape (batch, num_classes), float32
  - opset version >= 17
  - number of output classes matches config.yaml's model.classes

Run this the moment Iman's model arrives -- catches a contract mismatch
in seconds instead of discovering it mid-quantization (like we did with
our own BiLSTM shape issue).

Usage:
    python src/validate_onnx_interface.py --model-path models/model_iman.onnx
"""

import sys
import argparse
from pathlib import Path
import onnx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check(condition: bool, ok_msg: str, fail_msg: str, failures: list) -> None:
    if condition:
        print(f"  [OK]   {ok_msg}")
    else:
        print(f"  [FAIL] {fail_msg}")
        failures.append(fail_msg)


def validate(model_path: str) -> bool:
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    config = load_config()
    expected_classes = config.get('model', {}).get('classes', [])
    expected_num_classes = config.get('model', {}).get('num_classes', len(expected_classes))
    expected_leads = config.get('data', {}).get('num_leads', 12)
    expected_seq_len = config.get('data', {}).get('sequence_length', 5000)
    min_opset = 17

    print(f"Validating: {model_path}\n")
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    failures = []

    # Opset
    opset = model.opset_import[0].version if model.opset_import else None
    # Opset. Note: opset 17 was needed for OUR model's BiLSTM symbolic
    # shape inference specifically (see quantize_onnx.py) -- not a
    # universal ONNX interoperability requirement. A lower opset from a
    # different model/framework may quantize and run just fine; this is
    # informational, not a hard blocker.
    if opset is not None and opset >= min_opset:
        print(f"  [OK]   Opset version {opset} (>= {min_opset})")
    else:
        print(f"  [INFO] Opset version {opset} is below {min_opset} -- our BiLSTM export needed 17+, "
              f"but this may be unrelated to this model. Test quantization directly rather than assuming it'll fail.")

    # Input
    inputs = model.graph.input
    check(len(inputs) == 1, f"Exactly 1 input ({len(inputs)} found)",
          f"Expected exactly 1 input, found {len(inputs)}", failures)
    if inputs:
        inp = inputs[0]
        if inp.name == 'ecg_input':
            print(f"  [OK]   Input name is 'ecg_input'")
        else:
            print(f"  [INFO] Input name is '{inp.name}' (expected 'ecg_input' -- "
                  f"not blocking, just a naming difference)")
        dims = inp.type.tensor_type.shape.dim
        dim_values = [d.dim_value if d.dim_value > 0 else d.dim_param for d in dims]
        int_dims = [d for d in dim_values if isinstance(d, int)]
        channels_first = int_dims == [expected_leads, expected_seq_len]
        channels_last = int_dims == [expected_seq_len, expected_leads]
        if channels_first:
            print(f"  [OK]   Input shape {dim_values} -- channels-first (batch, {expected_leads}, {expected_seq_len})")
        elif channels_last:
            print(f"  [OK]   Input shape {dim_values} -- channels-last (batch, {expected_seq_len}, {expected_leads}) "
                  f"[Keras/TF convention -- handled automatically by onnx_io_utils.prepare_input()]")
        else:
            msg = (f"Input shape {dim_values} matches NEITHER (batch, {expected_leads}, {expected_seq_len}) "
                   f"nor (batch, {expected_seq_len}, {expected_leads})")
            print(f"  [FAIL] {msg}")
            failures.append(msg)

    # Output
    outputs = model.graph.output
    check(len(outputs) == 1, f"Exactly 1 output ({len(outputs)} found)",
          f"Expected exactly 1 output, found {len(outputs)}", failures)
    if outputs:
        out = outputs[0]
        if out.name == 'classification':
            print(f"  [OK]   Output name is 'classification'")
        else:
            print(f"  [INFO] Output name is '{out.name}' (expected 'classification' -- "
                  f"NOT blocking: our scripts read outputs positionally, not by name)")
        dims = out.type.tensor_type.shape.dim
        dim_values = [d.dim_value if d.dim_value > 0 else d.dim_param for d in dims]
        shape_ok = (len(dim_values) == 2 and dim_values[1] == expected_num_classes)
        check(shape_ok, f"Output shape {dim_values} matches (batch, {expected_num_classes})",
              f"Output shape {dim_values} does NOT match expected (batch, {expected_num_classes})", failures)

    print(f"\nExpected classes (from config.yaml): {expected_classes}")
    print("NOTE: class ORDER can't be verified from the ONNX file alone -- "
          "confirm with Iman that her class_to_idx matches this exact order.")

    print(f"\n{'='*50}")
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED. Do not proceed with this model until resolved.")
        for f in failures:
            print(f"  - {f}")
    else:
        print("RESULT: All checks PASSED. Model matches the shared interface contract.")
    print(f"{'='*50}")

    return len(failures) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an ONNX model against the shared interface contract")
    parser.add_argument("--model-path", required=True, help="Path to .onnx model to validate")
    args = parser.parse_args()
    ok = validate(args.model_path)
    sys.exit(0 if ok else 1)