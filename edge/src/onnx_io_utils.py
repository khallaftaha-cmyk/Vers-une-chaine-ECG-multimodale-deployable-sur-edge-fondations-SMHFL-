"""Shared helper: auto-detect whether an ONNX model expects channels-first
(batch, 12, 5000) -- the PyTorch/Conv1d convention used by our own models --
or channels-last (batch, 5000, 12) -- the Keras/TensorFlow Conv1D convention,
which is what Iman's model uses (confirmed via its 'dense_2' default output
name, a Keras auto-generated layer name).

Import this wherever a script loads an arbitrary ONNX model and needs to
feed it real (num_leads, sequence_length)-shaped ECG signals -- so the same
evaluation/benchmark/deployment code works for either model without manual
transposing or hardcoded assumptions.
"""

import numpy as np
import onnxruntime as ort


def detect_orientation(session: ort.InferenceSession, num_leads: int = 12, sequence_length: int = 5000) -> str:
    """Returns 'channels_first' if the model's input is (batch, leads, seq),
    or 'channels_last' if it's (batch, seq, leads). Raises if neither shape
    matches, since silently guessing wrong here would corrupt every result
    downstream."""
    shape = session.get_inputs()[0].shape  # e.g. ['batch_size', 12, 5000] or ['unk__754', 5000, 12]
    dims = [d for d in shape if isinstance(d, int)]  # drop the dynamic batch dim

    if dims == [num_leads, sequence_length]:
        return 'channels_first'
    if dims == [sequence_length, num_leads]:
        return 'channels_last'

    raise ValueError(
        f"Unrecognized input shape {shape} -- expected either "
        f"(batch, {num_leads}, {sequence_length}) or (batch, {sequence_length}, {num_leads})."
    )


def prepare_input(signals: np.ndarray, orientation: str) -> np.ndarray:
    """signals: (batch, 12, 5000) -- our own canonical channels-first storage
    format (matches what export_demo_signals.py / data_loader.py produce).
    Transposes to channels-last only if the target model needs it."""
    if orientation == 'channels_first':
        return signals
    elif orientation == 'channels_last':
        return np.transpose(signals, (0, 2, 1))  # (batch, 12, 5000) -> (batch, 5000, 12)
    else:
        raise ValueError(f"Unknown orientation '{orientation}'")


def load_adaptive_session(onnx_path: str, num_leads: int = 12, sequence_length: int = 5000):
    """Convenience wrapper: returns (session, input_name, orientation) so
    callers don't need to repeat the detection boilerplate."""
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    orientation = detect_orientation(session, num_leads, sequence_length)
    return session, input_name, orientation