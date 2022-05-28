import torch
import torch.nn.functional as F


def _assert_valid_input_type(s):
    assert s == "mulaw-quantize" or s == "mulaw" or s == "raw"


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == "mulaw-quantize"


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == "mulaw"


def is_raw(s):
    _assert_valid_input_type(s)
    return s == "raw"


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)


def zero_pad(input, left=0, right=0):
    left = max(0, left)
    right = max(0, right)
    return F.pad(input, (left, right), "constant", 0)


def window(waveform, head, lookback, dims=1):
    cut = waveform[max(0, head - lookback): head] if dims == 1 else waveform[:, max(0, head - lookback): head]
    padded = zero_pad(cut, left=lookback - head)
    return torch.unsqueeze(padded, 0)
