#!/usr/bin/env python
"""Export a trained PufferLib LSTM policy checkpoint to ONNX format.

Usage:
    python deploy/export_policy.py --checkpoint <path_to_.pt> [--output <path_to_.onnx>]
"""

import argparse
import json
import os
from pathlib import Path

import onnx
import torch
import torch.nn as nn


class LSTMPolicyONNXWrapper(nn.Module):
    """
    Wraps encoder + LSTM + split action heads for ONNX tracing.
    All shapes are derived from the loaded model — no hardcoded dims.
    Inputs:  obs[B, obs_dim], done[B], lstm_h[1, B, hidden], lstm_c[1, B, hidden]
    Outputs: (logits_0, ..., logits_N-1, lstm_h_out, lstm_c_out) where N = num_action_heads
    done masking: lstm_h *= (1 - done); this matches training behavior.
    """

    def __init__(
        self,
        encoder: nn.Sequential,
        lstm: nn.LSTM,
        action_heads: nn.ModuleList,
    ):
        super().__init__()
        self.encoder = encoder
        self.lstm = lstm
        self.action_heads = action_heads

    def forward(
        self,
        obs: torch.Tensor,       # [B, obs_dim]
        done: torch.Tensor,      # [B]
        lstm_h: torch.Tensor,    # [1, B, hidden]
        lstm_c: torch.Tensor,    # [1, B, hidden]
    ):
        h = self.encoder(obs.float())                           # [B, hidden]
        h_unsq = h.unsqueeze(0)                                 # [1, B, hidden]
        done_mask = (1.0 - done.float()).view(1, -1, 1)         # [1, B, 1]
        h_out, (h_new, c_new) = self.lstm(
            h_unsq, (done_mask * lstm_h, done_mask * lstm_c)
        )
        h_out = h_out.squeeze(0)                                # [B, hidden]

        logits = tuple(head(h_out) for head in self.action_heads)
        return logits + (h_new, c_new)


def build_model(state_dict: dict) -> tuple:
    """Reconstruct encoder, lstm, action_heads from a raw state_dict.

    Returns (wrapper, obs_dim, hidden_dim, action_sizes).
    """
    obs_dim = state_dict["encoder.0.weight"].shape[1]
    hidden_dim = state_dict["encoder.0.weight"].shape[0]

    num_heads = sum(
        1 for k in state_dict if k.startswith("action_heads.") and k.endswith(".weight")
    )
    action_sizes = [state_dict[f"action_heads.{i}.weight"].shape[0] for i in range(num_heads)]

    encoder = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )

    lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=False)

    action_heads = nn.ModuleList([nn.Linear(hidden_dim, sz) for sz in action_sizes])

    wrapper = LSTMPolicyONNXWrapper(encoder, lstm, action_heads)
    result = wrapper.load_state_dict(state_dict, strict=False)

    # Check for unexpected keys (excluding known value_head.*)
    unexpected = [k for k in result.unexpected_keys if not k.startswith("value_head.")]
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint (architecture mismatch?): {unexpected}")
    if result.missing_keys:
        raise RuntimeError(f"Missing keys — checkpoint does not match reconstructed model: {result.missing_keys}")

    wrapper.eval()

    return wrapper, obs_dim, hidden_dim, action_sizes


def main():
    parser = argparse.ArgumentParser(description="Export PufferLib LSTM policy to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--output",
        default="deploy/models/policy_lstm.onnx",
        help="Output path for .onnx file (default: deploy/models/policy_lstm.onnx)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    # Check checkpoint exists
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    # Validate checkpoint is a state_dict
    if not isinstance(state_dict, dict):
        raise SystemExit(
            f"Expected a state_dict (dict), got {type(state_dict).__name__}. "
            "If this is a full training checkpoint, extract the policy state_dict first."
        )

    wrapper, obs_dim, hidden_dim, action_sizes = build_model(state_dict)

    print(f"  obs_dim={obs_dim}, hidden_dim={hidden_dim}, action_sizes={action_sizes}")

    # Dummy inputs (batch=1)
    obs = torch.zeros(1, obs_dim)
    done = torch.zeros(1)
    lstm_h = torch.zeros(1, 1, hidden_dim)
    lstm_c = torch.zeros(1, 1, hidden_dim)

    input_names = ["obs", "done", "lstm_h", "lstm_c"]
    output_names = (
        [f"logits_{i}" for i in range(len(action_sizes))]
        + ["lstm_h_out", "lstm_c_out"]
    )

    dynamic_axes = {
        "obs": {0: "batch"},
        "done": {0: "batch"},
        "lstm_h": {1: "batch"},
        "lstm_c": {1: "batch"},
        "lstm_h_out": {1: "batch"},
        "lstm_c_out": {1: "batch"},
    }
    for name in output_names:
        if name.startswith("logits_"):
            dynamic_axes[name] = {0: "batch"}

    print(f"Exporting ONNX model to: {output_path}")
    torch.onnx.export(
        wrapper,
        (obs, done, lstm_h, lstm_c),
        str(output_path),
        opset_version=17,
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print("Validating ONNX model...")
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    print("  ONNX check passed.")

    sidecar_path = output_path.with_suffix(".json")
    sidecar = {
        "checkpoint": str(checkpoint_path),
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "action_sizes": action_sizes,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Sidecar JSON written: {sidecar_path}")

    print("\nModel summary:")
    print(f"  Path:    {output_path}")
    print("  Inputs:")
    for inp in model_proto.graph.input:
        shape = [
            (d.dim_param if d.dim_param else d.dim_value)
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"    {inp.name}: {shape}")
    print("  Outputs:")
    for out in model_proto.graph.output:
        shape = [
            (d.dim_param if d.dim_param else d.dim_value)
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"    {out.name}: {shape}")


if __name__ == "__main__":
    main()
