#!/usr/bin/env python
"""Export a trained PufferLib LSTM policy checkpoint to ONNX format.

⚠ DEPLOY SUSPENDED 2026-05-03 ⚠ — active development paused after Batch 3.5
(sim-only training take-priority). Last-known-good OBS_VERSION=v2-105dim.
Do NOT bump the obs_version literal or extend the export surface as sim
obs/action heads evolve. ONNX I/O may not match a future sim refactor;
expect this module to need a from-scratch revisit on resume. See gh #(filed).

Usage (run from repo root; from a worktree, prefix `env PYTHONPATH=<checkout>/src`):
    python -m cs2rl.deploy.export_policy --checkpoint <path_to_.pt> [--output <path_to_.onnx>]
"""

import argparse
import json
import os
from pathlib import Path

import onnx
import torch
import torch.nn as nn


class LSTMPolicyONNXWrapper(nn.Module):
    """Wraps encoder + LSTM + categorical heads + aim head for ONNX tracing.

    All shapes are derived from the loaded model — no hardcoded dims. The
    `done` masking (`lstm_h *= (1 - done)`) matches training behavior so the
    deployed graph picks up the same recurrent reset semantics.

    Batch 3 changes (vs Batch 2):
      - constructor takes optional `aim_mu` Linear and `max_turn_speed` float.
      - forward returns *logits + (mu_aim, h_new, c_new) — 10-tuple total
        for 7 logits. mu_aim is tanh*max_turn_speed (deterministic, μ-only;
        sigma stays Python-only per spec L6).
      - Backward-compat: if aim_mu is None, behaves exactly like Batch 2
        (returns *logits + (h_new, c_new) — 9-tuple) so old checkpoints
        still export. Used during the migration window.

    Inputs:  obs[B, obs_dim], done[B], lstm_h[1, B, hidden], lstm_c[1, B, hidden]
    Outputs (Batch 3): (logits_0, ..., logits_N-1, mu_aim, lstm_h_out, lstm_c_out)
    Outputs (Batch 2): (logits_0, ..., logits_N-1, lstm_h_out, lstm_c_out)
    """

    def __init__(
        self,
        encoder: nn.Sequential,
        lstm: nn.LSTM,
        action_heads: nn.ModuleList,
                                                       # Batch 3: aim_mu is the optional Linear head reconstructed from
                                                       # `aim_mu.*` keys; max_turn_speed mirrors sd->max_turn_speed and
                                                       # is baked in as a buffer (constant in the exported graph).
        aim_mu: nn.Linear | None = None,
        max_turn_speed: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.lstm = lstm
        self.action_heads = action_heads
        self.aim_mu = aim_mu
        if aim_mu is not None and max_turn_speed is None:
            raise ValueError("max_turn_speed must be supplied when aim_mu is present (Batch 3)")
                                                       # Stored as a buffer so it travels with the model and gets exported
                                                       # as a constant in the ONNX graph (no float-input plumbing needed).
                                                       # When aim_mu is absent, no buffer is registered — the Batch 2 graph
                                                       # has no place to splice the scalar in anyway.
        if max_turn_speed is not None:
            self.register_buffer("_max_turn_speed",
                                 torch.tensor(float(max_turn_speed), dtype=torch.float32))

    def forward(
            self,
            obs: torch.Tensor,                                         # [B, obs_dim]
            done: torch.Tensor,                                        # [B]
            lstm_h: torch.Tensor,                                      # [1, B, hidden]
            lstm_c: torch.Tensor,                                      # [1, B, hidden]
    ):
        h = self.encoder(obs.float())                                  # [B, hidden]
        h_unsq = h.unsqueeze(0)                                        # [1, B, hidden]
        done_mask = (1.0 - done.float()).view(1, -1, 1)                # [1, B, 1]
        h_out, (h_new, c_new) = self.lstm(h_unsq, (done_mask * lstm_h, done_mask * lstm_c))
        h_out = h_out.squeeze(0)                                       # [B, hidden]

        logits = tuple(head(h_out) for head in self.action_heads)
        if self.aim_mu is not None:
            # mu_aim: deterministic μ-only output, range [-max_turn_speed, +max_turn_speed].
            # Sigma (aim_log_std) is intentionally NOT exported per spec L6 — sampling
            # noise is added Python-side during training, deploy is greedy μ.
            # mu_aim shape [B, AIM_DIM]; 10-tuple total at AIM_DIM=1.
            mu_aim = torch.tanh(self.aim_mu(h_out)) * self._max_turn_speed
            return logits + (mu_aim, h_new, c_new)
        # Backward-compat path (Batch 2 checkpoints): 9-tuple, no mu_aim slot.
        return logits + (h_new, c_new)


def build_model(state_dict: dict) -> tuple:
    """Reconstruct encoder, lstm, action_heads (and aim_mu when present) from
    a raw state_dict.

    Returns (wrapper, obs_dim, hidden_dim, action_sizes, aim_dim).

    Batch 3: `aim_dim` is 0 for legacy Batch 2 checkpoints (no `aim_mu.*`
    keys) and equals the aim-head output width otherwise. The wrapper falls
    back to the Batch 2 9-output graph in the former case.
    """
    obs_dim = state_dict["encoder.0.weight"].shape[1]
    hidden_dim = state_dict["encoder.0.weight"].shape[0]

    num_heads = sum(1 for k in state_dict
                    if k.startswith("action_heads.") and k.endswith(".weight"))
    action_sizes = [state_dict[f"action_heads.{i}.weight"].shape[0] for i in range(num_heads)]

    encoder = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )

    # num_layers=1: PufferLib trains single-layer LSTMs; multi-layer checkpoints will
    # hit missing_keys below and raise. Change only if training config changes.
    lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=False)

    action_heads = nn.ModuleList([nn.Linear(hidden_dim, sz) for sz in action_sizes])

    # Batch 3: aim_mu Linear (single output dim AIM_DIM, default 1).
    # Detect presence by scanning the state_dict; absence is allowed
    # (backward-compat with Batch 2 checkpoints).
    aim_mu = None
    max_turn_speed = None
    if "aim_mu.weight" in state_dict:
        aim_dim = state_dict["aim_mu.weight"].shape[0]
        aim_mu = nn.Linear(hidden_dim, aim_dim)
        # max_turn_speed is stored as a buffer in the policy; if the checkpoint
        # has it, use that; otherwise default to π/4 (matches StaticData lock).
        # We WARN on the fallback because a Batch 3 checkpoint that lacks the
        # buffer is malformed — the buffer should always be saved by the
        # training loop. Silent fallback could mask a real bug (e.g. a future
        # training change that drops the buffer without updating this loader).
        # Batch 2 checkpoints never reach this branch (they lack aim_mu.weight)
        # so the warning only fires on actual Batch 3 misconfigs.
        if "max_turn_speed" in state_dict:
            max_turn_speed = float(state_dict["max_turn_speed"].item())
        else:
            import math
            import warnings
            max_turn_speed = math.pi / 4.0
            warnings.warn(
                "Batch 3 checkpoint (has aim_mu.weight) is missing the "
                "`max_turn_speed` buffer. Falling back to π/4. If the policy "
                "was trained with a different max_turn_speed, the exported "
                "ONNX will scale mu_aim incorrectly. Re-train or set the "
                "buffer manually before export.",
                RuntimeWarning,
                stacklevel=2,
            )
    # else: leave aim_mu=None — Batch 2 checkpoint, fall back to old graph.

    # Detect multi-layer LSTM checkpoints early with a clear error
    lstm_layer_keys = [k for k in state_dict if k.startswith("lstm.") and "l1" in k]
    if lstm_layer_keys:
        raise ValueError(f"Multi-layer LSTM detected in checkpoint (keys: {lstm_layer_keys[:3]}). "
                         "export_policy.py assumes num_layers=1. Update the LSTM reconstruction if "
                         "training config changed.")

    wrapper = LSTMPolicyONNXWrapper(encoder,
                                    lstm,
                                    action_heads,
                                    aim_mu=aim_mu,
                                    max_turn_speed=max_turn_speed)
    result = wrapper.load_state_dict(state_dict, strict=False)

    # Allow `aim_log_std` (state-independent param, not exported) and
    # `max_turn_speed` (buffer, set above via register_buffer in the wrapper)
    # as expected unexpected keys when aim_mu is present.
    unexpected = [
        k for k in result.unexpected_keys
        if not k.startswith("value_head.") and k not in ("aim_log_std", "max_turn_speed")
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint (architecture mismatch?): {unexpected}")
    if result.missing_keys:
        # aim_mu is optional; missing aim_mu.* is only an error if we expected it.
        # The wrapper's own `_max_turn_speed` buffer is hand-populated via
        # register_buffer above (the underscore prevents a collision with the
        # checkpoint's top-level `max_turn_speed` key, which we read out by
        # value, not via load_state_dict). Strict-load therefore reports it as
        # missing on every Batch 3 export — filter it explicitly.
        missing_unexpected = [
            k for k in result.missing_keys
            if not (aim_mu is None and k.startswith("aim_mu.")) and k != "_max_turn_speed"
        ]
        if missing_unexpected:
            raise RuntimeError("Missing keys — checkpoint does not match "
                               f"reconstructed model: {missing_unexpected}")

    wrapper.eval()

    aim_dim = aim_mu.weight.shape[0] if aim_mu is not None else 0
    return wrapper, obs_dim, hidden_dim, action_sizes, aim_dim


def main():
    """CLI entry point — parse args, call load_and_wrap_policy, export ONNX + JSON sidecar."""
    parser = argparse.ArgumentParser(prog="python -m cs2rl.deploy.export_policy",
                                     description="Export PufferLib LSTM policy to ONNX")
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
            "If this is a full training checkpoint, extract the policy state_dict first.")

    wrapper, obs_dim, hidden_dim, action_sizes, aim_dim = build_model(state_dict)

    print(f"  obs_dim={obs_dim}, hidden_dim={hidden_dim}, "
          f"action_sizes={action_sizes}, aim_dim={aim_dim}")

    # Dummy inputs (batch=1)
    obs = torch.zeros(1, obs_dim)
    done = torch.zeros(1)
    lstm_h = torch.zeros(1, 1, hidden_dim)
    lstm_c = torch.zeros(1, 1, hidden_dim)

    input_names = ["obs", "done", "lstm_h", "lstm_c"]
    if aim_dim > 0:
        # Batch 3+ contract: discrete logits, then mu_aim, then LSTM state.
        output_names = ([f"logits_{i}" for i in range(len(action_sizes))] +
                        ["mu_aim", "lstm_h_out", "lstm_c_out"])
    else:
        # Batch 2 backward-compat: discrete logits then LSTM state, no aim.
        output_names = ([f"logits_{i}"
                         for i in range(len(action_sizes))] + ["lstm_h_out", "lstm_c_out"])

    dynamic_axes = {
        "obs": {
            0: "batch"
        },
        "done": {
            0: "batch"
        },
        "lstm_h": {
            1: "batch"
        },
        "lstm_c": {
            1: "batch"
        },
        "lstm_h_out": {
            1: "batch"
        },
        "lstm_c_out": {
            1: "batch"
        },
    }
    for name in output_names:
        if name.startswith("logits_") or name == "mu_aim":
            dynamic_axes[name] = {0: "batch"}

    print(f"Exporting ONNX model to: {output_path}")
    torch.onnx.export(
        wrapper,
        (obs, done, lstm_h, lstm_c),
        str(output_path),
                                       # opset 17: required for OnnxRuntime 1.16+ and dynamic shapes
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
    # obs_version tags which observation schema was used at training time.
    # The C# plugin reads this to validate it loaded the correct mapdata JSON.
    # Must match OBS_VERSION in cs2rl/deploy/export_mapdata.py and the C# plugin constant.
    # Batch 3: bumped obs_version to v1-105dim to match the new role-bit
    # observation; aim_dim records whether this export carries the aim head
    # (0 = legacy Batch 2 graph, 1 = single-axis aim, 2+ = future Batch 3.5).
    # Batch 5 (map-verticality T5): bumped v1→v2 to signal centroids_z is now
    # present in the mapdata sidecar (spec §2 L4 / OBS_VERSION discovery row).
    sidecar = {
        "checkpoint": str(checkpoint_path),
        "obs_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "action_sizes": action_sizes,
        "aim_dim": aim_dim,
        "obs_version": "v2-105dim",
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Sidecar JSON written: {sidecar_path}")

    print("\nModel summary:")
    print(f"  Path:    {output_path}")
    print("  Inputs:")
    for inp in model_proto.graph.input:
        shape = [(d.dim_param if d.dim_param else d.dim_value)
                 for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    print("  Outputs:")
    for out in model_proto.graph.output:
        shape = [(d.dim_param if d.dim_param else d.dim_value)
                 for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")


if __name__ == "__main__":
    main()
