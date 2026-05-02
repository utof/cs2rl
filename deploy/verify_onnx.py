#!/usr/bin/env python
"""Verify ONNX export matches PyTorch reference model output.

Usage:
    python deploy/verify_onnx.py [--onnx deploy/models/policy_lstm.onnx] [--steps 100]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_policy import build_model


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX export against PyTorch reference")
    parser.add_argument(
        "--onnx",
        default="deploy/models/policy_lstm.onnx",
        help="Path to .onnx file (default: deploy/models/policy_lstm.onnx)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps to run (default: 100)",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise SystemExit(f"ONNX file not found: {onnx_path}")

    sidecar_path = onnx_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise SystemExit(f"Sidecar JSON not found: {sidecar_path}")

    # Load sidecar
    sidecar = json.loads(sidecar_path.read_text())
    checkpoint_path = Path(sidecar["checkpoint"])
    hidden_dim = sidecar["hidden_dim"]
    action_sizes = sidecar["action_sizes"]
    num_heads = len(action_sizes)
    # Batch 3: aim_dim is missing on legacy sidecars (treat as 0). When >0, the
    # ONNX graph emits an extra `mu_aim` slot between the last logit and the
    # LSTM state pair — see deploy/export_policy.py:LSTMPolicyONNXWrapper.forward.
    aim_dim = sidecar.get("aim_dim", 0)

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    # Load ORT session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Read obs_dim from session input metadata (index 0 = "obs") and cross-check sidecar
    obs_dim_from_session = session.get_inputs()[0].shape[1]
    obs_dim_from_sidecar = sidecar["obs_dim"]
    if obs_dim_from_session != obs_dim_from_sidecar:
        raise SystemExit(f"obs_dim mismatch: ONNX graph says {obs_dim_from_session}, "
                         f"sidecar JSON says {obs_dim_from_sidecar}. Re-export the model.")
    obs_dim = obs_dim_from_session

    # Validate output names match the deploy contract for this aim_dim.
    out_names = [o.name for o in session.get_outputs()]
    if aim_dim > 0:
        # Batch 3+: logits_0..N-1, mu_aim, lstm_h_out, lstm_c_out
        expected_names = ([f"logits_{i}"
                           for i in range(num_heads)] + ["mu_aim", "lstm_h_out", "lstm_c_out"])
    else:
        # Batch 2 backward-compat: logits_0..N-1, lstm_h_out, lstm_c_out
        expected_names = ([f"logits_{i}" for i in range(num_heads)] + ["lstm_h_out", "lstm_c_out"])
    if out_names != expected_names:
        raise SystemExit(f"ONNX output names {out_names} != expected {expected_names}. "
                         "Re-export the model.")

    print(f"ONNX:        {onnx_path}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"obs_dim={obs_dim}, hidden_dim={hidden_dim}, "
          f"action_sizes={action_sizes}, aim_dim={aim_dim}")

    # Load PyTorch wrapper. build_model returns 5-tuple (Batch 3+); the trailing
    # aim_dim is unused here because the sidecar is the source of truth for
    # this script's structural assertions.
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    wrapper, _, _, _, _ = build_model(state_dict)
    wrapper.eval()

    # Batch 3 range check: mu_aim must lie in [-max_turn_speed, +max_turn_speed].
    # We default to π/4 (StaticData lock); the actual buffer in the wrapper is
    # the source of truth for the PyTorch reference path.
    max_turn_speed = float(np.pi / 4.0)
    if aim_dim > 0 and hasattr(wrapper, "_max_turn_speed"):
        max_turn_speed = float(wrapper._max_turn_speed.item())

    # Initialize LSTM states
    ort_h = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    ort_c = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    pt_h = torch.zeros(1, 1, hidden_dim)
    pt_c = torch.zeros(1, 1, hidden_dim)

    passed = 0
    failed = 0
    done_steps = {10, 50}

    np.random.seed(42)

    for step in range(args.steps):
        obs_np = np.random.randn(1, obs_dim).astype(np.float32)
        done_val = 1.0 if step in done_steps else 0.0
        done_np = np.array([done_val], dtype=np.float32)

        # ORT inference
        ort_outputs = session.run(
            None,
            {
                "obs": obs_np,
                "done": done_np,
                "lstm_h": ort_h,
                "lstm_c": ort_c,
            },
        )
        # ort_outputs ordering depends on aim_dim:
        #   aim_dim == 0: [logits_0, ..., logits_N-1, lstm_h_out, lstm_c_out]
        #   aim_dim >  0: [logits_0, ..., logits_N-1, mu_aim, lstm_h_out, lstm_c_out]
        ort_logits = ort_outputs[:num_heads]
        if aim_dim > 0:
            ort_mu_aim = ort_outputs[num_heads]
            ort_h_new = ort_outputs[num_heads + 1]
            ort_c_new = ort_outputs[num_heads + 2]
        else:
            ort_mu_aim = None
            ort_h_new = ort_outputs[num_heads]
            ort_c_new = ort_outputs[num_heads + 1]

        # PyTorch inference
        pt_done = torch.tensor([done_val], dtype=torch.float32)
        with torch.no_grad():
            pt_outputs = wrapper(torch.from_numpy(obs_np), pt_done, pt_h, pt_c)
        # pt_outputs mirrors the ORT ordering above.
        pt_logits = [pt_outputs[i].numpy() for i in range(num_heads)]
        if aim_dim > 0:
            pt_mu_aim = pt_outputs[num_heads].numpy()
            pt_h_new = pt_outputs[num_heads + 1].numpy()
            pt_c_new = pt_outputs[num_heads + 2].numpy()
        else:
            pt_mu_aim = None
            pt_h_new = pt_outputs[num_heads].numpy()
            pt_c_new = pt_outputs[num_heads + 1].numpy()

        # Compare all outputs
        step_max_diff = 0.0
        step_pass = True

        for i in range(num_heads):
            diff = float(np.max(np.abs(pt_logits[i] - ort_logits[i])))
            step_max_diff = max(step_max_diff, diff)
            if diff >= 1e-4:
                step_pass = False

        if aim_dim > 0:
            mu_diff = float(np.max(np.abs(pt_mu_aim - ort_mu_aim)))
            step_max_diff = max(step_max_diff, mu_diff)
            if mu_diff >= 1e-4:
                step_pass = False
            # Range check: tanh*max_turn_speed must stay inside the
            # locked interval. 1e-5 slack absorbs onnx fp32 wobble.
            if float(np.abs(ort_mu_aim).max()) > max_turn_speed + 1e-5:
                step_pass = False

        h_diff = float(np.max(np.abs(pt_h_new - ort_h_new)))
        c_diff = float(np.max(np.abs(pt_c_new - ort_c_new)))
        step_max_diff = max(step_max_diff, h_diff, c_diff)
        if h_diff >= 1e-4 or c_diff >= 1e-4:
            step_pass = False

        # Propagate LSTM state. The 5-tuple/4-tuple slot indices for the
        # PyTorch path mirror the ORT layout selected above.
        ort_h = ort_h_new
        ort_c = ort_c_new
        if aim_dim > 0:
            pt_h = pt_outputs[num_heads + 1]
            pt_c = pt_outputs[num_heads + 2]
        else:
            pt_h = pt_outputs[num_heads]
            pt_c = pt_outputs[num_heads + 1]

        if step_pass:
            passed += 1
            print(f"step {step:3d}: PASS (max_diff={step_max_diff:.1e})")
        else:
            failed += 1
            print(f"step {step:3d}: FAIL (max_diff={step_max_diff:.1e})")

    print()
    if failed == 0:
        print(f"All {args.steps} steps passed.")
        sys.exit(0)
    else:
        print(f"{failed}/{args.steps} steps FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
