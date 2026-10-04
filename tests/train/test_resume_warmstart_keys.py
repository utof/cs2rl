"""Checkpoint key migration must preserve the next warmstart entropy update.

Use real three-file checkpoints and CPU harness trainers. Env RNG is not saved,
so continuation compares schedules and alpha optimizer progress, not PPO losses.
"""
from pathlib import Path

import pytest
import torch

from cs2rl.train.resume import load_full_resume, resolve_resume_run
from tests.train.test_resume_state import _make

# Literal old-format fixture, independent of the production migration table.
_WARMSTART_KEYS = (
    ("_batch1_warmstart_phase", "_warmstart_phase"),
    ("_batch1_last_entropy_mean", "_last_entropy_mean"),
    ("_batch1_log_alpha_reset_done", "_log_alpha_reset_done"),
    ("_batch1_current_target_entropy", "_current_target_entropy"),
    ("_batch1_warmstart_h_anchor", "_warmstart_h_anchor"),
    ("_batch1_warmstart_h0", "_warmstart_h0"),
    ("_batch1_warmstart_warn_epoch", "_warmstart_warn_epoch"),
)


@pytest.mark.training
@pytest.mark.parametrize("layout", ["old", "current", "old-first", "current-first"])
def test_disk_resume_preserves_warmstart_update_and_saves_current_keys(layout):
    """Missing migration loses the ramp anchor or repeats the one-shot alpha reset.

    Mixed-key files must prefer current values in either insertion order. A
    save after old-format load must emit current keys and keep tensors aliased
    to the optimizer that subsequently steps them.
    """
    control, manager, cleanup = _make()
    resumed = None
    cleanup_resumed = None
    try:
        control.config.update(warmstart_entropy=True,
                              warmstart_grace_steps=0,
                              warmstart_ramp_steps=4 * control.config["batch_size"],
                              entropy_target_warmup_steps=1)
        for _ in range(2):
            control.evaluate()
            control.train()
        # Distinct values detect omitted restores, including a reset flag whose
        # omission would overwrite the saved alpha during the next update.
        with torch.no_grad():
            control._log_alpha_tensor.fill_(-1.37)
        control._warmstart_warn_epoch = 7
        expected = {current: getattr(control, current) for _, current in _WARMSTART_KEYS}
        assert expected["_warmstart_phase"] == 1                                             # RAMP, with a captured anchor
        assert expected["_warmstart_h_anchor"] is not None
        control.save_checkpoint()
        paths = resolve_resume_run(Path(control.config["data_dir"]), run_id="rid-test")
        state = torch.load(paths["train_state_path"], weights_only=False)
        current = state["warmstart"]
        legacy = {old: current[new] for old, new in _WARMSTART_KEYS}
        if layout == "old":
            state["warmstart"] = legacy
        elif layout != "current":
                                                                                             # Every stale legacy value disagrees with the corresponding current
                                                                                             # one. The current keys win even if the legacy key is visited last.
            stale = {old: None for old, _ in _WARMSTART_KEYS}
            state["warmstart"] = (dict(stale, **current) if layout == "old-first" else dict(
                current, **stale))
        torch.save(state, paths["train_state_path"])

        resumed, manager_resumed, cleanup_resumed = _make(seed=99)
        resumed.config.update({
            key: control.config[key]
            for key in ("warmstart_entropy", "warmstart_grace_steps", "warmstart_ramp_steps",
                        "entropy_target_warmup_steps")
        })
        alpha = resumed._log_alpha_tensor
        return_tensors = (resumed._ret_mean, resumed._ret_var, resumed._ret_count)
        # The CLI loads policy weights before load_full_resume restores the
        # optimizer/counters/sidecar; mirror that ordering on this real set.
        resumed.policy.load_state_dict(torch.load(paths["model_path"], weights_only=True))
        load_full_resume(resumed, manager_resumed, paths)
        assert {new: getattr(resumed, new) for _, new in _WARMSTART_KEYS} == expected
        assert all(not hasattr(resumed, old) for old, _ in _WARMSTART_KEYS)
        assert resumed._log_alpha_tensor is alpha
        assert resumed._alpha_optimizer.param_groups[0]["params"][0] is alpha
        for attr, tensor, key in zip(("_ret_mean", "_ret_var", "_ret_count"),
                                     return_tensors, ("ret_mean", "ret_var", "ret_count"),
                                     strict=True):
            assert getattr(resumed, attr) is tensor
            assert torch.equal(tensor, state[key])
        assert torch.equal(alpha.detach(), state["log_alpha"])
        saved_alpha_state = state["alpha_optimizer"]["state"][0]
        for key, value in saved_alpha_state.items():
            assert torch.equal(resumed._alpha_optimizer.state[alpha][key], value)
        alpha_step = resumed._alpha_optimizer.state[alpha]["step"].item()

        control.evaluate()
        control.train()
        resumed.evaluate()
        resumed.train()
        assert (resumed.global_step, resumed.epoch,
                resumed._warmstart_phase, resumed._current_target_entropy,
                resumed.scheduler.get_last_lr()) == (control.global_step, control.epoch,
                                                     control._warmstart_phase,
                                                     control._current_target_entropy,
                                                     control.scheduler.get_last_lr())
        assert resumed._alpha_optimizer.state[alpha]["step"].item() > alpha_step
        assert resumed._log_alpha_tensor is alpha
        assert alpha.item() == pytest.approx(-1.37, abs=0.1)
        saved_model = Path(resumed.save_checkpoint())
        saved = torch.load(saved_model.parent / "train_state.pt", weights_only=False)["warmstart"]
        assert set(saved) == {new for _, new in _WARMSTART_KEYS}
        assert all(saved[new] == getattr(resumed, new) for _, new in _WARMSTART_KEYS)
    finally:
        if cleanup_resumed is not None:
            cleanup_resumed()
        cleanup()
