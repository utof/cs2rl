"""Atomic checkpoint save — crash-safety pin (2026-08-13).

WHY this exists: the user's GPU intermittently overheats and falls off the
PCI bus, hard-crashing the machine mid-run. The periodic save in train()
overwrites the SAME file (dust2_policy.pt) every --save_every_sec, so a
crash landing inside torch.save() would corrupt the only recovery point.
_atomic_save_state_dict writes to a sibling .tmp and os.replace()s it into
place — POSIX rename is atomic, so the destination is always either the old
complete checkpoint or the new complete checkpoint, never a torn write.

PITFALL: do NOT "simplify" the helper back to a bare torch.save(path).
Every periodic/final save site in train() must go through the helper.
"""

import pytest
import torch


def test_atomic_save_roundtrip_and_no_tmp_left(tmp_path):
    """Saved dict loads back intact; no .tmp sibling survives."""
    from cs2rl.train.resume import _atomic_save_state_dict

    path = tmp_path / "policy.pt"
    state = {"w": torch.tensor([1.0, 2.0, 3.0])}
    _atomic_save_state_dict(state, path)

    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["w"], state["w"])
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], f"tmp file leaked: {leftovers}"


def test_atomic_save_preserves_old_checkpoint_on_crash(tmp_path, monkeypatch):
    """A crash mid-torch.save must leave the previous checkpoint readable.

    Simulates the power-loss/bus-drop by making torch.save die after
    opening the tmp file — the destination must still hold the OLD state.
    """
    from cs2rl.train.resume import _atomic_save_state_dict

    path = tmp_path / "policy.pt"
    old = {"w": torch.tensor([1.0])}
    _atomic_save_state_dict(old, path)

    def exploding_save(obj, f, *a, **kw):
        # Touch the tmp target (partial write), then die like a bus drop.
        with open(f, "wb") as fh:
            fh.write(b"garbage")
        raise RuntimeError("simulated GPU fell off the bus")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError):
        _atomic_save_state_dict({"w": torch.tensor([2.0])}, path)

    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["w"], old["w"]), "old checkpoint was clobbered"
