"""sim-v4 warm start at the trainer's load sites: --resume and the self-play pool.

What each test catches:
  * _resume_policy_init (cs2rl.train.loop; also the Modal runner's remote --resume) not
    widening a 110-wide checkpoint: build_trainer's strict load would raise a size mismatch;
  * the trunk warm split (--tct-split-trunk on a file without one): both per-team encoder
    copies must come out widened, each holding the file's encoder on the old columns;
  * --resume-run (full_state) widening weights whose Adam moments stay 110 wide;
  * SelfPlayManager.load_past_policy not widening: sp_seed.pt is a copy of the --resume
    checkpoint, so a fine-tune from a sim-v3 file would crash at its first past-policy epoch.
"""
from types import SimpleNamespace

import pytest
import torch

from cs2rl import policy as policy_mod
from cs2rl.env.c.cs2_env import make_env
from cs2rl.spec import obs as spec_obs
from cs2rl.train import loop as train_loop
from cs2rl.train import selfplay as train_selfplay

OLD = policy_mod.SIM_V3_OBS_DIM


@pytest.fixture(scope="module")
def env():
    e = make_env(seed=0)
    try:
        yield e
    finally:
        e.close()


def _save_sim_v3(env, path, **arch):
    torch.manual_seed(7)
    sd = policy_mod.build_policy(env, "cpu", obs_dim_override=OLD, **arch).state_dict()
    torch.save(sd, path)
    return sd


SHARED = ({}, {})
HEADS_FILE_SPLIT_TRUNK = (dict(tct_split_heads=True), dict(tct_split_trunk=True))


@pytest.mark.parametrize("arch, flags", [SHARED, HEADS_FILE_SPLIT_TRUNK],
                         ids=["shared", "heads-file-warm-split-trunk"])
def test_resume_widens_a_sim_v3_checkpoint(env, tmp_path, capsys, arch, flags):
    path = tmp_path / "f2x.pt"
    old = _save_sim_v3(env, path, **arch)
    capsys.readouterr()
    init = train_loop._resume_policy_init(SimpleNamespace(resume=str(path), **flags),
                                          full_state=False)
    assert "[Train] Obs widen: f2x.pt reads the 110-wide sim-v3 obs" in capsys.readouterr().out
    sd = init.state_dict
    assert sd is not None
    cols = policy_mod.sim_v3_obs_columns()
    keys = [k for k in ("encoder.0.weight", "encoder_t.0.weight", "encoder_ct.0.weight") if k in sd]
    assert keys == (["encoder_t.0.weight", "encoder_ct.0.weight"]
                    if flags else ["encoder.0.weight"])
    for k in keys:
        assert sd[k].shape[1] == spec_obs.OBS_DIM, k
        assert torch.equal(sd[k][:, cols], old["encoder.0.weight"]), k
    pol = policy_mod.build_policy(env,
                                  "cpu",
                                  tct_split_heads=init.tct_split_heads,
                                  tct_split_trunk=init.tct_split_trunk)
    policy_mod.load_state_dict_arch_checked(pol, sd, source=init.source)


def test_resume_of_a_current_checkpoint_does_not_widen(env, tmp_path, capsys):
    path = tmp_path / "cur.pt"
    torch.save(policy_mod.build_policy(env, "cpu").state_dict(), path)
    capsys.readouterr()
    init = train_loop._resume_policy_init(SimpleNamespace(resume=str(path)), full_state=False)
    assert "Obs widen" not in capsys.readouterr().out
    assert init.state_dict is not None
    assert init.state_dict["encoder.0.weight"].shape[1] == spec_obs.OBS_DIM


def test_full_state_resume_of_a_sim_v3_run_is_refused(env, tmp_path):
    path = tmp_path / "model_000010.pt"
    _save_sim_v3(env, path)
    with pytest.raises(SystemExit, match="--resume-run restores optimizer moments shaped for it"):
        train_loop._resume_policy_init(SimpleNamespace(resume=str(path)), full_state=True)


def test_past_policy_pool_widens_a_sim_v3_snapshot(env, tmp_path):
    path = tmp_path / "sp_seed.pt"
    _save_sim_v3(env, path)
    mgr = train_selfplay.SelfPlayManager()
    mgr.pool = [path]
    past = mgr.load_past_policy("cpu", env)
    assert past is not None and past.obs_dim == spec_obs.OBS_DIM
    w = past.state_dict()["encoder.0.weight"]
    vel = [
        spec_obs.OBS_BLOCKS["enemy"][0] + spec_obs.OBS_ENEMY_STRIDE * s + f
        for s in range(spec_obs.OBS_ENEMY_COUNT) for f in (8, 9)
    ]
    assert not w[:, vel].any(), "the velocity columns of a widened snapshot are not zero"
