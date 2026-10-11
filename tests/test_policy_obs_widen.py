"""sim-v4 warm start: a 110-wide (sim-v3) checkpoint widened onto the 120-wide obs.

What each test catches (cs2rl.policy.sim_v3_obs_columns / widen_sim_v3_encoder):
  * the column map: an old column copied to the wrong new column (enemy slot or global
    block off by one), or the map built from the wrong stride;
  * the identity: the widened policy's outputs differ from the 110-wide policy's on the old
    features, whatever the new velocity columns hold (new weights not zero, a column moved);
  * every encoder copy: the per-team encoder_t / encoder_ct widened like the shared one;
  * the guard: sim_v3_obs_columns on a layout of another shape raises instead of mapping;
  * no silent partial load: an unknown width passes through and fails in load_state_dict;
  * the load site: load_policy_from_checkpoint widens a 110-wide file and says so.
"""
import pytest
import torch

from cs2rl import policy as policy_mod
from cs2rl.env.c.cs2_env import make_env
from cs2rl.spec import obs as spec_obs

OLD = policy_mod.SIM_V3_OBS_DIM


@pytest.fixture(scope="module")
def env():
    e = make_env(seed=0)
    try:
        yield e
    finally:
        e.close()


def _velocity_columns():
    b0, st = spec_obs.OBS_BLOCKS["enemy"][0], spec_obs.OBS_ENEMY_STRIDE
    return {b0 + st * s + f for s in range(spec_obs.OBS_ENEMY_COUNT) for f in (8, 9)}


def test_sim_v3_columns_map_every_old_field_to_its_new_slot():
    cols = policy_mod.sim_v3_obs_columns()
    assert len(cols) == OLD == 110 and len(set(cols)) == OLD
    assert cols == sorted(cols), "the map must keep the old column order"
    assert cols[:56] == list(range(56)), "self and teammate blocks do not move"
    for s in range(5):                 # enemy slot s, fields 0..7
        assert cols[56 + 8 * s:56 + 8 * s + 8] == list(range(56 + 10 * s, 56 + 10 * s + 8)), s
    assert cols[96:] == list(range(106, 120)), "the global block moves by +10"
    assert set(range(spec_obs.OBS_DIM)) - set(cols) == _velocity_columns()


def _layout(self_w=28, count=5, stride=10, global_w=14):
    """A tiled OBS_BLOCKS of these widths (teammate 28 as in every layout so far)."""
    en0 = self_w + 28
    g0 = en0 + count * stride
    blocks = {
        "self": (0, self_w),
        "teammate": (self_w, en0),
        "enemy": (en0, g0),
        "global": (g0, g0 + global_w)
    }
    return blocks, count, stride


# Planted siblings of the sim-v4 layout: each changes ONE shape the column map assumes.
@pytest.mark.parametrize(
    "blocks, count, stride",
    [_layout(self_w=30),
     _layout(stride=7),
     _layout(count=6),
     _layout(global_w=16)],
    ids=["self-grew", "enemy-stride-shrank", "enemy-count-grew", "global-grew"])
def test_sim_v3_columns_refuse_a_layout_of_another_shape(monkeypatch, blocks, count, stride):
    # Positive control: _layout() with no change IS the live layout, which maps.
    assert _layout() == (spec_obs.OBS_BLOCKS, spec_obs.OBS_ENEMY_COUNT, spec_obs.OBS_ENEMY_STRIDE)
    monkeypatch.setattr(spec_obs, "OBS_BLOCKS", blocks)
    monkeypatch.setattr(spec_obs, "OBS_ENEMY_COUNT", count)
    monkeypatch.setattr(spec_obs, "OBS_ENEMY_STRIDE", stride)
    with pytest.raises(RuntimeError, match="add a mapping for the new layout"):
        policy_mod.sim_v3_obs_columns()


def _old_and_new_obs(n, seed):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn(n, OLD, generator=g)
    old[:, 24] = (torch.arange(n) % 2).float()                         # team bit: both T and CT rows
    new = torch.zeros(n, spec_obs.OBS_DIM)
    new[:, policy_mod.sim_v3_obs_columns()] = old
    vel = sorted(_velocity_columns())
    new[:, vel] = torch.randn(n, len(vel), generator=g) * 2.0          # the widened policy must ignore
    return old, new


@pytest.mark.parametrize("heads, trunk", [(False, False), (True, True)],
                         ids=["shared", "split-heads-and-trunk"])
def test_widened_policy_matches_the_sim_v3_policy_on_the_old_features(env, heads, trunk):
    torch.manual_seed(3)
    old_pol = policy_mod.build_policy(env,
                                      "cpu",
                                      obs_dim_override=OLD,
                                      tct_split_heads=heads,
                                      tct_split_trunk=trunk)
    sd, widened = policy_mod.widen_sim_v3_encoder(old_pol.state_dict())
    assert widened
    keys = [k for k in ("encoder.0.weight", "encoder_t.0.weight", "encoder_ct.0.weight") if k in sd]
    assert len(keys) == (2 if trunk else 1)
    cols = policy_mod.sim_v3_obs_columns()
    for k in keys:
        assert sd[k].shape == (old_pol.hidden_size, spec_obs.OBS_DIM)
        assert torch.equal(sd[k][:, cols], old_pol.state_dict()[k]), k
        assert not sd[k][:, sorted(_velocity_columns())].any(), f"{k}: new columns not zero"
    new_pol = policy_mod.build_policy(env, "cpu", tct_split_heads=heads, tct_split_trunk=trunk)
    policy_mod.load_state_dict_arch_checked(new_pol, sd, source="widened")
    old_pol.eval()
    new_pol.eval()
    st_old, st_new = {}, {}
    with torch.no_grad():
        for tick in range(3):          # the LSTM state carries across ticks
            x_old, x_new = _old_and_new_obs(10, seed=tick)
            out_old = old_pol.forward_eval(x_old, st_old)
            out_new = new_pol.forward_eval(x_new, st_new)
            for a, b in zip([*out_old[0], *out_old[1:]], [*out_new[0], *out_new[1:]], strict=True):
                torch.testing.assert_close(b, a, rtol=0.0, atol=1e-5)
            torch.testing.assert_close(st_new["lstm_h"], st_old["lstm_h"], rtol=0.0, atol=1e-5)


def test_widen_leaves_a_current_or_unknown_width_alone_and_load_refuses_it(env):
    cur = policy_mod.build_policy(env, "cpu").state_dict()
    same, widened = policy_mod.widen_sim_v3_encoder(cur)
    assert same is cur and not widened
    old105 = policy_mod.build_policy(env, "cpu", obs_dim_override=105).state_dict()
    same, widened = policy_mod.widen_sim_v3_encoder(old105)
    assert same is old105 and not widened
    with pytest.raises(RuntimeError, match="size mismatch"):
        policy_mod.load_state_dict_arch_checked(policy_mod.build_policy(env, "cpu"),
                                                old105,
                                                source="105")


def test_load_policy_from_checkpoint_widens_a_sim_v3_file(env, tmp_path, capsys):
    torch.manual_seed(5)
    old_pol = policy_mod.build_policy(env, "cpu", obs_dim_override=OLD)
    path = tmp_path / "sim_v3.pt"
    torch.save(old_pol.state_dict(), path)
    capsys.readouterr()
    pol = policy_mod.load_policy_from_checkpoint(path, "cpu")
    assert "[Policy] Obs widen: sim_v3.pt reads the 110-wide sim-v3 obs" in capsys.readouterr().out
    assert pol.obs_dim == spec_obs.OBS_DIM
    old_pol.eval()
    x_old, x_new = _old_and_new_obs(4, seed=9)
    with torch.no_grad():
        mu_old = old_pol.forward_eval(x_old, {})[1]
        mu_new = pol.forward_eval(x_new, {})[1]
    torch.testing.assert_close(mu_new, mu_old, rtol=0.0, atol=1e-5)


def test_select_policy_actions_native_refuses_a_width_mismatch(env):
    pol = policy_mod.build_policy(env, "cpu", obs_dim_override=OLD)
    obs = torch.zeros(10, spec_obs.OBS_DIM).numpy()
    with pytest.raises(ValueError, match="obs width 120 != policy obs_dim 110"):
        policy_mod.select_policy_actions_native(pol, obs, "cpu",
                                                policy_mod.init_policy_state(pol, "cpu"), "greedy")
