# tests/test_reward.py
import numpy as np
import pytest
import sim as sim_module
from sim import Dust2Env, ROUND_TIME
from train import TRAINING_CONFIG


@pytest.fixture(autouse=True)
def reset_team_spirit():
    """Reset module-level _TEAM_SPIRIT to 0.0 before every test."""
    sim_module._TEAM_SPIRIT = 0.0
    yield
    sim_module._TEAM_SPIRIT = 0.0


def _make_env_and_state():
    env = Dust2Env()
    obs, _ = env.reset(seed=0)
    return env, env.state


def test_potential_all_alive_equal():
    """Φ(s, T) == −Φ(s, CT) when both teams have equal HP and no site presence."""
    env, gs = _make_env_and_state()
    phi_t = env._potential(gs, 0)
    phi_ct = env._potential(gs, 1)
    assert phi_t == pytest.approx(-phi_ct, abs=1e-4), (
        f"Φ(T)={phi_t:.4f} should equal -Φ(CT)={phi_ct:.4f} at balanced state"
    )


def test_potential_team_advantage_increases():
    """Φ(T) increases when a CT agent dies."""
    env, gs = _make_env_and_state()
    phi_t_before = env._potential(gs, 0)
    gs.agents[5].alive = False
    gs.agents[5].hp = 0
    phi_t_after = env._potential(gs, 0)
    assert phi_t_after > phi_t_before, (
        f"Killing a CT should increase Φ(T): {phi_t_before:.4f} -> {phi_t_after:.4f}"
    )


def test_potential_site_control():
    """Φ(T) increases when a T agent moves onto a bombsite."""
    env, gs = _make_env_and_state()
    phi_t_before = env._potential(gs, 0)
    gs.agents[0].area_id = env.a_site_areas[0]
    phi_t_after = env._potential(gs, 0)
    assert phi_t_after > phi_t_before, (
        f"T on bombsite should increase Φ(T): {phi_t_before:.4f} -> {phi_t_after:.4f}"
    )


def test_team_spirit_module_default_zero():
    """sim._TEAM_SPIRIT must be 0.0 at module load so existing training is unchanged."""
    assert sim_module._TEAM_SPIRIT == 0.0


def test_pbrs_rewards_are_finite():
    """PBRS must not produce NaN or inf over a full episode."""
    env = Dust2Env()
    obs, _ = env.reset(seed=7)
    for step_n in range(500):
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for aid, r in rewards.items():
            assert np.isfinite(r), f"Non-finite reward at step {step_n} {aid}: {r}"
        if all(terms.get(aid, False) for aid in env.possible_agents):
            break


def test_pbrs_shaping_positive_on_kill():
    """The PBRS shaping component alone is positive for T when CT is killed."""
    env = Dust2Env()
    obs, _ = env.reset(seed=3)

    # Directly verify shaping math: killing CT always increases T potential,
    # so shaping = γΦ(after) − Φ(before) must be positive.
    # We don't rely on step() to produce the kill — we manipulate state directly.
    phi_before_t = env._potential(env.state, 0)
    env.state.agents[5].alive = False
    env.state.agents[5].hp = 0
    phi_after_t = env._potential(env.state, 0)
    gamma = TRAINING_CONFIG["gamma"]
    shaping = gamma * phi_after_t - phi_before_t
    assert shaping > 0, f"Killing CT gives negative shaping: {shaping:.4f}"


def test_team_spirit_zero_unchanged():
    """At _TEAM_SPIRIT=0.0, step() rewards are identical to a reference call."""
    # Run twice with the same seed — results must be identical.
    env1 = Dust2Env()
    env1.reset(seed=42)
    actions1 = {aid: np.array([0, 0, 0, 0]) for aid in env1.agents}
    _, rewards1, _, _, _ = env1.step(actions1)

    sim_module._TEAM_SPIRIT = 0.0  # explicit (fixture already ensures this)
    env2 = Dust2Env()
    env2.reset(seed=42)
    actions2 = {aid: np.array([0, 0, 0, 0]) for aid in env2.agents}
    _, rewards2, _, _, _ = env2.step(actions2)

    for aid in env1.possible_agents:
        assert rewards1[aid] == pytest.approx(rewards2[aid], abs=1e-6), (
            f"Reward mismatch at τ=0 for {aid}: {rewards1[aid]} vs {rewards2[aid]}"
        )


def test_team_spirit_one_equalizes_alive_team():
    """At _TEAM_SPIRIT=1.0, all alive agents on same team get equal rewards."""
    sim_module._TEAM_SPIRIT = 1.0
    env = Dust2Env()
    env.reset(seed=11)
    # No-op actions — no deaths, no kills, all agents stay alive
    actions = {aid: np.array([0, 0, 0, 0]) for aid in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    t_alive = [f"t{i}" for i in range(5) if env.state.agents[i].alive]
    ct_alive = [f"ct{i}" for i in range(5) if env.state.agents[5 + i].alive]

    if len(t_alive) > 1:
        t_rewards = [rewards[aid] for aid in t_alive]
        assert all(abs(r - t_rewards[0]) < 1e-5 for r in t_rewards), (
            f"T alive rewards not equal at τ=1: {dict(zip(t_alive, t_rewards))}"
        )

    if len(ct_alive) > 1:
        ct_rewards = [rewards[aid] for aid in ct_alive]
        assert all(abs(r - ct_rewards[0]) < 1e-5 for r in ct_rewards), (
            f"CT alive rewards not equal at τ=1: {dict(zip(ct_alive, ct_rewards))}"
        )


def test_team_spirit_callback_anneals():
    """TeamSpiritCallback linearly anneals sim._TEAM_SPIRIT from 0→1."""
    from train import TeamSpiritCallback

    sim_module._TEAM_SPIRIT = 0.0
    cb = TeamSpiritCallback(anneal_steps=1_000_000)

    cb.num_timesteps = 0
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(0.0, abs=1e-6)

    cb.num_timesteps = 500_000
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(0.5, abs=1e-3)

    cb.num_timesteps = 1_000_000
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(1.0, abs=1e-3)

    cb.num_timesteps = 2_000_000
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(1.0, abs=1e-3), (
        "team_spirit must not exceed 1.0 after anneal_steps"
    )
