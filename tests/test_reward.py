# tests/test_reward.py
import numpy as np
import pytest
import sim as sim_module
from sim import Dust2Env, ROUND_TIME


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
