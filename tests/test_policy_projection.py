"""Public projection paths must agree with legacy action-density evaluation."""

import pytest
import torch

from cs2rl.env.c.cs2_env import make_env
from cs2rl.policy import LOG_STD_MIN, _hybrid_sample_logits, build_policy
from cs2rl.spec.obs import OBS_DIM


@pytest.fixture(scope="module")
def env():
    """Use the real observation and turn-speed contract for the policy."""
    instance = make_env(seed=0)
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("split_heads,split_trunk", [(False, False), (True, False), (False, True),
                                                     (True, True)])
@pytest.mark.parametrize("teams", [(1., 1.), (0., 0.), (1., 0.)])
def test_projection_entrypoints_preserve_action_density_and_gradients(env, split_heads, split_trunk,
                                                                      teams):
    """Routing or clamp drift must not change PPO's reevaluated action density.

    The legacy distribution-based sampler is an independent public path. Use
    supplied actions so this checks density and gradients without sampling
    noise; opposing cap violations exercise both saturation bounds.
    """
    torch.manual_seed(12)
    policy = build_policy(env,
                          "cpu",
                          tct_split_heads=split_heads,
                          tct_split_trunk=split_trunk,
                          aim_log_std_max=-1.5,
                          pin_pitch=True)
    with torch.no_grad():
        if split_heads:
            policy.aim_log_std_t.copy_(torch.tensor([LOG_STD_MIN - 1, -2.]))
            policy.aim_log_std_ct.copy_(torch.tensor([0., -3.]))
        else:
            policy.aim_log_std.copy_(torch.tensor([0., LOG_STD_MIN - 1]))
    obs = torch.randn(2, OBS_DIM)
    obs[:, 24] = torch.tensor(teams)
    actions = torch.zeros(2, 7, dtype=torch.long)
    aim = torch.tensor([[.04, -.01], [-.03, .02]])
    h, c = torch.randn(2, policy.hidden_size), torch.randn(2, policy.hidden_size)
    done = torch.tensor([0., 1.])
    legacy = policy.get_action_and_value(obs, (h, c), done, actions, aim)
    reference_loss = (legacy[2] + legacy[3] + legacy[4].squeeze(-1)).sum()
    reference_loss.backward()
    reference_grads = {
        name: None if p.grad is None else p.grad.clone()
        for name, p in policy.named_parameters()
    }

    for training in (False, True):
        policy.zero_grad(set_to_none=True)
        if training:
            output = policy(obs[:, None, :], {"lstm_h": h, "lstm_c": c, "terminals": done[:, None]})
        else:
            state = {"lstm_h": h.clone(), "lstm_c": c.clone(), "done": done}
            output = policy.forward_eval(obs, state)
            torch.testing.assert_close(state["lstm_h"], legacy[5][0], rtol=0, atol=0)
            torch.testing.assert_close(state["lstm_c"], legacy[5][1], rtol=0, atol=0)
        sampled = _hybrid_sample_logits(output,
                                        action=actions,
                                        continuous_action=aim,
                                        aim_dim_mask=policy.aim_dim_mask)
        log_prob, entropy = sampled[2] + sampled[3], sampled[4] + sampled[5]
        torch.testing.assert_close(log_prob, legacy[2], rtol=2e-6, atol=2e-5)
        torch.testing.assert_close(entropy, legacy[3], rtol=2e-6, atol=2e-5)
        torch.testing.assert_close(output[3], legacy[4], rtol=0, atol=0)
        assert output[2].shape == aim.shape
        (log_prob + entropy + output[3].squeeze(-1)).sum().backward()
        for name, parameter in policy.named_parameters():
            expected = reference_grads[name]
            if expected is None:
                assert parameter.grad is None, name
            else:
                actual = parameter.grad
                assert actual is not None, name
                torch.testing.assert_close(actual,
                                           expected,
                                           rtol=2e-5,
                                           atol=2e-5,
                                           msg=lambda msg, name=name: f"{name}: {msg}")
                if split_heads and len(set(teams)) == 1:
                    unused = "_ct" if teams[0] == 1 else "_t"
                    if name.startswith(
                        (f"action_heads{unused}.", f"aim_mu{unused}.", f"aim_log_std{unused}")):
                        assert torch.count_nonzero(actual) == 0, name
