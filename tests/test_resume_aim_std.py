"""gh#91 regression tests — aim_log_std re-init when resuming a BC checkpoint.

Background (root-caused 2026-08-01): BC training freezes ``aim_log_std`` at
LOG_STD_INIT = log(0.1) (spec D-6 detach) while fitting strongly
obs-dependent aim means. Resuming PPO from such a checkpoint puts a
near-deterministic Gaussian under the continuous-KL microscope: ONE lr=3e-4
Adam step moves mu by a full sigma, continuous approx_kl hits ~1.4 vs the
0.03 target, and the KL early-stop throttles every update to ~1 minibatch
for ~85 epochs. Widening sigma to 0.3 at resume drops the per-step KL ~9x.

Contract pinned here (train.reinit_frozen_aim_log_std):
  * a state_dict whose aim_log_std is (still) exactly LOG_STD_INIT — the
    BC-frozen signature — gets it re-initialized to AIM_LOG_STD_RESUME_INIT
    = log(0.3), and the function reports True;
  * a state_dict whose aim_log_std has moved (any RL-trained checkpoint)
    is left byte-identical, and the function reports False;
  * nothing else in the state_dict is touched either way.
"""

import math

import torch


def test_frozen_aim_log_std_is_reinitialized():
    from cs2rl.train import AIM_LOG_STD_RESUME_INIT, LOG_STD_INIT, reinit_frozen_aim_log_std

    sd = {
        "aim_log_std": torch.full((2, ), LOG_STD_INIT),
        "some.other.weight": torch.randn(3, 3),
    }
    other_before = sd["some.other.weight"].clone()

    changed = reinit_frozen_aim_log_std(sd)

    assert changed is True
    assert torch.allclose(sd["aim_log_std"], torch.full((2, ),
                                                        AIM_LOG_STD_RESUME_INIT)), sd["aim_log_std"]
    assert math.isclose(AIM_LOG_STD_RESUME_INIT, math.log(0.3), rel_tol=1e-12)
    assert torch.equal(sd["some.other.weight"],
                       other_before), ("reinit must not touch unrelated tensors")


def test_trained_aim_log_std_is_left_alone():
    from cs2rl.train import LOG_STD_INIT, reinit_frozen_aim_log_std

    # An RL-trained checkpoint: sigma has moved off the init (e.g. the 30M
    # validation run ended around log_std_pitch ~ -0.98). Must be untouched.
    trained = torch.tensor([-0.98, -1.7])
    sd = {"aim_log_std": trained.clone()}

    changed = reinit_frozen_aim_log_std(sd)

    assert changed is False
    assert torch.equal(sd["aim_log_std"], trained)
    # Guard the guard: the test above is only meaningful if the trained
    # values genuinely differ from the frozen signature.
    assert not torch.allclose(trained, torch.full((2, ), LOG_STD_INIT))


def test_missing_aim_log_std_is_a_noop():
    from cs2rl.train import reinit_frozen_aim_log_std

    sd = {"encoder.weight": torch.randn(4, 4)}
    assert reinit_frozen_aim_log_std(sd) is False
