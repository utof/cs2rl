"""#420: --resume-optimizer loads Adam's moments for a weights-only --resume.

Pure torch, no trainer: a two-group Adam (like the run's σ split) trains a few steps
under a cosine schedule that anneals its lr to 0, and its checkpoint set is written the
way Cs2PuffeRL.save_checkpoint names it. A fresh optimizer then takes the moments.
"""
import pytest
import torch

from cs2rl.train.resume import load_optimizer_moments


def _model_and_adam(lr):
    model = torch.nn.Linear(4, 3)
    adam = torch.optim.Adam([{
        "params": [model.weight]
    }, {
        "params": [model.bias],
        "weight_decay": 0.0
    }],
                            lr=lr)
    return model, adam


@pytest.fixture
def source_set(tmp_path):
    """A trained (model, adam) whose lr has annealed to 0, saved as a checkpoint set."""
    torch.manual_seed(0)
    model, adam = _model_and_adam(3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=5)
    for _ in range(5):
        adam.zero_grad()
        model(torch.randn(8, 4)).pow(2).sum().backward()
        adam.step()
        sched.step()
    torch.save(model.state_dict(), tmp_path / "model_000005.pt")
    torch.save({
        "model_name": "model_000005.pt",
        "optimizer_state_dict": adam.state_dict()
    }, tmp_path / "trainer_state.pt")
    return model, adam, tmp_path


def test_moments_load_and_this_runs_lr_stays(source_set):
    """The moments arrive; the source's annealed lr (0.0) does not replace this run's."""
    src_model, src_adam, d = source_set
    assert [g["lr"] for g in src_adam.param_groups] == [0.0, 0.0]
    model, adam = _model_and_adam(1e-4)
    model.load_state_dict(src_model.state_dict())
    info = load_optimizer_moments(adam, d, d / "model_000005.pt")
    assert info["params"] == 2 and info["step"] == 5.0
    assert torch.equal(adam.state[model.weight]["exp_avg_sq"],
                       src_adam.state[src_model.weight]["exp_avg_sq"])
    assert [g["lr"] for g in adam.param_groups] == [1e-4, 1e-4]


def test_moments_of_other_weights_are_refused(source_set):
    """The moments belong to the model trainer_state.pt names; other weights are refused."""
    src_model, _, d = source_set
    torch.save({k: v + 1 for k, v in src_model.state_dict().items()}, d / "other.pt")
    _, adam = _model_and_adam(1e-4)
    with pytest.raises(SystemExit, match="is not"):
        load_optimizer_moments(adam, d / "trainer_state.pt", d / "other.pt")
