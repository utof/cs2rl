#!/usr/bin/env python
"""Behaviour-cloning warm-start trainer (Batch 6 Task 4 — spec D-6/D-7, §8 gate 1).

Reads the scripted-expert demonstrations written by scripts/gen_bc_demos.py,
fits the SAME policy architecture PPO uses (build_policy) to them by maximum
likelihood, writes a bare state_dict to outputs/checkpoints/bc_warmstart.pt,
and then runs the BC-only eval gate: greedy rollouts on the demo-generation
spawn set, reporting the plant rate BEFORE any RL touches the weights.

    uv run python src/train_bc.py                       # train + eval gate
    uv run python src/train_bc.py --skip-eval           # train only
    uv run python src/train_bc.py --eval-only <ckpt>    # re-run the gate

The checkpoint is consumed with zero extra code by
    uv run python -m src.train --train --resume outputs/checkpoints/bc_warmstart.pt ...
(spec D-8: `--resume` torch.load()s a bare state_dict).

WHY this shape
--------------
* **Per-tick, stateless (spec D-7).** Training shuffles individual
  (obs, action) ticks and runs the policy with NO LSTM state (`forward(x, {})`
  → T=1 from zero state). The plant task is Markovian given the obs (spec §3),
  the expert is deterministic (a sequence model would happily overfit
  "tick index → action"), and stateless matches what the PPO update itself
  does, so the warm-start cannot drift on a BC↔PPO contract mismatch.
* **Loss reuses `_hybrid_sample_logits` verbatim (spec D-6)** — the exact
  function PPO evaluates log-probs with, called in its "action supplied"
  mode. BC therefore optimises precisely the quantity PPO will later measure:
      L = mean(-(log_prob_d + log_prob_c)) - λ_ent · mean(entropy_d + entropy_c)
* **`aim_log_std` is frozen** (detached before the loss): with σ fixed at
  exp(LOG_STD_INIT)=0.1 the Gaussian NLL degenerates to an auto-scaled MSE on
  Δyaw. If σ were trainable, the fastest way to cut the loss would be to
  shrink σ toward zero, which hands PPO a policy that cannot explore. σ is
  left at its init so PPO resumes with the exploration noise it expects.

PITFALLS (each one cost time; do not "simplify" them away)
----------------------------------------------------------
* Demos are **self-identifying** — always assert OBS_DIM/ACTION_DIM/AIM_DIM,
  the map tag and the git sha against the LIVE constants before training
  (spec R7: `--resume` has no shape guard, so a stale demo set would surface
  as a raw PyTorch shape error 40 minutes into a GPU run).
* The demo set is **5 unique trajectories duplicated ×10** — spawns are
  deterministic area centroids, seeds only permute which agent gets which
  spawn (plan Task 1 RESULT). We dedupe byte-identical episodes by default;
  it changes nothing statistically (the duplication is perfectly uniform) and
  makes each epoch 10× cheaper.
* Eval obs must be masked **exactly** like the recorded obs (spec R8:
  teammate + enemy blocks zeroed) — otherwise the greedy rollout feeds the
  clone dims it never saw. `mask_idle_agent_blocks` mirrors gen_bc_demos.py;
  `tests/test_train_bc_smoke.py` pins the two together by comparing a real
  recorded obs against a live eval-rollout obs.
* `Cs2Env.reset(seed=)` **ignores** its seed (the C RNG is seeded at init) —
  per-episode determinism needs a fresh `make_env(seed=...)`.
* `env.reset()` does not fill `env.observations` (compute_observations runs
  inside step) — a priming zero-action step is required before the first obs
  read, exactly as in demo generation.
* Eval runs the SAME stateless forward as training (`forward(x, {})`), not
  `forward_eval` with carried LSTM state: a clone trained from zero state
  must be evaluated from zero state.
"""
import os

# Match src/train.py: BLAS thread caps must be set before torch/numpy spin up
# their pools, or a CPU BC run oversubscribes every core on this box.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse                                        # noqa: E402
import sys                                             # noqa: E402
from dataclasses import dataclass, field               # noqa: E402
from pathlib import Path                               # noqa: E402

import numpy as np                     # noqa: E402

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:       # allows `python src/train_bc.py`
    sys.path.insert(0, str(SRC_DIR))   # AND `import train_bc` from tests

from _action_spec import ACTION_DIM, AIM_DIM                           # noqa: E402
from _obs_spec import OBS_BLOCKS, OBS_DIM                              # noqa: E402
from map import make_simple_map                                        # noqa: E402
from nav import MAX_TURN_SPEED_RAD, N_AGENTS, ROUND_TIME, TEAM_SIZE    # noqa: E402
from paths import CHECKPOINTS_DIR                                      # noqa: E402
from scripted_expert import setup_bomb_carrier                         # noqa: E402

# The map tag every demo must carry. Demos are only meaningful for BC → PPO on
# the map they were generated on (geometry + per-map obs normalisation), so a
# mismatch is a hard error rather than a warning. Mirrors gen_bc_demos.MAP_NAME.
EXPECTED_MAP = "simple_v1"

DEFAULT_DEMO_DIR = SRC_DIR.parent / "outputs" / "demos"
DEFAULT_CHECKPOINT = Path(CHECKPOINTS_DIR) / "bc_warmstart.pt"

# λ_ent from spec D-6 (imitation-lib default). Small on purpose: it only keeps
# the heads from collapsing to one-hot certainty on a 365-tick dataset, it is
# not meant to compete with the likelihood term.
DEFAULT_ENTROPY_COEF = 1e-3

# Epoch count is MEASURED, not guessed (2026-08-01, this box, CPU, lr 3e-4,
# batch 64, the 5-unique-trajectory demo set): 300 epochs reaches nll_d≈0.03
# and a 0.800 greedy plant rate — 4 of the 5 routes — while 900 epochs reaches
# nll_d≈0.02 and 1.000. The last route needs the extra fit because greedy
# rollout compounds error off-route (spec review F6): a slightly-wrong Δyaw
# early puts the clone in states the expert never visited. ~3.5 min on CPU.
DEFAULT_EPOCHS = 900

# Spec §8 gate 1 asks for a "non-trivial" plant rate on the demo distribution
# before any RL; it does not name a number. 0.5 is our operationalisation:
# the cold-start baseline is 0.000, the expert is 1.000, and a clone that
# plants on the majority of the 5 demonstrated start states has demonstrably
# learned "walk to the site and press USE" rather than a lucky single route.
# NEVER loosen this to make a run pass — a failing gate means BC is broken
# (spec §8: "fix before Task 5").
DEFAULT_GATE_PLANT_RATE = 0.5


def mask_idle_agent_blocks(obs: np.ndarray) -> np.ndarray:
    """Zero the teammate + enemy obs blocks in-place and return `obs`.

    Spec R8. During demo generation the other 9 agents are frozen at spawn, so
    those ~68 dims describe a static world that never occurs at RL time; the
    generator zeroes them and so must every consumer that feeds the clone an
    obs, or eval/rollout obs land off-distribution.

    Boundaries come from _obs_spec.OBS_BLOCKS (generated from cs2_types.h) —
    hardcoding 28/56/96 here would silently check the wrong slots after the
    next layout bump (they were 25/53/93 before Task 2.5). MUST stay identical
    to scripts/gen_bc_demos.py's mask; tests/test_train_bc_smoke.py pins them
    together against a real recorded demo.
    """
    obs[..., slice(*OBS_BLOCKS["teammate"])] = 0.0
    obs[..., slice(*OBS_BLOCKS["enemy"])] = 0.0
    return obs


@dataclass
class DemoSet:
    """Flat per-tick BC dataset (spec D-7) plus the provenance of its episodes.

    obs/discrete/continuous are concatenated across episodes; `dones` is kept
    only for diagnostics — per-tick BC never crosses episode boundaries
    because it does not use time at all.
    """
    obs: np.ndarray                                    # float32 [N, OBS_DIM]
    discrete: np.ndarray                               # int64   [N, ACTION_DIM]
    continuous: np.ndarray                             # float32 [N, AIM_DIM]
    dones: np.ndarray                                  # bool    [N]
    episodes: list = field(default_factory=list)       # per-kept-episode metadata dicts
    n_files: int = 0                                   # files on disk (pre-dedupe)
    n_duplicates: int = 0                              # byte-identical episodes dropped

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def spawn_areas(self) -> list:
        return sorted({int(e["spawn_area"]) for e in self.episodes})


def load_demos(demo_dir, dedupe: bool = True, expected_map: str = EXPECTED_MAP) -> DemoSet:
    """Load every .npz in `demo_dir`, assert its schema against the LIVE
    constants, and concatenate into one flat per-tick dataset.

    The schema assert is the whole reason demos are self-identifying (spec §7):
    obs layout is a moving target (107→110 happened *inside* this batch), and
    dims alone do not catch semantic slot changes — hence the git_sha + map tag
    checks too. A mismatch raises here, at second zero, instead of surfacing as
    a shape error inside a long `--resume` run (spec R7).

    `dedupe` drops episodes whose (obs, discrete, continuous) bytes exactly
    match one already loaded. Spawns are deterministic centroids, so a
    10-seed × 5-carrier set is 5 unique trajectories repeated 10× each
    (plan Task 1 RESULT); dropping exact duplicates is statistically a no-op
    (the duplication is uniform across the 5 routes) and cuts epoch cost 10×.
    Only BYTE-identical episodes are dropped — anything that differs, however
    slightly, is real data and is kept.
    """
    demo_dir = Path(demo_dir)
    files = sorted(demo_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz demos in {demo_dir} — generate them first: "
                                f"uv run python scripts/gen_bc_demos.py --seeds 10")

    obs_parts, disc_parts, cont_parts, done_parts = [], [], [], []
    episodes, seen = [], {}
    n_duplicates = 0
    for path in files:
        d = np.load(path)
        # ── schema vs live constants (spec §7/R7) ──
        if int(d["OBS_DIM"]) != OBS_DIM or int(d["ACTION_DIM"]) != ACTION_DIM or int(
                d["AIM_DIM"]) != AIM_DIM:
            raise ValueError(
                f"{path.name}: demo schema (OBS_DIM={int(d['OBS_DIM'])}, "
                f"ACTION_DIM={int(d['ACTION_DIM'])}, AIM_DIM={int(d['AIM_DIM'])}) != live "
                f"({OBS_DIM}, {ACTION_DIM}, {AIM_DIM}). The obs/action layout changed since "
                f"generation — regenerate the demo set.")
        if str(d["map"]) != expected_map:
            raise ValueError(f"{path.name}: demo map={str(d['map'])!r} != expected "
                             f"{expected_map!r}; demos do not transfer across maps.")
        if len(str(d["git_sha"])) != 40:
            raise ValueError(f"{path.name}: missing/short git_sha — demo is not self-identifying.")

        obs, disc, cont = d["obs"], d["discrete_actions"], d["continuous_actions"]
        dones = d["dones"]
        T = int(d["tick_count"])
        if obs.shape != (T, OBS_DIM) or disc.shape != (T, ACTION_DIM) or cont.shape != (T, AIM_DIM):
            raise ValueError(f"{path.name}: array shapes disagree with tick_count={T}")
        # The expert pre-clamps Δyaw to what the env actually applied, so this
        # is an assert, not a re-derivation: if a label ever exceeds the cap,
        # the recorded action is NOT the executed one and the labels are lies.
        if np.abs(cont[:, 0]).max() > MAX_TURN_SPEED_RAD + 1e-6:
            raise ValueError(f"{path.name}: |Δyaw| label exceeds MAX_TURN_SPEED_RAD — the "
                             f"recorded action is not what the env executed.")

        if dedupe:
            key = (obs.tobytes(), disc.tobytes(), cont.tobytes())
            if key in seen:
                n_duplicates += 1
                continue
            seen[key] = path.name

        obs_parts.append(obs.astype(np.float32, copy=False))
        disc_parts.append(disc.astype(np.int64, copy=False))
        cont_parts.append(cont.astype(np.float32, copy=False))
        done_parts.append(dones.astype(bool, copy=False))
        episodes.append({
            "file": path.name,
            "seed": int(d["seed"]),
            "carrier_idx": int(d["carrier_idx"]),
            "spawn_area": int(d["spawn_area"]),
            "tick_count": T,
            "git_sha": str(d["git_sha"]),
        })

    return DemoSet(
        obs=np.concatenate(obs_parts),
        discrete=np.concatenate(disc_parts),
        continuous=np.concatenate(cont_parts),
        dones=np.concatenate(done_parts),
        episodes=episodes,
        n_files=len(files),
        n_duplicates=n_duplicates,
    )


def make_bc_env(seed: int = 0):
    """Fresh single C env on the SIMPLE map, configured exactly like demo-gen.

    auto_reset=False: a silent mid-episode round reset would splice two rounds
    and make a plant count as happening on the wrong start state. The simple
    map is mandatory (plan §Target map) — bare make_env() defaults to real
    de_dust2, whose geometry the demos say nothing about.
    """
    from c_env.cs2_env import make_env as make_c_env
    return make_c_env(seed=seed, map_data=make_simple_map(), auto_reset=False)


def build_bc_policy(device="cpu", seed: int = 0):
    """Build the RL policy architecture (src/train.build_policy) for BC.

    Arch-identical to training is the entire point of the warm-start: the
    saved state_dict must load into the policy `--resume` constructs. We build
    it against a throwaway env because build_policy pulls both obs_dim and the
    env's max_turn_speed off the env; the env is closed immediately after.
    """
    from train import build_policy
    env = make_bc_env(seed=seed)
    try:
        return build_policy(env, device)
    finally:
        env.close()


def bc_loss(policy, obs_t, disc_t, cont_t, entropy_coef: float = DEFAULT_ENTROPY_COEF):
    """BC loss for one minibatch (spec D-6). Returns (loss, stats dict).

        L = mean(-(log_prob_d + log_prob_c)) - λ_ent · mean(entropy_d + entropy_c)

    `_hybrid_sample_logits` is called in evaluate-mode (both actions supplied),
    which is byte-for-byte the same code path PPO's update uses to score
    stored actions — so BC maximises exactly the likelihood PPO later reads.

    `aim_log_std` is DETACHED before the loss (spec D-6 "freeze aim_log_std").
    Both the Gaussian NLL and the continuous entropy depend on σ; without the
    detach, gradient descent's cheapest win is to shrink σ, and PPO would
    resume from a policy that cannot explore. Detaching (rather than removing
    the parameter from the optimizer) also keeps `entropy_c` a constant, so
    the entropy bonus acts only on the discrete heads where it is wanted.

    No tanh/atanh change-of-variables: the head tanh-squashes the MEAN only
    and samples a plain Normal (train.py forward()), so the density is the
    plain Gaussian one — see spec D-6, verified against _hybrid_sample_logits.
    """
    from train import _hybrid_sample_logits

    logits, mu_aim, log_std, value = policy(obs_t, {})
    _a, _c, log_prob_d, log_prob_c, entropy_d, entropy_c = _hybrid_sample_logits(
        (logits, mu_aim, log_std.detach(), value),
        action=disc_t,
        continuous_action=cont_t,
    )
    nll = -(log_prob_d + log_prob_c).mean()
    entropy = (entropy_d + entropy_c).mean()
    loss = nll - entropy_coef * entropy
    return loss, {
        "loss": float(loss.detach()),
        "nll": float(nll.detach()),
        "nll_d": float(-log_prob_d.mean().detach()),
        "nll_c": float(-log_prob_c.mean().detach()),
        "entropy_d": float(entropy_d.mean().detach()),
        "entropy_c": float(entropy_c.mean().detach()),
    }


def train_bc(demos: DemoSet,
             policy=None,
             epochs: int = DEFAULT_EPOCHS,
             batch_size: int = 64,
             lr: float = 3e-4,
             entropy_coef: float = DEFAULT_ENTROPY_COEF,
             device: str = "cpu",
             seed: int = 0,
             log_every: int = 25,
             verbose: bool = True):
    """Fit `policy` to `demos` by shuffled per-tick maximum likelihood (D-7).

    Returns (policy, history) where history is a list of per-epoch stat dicts
    (mean over minibatches) — the caller checks that the loss actually fell.

    Adam over ALL policy parameters, including the value head and the LSTM.
    The value head gets no BC signal (there is no return label) so it stays at
    init; that is fine and intended — PPO relearns V from scratch and its
    warm-up epochs are dominated by value loss anyway. The LSTM does receive
    gradient (it sits in the forward path even at T=1 from zero state), which
    is exactly what makes the BC weights usable by the recurrent PPO policy.
    """
    import torch

    if policy is None:
        policy = build_bc_policy(device=device, seed=seed)
    torch.manual_seed(seed)

    obs_t = torch.as_tensor(demos.obs, dtype=torch.float32, device=device)
    disc_t = torch.as_tensor(demos.discrete, dtype=torch.int64, device=device)
    cont_t = torch.as_tensor(demos.continuous, dtype=torch.float32, device=device)
    n = obs_t.shape[0]

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator).to(device)
        epoch_stats, n_batches = {}, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss, stats = bc_loss(policy,
                                  obs_t[idx],
                                  disc_t[idx],
                                  cont_t[idx],
                                  entropy_coef=entropy_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Same 0.5 cap the PPO trainer uses; BC on a tiny dataset can
            # otherwise take one huge step early and land in a saturated
            # tanh regime for the aim head.
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats.get(k, 0.0) + v
            n_batches += 1
        epoch_stats = {k: v / n_batches for k, v in epoch_stats.items()}
        epoch_stats["epoch"] = epoch
        history.append(epoch_stats)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"[train_bc] epoch {epoch:4d}/{epochs}  loss={epoch_stats['loss']:.4f}  "
                  f"nll_d={epoch_stats['nll_d']:.4f}  nll_c={epoch_stats['nll_c']:.4f}  "
                  f"H_d={epoch_stats['entropy_d']:.3f}")
    policy.eval()
    return policy, history


# ── BC-only eval gate (spec §8 gate 1) ─────────────────────────────────────


def greedy_carrier_action(policy, obs_row, device="cpu"):
    """Greedy action for ONE agent's obs row: argmax over each discrete head +
    μ_aim (no Gaussian sampling).

    Deliberately runs the STATELESS training forward (`policy(x, {})` → T=1
    from zero LSTM state), not `forward_eval` with carried state: the clone was
    fitted from zero state on every tick (D-7), so evaluating it with carried
    state would score a function BC never optimised.

    Returns (discrete int64[ACTION_DIM], continuous float32[AIM_DIM]).
    """
    import torch

    with torch.no_grad():
        x = torch.as_tensor(obs_row, dtype=torch.float32, device=device).unsqueeze(0)
        logits, mu_aim, _log_std, _value = policy(x, {})
        disc = np.array([int(h.argmax(dim=-1)[0]) for h in logits], dtype=np.int64)
        cont = mu_aim[0].cpu().numpy().astype(np.float32)
    return disc, cont


def rollout_episode(policy, seed: int, carrier_idx: int, device="cpu", max_ticks=None):
    """One greedy BC rollout on a demo-generation start state.

    Mirrors gen_bc_demos.generate_episode exactly EXCEPT that the policy, not
    ScriptedBomber, drives: fresh seeded env on the simple map, the carrier
    pokes (bomb + knife + designated-carrier role bit), a priming zero-action
    step so env.observations is real (reset() only zeroes the buffer), then
    per-tick greedy actions for the carrier row with the other 9 agents held
    at all-zero actions — the same frozen-idle world the demos recorded.

    Returns a dict: planted (bool), ticks, reached-plant tick or None.
    """
    if max_ticks is None:
        max_ticks = ROUND_TIME - 1     # the priming step already spent one tick

    env = make_bc_env(seed=seed)
    try:
        env.reset(seed=seed)           # NB: the seed here is ignored by design (see module doc)
        setup_bomb_carrier(env, carrier_idx)
        disc = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        cont = np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32)
        env.step(disc, cont)           # priming step: populates env.observations

        planted, ticks = False, 0
        for tick in range(max_ticks):
            obs_row = env.observations[carrier_idx].astype(np.float32, copy=True)
            mask_idle_agent_blocks(obs_row)            # spec R8 — must match the demos
            a_disc, a_cont = greedy_carrier_action(policy, obs_row, device=device)
            disc[:] = 0
            cont[:] = 0.0
            disc[carrier_idx] = a_disc
                                                       # Δyaw is clamped silently by the C env; clamp here too so what we
                                                       # feed equals what executes (same reason the expert pre-clamps).
            cont[carrier_idx] = np.clip(a_cont, -MAX_TURN_SPEED_RAD, MAX_TURN_SPEED_RAD)
            env.step(disc, cont)
            ticks = tick + 1
            if bool(env._c_env.game.bomb_planted):
                planted = True
                break
        return {"seed": seed, "carrier_idx": carrier_idx, "planted": planted, "ticks": ticks}
    finally:
        env.close()


def eval_plant_rate(policy, seeds, carriers=None, device="cpu", verbose=True, label=""):
    """Greedy plant rate over the (seed × carrier) start-state grid.

    NOTE ON "HELD-OUT" (plan Task 1 RESULT): spawns are deterministic area
    centroids and seeds only permute which agent gets which spawn area, so
    unseen seeds produce the SAME 5 carrier start states as the demo set.
    Evaluating on other seeds is therefore a determinism/robustness check, not
    a generalization measurement — it is reported separately and labelled as
    degenerate rather than dressed up as held-out.
    """
    if carriers is None:
        carriers = range(TEAM_SIZE)
    results = [
        rollout_episode(policy, seed=s, carrier_idx=c, device=device) for s in seeds
        for c in carriers
    ]
    planted = [r for r in results if r["planted"]]
    plant_rate = len(planted) / max(1, len(results))
    ticks = [r["ticks"] for r in planted]
    summary = {
        "label": label,
        "episodes": len(results),
        "planted": len(planted),
        "plant_rate": plant_rate,
        "ticks_min": int(min(ticks)) if ticks else None,
        "ticks_median": int(np.median(ticks)) if ticks else None,
        "ticks_max": int(max(ticks)) if ticks else None,
        "results": results,
    }
    if verbose:
        print(f"[bc-eval] {label}: plant_rate={plant_rate:.3f} "
              f"({len(planted)}/{len(results)} episodes) "
              f"ticks(min/median/max)={summary['ticks_min']}/{summary['ticks_median']}/"
              f"{summary['ticks_max']}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    # CPU is the default on purpose: the dataset is ~365 unique ticks and the
    # GPU on this box is usually busy with a PPO run.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-dedupe",
                        action="store_true",
                        help="keep byte-identical duplicate episodes (default: drop them)")
    parser.add_argument("--eval-seeds",
                        type=int,
                        default=10,
                        help="demo-distribution eval: seeds 0..N-1 × 5 carrier slots")
    parser.add_argument("--holdout-seeds",
                        type=int,
                        default=5,
                        help="extra seeds evaluated as a (degenerate) held-out check; 0 to skip")
    parser.add_argument("--gate-plant-rate", type=float, default=DEFAULT_GATE_PLANT_RATE)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-only",
                        type=Path,
                        default=None,
                        help="skip training; load this checkpoint and run the gate")
    args = parser.parse_args(argv)

    import torch

    if args.eval_only is not None:
        policy = build_bc_policy(device=args.device, seed=args.seed)
        policy.load_state_dict(torch.load(args.eval_only, map_location=args.device))
        policy.eval()
        print(f"[train_bc] loaded {args.eval_only} for eval only")
    else:
        demos = load_demos(args.demos, dedupe=not args.no_dedupe)
        print(f"[train_bc] {len(demos)} ticks from {len(demos.episodes)} episodes "
              f"({demos.n_files} files, {demos.n_duplicates} byte-identical duplicates dropped); "
              f"spawn areas {demos.spawn_areas()}; OBS_DIM={OBS_DIM} ACTION_DIM={ACTION_DIM} "
              f"AIM_DIM={AIM_DIM}")
        policy, history = train_bc(demos,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   lr=args.lr,
                                   entropy_coef=args.entropy_coef,
                                   device=args.device,
                                   seed=args.seed)
        print(f"[train_bc] loss {history[0]['loss']:.4f} → {history[-1]['loss']:.4f}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # BARE state_dict — the format all three load sites expect, including
        # `--resume` (spec D-8). Do NOT wrap it in a {"model": ...} dict and do
        # NOT write to dust2_policy.pt (that is the RL run's checkpoint).
        torch.save(policy.state_dict(), args.out)
        print(f"[train_bc] saved warm-start state_dict → {args.out}")

    if args.skip_eval:
        return 0

    demo_eval = eval_plant_rate(policy,
                                seeds=range(args.eval_seeds),
                                device=args.device,
                                label="demo-distribution spawns")
    if args.holdout_seeds > 0:
        eval_plant_rate(policy,
                        seeds=range(1000, 1000 + args.holdout_seeds),
                        device=args.device,
                        label="unseen seeds (DEGENERATE: deterministic spawns → same 5 states)")

    passed = demo_eval["plant_rate"] >= args.gate_plant_rate
    print(f"[bc-eval] GATE 1 (spec §8): plant_rate={demo_eval['plant_rate']:.3f} vs threshold "
          f"{args.gate_plant_rate:.3f} → {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("[bc-eval] FAIL means BC is broken — fix it before the PPO handoff (Task 5). "
              "Do not lower the threshold.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
