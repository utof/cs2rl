#!/usr/bin/env python
"""Behaviour-cloning warm-start trainer (Batch 6 Task 4 — spec D-6, §8 gate 1).

Reads the scripted-expert demonstrations written by scripts/gen_bc_demos.py,
fits the SAME policy architecture PPO uses (build_policy) to them by maximum
likelihood **through the same forward contract PPO uses**, writes a bare
state_dict to outputs/checkpoints/bc_warmstart.pt, and then runs the BC-only
eval gate: greedy rollouts on the demo-generation spawn set, reporting the
plant rate BEFORE any RL touches the weights.

    uv run python src/train_bc.py                       # train + eval gate
    uv run python src/train_bc.py --skip-eval           # train only
    uv run python src/train_bc.py --eval-only <ckpt>    # re-run the gate

The checkpoint is consumed with zero extra code by
    uv run python -m src.train --train --resume outputs/checkpoints/bc_warmstart.pt ...
(spec D-8: `--resume` torch.load()s a bare state_dict).

WHY BC MUST TRAIN THROUGH PPO'S FORWARD CONTRACT (read before "simplifying")
---------------------------------------------------------------------------
The original implementation followed spec D-7: shuffle individual (obs, action)
ticks and run the policy STATELESSLY — `policy(x, {})`, i.e. a T=1 sequence
from a ZERO LSTM state on every tick. That premise is **stale**: it predates
commit 456c361, which made PPO do true BPTT. Today

  * PPO's rollout (src/train.py `forward_eval`) CARRIES lstm_h/lstm_c from
    tick to tick within an episode, and
  * PPO's update (src/train.py `Dust2Policy.forward` → `_lstm_bptt`) unrolls a
    whole 64-tick segment through the LSTM in one call.

So a stateless clone optimises a function PPO never evaluates. Measured on the
old shipped bc_warmstart.pt over the 5 demo start states:

      stateless greedy plant rate = 1.000   (what the old gate reported)
      carried-state plant rate    = 0.200   (what PPO actually inherits)

Four of the five routes silently died the moment the LSTM was allowed to
remember anything. The gate was green and the artefact was junk.

The fix, and the shape of this file:

  * **Training is sequence training.** One demo episode = one segment. Each
    episode is fed as a (1, T, OBS_DIM) slice of a padded (B, T_max, OBS_DIM)
    batch straight into `policy(x, {})` — the SAME `Dust2Policy.forward` the
    PPO update calls, with the same segment-major/time-minor flattening of its
    output. The LSTM state therefore evolves across the episode exactly as it
    does at rollout time. Padding ticks are excluded from the loss by an
    explicit validity mask (they sit AFTER every real tick, so they cannot
    contaminate the hidden states that matter).
  * **The gate is a carried-state rollout.** `rollout_episode(..., carry_state=
    True)` drives the env with `forward_eval` and a persistent state dict,
    byte-for-byte the loop src/train.py's rollout runs. The stateless number is
    still printed, clearly labelled as a diagnostic — it is the quantity that
    lied, so it is worth watching, but it is NOT the gate.

  Episodes are 54–94 ticks, i.e. shorter than PPO's 64-tick bptt_horizon in the
  median case. We deliberately do NOT chunk them to 64: PPO's zero-initial-state
  assumption is only valid because each agent row fills exactly one segment per
  evaluate() (see `Dust2Policy.forward`'s "WHY zero initial state is CORRECT"
  note). Feeding one whole episode per segment from zero state satisfies the
  same invariant and matches the eval rollout, which also starts from an empty
  state dict. `terminals` is left None because a segment never contains an
  episode boundary — the only done is the final tick.

REST OF THE SHAPE
-----------------
* **Loss reuses `_hybrid_sample_logits` (spec D-6)** — the exact function PPO
  evaluates log-probs with, called in its "action supplied" mode. BC therefore
  optimises precisely the quantity PPO will later measure:
      L = mean_valid(-(log_prob_d + log_prob_c)) - λ_ent · mean_valid(H_d + H_c)
  It is NOT byte-for-byte identical to PPO's update, which additionally applies
  the C-computed action masks (`mb_masks`); BC leaves the loss unmasked because
  expert actions are always valid, so masking cannot change the label's
  log-prob rank. The greedy EVAL does apply the masks (see
  `greedy_carrier_action`) so argmax can never select a bin the env forbids.
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
  as a raw PyTorch shape error 40 minutes into a GPU run). The sha check is a
  real comparison against HEAD, not a length check — see `check_demo_sha`.
* The demo set is **5 unique trajectories duplicated ×10** — spawns are
  deterministic area centroids, seeds only permute which agent gets which
  spawn (plan Task 1 RESULT). We dedupe byte-identical episodes by default;
  it changes nothing statistically (the duplication is perfectly uniform) and
  makes each epoch 10× cheaper. The eval gate dedupes the same way, on the
  priming obs, so the printed denominator is the number of *distinct* start
  states rather than a 10×-inflated 50.
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
"""
import os

# Match src/train.py: BLAS thread caps must be set before torch/numpy spin up
# their pools, or a CPU BC run oversubscribes every core on this box.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse                                        # noqa: E402
import math                                            # noqa: E402
import subprocess                                      # noqa: E402
import sys                                             # noqa: E402
from dataclasses import dataclass, field               # noqa: E402
from pathlib import Path                               # noqa: E402

import numpy as np                     # noqa: E402

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:       # allows `python src/train_bc.py`
    sys.path.insert(0, str(SRC_DIR))   # AND `import train_bc` from tests

REPO_ROOT = SRC_DIR.parent

from _action_spec import ACTION_DIM, ACTION_MASK_DIM, AIM_DIM          # noqa: E402
from _obs_spec import OBS_BLOCKS, OBS_DIM                              # noqa: E402
from map import make_simple_map                                        # noqa: E402
from nav import MAX_TURN_SPEED_RAD, N_AGENTS, ROUND_TIME, TEAM_SIZE    # noqa: E402
from paths import CHECKPOINTS_DIR                                      # noqa: E402
from scripted_expert import setup_bomb_carrier                         # noqa: E402

# The map tag every demo must carry. Demos are only meaningful for BC → PPO on
# the map they were generated on (geometry + per-map obs normalisation), so a
# mismatch is a hard error rather than a warning. Mirrors gen_bc_demos.MAP_NAME.
EXPECTED_MAP = "simple_v1"

DEFAULT_DEMO_DIR = REPO_ROOT / "outputs" / "demos"
DEFAULT_CHECKPOINT = Path(CHECKPOINTS_DIR) / "bc_warmstart.pt"

# Files whose contents determine what a demo *means*: the C env (dynamics,
# observations, masks), the generated obs/action layouts, the map the demos
# were recorded on, and the expert that produced the labels. A demo recorded at
# a different commit is only genuinely stale if one of these changed — see
# check_demo_sha for why we diff this surface instead of comparing shas
# verbatim.
DEMO_RELEVANT_PATHS = (
    "src/c_env",
    "src/_obs_spec.py",
    "src/_action_spec.py",
    "src/map.py",
    "src/nav.py",
    "src/scripted_expert.py",
    "scripts/gen_bc_demos.py",
)

# λ_ent from spec D-6 (imitation-lib default). Small on purpose: it only keeps
# the heads from collapsing to one-hot certainty on a 369-tick dataset, it is
# not meant to compete with the likelihood term.
DEFAULT_ENTROPY_COEF = 1e-3

# Epoch count is MEASURED, not guessed (2026-08-01, this box, CPU, lr 3e-4, the
# 5-unique-episode demo set, one gradient step per epoch because all 5 episodes
# fit in a single sequence minibatch). Sequence training needs an order of
# magnitude more epochs than the old per-tick loop, purely because an epoch is
# now 1 optimizer step instead of ~6.
#
# Sweep, CARRIED-STATE plant rate at seed 0:
#     100 / 200 / 300 / 400 epochs → 0.000   (gate FAILs)
#     500 / 1000 / 2000 / 3000 / 4000 epochs → 1.000
# The cliff sits between 400 and 500, so 500 is NOT a safe default. 2000 gives
# 4× margin over the cliff, costs ~100 s, and held 1.000 at seeds 1, 2 and 3 as
# well. Do not lower it to save a minute.
#
# The stateless diagnostic over those same runs read 1.000, 0.000, 1.000, 1.000,
# 0.400 — it wanders, because nothing optimises it any more. That is the
# expected signature of a genuinely recurrent clone, not a regression.
DEFAULT_EPOCHS = 2000

# Spec §8 gate 1 asks for a "non-trivial" plant rate on the demo distribution
# before any RL; it does not name a number. 0.5 is our operationalisation:
# the cold-start baseline is 0.000, the expert is 1.000, and a clone that
# plants on the majority of the 5 demonstrated start states has demonstrably
# learned "walk to the site and press USE" rather than a lucky single route.
# NEVER loosen this to make a run pass — a failing gate means BC is broken
# (spec §8: "fix before Task 5"). The gate reads the CARRIED-STATE rate; the
# stateless rate is a diagnostic and is never gated on (module docstring).
DEFAULT_GATE_PLANT_RATE = 0.5

# The C env clamps the two continuous dims DIFFERENTLY (cs2_env.h ~line 146 and
# ~line 262): dim 0 is a Δyaw *delta*, clamped to ±max_turn_speed; dim 1 is an
# ABSOLUTE pitch target, clamped to ±π/2. Mirroring the C clamp Python-side
# keeps what we feed equal to what executes (same reason the expert pre-clamps).
# Today it is inert — the policy's own tanh already bounds both dims to
# ±max_turn_speed — but a wrong clamp here would silently truncate pitch the
# moment the aim head's scaling changes.
AIM_CLAMP_LO = np.array([-MAX_TURN_SPEED_RAD, -math.pi / 2], dtype=np.float32)
AIM_CLAMP_HI = np.array([MAX_TURN_SPEED_RAD, math.pi / 2], dtype=np.float32)


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


# ── Demo provenance ────────────────────────────────────────────────────────


def _git(*args, cwd=REPO_ROOT):
    """Run a git command, returning (returncode, stdout). Never raises — a
    missing/foreign git checkout must degrade to "cannot verify", not crash a
    training run."""
    try:
        p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except (OSError, ValueError):
        return 1, ""
    return p.returncode, p.stdout.strip()


def check_demo_sha(sha: str, allow_stale: bool = False, name: str = "demo") -> str | None:
    """Verify a demo's recorded git sha against the working tree. Returns a
    human-readable note when the demo is off-HEAD but still trustworthy, None
    when it is exactly current; raises ValueError when it is genuinely stale.

    WHY not a plain `sha == HEAD` equality: every commit to this repo — a
    docstring fix, a test tweak — would invalidate a demo set that is still
    bit-for-bit reproducible, and the escape hatch would become the normal
    path (at which point it guards nothing). Instead we ask the question that
    actually matters: *did anything that determines the demo's contents change
    between then and now?* That surface is DEMO_RELEVANT_PATHS — the C env,
    the obs/action layouts, the map, and the expert. `git diff --quiet` over
    exactly those paths answers it exactly.

    The predecessor of this function checked `len(sha) != 40` and nothing else,
    which passed any 40-character string including the sha of an env revision
    whose obs layout no longer exists (review finding 5).

    PITFALLS:
      * A sha git does not know (foreign checkout, shallow clone, demos copied
        from another machine) is treated as STALE, not as "probably fine" — we
        cannot diff against a commit we do not have.
      * Uncommitted edits to DEMO_RELEVANT_PATHS also make demos stale, and
        `git diff <sha> HEAD` cannot see them: we diff the WORKING TREE
        (`git diff <sha> -- paths`), so a dirty c_env is caught too.
      * `allow_stale=True` downgrades the error to a printed warning. It exists
        for deliberate experiments ("does the old demo set still transfer?"),
        not for silencing the check on the happy path.
    """
    if len(sha) != 40:
        raise ValueError(f"{name}: missing/short git_sha {sha!r} — demo is not self-identifying.")

    rc, head = _git("rev-parse", "HEAD")
    if rc != 0:
        return f"{name}: not a git checkout — demo provenance unverifiable"
    if sha == head and _git("diff", "--quiet", "--", *DEMO_RELEVANT_PATHS)[0] == 0:
        return None

    known = _git("cat-file", "-e", f"{sha}^{{commit}}")[0] == 0
    if known and _git("diff", "--quiet", sha, "--", *DEMO_RELEVANT_PATHS)[0] == 0:
        return (f"{name}: recorded at {sha[:9]}, HEAD is {head[:9]} — but nothing under "
                f"{', '.join(DEMO_RELEVANT_PATHS)} changed since, so the demos are still "
                f"reproducible byte-for-byte.")

    why = ("that commit is unknown to this checkout"
           if not known else "the env/obs/map/expert sources changed since then")
    if allow_stale:
        return (f"{name}: STALE (recorded at {sha[:9]}, HEAD {head[:9]}; {why}) — "
                f"proceeding because --allow-stale-demos was given.")
    raise ValueError(
        f"{name}: STALE DEMO — recorded at git sha {sha[:9]} but {why}. The recorded "
        f"observations and expert actions may no longer describe this env. Regenerate: "
        f"uv run python scripts/gen_bc_demos.py --seeds 10 "
        f"(or pass --allow-stale-demos if you really mean to train on them).")


@dataclass
class DemoSet:
    """BC dataset: episodes concatenated flat, plus the per-episode lengths that
    let `as_sequences` rebuild them.

    Both views are kept on purpose. The flat arrays are what the files contain
    and what the schema asserts are written against; `lengths` is what makes
    sequence training possible, and it is the field that must never drift —
    `sum(lengths) == len(obs)` is checked in __post_init__ because a silent
    off-by-one there would shear every episode boundary and train the LSTM on
    spliced trajectories.
    """
    obs: np.ndarray                                    # float32 [N, OBS_DIM]
    discrete: np.ndarray                               # int64   [N, ACTION_DIM]
    continuous: np.ndarray                             # float32 [N, AIM_DIM]
    dones: np.ndarray                                  # bool    [N]
    lengths: np.ndarray = None                         # int64   [E] ticks per episode
    episodes: list = field(default_factory=list)       # per-kept-episode metadata dicts
    n_files: int = 0                                   # files on disk (pre-dedupe)
    n_duplicates: int = 0                              # byte-identical episodes dropped

    def __post_init__(self):
        if self.lengths is None:
            # One episode covering everything — the degenerate case a synthetic
            # test constructs. Real demo sets always pass explicit lengths.
            self.lengths = np.array([self.obs.shape[0]], dtype=np.int64)
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        if int(self.lengths.sum()) != self.obs.shape[0]:
            raise ValueError(f"DemoSet: lengths sum to {int(self.lengths.sum())} but there are "
                             f"{self.obs.shape[0]} ticks — episode boundaries are wrong.")

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_episodes(self) -> int:
        return int(self.lengths.shape[0])

    def spawn_areas(self) -> list:
        return sorted({int(e["spawn_area"]) for e in self.episodes})

    def as_sequences(self):
        """Right-pad the episodes into (obs, discrete, continuous, valid) arrays
        of shape [E, T_max, ...] / [E, T_max].

        Padding is zeros and `valid` is False there. RIGHT-padding specifically:
        the LSTM runs over the padded tail too (it is one nn.LSTM call), so pads
        must come after every real tick or they would poison the hidden state
        the real ticks depend on. The loss then drops them via `valid`.
        """
        e, t_max = self.n_episodes, int(self.lengths.max())
        obs = np.zeros((e, t_max, self.obs.shape[1]), dtype=np.float32)
        disc = np.zeros((e, t_max, self.discrete.shape[1]), dtype=np.int64)
        cont = np.zeros((e, t_max, self.continuous.shape[1]), dtype=np.float32)
        valid = np.zeros((e, t_max), dtype=bool)
        off = 0
        for i, n in enumerate(self.lengths.tolist()):
            obs[i, :n] = self.obs[off:off + n]
            disc[i, :n] = self.discrete[off:off + n]
            cont[i, :n] = self.continuous[off:off + n]
            valid[i, :n] = True
            off += n
        return obs, disc, cont, valid


def load_demos(demo_dir,
               dedupe: bool = True,
               expected_map: str = EXPECTED_MAP,
               allow_stale: bool = False,
               verbose: bool = True) -> DemoSet:
    """Load every .npz in `demo_dir`, assert its schema against the LIVE
    constants, and concatenate into one dataset (flat + per-episode lengths).

    The schema assert is the whole reason demos are self-identifying (spec §7):
    obs layout is a moving target (107→110 happened *inside* this batch), and
    dims alone do not catch semantic slot changes — hence the git_sha + map tag
    checks too (see `check_demo_sha`). A mismatch raises here, at second zero,
    instead of surfacing as a shape error inside a long `--resume` run (spec R7).

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

    obs_parts, disc_parts, cont_parts, done_parts, lengths = [], [], [], [], []
    episodes, seen = [], {}
    n_duplicates = 0
    sha_notes = set()
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
        note = check_demo_sha(str(d["git_sha"]), allow_stale=allow_stale, name=path.name)
        if note is not None:
            # One note per distinct sha, not per file: 50 identical warnings
            # would bury the schema line that follows.
            sha_notes.add(note.split(":", 1)[1].strip())

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
        lengths.append(T)
        episodes.append({
            "file": path.name,
            "seed": int(d["seed"]),
            "carrier_idx": int(d["carrier_idx"]),
            "spawn_area": int(d["spawn_area"]),
            "tick_count": T,
            "git_sha": str(d["git_sha"]),
        })

    if verbose:
        for note in sorted(sha_notes):
            print(f"[train_bc] demo provenance: {note}")

    return DemoSet(
        obs=np.concatenate(obs_parts),
        discrete=np.concatenate(disc_parts),
        continuous=np.concatenate(cont_parts),
        dones=np.concatenate(done_parts),
        lengths=np.array(lengths, dtype=np.int64),
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

    `torch.manual_seed` is called HERE, before build_policy, not later in
    train_bc: layer_init draws from the global RNG at construction time, so
    seeding afterwards left two `build_bc_policy(seed=0)` calls with different
    weights and made `--seed` a lie (review finding 8).

    Batch 7 (spec 2026-08-13 §3.3): BC always builds the LEGACY architecture
    (tct_split_heads defaults False). Its product, bc_warmstart.pt, is the
    input to the legacy→split warm conversion, so a split BC policy would have
    no consumer — and the RL side infers architecture from checkpoint keys, so
    a legacy BC checkpoint resumes into either architecture correctly.
    """
    import torch

    from train import build_policy
    torch.manual_seed(seed)
    env = make_bc_env(seed=seed)
    try:
        return build_policy(env, device)
    finally:
        env.close()


def bc_loss(policy, obs_t, disc_t, cont_t, valid=None, entropy_coef: float = DEFAULT_ENTROPY_COEF):
    """BC loss for one minibatch of SEQUENCES (spec D-6). Returns (loss, stats).

        L = mean_valid(-(log_prob_d + log_prob_c)) - λ_ent · mean_valid(H_d + H_c)

    Shapes
    ------
    obs_t   (B, T, OBS_DIM) — B episode segments of T ticks. A 2D (N, OBS_DIM)
            input is also accepted and behaves as T=1, which is what the ONNX /
            single-tick path in `Dust2Policy.forward` does; it is kept working
            so the loss can be unit-tested without a sequence harness.
    disc_t  (B, T, ACTION_DIM) int64      cont_t (B, T, AIM_DIM) float32
    valid   (B, T) bool or None — False on right-padding ticks, which are
            dropped from both means. None = every tick counts.

    WHY sequences and not shuffled ticks: `policy(x, {})` IS PPO's update-path
    forward (`Dust2Policy.forward` → `_lstm_bptt`), which unrolls the LSTM
    along T. Feeding it T=1 slices trains a stateless function that PPO never
    evaluates — the failure that took the shipped checkpoint from a 1.000
    stateless plant rate to 0.200 carried-state. See the module docstring.

    The flat output of `Dust2Policy.forward` is segment-major, time-minor
    (`hidden_out = h.transpose(0,1).reshape(B*T, H)`), so `disc_t.reshape(-1,
    ACTION_DIM)` lines up row-for-row. This is the same alignment
    `_hybrid_ppo_loss` relies on; changing either side silently mispairs every
    label with the wrong tick's logits.

    `_hybrid_sample_logits` is called in evaluate-mode (both actions supplied)
    — the same code path PPO's update uses to score stored actions, minus the
    action masks (PPO passes `mb_masks`; expert actions are always valid so the
    label's log-prob is unaffected — module docstring).

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
    import torch

    from train import _hybrid_sample_logits

    flat_disc = disc_t.reshape(-1, disc_t.shape[-1])
    flat_cont = cont_t.reshape(-1, cont_t.shape[-1])

    logits, mu_aim, log_std, value = policy(obs_t, {})
    # R0-E.2 (#131): a pin_pitch policy carries aim_dim_mask=[1,0]; forward it
    # so nll_c / entropy_c cover the yaw dim only (the env ignores cont[:,1]).
    _a, _c, log_prob_d, log_prob_c, entropy_d, entropy_c = _hybrid_sample_logits(
        (logits, mu_aim, log_std.detach(), value),
        action=flat_disc,
        continuous_action=flat_cont,
        aim_dim_mask=getattr(policy, "aim_dim_mask", None),
    )

    if valid is None:
        w = torch.ones_like(log_prob_d)
    else:
        w = valid.reshape(-1).to(log_prob_d.dtype)
    denom = w.sum().clamp(min=1.0)

    def _mean(x):
        return (x * w).sum() / denom

    nll_d, nll_c = -_mean(log_prob_d), -_mean(log_prob_c)
    h_d, h_c = _mean(entropy_d), _mean(entropy_c)
    nll = nll_d + nll_c
    loss = nll - entropy_coef * (h_d + h_c)
    return loss, {
        "loss": float(loss.detach()),
        "nll": float(nll.detach()),
        "nll_d": float(nll_d.detach()),
        "nll_c": float(nll_c.detach()),
        "entropy_d": float(h_d.detach()),
        "entropy_c": float(h_c.detach()),
    }


def train_bc(demos: DemoSet,
             policy=None,
             epochs: int = DEFAULT_EPOCHS,
             batch_size: int = 64,
             lr: float = 3e-4,
             entropy_coef: float = DEFAULT_ENTROPY_COEF,
             device: str = "cpu",
             seed: int = 0,
             log_every: int = 200,
             verbose: bool = True):
    """Fit `policy` to `demos` by sequence maximum likelihood.

    Returns (policy, history) where history is a list of per-epoch stat dicts
    (mean over minibatches) — the caller checks that the loss actually fell.

    `batch_size` counts EPISODES, not ticks: one row of a minibatch is one
    whole demo episode unrolled through the LSTM. With the 5-unique-episode
    demo set the default 64 means "one minibatch per epoch", which is why
    DEFAULT_EPOCHS is an order of magnitude larger than the old per-tick
    loop's — an epoch is now a single optimizer step, not ~6.

    Adam over ALL policy parameters, including the value head and the LSTM.
    The value head gets no BC signal (there is no return label) so it stays at
    init; that is fine and intended — PPO relearns V from scratch and its
    warm-up epochs are dominated by value loss anyway. The LSTM DOES receive
    through-time gradient here, which is the whole point: it is the recurrent
    weights that decide whether the warm-start survives contact with PPO's
    carried-state rollout.
    """
    import torch

    if policy is None:
        policy = build_bc_policy(device=device, seed=seed)
    torch.manual_seed(seed)

    seq_obs, seq_disc, seq_cont, seq_valid = demos.as_sequences()
    obs_t = torch.as_tensor(seq_obs, dtype=torch.float32, device=device)
    disc_t = torch.as_tensor(seq_disc, dtype=torch.int64, device=device)
    cont_t = torch.as_tensor(seq_cont, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(seq_valid, dtype=torch.bool, device=device)
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
                                  valid=valid_t[idx],
                                  entropy_coef=entropy_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Same 0.5 cap the PPO trainer uses; BPTT over a 90-tick episode
            # can otherwise deliver one huge step early and land the aim head
            # in a saturated tanh regime it never recovers from.
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


def greedy_carrier_action(policy, obs_row, state=None, action_mask=None, device="cpu"):
    """Greedy action for ONE agent's obs row: argmax over each discrete head +
    μ_aim (no Gaussian sampling).

    `state` selects the forward contract, and this choice is the entire subject
    of the module docstring:
      * dict  → `policy.forward_eval(x, state)`, which READS and WRITES
        state["lstm_h"]/["lstm_c"] — the LSTM state is carried tick-to-tick,
        exactly as src/train.py's rollout does. This is what the gate uses.
      * None  → `policy(x, {})`, a T=1 unroll from ZERO state every tick. Kept
        ONLY as the labelled diagnostic; it is the number that read 1.000 while
        the carried-state truth was 0.200.

    `action_mask` is the C-computed (ACTION_MASK_DIM,) validity row for this
    agent (env._masks_view). When given, invalid bins are pushed to
    finfo.min/2 by train._apply_action_masks BEFORE the argmax, so greedy eval
    can never pick a bin the env would reject — the same masking PPO's sampler
    applies (review finding 7).

    Returns (discrete int64[ACTION_DIM], continuous float32[AIM_DIM]).
    """
    import torch

    from train import _apply_action_masks

    with torch.no_grad():
        x = torch.as_tensor(obs_row, dtype=torch.float32, device=device).unsqueeze(0)
        if state is None:
            logits, mu_aim, _log_std, _value = policy(x, {})
        else:
            logits, mu_aim, _log_std, _value = policy.forward_eval(x, state)
        if action_mask is not None:
            m = torch.as_tensor(np.asarray(action_mask), device=device).reshape(1, -1) != 0
            logits = _apply_action_masks(logits, m)
        disc = np.array([int(h.argmax(dim=-1)[0]) for h in logits], dtype=np.int64)
        cont = mu_aim[0].cpu().numpy().astype(np.float32)
    return disc, cont


def rollout_episode(policy,
                    seed: int,
                    carrier_idx: int,
                    device="cpu",
                    max_ticks=None,
                    carry_state: bool = True):
    """One greedy BC rollout on a demo-generation start state.

    Mirrors gen_bc_demos.generate_episode exactly EXCEPT that the policy, not
    ScriptedBomber, drives: fresh seeded env on the simple map, the carrier
    pokes (bomb + knife + designated-carrier role bit), a priming zero-action
    step so env.observations is real (reset() only zeroes the buffer), then
    per-tick greedy actions for the carrier row with the other 9 agents held
    at all-zero actions — the same frozen-idle world the demos recorded.

    `carry_state=True` (the DEFAULT, and what the gate measures) threads one
    LSTM state dict through the whole episode, which is what src/train.py's
    rollout does and therefore what PPO inherits. `carry_state=False` resets to
    zero state every tick; it exists only to print the historical stateless
    diagnostic alongside the real number. Never gate on the False variant.

    Returns a dict: planted (bool), ticks, and `first_obs` — the masked priming
    observation, i.e. the start state this episode actually ran from. It is
    reported for diagnostics and so a caller can confirm that two rollouts it
    believes are duplicates really did start identically (`probe_start_state`
    computes the same bytes without paying for the rollout).
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

        state = {} if carry_state else None
        planted, ticks, first_obs = False, 0, None
        for tick in range(max_ticks):
            obs_row = env.observations[carrier_idx].astype(np.float32, copy=True)
            mask_idle_agent_blocks(obs_row)                              # spec R8 — must match the demos
            if first_obs is None:
                first_obs = obs_row.copy()
            mask_row = np.asarray(env._masks_view).reshape(N_AGENTS, ACTION_MASK_DIM)[carrier_idx]
            a_disc, a_cont = greedy_carrier_action(policy,
                                                   obs_row,
                                                   state=state,
                                                   action_mask=mask_row,
                                                   device=device)
            disc[:] = 0
            cont[:] = 0.0
            disc[carrier_idx] = a_disc
                                                                         # Per-dim clamp: Δyaw ±max_turn_speed, ABSOLUTE pitch ±π/2 — see
                                                                         # AIM_CLAMP_LO/HI for why the two dims differ.
            cont[carrier_idx] = np.clip(a_cont, AIM_CLAMP_LO, AIM_CLAMP_HI)
            env.step(disc, cont)
            ticks = tick + 1
            if bool(env._c_env.game.bomb_planted):
                planted = True
                break
        return {
            "seed": seed,
            "carrier_idx": carrier_idx,
            "planted": planted,
            "ticks": ticks,
            "first_obs": first_obs,
        }
    finally:
        env.close()


def probe_start_state(seed: int, carrier_idx: int) -> bytes:
    """Identity of the (seed, carrier) start state: the bytes of the masked
    priming obs.

    WHY this and not (seed, carrier): spawns are deterministic area centroids
    and seeds only permute WHICH agent gets which area (plan Task 1 RESULT), so
    10 seeds × 5 carriers is 5 distinct start states repeated 10× — and every
    rollout from a repeat is bit-identical, because the env and the greedy
    policy are both deterministic. Reporting "5/50 episodes" was therefore a
    10×-inflated denominator dressed up as sample size (review finding 6).

    The priming obs is the right key rather than `spawn_area` because it is the
    *complete* input the deterministic rollout consumes: if two start states
    ever differ in anything the policy can see, they get different keys, and
    the dedupe degrades to no dedupe rather than to a silently wrong average.

    Costs one env construction + one step per (seed, carrier) — negligible
    against the up-to-639-tick rollout it saves.
    """
    env = make_bc_env(seed=seed)
    try:
        env.reset(seed=seed)
        setup_bomb_carrier(env, carrier_idx)
        env.step(np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32),
                 np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32))
        obs_row = env.observations[carrier_idx].astype(np.float32, copy=True)
        return mask_idle_agent_blocks(obs_row).tobytes()
    finally:
        env.close()


def eval_plant_rate(policy,
                    seeds,
                    carriers=None,
                    device="cpu",
                    verbose=True,
                    label="",
                    carry_state: bool = True,
                    dedupe_start_states: bool = True):
    """Greedy plant rate over the DISTINCT start states in the (seed × carrier)
    grid.

    `carry_state` picks the forward contract — True (default) is the carried
    LSTM state PPO's rollout uses and the only thing the gate may read; False
    is the stateless diagnostic. See `greedy_carrier_action`.

    `dedupe_start_states` collapses (seed, carrier) pairs whose priming obs are
    identical, running ONE rollout per distinct state and reporting the
    duplicate multiplier separately. Spawns are deterministic centroids and
    both env and greedy policy are deterministic, so the duplicates would
    contribute literally the same episode; averaging over them inflates the
    denominator without adding a single bit of evidence (review finding 6).
    `summary["episodes"]` is therefore the number of DISTINCT states — the
    honest denominator — and `summary["duplicates"]` records what was dropped.

    NOTE ON "HELD-OUT" (plan Task 1 RESULT): seeds only permute which agent
    gets which spawn area, so unseen seeds produce the SAME 5 carrier start
    states as the demo set. Evaluating on other seeds is a determinism check,
    not a generalization measurement — it is reported separately and labelled
    as degenerate rather than dressed up as held-out. With dedupe on, an
    "unseen seeds" eval collapses to the same 5 states, which makes the
    degeneracy visible in the printed counts instead of merely documented.
    """
    if carriers is None:
        carriers = range(TEAM_SIZE)
    grid = [(s, c) for s in seeds for c in carriers]

    if dedupe_start_states:
        unique, seen = [], {}
        for s, c in grid:
            key = probe_start_state(s, c)
            if key in seen:
                continue
            seen[key] = (s, c)
            unique.append((s, c))
    else:
        unique = grid

    results = [
        rollout_episode(policy, seed=s, carrier_idx=c, device=device, carry_state=carry_state)
        for s, c in unique
    ]
    planted = [r for r in results if r["planted"]]
    plant_rate = len(planted) / max(1, len(results))
    ticks = [r["ticks"] for r in planted]
    summary = {
        "label": label,
        "carry_state": carry_state,
        "episodes": len(results),
        "grid_size": len(grid),
        "duplicates": len(grid) - len(results),
        "planted": len(planted),
        "plant_rate": plant_rate,
        "ticks_min": int(min(ticks)) if ticks else None,
        "ticks_median": int(np.median(ticks)) if ticks else None,
        "ticks_max": int(max(ticks)) if ticks else None,
        "results": results,
    }
    if verbose:
        contract = "CARRIED LSTM state (PPO's rollout contract)" if carry_state \
            else "STATELESS zero-state-per-tick (DIAGNOSTIC ONLY)"
        dupes = (f", {summary['duplicates']} duplicate start states in the "
                 f"{len(grid)}-cell grid collapsed" if summary["duplicates"] else "")
        print(f"[bc-eval] {label} [{contract}]: plant_rate={plant_rate:.3f} "
              f"({len(planted)}/{len(results)} distinct start states{dupes}) "
              f"ticks(min/median/max)={summary['ticks_min']}/{summary['ticks_median']}/"
              f"{summary['ticks_max']}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",
                        type=int,
                        default=64,
                        help="EPISODES (sequences) per minibatch, not ticks")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    # CPU is the default on purpose: the dataset is ~369 unique ticks and the
    # GPU on this box is usually busy with a PPO run.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-dedupe",
                        action="store_true",
                        help="keep byte-identical duplicate episodes (default: drop them)")
    parser.add_argument("--allow-stale-demos",
                        action="store_true",
                        help="train on demos recorded before an env/obs/map/expert change "
                        "(default: hard error — the labels may no longer be executable)")
    parser.add_argument("--eval-seeds",
                        type=int,
                        default=10,
                        help="demo-distribution eval: seeds 0..N-1 × 5 carrier slots, "
                        "deduped to the distinct start states")
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
        # weights_only=True matches src/train.py's load sites: a checkpoint is
        # data, and unpickling arbitrary objects out of one is a code-execution
        # surface we have no reason to keep open.
        policy.load_state_dict(
            torch.load(args.eval_only, map_location=args.device, weights_only=True))
        policy.eval()
        print(f"[train_bc] loaded {args.eval_only} for eval only")
    else:
        demos = load_demos(args.demos,
                           dedupe=not args.no_dedupe,
                           allow_stale=args.allow_stale_demos)
        print(f"[train_bc] {len(demos)} ticks from {demos.n_episodes} episodes "
              f"({demos.n_files} files, {demos.n_duplicates} byte-identical duplicates dropped); "
              f"episode ticks {int(demos.lengths.min())}–{int(demos.lengths.max())}; "
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

    # THE GATE: carried LSTM state, because that is the function PPO's rollout
    # and update both evaluate. Do not swap this for the stateless number.
    demo_eval = eval_plant_rate(policy,
                                seeds=range(args.eval_seeds),
                                device=args.device,
                                carry_state=True,
                                label="demo-distribution spawns")
    # Secondary diagnostic only. A large gap between the two lines means the
    # clone is leaning on (or being wrecked by) its recurrent state; a gap in
    # the direction stateless >> carried is the exact bug this file was
    # rewritten to kill.
    eval_plant_rate(policy,
                    seeds=range(args.eval_seeds),
                    device=args.device,
                    carry_state=False,
                    label="demo-distribution spawns (stateless diagnostic)")
    if args.holdout_seeds > 0:
        eval_plant_rate(policy,
                        seeds=range(1000, 1000 + args.holdout_seeds),
                        device=args.device,
                        carry_state=True,
                        label="unseen seeds (DEGENERATE: deterministic spawns → same 5 states)")

    passed = demo_eval["plant_rate"] >= args.gate_plant_rate
    print(f"[bc-eval] GATE 1 (spec §8, CARRIED-STATE rate): "
          f"plant_rate={demo_eval['plant_rate']:.3f} vs threshold "
          f"{args.gate_plant_rate:.3f} → {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("[bc-eval] FAIL means BC is broken — fix it before the PPO handoff (Task 5). "
              "Do not lower the threshold and do not gate on the stateless number.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
