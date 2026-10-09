"""Self-play: the past-policy pool (`SelfPlayManager`) and its one
builder, `build_selfplay_manager` (moved up from env.factory, #92).
"""

import random
from pathlib import Path
from typing import TYPE_CHECKING

from cs2rl.policy import (
    build_policy,
    load_state_dict_arch_checked,
    state_dict_is_split,
    state_dict_is_trunk_split,
)
from cs2rl.train.config import OPPONENT_MODES

if TYPE_CHECKING:
    # Only the `-> "torch.Tensor"` annotation reads it; the flat train.py got the name from the
    # `import torch` in its __main__ block, and a module-scope import would load torch.
    import torch

# ── SECTION: Self-Play ─────────────────────────────────────────────────────


def self_play_used_past_metric(trainer) -> float:
    """0.0/1.0 for metrics.jsonl. Persist filter drops non-floats.

    WHAT: expose whether this epoch's evaluate() rollout used a past-policy
      opponent (`trainer._selfplay_used_past`, set in
      `cs2rl.train.trainer.Cs2PuffeRL._draw_past_policy`).
    WHY: the persist filter on the outer logs dict drops non-floats, so a
      bool never reaches metrics.jsonl. Callers write the returned float
      onto the outer dict next to self_play/pool_size — never under
      losses/. The trainer declares the flag False at construction
      (`_init_selfplay`), so it is always there; no getattr default, which
      would turn a renamed attribute into a silent 0.0.
    PITFALL: do not log self_play/opponent_id. `load_past_policy` keeps
      the chosen path as a local; a string would also be dropped by the
      persist filter, and inventing a pool schema is out of scope.
    """
    return float(trainer._selfplay_used_past)


class SelfPlayManager:
    """Manages a pool of past checkpoints for self-play training.

    Every save_every_epochs epochs (or when the hero team win_rate > win_threshold),
    the current policy is saved to a pool.  With probability p_past, a random past
    checkpoint is used to supply actions for the *opponent* team during rollout
    collection.  This prevents the co-adaptation collapse that arises when both
    teams train against only the latest version of each other.

    Agent layout (per env, 10 agents total):
        slots 0-4  → T team
        slots 5-9  → CT team
    """

    AGENTS_PER_ENV = 10
    T_SLOTS = slice(0, 5)
    CT_SLOTS = slice(5, 10)
    # The team that plays OPPONENT at construction (CT attacks second). A
    # constant, not a bare literal in __init__, because Rung 1a T3 builds the
    # participation vector for the complement team BEFORE any manager exists —
    # see initial_hero_team().
    INITIAL_OPPONENT_TEAM = "ct"

    @classmethod
    def initial_hero_team(cls) -> str:
        """Team the HERO policy plays before any maybe_switch_teams flip.

        Rung 1a T3: build_participating_rows needs this at trainer-construction
        time, which is upstream of the SelfPlayManager instance. Deriving it
        from INITIAL_OPPONENT_TEAM (rather than hardcoding "t" at the call
        site) is what keeps the participation vector and the statue mask from
        silently disagreeing if the initial sides are ever swapped.
        Under a scripted --opponent (noop, walker) the value is constant for the whole run: the mode
        forbids self-play, and maybe_switch_teams is the only thing that flips
        opponent_team.
        """
        return "t" if cls.INITIAL_OPPONENT_TEAM == "ct" else "ct"

    def __init__(
        self,
        pool_size: int = 15,
        p_past: float = 0.3,
        save_every_epochs: int = 25,
        win_threshold: float = 0.6,
        phase_length: int = 50,
        aim_log_std_max=None,
        pin_pitch: bool = False,
        opponent_mode: str = "self",
    ):
        # R0-E (#131): run properties re-applied to every past policy built by
        # load_past_policy (they are non-persistent on the policy, so the
        # snapshot cannot carry them). A past opponent with an unpinned mask
        # would sample a live pitch dim the env ignores — harmless for the
        # env, but its stored logprob_c would include a factor the live
        # policy's does not, and self-play ratio_c would silently drift.
        self.aim_log_std_max = aim_log_std_max
        self.pin_pitch = bool(pin_pitch)
        # Rung 1a T3: "noop" makes the patched evaluate() overwrite this team's
        # actions with the no-op bin on every head (see _patch_trainer_with_
        # selfplay); "walker" does the same except the move head walks.
        # Validated here so a typo'd mode cannot reach the rollout
        # as a silently-inactive branch. Callers that pass "noop" or "walker" MUST also
        # have passed assert_opponent_self_play_compatible.
        if opponent_mode not in OPPONENT_MODES:
            raise ValueError(f"opponent_mode={opponent_mode!r} must be one of {OPPONENT_MODES}")
        self.opponent_mode = opponent_mode
        self.pool: list[Path] = []
        self.pool_size = pool_size
        self.p_past = p_past
        self.save_every_epochs = save_every_epochs
        self.win_threshold = win_threshold
        self.phase_length = phase_length
        self.opponent_team = self.INITIAL_OPPONENT_TEAM                # CT is opponent first; T learns to attack
        self._milestone_count = 0
        self._last_save_epoch = -1

    def state_dict(self) -> dict:
        """R0-C (#134): everything a full-state resume must restore.

        Paths are stringified AND resolve()d: the pool is filled with paths
        relative to --checkpoint-dir as given, while --resume-run resolves the
        run dir to absolute — a resume from another cwd would otherwise fail
        every exists() check in load_state_dict and empty the pool. The knobs
        (pool_size, p_past, ...) are NOT saved — they are rebuilt from args and
        guarded by check_resume_config via config.json.
        """
        return {
            "pool": [str(Path(p).resolve()) for p in self.pool],
            "opponent_team": self.opponent_team,
            "_milestone_count": self._milestone_count,
            "_last_save_epoch": self._last_save_epoch,
        }

    def load_state_dict(self, state: dict):
        """Inverse of state_dict. Pool entries whose file vanished are dropped
        (a later past-policy draw would crash on torch.load). opponent_team is
        restored explicitly: rebuilt-at-default would invert every later
        maybe_switch_teams toggle relative to the pre-crash run."""
        self.pool = [Path(p) for p in state["pool"] if Path(p).exists()]
        dropped = len(state["pool"]) - len(self.pool)
        if dropped:
            print(f"[SelfPlay] WARN: dropped {dropped}/{len(state['pool'])} pool entries whose "
                  "file no longer exists")
        print(f"[SelfPlay] pool restored: {len(self.pool)} entries, "
              f"opponent_team={state['opponent_team']}")
        self.opponent_team = state["opponent_team"]
        self._milestone_count = int(state["_milestone_count"])
        self._last_save_epoch = int(state["_last_save_epoch"])

    def maybe_save(
        self,
        policy,
        checkpoint_dir: Path,
        epoch: int,
        win_rate_t: float,
        win_rate_ct: float,
    ):
        """Save current policy to the pool if conditions are met."""
        import torch

        hero_win = win_rate_t if self.opponent_team == "ct" else win_rate_ct
        # Schedule: every save_every_epochs epochs, OR when hero is dominating
        # (win_threshold) but only if enough epochs have passed since last save.
        since_last = epoch - self._last_save_epoch
        scheduled = epoch % self.save_every_epochs == 0
        dominant = hero_win > self.win_threshold and since_last >= self.save_every_epochs // 2
        if scheduled or dominant:
            path = checkpoint_dir / f"sp_{epoch:06d}.pt"
            torch.save(policy.state_dict(), path)
            self._add_to_pool(path)
            self._milestone_count += 1
            self._last_save_epoch = epoch
            print(f"[SelfPlay] Saved checkpoint → {path.name}  "
                  f"(pool={len(self.pool)}, hero_win={hero_win:.2f})")

    def _add_to_pool(self, path: Path):
        self.pool.append(path)
        if len(self.pool) > self.pool_size:
            # Keep every 5th entry as milestone; evict the most recent non-milestone
            non_milestones = [i for i in range(len(self.pool) - 1) if i % 5 != 0]
            evict = non_milestones[-1] if non_milestones else 0
            evicted = self.pool.pop(evict)
            if evicted.exists():
                evicted.unlink(missing_ok=True)

    def maybe_switch_teams(self, epoch: int):
        if epoch > 0 and epoch % self.phase_length == 0:
            old = self.opponent_team
            self.opponent_team = "ct" if self.opponent_team == "t" else "t"
            print(f"[SelfPlay] Epoch {epoch}: opponent {old} → {self.opponent_team}")

    def should_use_past(self) -> bool:
        return bool(self.pool) and random.random() < self.p_past

    def load_past_policy(self, device, vecenv):
        """Load a random past checkpoint. Returns the policy module or None.

        Batch 7 (spec §3.3): the state_dict is read BEFORE build_policy so
        BOTH architecture bits (heads + trunk) can be inferred from its keys.
        This method receives no config and no flag — during a split run the
        pool fills with split snapshots, and a flag-only design would raise
        here on ~30% of epochs (p_past=0.3), hours into the run. Inference
        also lets a split run mix in pre-split snapshots from an older pool.
        """
        import torch

        if not self.pool:
            return None
        path = random.choice(self.pool)
        if not path.exists():
            self.pool.remove(path)
            return None
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Both bits inferred from keys — this method receives no config.
        policy = build_policy(vecenv,
                              device,
                              tct_split_heads=state_dict_is_split(state_dict),
                              tct_split_trunk=state_dict_is_trunk_split(state_dict),
                              aim_log_std_max=self.aim_log_std_max,
                              pin_pitch=self.pin_pitch)
        load_state_dict_arch_checked(policy, state_dict, source=str(path))
        policy.eval()
        return policy

    def get_opponent_mask(self, batch_n: int, device) -> "torch.Tensor":
        """Bool mask of shape (batch_n,): True for every opponent-team agent slot."""
        import torch

        n_envs = batch_n // self.AGENTS_PER_ENV
        mask = torch.zeros(batch_n, dtype=torch.bool, device=device)
        slots = self.CT_SLOTS if self.opponent_team == "ct" else self.T_SLOTS
        for e in range(n_envs):
            base = e * self.AGENTS_PER_ENV
            mask[base + slots.start:base + slots.stop] = True
        return mask


def build_selfplay_manager(*, self_play_enabled, aim_log_std_max, pin_pitch, opponent_mode):
    """Construct the run's `SelfPlayManager`. The only `SelfPlayManager(...)` call site.

    ONE BUILDER, NOT A ROLE ENUM, because the three pre-migration sites —
    `train()` and both branches of `_build_trainer_for_test` — differed in
    `p_past` and nothing else, and even that was the same rule twice:
    ``0.3 if <self-play on> else 0.0``. train() wrote it as a conditional
    expression, the harness spelled the two values out as separate constructions
    under ``if not with_selfplay:``. Naming a role per site would have frozen a
    distinction that does not exist and invited the next knob to be added to one
    "role" only — the exact drift this module exists to stop.
    `tests/fixtures/selfplay_kwargs_pre_w3.json` records all three shapes as they
    stood before this function, so the collapse is asserted rather than assumed.

    ``self_play_enabled`` is the flag, not ``p_past`` itself: p_past is derived
    policy, and a caller that could pass it directly could set 0.15 at one site
    and 0.3 at another without anything noticing.

    ``bool(pin_pitch)`` here rather than at the callers, which is where both
    pre-migration sites did it (`bool(args.pin_pitch)` / `bool(pin_pitch)`).
    `SelfPlayManager.__init__` also coerces, so this is belt-and-braces on the
    class's side — but it keeps the KWARG the callers produce identical to the
    captured one, which is what the fixture compares.

    ``aim_log_std_max`` is required rather than defaulted to None: train() reads
    it as ``getattr(args, "aim_log_std_max", None)`` and the harness takes it as a
    parameter, so both callers always have a value to pass, and a default here
    would let a future caller silently un-pin the aim head's log-std cap on past
    policies (R0-E, #131) — a divergence no gate on this branch can see.

    The four pool constants are literals with the call sites' own comments
    attached; they were identical across all three sites and are now stated once.
    """
    # #92: this builder lived in env.factory and imported SelfPlayManager from train
    # function-locally, an upward edge pyproject.toml had to ignore. It now sits beside
    # the class, and reads it as a module global: tests patch `train_selfplay.SelfPlayManager`.
    return SelfPlayManager(
        pool_size=15,
        p_past=0.3 if self_play_enabled else 0.0,
        save_every_epochs=25,                          # ~2M steps per save at batch_size=81920
        win_threshold=0.6,
        phase_length=50,                               # switch opponent team every ~4M steps
        aim_log_std_max=aim_log_std_max,
        pin_pitch=bool(pin_pitch),
                                                       # Rung 1a T3: "noop" ⇒ the patched evaluate() drives opponent_team as a
                                                       # statue. train() guards that it cannot combine with self-play, so
                                                       # opponent_team stays INITIAL_OPPONENT_TEAM — the same team
                                                       # build_participating_rows masked out. The harness mirrors that guard
                                                       # (assert_opponent_self_play_compatible) and passes the mode anyway, so
                                                       # the self-play and no-self-play paths cannot drift if the pairing is
                                                       # ever allowed.
        opponent_mode=opponent_mode,
    )
