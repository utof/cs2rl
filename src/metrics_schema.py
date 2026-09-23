"""The single registry of every metrics key this repo emits, derives or reads.

WHAT: `REGISTRY` maps a metrics key (or an f-string key FAMILY template) to a
`MetricSpec` carrying `kind`, `aggregation`, `units`, `consumers` — plus
`inputs` for derived columns, `members` for statically-closed families, and
`notes`. It also owns `EVAL_KEYS`, moved here from `eval_baselines` (see below).

WHY it is here and not a docstring: before W4 the aggregation contract of a key
lived in a comment next to its emitter, and the list of keys a gate script reads
lived in the gate script. Neither side could notice the other drifting. The
registry is the one place both are written down, and
`tests/test_metrics_schema.py` checks it against the SOURCE of the emitters and
of the readers (via the AST census in `tests/metrics_census.py`) in BOTH
directions — an unregistered key fails, and a registered key nothing emits fails
too. A registry that could only be wrong by omission would be a docstring again.

WHY THIS MODULE MUST STAY IMPORT-LIGHT (spec §2 W1, guarded by
`tests/test_w1_modules.py`): `EVAL_KEYS` used to live in `src/eval_baselines.py`,
which imports torch and `c_env.cs2_env` at module scope. Any consumer that wanted
those eight strings — including this registry — paid ~30 s of torch import for a
tuple of strings. So the ownership is INVERTED: `EVAL_KEYS` lives here and
`eval_baselines` does `from metrics_schema import EVAL_KEYS` (never the reverse;
that direction is what the import-lightness test exists to catch). The only
import in this module's scope is `_action_spec`, itself nothing but literals
auto-generated from `cs2_types.h`.

THE THREE KINDS
  emitted  — written by one of OUR emitters (the named island in
             `tests/metrics_census.EMITTER_SITES`). The structural aggregation
             assert applies to these.
  family   — an f-string-built key template, placeholders written `*`. CLOSED
             families additionally declare `members` (the exact concrete keys),
             which are ALSO registered individually so a reader can be attached
             to one of them; OPEN families (a genuinely runtime-valued
             placeholder — a module's named_parameters(), a `zip` over live
             objects) declare no members and the concrete keys are unenumerable
             by construction. Prefer closed: an OPEN template is accepted as a
             glob alibi by the reverse-completeness test, so every key under it
             goes unchecked.
  derived  — a registered NAME that no emitter writes: a gate-report column
             computed from emitted keys (`hit_per_facing`), a renamed statistic
             (`losses/approx_kl_p90`), or an OPTIONAL upstream key a reader
             presence-gates on but nothing currently produces
             (`environment/use_at_site_frac`). `inputs` names its sources; an
             empty `inputs` means "nothing in this repo produces it", which the
             notes then explain.

AGGREGATION IS STRUCTURAL, NOT EDITORIAL. The declared value must match the
SHAPE of the write, per `tests/metrics_census.SHAPES`:
  `self.stats[k] = [scalar]`  one-element list  → `last`   (PufferLib's np.mean
                                                            over a 1-list is an
                                                            identity; this is
                                                            `environment/episodes`)
  `self.stats[k]` via the append/extend collection loop → `window-mean-pufferlib`
  `losses[k] += ...` BEFORE the gh#90 divisor loop      → `mean`
  `losses[k] = ...`  AFTER  the gh#90 divisor loop      → `last`
  a write into `logs` / `log_entry` after `mean_and_log` → `last`
  a `game/*` re-key of a value `_get()` read out of `logs` → `window-mean-pufferlib`
Anything declared here that contradicts its emitter's shape fails the assert BY
KEY NAME. What is NOT claimed: per-key semantic correctness of a
`window-mean-pufferlib` value — that aggregation happens inside PufferLib's
`mean_and_log`, which this repo does not own, so those declarations are
documentation. See the spec's "honestly scoped" wording.

PITFALL — the readers are FROZEN, so the registry chases them. `scripts/
rung1_gate.py` and `scripts/rung1a_smoke_read.py` are registered evidence and are
never migrated to import from here; the completeness test AST-parses their key
literals instead. Their ROW LOADER (`scripts/analyze_tplant.py`, plus
`analyze_experiment.py` and `_iter_metrics_steps` in
`scripts/modal_runner/checkpoint.py`) is deliberately OUT of scope — recorded in
#155 as the future-reader residue.
"""
from typing import NamedTuple

from _action_spec import ACTION_HEAD_NAMES

# ── Vocabularies ──────────────────────────────────────────────────────────
#
# Closed sets, asserted in tests/test_metrics_schema.py. A free-text `units`
# column is the part of a registry nobody can check and everybody stops
# maintaining; making it a vocabulary means a new unit is a deliberate edit here.

KINDS = frozenset({"emitted", "derived", "family"})

# The spec's five, plus ONE documented extension. `dropped-non-numeric` is not an
# aggregation — it records that the value is not a number at all, so PufferLib's
# `mean_and_log` np.mean raises, and train.py's `isinstance(v, (int, float))`
# persist filter drops it before metrics.jsonl. Exactly one key is in this class
# (`environment/step_stats`). Declaring it `window-mean-pufferlib` instead would
# be the registry telling an analyst to interpret a ctypes view as a mean.
AGGREGATIONS = frozenset({
    "mean",
    "max",
    "sum",
    "last",
    "window-mean-pufferlib",
    "dropped-non-numeric",
})

UNITS = frozenset({
    "count",                           # integer event/entity count
    "fraction",                        # 0..1 rate
    "flag",                            # 0/1 indicator
    "ratio",                           # dimensionless quotient of two metrics
    "dimensionless",                   # a number with no natural unit (KL, cosine, norm)
    "nats",                            # entropy
    "reward",                          # reward units (the C reward scale)
    "hp",                              # health points of damage
    "units",                           # Source-engine distance units
    "ticks",                           # sim ticks
    "radians",                         # aim delta (Δyaw / absolute pitch), clamped
    "radians^2",                       # the Welford sum-of-squares companion
    "log-radians",                     # natural log of an aim σ expressed in radians
    "steps",                           # agent steps
    "epochs",                          # epoch index or count
    "seconds",
    "milliseconds",
    "id",                              # opaque string identifier
    "struct",                          # a non-numeric payload object
})

# Named in-repo readers of a metrics row. Both frozen gate scripts plus the three
# train-side consumers; `tests/metrics_census.consumer_key_reads()` derives the
# actual reads from source and the test compares BOTH directions.
CONSUMERS = frozenset({
    "rung1_gate",
    "rung1a_smoke_read",
    "format_train_status",
    "elimination_only_win_rates",
    "compute_game_metrics",
})

# Who COMPUTES a `derived` column. Distinct from `consumers` on purpose:
# `kills_per_episode` the gate column is written BY rung1_gate out of keys it
# reads, so listing rung1_gate as its consumer would claim the gate reads a
# column it invents — and the both-directions consumer check would then have to
# be weakened to let that claim through.
PRODUCERS = frozenset({"rung1_gate"})


class MetricSpec(NamedTuple):
    """One registry row. See the module docstring for the field contracts."""
    kind: str
    aggregation: str
    units: str
    consumers: tuple = ()              # in-repo readers of this key (never its author)
    inputs: tuple = ()                 # derived only: the keys it is computed from
    members: tuple = ()                # family only: closed member list, () when open
    producer: str = ""                 # derived only: the script that computes it
    notes: str = ""


def _e(aggregation, units, consumers=(), notes=""):
    """An `emitted` entry."""
    return MetricSpec("emitted", aggregation, units, tuple(consumers), notes=notes)


def _d(units, inputs=(), producer="", consumers=(), notes=""):
    """A `derived` entry. Aggregation is `last` — a derived column is computed
    once per report over an already-aggregated window, never re-aggregated."""
    return MetricSpec("derived",
                      "last",
                      units,
                      tuple(consumers),
                      inputs=tuple(inputs),
                      producer=producer,
                      notes=notes)


def _f(aggregation, units, members=(), consumers=(), notes=""):
    """A `family` entry keyed by its `*`-placeholder template."""
    return MetricSpec("family",
                      aggregation,
                      units,
                      tuple(consumers),
                      members=tuple(members),
                      notes=notes)


# ── EVAL_KEYS — moved here from eval_baselines (spec §2 W4) ───────────────

# eval/* keys emitted by BaselineEvaluator.evaluate — the analysis contract.
# Lives HERE, not in eval_baselines: that module imports torch and
# c_env.cs2_env at module scope, so `from eval_baselines import EVAL_KEYS`
# would make a tuple of 8 strings cost a torch import. eval_baselines imports
# it back and re-exports it for existing consumers.
EVAL_KEYS = (
    "eval/win_vs_random",
    "eval/win_vs_random_as_t",
    "eval/win_vs_random_as_ct",
    "eval/kills_per_episode_vs_random",
    "eval/win_vs_oracle",
    "eval/win_vs_oracle_as_t",
    "eval/win_vs_oracle_as_ct",
    "eval/kills_per_episode_vs_oracle",
)

# ScheduledEval stamps these two ON TOP of EVAL_KEYS before handing the buffer to
# the next logged row (train_metrics.ScheduledEval.after_train). The registry's
# eval/* surface is the union — EVAL_KEYS alone would under-declare it by two.
EVAL_EXTRA_KEYS = ("eval/epoch", "eval/wall_s")
EVAL_SURFACE = EVAL_KEYS + EVAL_EXTRA_KEYS

# ── The registry ──────────────────────────────────────────────────────────

REGISTRY = {}

# ── PufferLib's own row keys ──────────────────────────────────────────────
# Written by pufferl.mean_and_log itself, not by us: this repo cannot enforce
# their aggregation, only record it. `agent_steps` is the PARTICIPATING step
# counter both gate scripts key their window on (rung1_gate.gate_window) — the
# single most load-bearing key in the file that no emitter of ours writes.
#
# PUFFERLIB_OWNED is the declared provenance the completeness test needs: these
# entries are legitimately absent from the emitter census, and saying so HERE
# rather than as a skip-list inside the test keeps the exemption reviewable next
# to the entries it exempts. The test also parses pufferl.mean_and_log's own
# `logs = {...}` literal, so a PufferLib rename of `agent_steps` — which both
# frozen gate scripts key their window on — fails rather than silently emptying
# every gate window. `epoch` is NOT here: mean_and_log writes it, but so does
# train.py's log_entry literal, and ours is the one that lands in the row.
PUFFERLIB_OWNED = ("agent_steps", "SPS", "uptime", "learning_rate", "performance/*")

REGISTRY.update({
    "agent_steps":
    _e(
        "last", "steps", ("rung1_gate", "rung1a_smoke_read"),
        "PufferLib global_step. rung1a_smoke_read reads it inline at three sites; "
        "no *_KEY constant exists for it in either script."),
    "SPS":
    _e("last", "count", ("format_train_status", ),
       "PufferLib steps-per-second. Dropped by the §3 gate's row filter (timing)."),
    "uptime":
    _e("last", "seconds", (), "PufferLib wall clock. Dropped by the §3 gate's row filter."),
    "learning_rate":
    _e("last", "dimensionless", (), "PufferLib optimizer LR readback."),
    "performance/*":
    _f(
        "last", "seconds", (), (),
        "PufferLib's own profiler sections (env, eval, eval_copy, eval_forward, "
        "eval_misc, learn, train, train_copy, train_forward, train_misc). OPEN: the "
        "member list is whatever profile() was called with. Dropped by the §3 filter."),
})

# ── train.py's persisted row literals ─────────────────────────────────────
# Written straight into `log_entry`, never through PufferLib. `epoch` is also a
# mean_and_log key; train.py's literal overwrites it with the same value.
REGISTRY.update({
    "run_id":
    _e(
        "last", "id", (),
        "String key — segment runs by this, not by agent_steps, which resets on resume. "
        "The only non-numeric key that survives the persist filter (it is a log_entry "
        "literal, not a `logs` entry). Dropped by the §3 gate's row filter."),
    "step":
    _e("last", "steps", (), "trainer.global_step at the moment the row was written."),
    "epoch":
    _e("last", "epochs", ()),
    "team_spirit":
    _e("last", "fraction", (), "Current team-spirit schedule value."),
    "resumed_from_step":
    _e(
        "last", "steps", (),
        "PRESENT ONLY on the first row after a --resume-run (R0-C analysis seam "
        "marker); absent from every other row by design."),
})

# ── environment/* — Cs2Env._build_terminal_info, one entry per episode ────
# Every key here is appended into `self.stats` by train.py's info-collection loop
# and np.mean'd over the collection window by mean_and_log, hence
# window-mean-pufferlib. `environment/episodes` is the one exception (below).
_GAME_METRICS = ("compute_game_metrics", )
REGISTRY.update({
    "environment/bomb_planted":
    _e("window-mean-pufferlib", "flag", ("compute_game_metrics", "format_train_status")),
    "environment/bomb_defused":
    _e("window-mean-pufferlib", "flag", _GAME_METRICS),
    "environment/kills_t":
    _e("window-mean-pufferlib", "count", ("compute_game_metrics", "format_train_status")),
    "environment/kills_ct":
    _e("window-mean-pufferlib", "count", ("compute_game_metrics", "format_train_status")),
    "environment/blocked_moves_t":
    _e("window-mean-pufferlib", "count", ()),
    "environment/blocked_moves_ct":
    _e("window-mean-pufferlib", "count", ()),
    "environment/aim_delta_sum":
    _e(
        "window-mean-pufferlib", "radians", (),
        "Welford triple with aim_delta_sq_sum/aim_delta_count: mean = sum/count, "
        "var = sq_sum/count - mean². Accumulates the CLAMPED Δyaw the env executed."),
    "environment/aim_delta_sq_sum":
    _e("window-mean-pufferlib", "radians^2", ()),
    "environment/aim_delta_count":
    _e("window-mean-pufferlib", "count", ()),
    "environment/aim_delta_pitch_sum":
    _e("window-mean-pufferlib", "radians", (),
       "Pitch is ABSOLUTE (no accumulator), unlike the Δyaw triple above."),
    "environment/aim_delta_pitch_sq_sum":
    _e("window-mean-pufferlib", "radians^2", ()),
    "environment/aim_delta_pitch_count":
    _e("window-mean-pufferlib", "count", ()),
    "environment/winner":
    _e("window-mean-pufferlib", "flag", ()),
    "environment/winner_t":
    _e("window-mean-pufferlib", "flag",
       ("compute_game_metrics", "elimination_only_win_rates", "format_train_status")),
    "environment/winner_ct":
    _e(
        "window-mean-pufferlib", "flag",
        ("compute_game_metrics", "elimination_only_win_rates", "format_train_status"),
        "Counts TIMEOUTS as CT wins in C — which is why elimination_only_win_rates "
        "subtracts environment/timed_out before feeding self-play."),
    "environment/timed_out":
    _e("window-mean-pufferlib", "flag",
       ("compute_game_metrics", "elimination_only_win_rates", "format_train_status")),
    "environment/alive_t_end":
    _e("window-mean-pufferlib", "count", ()),
    "environment/alive_ct_end":
    _e("window-mean-pufferlib", "count", ()),
    "environment/round_length":
    _e("window-mean-pufferlib", "ticks", ("compute_game_metrics", "format_train_status")),
    "environment/plant_tick":
    _e(
        "window-mean-pufferlib", "ticks", _GAME_METRICS,
        "Observe-only C field; 0 = never planted. PRESENCE-GATED downstream: "
        "compute_game_metrics only re-keys it when it is already in logs."),
    "environment/win_by_detonation":
    _e("window-mean-pufferlib", "flag", _GAME_METRICS, "Presence-gated downstream."),
    "environment/win_by_defuse":
    _e("window-mean-pufferlib", "flag", _GAME_METRICS, "Presence-gated downstream."),
    "environment/reward_win":
    _e(
        "window-mean-pufferlib", "reward", (),
        "Cross-team SUM; nets ~0 by the zero-sum identity, which is why #128 emits no "
        "game/reward/win and the one-sided reward_win_t/_ct carry the signal."),
    "environment/reward_kills":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_deaths":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_bomb":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_pbrs":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_shots":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_survival":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_inaction":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_win_t":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/reward_win_ct":
    _e("window-mean-pufferlib", "reward", _GAME_METRICS),
    "environment/shots_fired":
    _e("window-mean-pufferlib", "count", _GAME_METRICS),
    "environment/shots_with_enemy_in_los":
    _e("window-mean-pufferlib", "count", _GAME_METRICS),
    "environment/shots_facing_enemy":
    _e("window-mean-pufferlib", "count", _GAME_METRICS),
    "environment/shots_on_target":
    _e("window-mean-pufferlib", "count", _GAME_METRICS),
    "environment/shots_hit":
    _e("window-mean-pufferlib", "count", _GAME_METRICS),
    "environment/shots_stance_blocked":
    _e("window-mean-pufferlib", "count", _GAME_METRICS),
    "environment/damage_dealt":
    _e("window-mean-pufferlib", "hp", _GAME_METRICS),
    "environment/mutual_vis_pair_ticks":
    _e("window-mean-pufferlib", "ticks", _GAME_METRICS),
    "environment/agent_ticks_with_visible_enemy":
    _e("window-mean-pufferlib", "ticks", _GAME_METRICS),
    "environment/min_enemy_distance_sum":
    _e(
        "window-mean-pufferlib", "units", _GAME_METRICS,
        "The 1e30 no-contact sentinel is resolved PER EPISODE in cs2_env (emit 0.0 and "
        "flag it invalid) — one sentinel inside a 40-episode window would otherwise "
        "average to ~2.5e28."),
    "environment/min_enemy_distance_valid":
    _e("window-mean-pufferlib", "flag", _GAME_METRICS,
       "Denominator of the game/min_enemy_distance conditional mean."),
                                                                                           # The one non-window key on this path, and the one the aggregation assert has
                                                                                           # an explicit case for: train_update.py writes it as a ONE-ELEMENT LIST so
                                                                                           # PufferLib's np.mean is an identity. It is the denominator of every
                                                                                           # weighted_sum(...)/episodes ratio in both gate scripts.
    "environment/episodes":
    _e(
        "last", "count", ("rung1_gate", "rung1a_smoke_read"),
        "ONE-ELEMENT-LIST wrap in train_update._train_with_return_norm — np.mean over a "
        "1-list is an identity, so this is the episode COUNT of the window, not a mean. "
        "Turning it into a bare write would silently divide every gate ratio by the "
        "window length; that is the case the aggregation assert pins by name."),
                                                                                           # Not a metric: a ctypes view merged into the terminal info when
                                                                                           # include_step_stats_in_info=True (the test harness; train.py passes False).
    "environment/step_stats":
    _e(
        "dropped-non-numeric", "struct", (),
        "StepStatsView, not a scalar: mean_and_log's np.mean raises and train.py's "
        "isinstance(v, (int, float)) persist filter drops it, so it never reaches "
        "metrics.jsonl. Registered because it IS written to the info dict — an "
        "unregistered write is exactly what this file exists to make impossible."),
})

# environment/action_* — per-bin action histograms, one f-string family per head.
# The bin counts come from _action_spec (auto-generated from cs2_types.h), and the
# census resolves every member statically, so a head gaining a bin fails the
# closed-member assert rather than silently going unregistered.
# NOTE: rung1a_smoke_read builds `environment/action_move_*` by REGEX
# (`MOVE_KEY_RE`, no key literal to extract) and additionally hard-codes
# `environment/action_move_0`; format_train_status reads `environment/action_move_1`.
REGISTRY.update({
    "environment/action_move_*":
    _f("window-mean-pufferlib", "count", tuple(f"environment/action_move_{i}" for i in range(9)),
       (), "9 move bins. rung1a_smoke_read regenerates this family with MOVE_KEY_RE."),
    "environment/action_shoot_*":
    _f("window-mean-pufferlib", "count", tuple(f"environment/action_shoot_{i}" for i in range(2))),
    "environment/action_use_*":
    _f("window-mean-pufferlib", "count", tuple(f"environment/action_use_{i}" for i in range(2))),
    "environment/action_reload_*":
    _f("window-mean-pufferlib", "count", tuple(f"environment/action_reload_{i}" for i in range(2))),
    "environment/action_weapon_*":
    _f("window-mean-pufferlib", "count", tuple(f"environment/action_weapon_{i}" for i in range(3))),
    "environment/action_crouch_*":
    _f("window-mean-pufferlib", "count", tuple(f"environment/action_crouch_{i}" for i in range(2))),
    "environment/action_jump_*":
    _f("window-mean-pufferlib", "count", tuple(f"environment/action_jump_{i}" for i in range(2))),
})
for _head, _bins in (("move", 9), ("shoot", 2), ("use", 2), ("reload", 2), ("weapon", 3),
                     ("crouch", 2), ("jump", 2)):
    for _i in range(_bins):
        _cons = ()
        if _head == "move" and _i == 0:
            _cons = ("rung1a_smoke_read", )
        elif _head == "move" and _i == 1:
            _cons = ("format_train_status", )
        REGISTRY[f"environment/action_{_head}_{_i}"] = _e("window-mean-pufferlib", "count", _cons)

# ── losses/* — trainer.losses, prefixed by mean_and_log ───────────────────
# `mean` = accumulated across minibatches then divided by the EXECUTED minibatch
# count by the gh#90 divisor loop. `last` = written AFTER that loop, so it is an
# absolute epoch scalar. Moving a write across that loop changes the aggregation,
# which is exactly what the structural assert re-derives from source every run.
REGISTRY.update({
    "losses/policy_loss":
    _e("mean", "dimensionless", ()),
    "losses/value_loss":
    _e("mean", "dimensionless", ()),
    "losses/entropy":
    _e("mean", "nats", (), "Masked mean entropy over participating rows."),
    "losses/entropy_unmasked":
    _e("mean", "nats", ()),
    "losses/alpha":
    _e("mean", "dimensionless", (), "Entropy-coefficient α (the value used)."),
    "losses/alpha_loss":
    _e("mean", "dimensionless", ()),
    "losses/old_approx_kl":
    _e("mean", "dimensionless", ()),
    "losses/approx_kl":
    _e("mean", "dimensionless", ("rung1_gate", ),
       "rung1_gate reads this as the SOURCE of its losses/approx_kl_p90 column."),
    "losses/clipfrac":
    _e("mean", "fraction", ()),
    "losses/clipfrac_d":
    _e("mean", "fraction", (), "Discrete-head clip fraction."),
    "losses/clipfrac_c":
    _e("mean", "fraction", (), "Continuous (aim) clip fraction."),
    "losses/importance":
    _e("mean", "dimensionless", ()),
    "losses/entropy/total":
    _e("mean", "nats", (), "Accumulated alongside the per-head entropy/<head> family."),
    "losses/minibatches_run":
    _e(
        "last", "count", (),
        "Inserted immediately AFTER the gh#90 divisor loop — it is the divisor itself "
        "and must not be divided by it."),
    "losses/entropy_floor_fires":
    _e("last", "count", ()),
    "losses/empty_minibatches":
    _e("last", "count", ("rung1_gate", )),
    "losses/participating_rows":
    _e("last", "count", ("rung1a_smoke_read", )),
    "losses/warmstart_phase":
    _e("last", "flag", ()),
    "losses/warmstart_h_over_h0":
    _e("last", "ratio", (), "Post-divisor losses/entropy over the warm-start reference H0."),
    "losses/explained_variance":
    _e("last", "dimensionless", ()),
    "losses/ret_mean":
    _e("last", "dimensionless", ()),
    "losses/ret_std":
    _e("last", "dimensionless", ()),
    "losses/event_oversample_fraction":
    _e("last", "fraction", ()),
    "losses/log_alpha":
    _e("last", "dimensionless", ()),
    "losses/effective_alpha":
    _e("last", "dimensionless", ("rung1_gate", )),
    "losses/entropy/*":
    _f(
        "mean", "nats", tuple(f"losses/entropy/{h}" for h in ACTION_HEAD_NAMES), (),
        "Per-discrete-head entropy. OPEN in the AST census (the loop iterates the "
        "imported ACTION_HEAD_NAMES, not a literal), so the member list is taken from "
        "_action_spec here — the same auto-generated tuple the emitter zips over, which "
        "is what makes a new head in cs2_types.h propagate instead of drifting."),
})
for _head in ACTION_HEAD_NAMES:
    REGISTRY[f"losses/entropy/{_head}"] = _e("mean", "nats",
                                             ("rung1_gate", ) if _head == "shoot" else ())

# ── game/* and actions/* — compute_game_metrics ───────────────────────────
# Every value here is re-keyed out of a PufferLib window mean that mean_and_log
# already computed, so the aggregation belongs upstream (pass-through). The
# arithmetic is linear except game/min_enemy_distance, noted at its entry.
REGISTRY.update({
    "game/win_rate_t":
    _e("window-mean-pufferlib", "fraction", ()),
    "game/win_rate_ct":
    _e("window-mean-pufferlib", "fraction", (),
       "CT win rate INCLUDING timeouts (see environment/winner_ct)."),
    "game/timeout_rate":
    _e("window-mean-pufferlib", "fraction", ()),
    "game/kills_per_episode":
    _e(
        "window-mean-pufferlib", "count", ("rung1_gate", "rung1a_smoke_read"),
        "kills_t + kills_ct — both window means, so their sum is the window mean of the "
        "sum. Do NOT confuse with rung1_gate's `kills_per_episode` COLUMN, which is the "
        "episode-weighted ratio over the whole gate window (registered separately)."),
    "game/bomb_plant_rate":
    _e("window-mean-pufferlib", "fraction", ()),
    "game/avg_episode_length":
    _e("window-mean-pufferlib", "ticks", ()),
    "game/defuse_rate":
    _e("window-mean-pufferlib", "fraction", ()),
    "game/kills_t":
    _e("window-mean-pufferlib", "count", ()),
    "game/kills_ct":
    _e("window-mean-pufferlib", "count", ("rung1a_smoke_read", )),
    "game/reward/kills":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/deaths":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/bomb":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/pbrs":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/shots":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/survival":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/inaction":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/win_t":
    _e("window-mean-pufferlib", "reward", ()),
    "game/reward/win_ct":
    _e("window-mean-pufferlib", "reward", (),
       "There is deliberately no game/reward/win: the C cross-team sum nets ~0 (#128)."),
    "game/min_enemy_distance":
    _e(
        "window-mean-pufferlib", "units", ("rung1_gate", ),
        "RATIO of two window means (sum / valid) = the conditional mean over episodes "
        "where a pair coexisted. Declared window-mean-pufferlib because both operands "
        "are; the quotient itself is not a plain window mean. A max(·,1) guard would "
        "silently return the unconditional mean, hence the explicit zero-valid branch."),
    "game/min_enemy_distance_valid_frac":
    _e("window-mean-pufferlib", "fraction", ("rung1_gate", )),
    "game/plant_tick":
    _e(
        "window-mean-pufferlib", "ticks", (),
        "PRESENCE-GATED: emitted only when environment/plant_tick is already in logs. A "
        "synthetic 0.0 would make an old-format row look new-format. Absent from the §3 "
        "two-epoch row, so the emission half of this registry is blind to it — which is "
        "why the census reads the emitter's SOURCE."),
    "game/win_by_detonation":
    _e("window-mean-pufferlib", "fraction", (), "Presence-gated (see game/plant_tick)."),
    "game/win_by_defuse":
    _e("window-mean-pufferlib", "fraction", (), "Presence-gated (see game/plant_tick)."),
    "game/*":
    _f(
        "window-mean-pufferlib", "count",
        tuple(f"game/{k}" for k in (
            "shots_fired",
            "shots_with_enemy_in_los",
            "shots_facing_enemy",
            "shots_on_target",
            "shots_hit",
            "shots_stance_blocked",
            "damage_dealt",
            "mutual_vis_pair_ticks",
            "agent_ticks_with_visible_enemy",
        )), (), "The combat-counter re-key loop. CLOSED (the loop iterates a literal tuple), so "
        "every member is registered individually below and a tenth counter fails here."),
    "game/shots_fired":
    _e(
        "window-mean-pufferlib", "count", ("rung1_gate", "rung1a_smoke_read"),
        "rung1_gate's key-drift tripwire: absent from EVERY window row → INVALID, not a "
        "gate failure."),
    "game/shots_with_enemy_in_los":
    _e("window-mean-pufferlib", "count", ("rung1_gate", )),
    "game/shots_facing_enemy":
    _e("window-mean-pufferlib", "count", ("rung1_gate", )),
    "game/shots_on_target":
    _e("window-mean-pufferlib", "count", ("rung1_gate", "rung1a_smoke_read")),
    "game/shots_hit":
    _e("window-mean-pufferlib", "count", ("rung1_gate", )),
    "game/shots_stance_blocked":
    _e("window-mean-pufferlib", "count", ("rung1_gate", )),
    "game/damage_dealt":
    _e("window-mean-pufferlib", "hp", ()),
    "game/mutual_vis_pair_ticks":
    _e("window-mean-pufferlib", "ticks", ("rung1_gate", )),
    "game/agent_ticks_with_visible_enemy":
    _e("window-mean-pufferlib", "ticks", ("rung1_gate", )),
    "actions/use_at_site_frac":
    _e(
        "window-mean-pufferlib", "fraction", (),
        "Emitted only when environment/use_at_site_frac is present in logs — nothing "
        "writes that source today, so this key never appears in a current run. Its own "
        "namespace, and the hand census missed it entirely."),
})

# ── policy/* — log_aim_log_std ────────────────────────────────────────────
# Written into the outer `logs` dict after mean_and_log, one value per logged
# row. Under a SPLIT policy the eight _t/_ct[_raw] keys exist and the unsuffixed
# pair becomes mean-of-clamped / MAX-of-raw; under pin_pitch every `pitch` key is
# omitted rather than emitted as NaN.
REGISTRY.update({
    "policy/aim_log_std_yaw":
    _e(
        "last", "log-radians", ("rung1_gate", "format_train_status"),
        "Clamped to the RUN's aim_log_std_max, not the module constant. Split policy: "
        "MEAN of the two clamped copies."),
    "policy/aim_log_std_pitch":
    _e(
        "last", "log-radians", ("format_train_status", ),
        "Omitted under pin_pitch (aim_dim_mask[1] == 0): the parameter is dead, and a "
        "NaN would read as live-but-broken on a dashboard."),
    "policy/aim_log_std_yaw_raw":
    _e(
        "last", "log-radians", ("rung1a_smoke_read", ),
        "UNCLAMPED. Split policy: MAX of the two copies, deliberately not the mean — the "
        "question is 'did any copy overshoot the cap and go gradient-dead'."),
    "policy/aim_log_std_pitch_raw":
    _e("last", "log-radians", ()),
    "policy/aim_log_std_yaw_t":
    _e("last", "log-radians", (), "SPLIT-ARCHITECTURE ONLY (policy has aim_log_std_t)."),
    "policy/aim_log_std_yaw_ct":
    _e("last", "log-radians", (), "Split-architecture only."),
    "policy/aim_log_std_yaw_t_raw":
    _e("last", "log-radians", (), "Split-architecture only."),
    "policy/aim_log_std_yaw_ct_raw":
    _e("last", "log-radians", (), "Split-architecture only."),
    "policy/aim_log_std_pitch_t":
    _e("last", "log-radians", (), "Split-architecture only; also omitted under pin_pitch."),
    "policy/aim_log_std_pitch_ct":
    _e("last", "log-radians", (), "Split-architecture only; also omitted under pin_pitch."),
    "policy/aim_log_std_pitch_t_raw":
    _e("last", "log-radians", (), "Split-architecture only; also omitted under pin_pitch."),
    "policy/aim_log_std_pitch_ct_raw":
    _e("last", "log-radians", (), "Split-architecture only; also omitted under pin_pitch."),
})

# ── split/* — architecture labelling and divergence ───────────────────────
REGISTRY.update({
    "split/active":
    _e(
        "last", "flag", (),
        "UNCONDITIONAL — every logged row must carry it, including in a non-split run "
        "(where it is 0.0). Derived from the live policy object, never config.json, "
        "which is rewritten at every launch and lies after a flag-less resume. Its "
        "placement OUTSIDE the --tag-diagnostic hook is part of the contract."),
    "split/trunk_active":
    _e("last", "flag", (), "Unconditional twin of split/active, keyed on encoder_t."),
    "split/head_l2_rel/*":
    _f(
        "last", "ratio",
        tuple(f"split/head_l2_rel/{n}" for n in ("action_heads", "aim_mu", "aim_log_std")), (),
        "ARCHITECTURE-GATED: compute_head_divergence returns {} without action_heads_t, "
        "so these keys are genuinely absent in a non-split run — unlike split/active."),
    "split/trunk_l2_rel/*":
    _f("last", "ratio", tuple(f"split/trunk_l2_rel/{n}" for n in ("encoder", "lstm")), (),
       "Architecture-gated on encoder_t. Modules are GROUPED, not per-tensor."),
})
for _n in ("action_heads", "aim_mu", "aim_log_std"):
    REGISTRY[f"split/head_l2_rel/{_n}"] = _e(
        "last", "ratio", (), "‖W_t − W_ct‖ / (0.5‖W_t‖ + 0.5‖W_ct‖); NaN propagates rather "
        "than reading as 'teams identical'.")
for _n in ("encoder", "lstm"):
    REGISTRY[f"split/trunk_l2_rel/{_n}"] = _e("last", "ratio", (),
                                              "Same relative-L2 formula as the head family.")

# ── health/* — compute_network_health ─────────────────────────────────────
REGISTRY["health/weight_norm_*"] = _f(
    "last", "dimensionless", (), (),
    "L2 norm per named parameter, key built from name.replace('.', '_'). OPEN: the "
    "member list is the model's named_parameters(), which depends on the architecture. "
    "Gated on epoch % 5 at the call site, so it is absent from most rows.")

# ── tag/* — TAG gradient diagnostics (--tag-diagnostic) ───────────────────
# tag_grad_cossim (train_update.py) builds these; _inject_tag_metrics merges its
# dict into logs after mean_and_log and emits NO keys of its own — so a census
# built from the merging helper alone would register zero tag/* entries and still
# pass. The families are CLOSED on both placeholders, and neither axis is visible
# in the emitter body alone:
#   group axis  `for g in pg_group_names`, one hop back to the literal
#               `pg_group_names = ("trunk", "policy_heads")` in the same function.
#   label axis  `mb_label` is a PARAMETER; the single call site passes
#               `mb_label="mb0" if _tag_mb0 else "mbL"` (mb0 = the epoch's first
#               minibatch, mbL = the throttled later one).
# tests/metrics_census.py resolves both from source and
# test_metrics_schema.test_tag_families_are_census_closed_on_both_axes pins that
# they stay resolved — an OPEN tag/* template would alibi any `tag/...` entry the
# registry cared to invent, which is the accumulation the reverse-completeness
# half exists to stop.
TAG_PARAM_GROUPS = ("trunk", "policy_heads")
TAG_MB_LABELS = ("mb0", "mbL")
_TAG_NOTE = ("FLAG-GATED on --tag-diagnostic and throttled by --tag-every, so absent from "
             "a default run's rows. Carries deliberate NaNs for zero-norm subsets — which "
             "is why injection must happen AFTER dead_run_detector.check.")
_TAG_STATS = ("cossim_cross", "cossim_cross_half", "cossim_within_t", "cossim_within_ct", "gnorm_t",
              "gnorm_ct")


def _tag_members(stat):
    return tuple(f"tag/{stat}/{g}/{lbl}" for g in TAG_PARAM_GROUPS for lbl in TAG_MB_LABELS)


REGISTRY.update({
    f"tag/{_s}/*/*": _f("last", "dimensionless", _tag_members(_s), (), _TAG_NOTE)
    for _s in _TAG_STATS
})
REGISTRY.update({
    "tag/cossim_vf/*":
    _f("last", "dimensionless", tuple(f"tag/cossim_vf/{lbl}" for lbl in TAG_MB_LABELS), (),
       _TAG_NOTE),
    "tag/mbL_index":
    _e(
        "last", "count", (),
        "Written onto trainer._tag_metrics inside the minibatch loop but merged into "
        "logs AFTER mean_and_log — never routed through `losses`, whose keys are divided "
        "by the minibatch count (gh#90) and lag environment/* by one epoch."),
    "tag/selfplay_active":
    _e("last", "flag", ()),
})
# The 26 concrete tag/* keys, one entry per closed-family member (a closed
# family's members carry their own entries — that is what lets a reader be
# attached to one, and what makes a member declared differently from its
# template a failure). Notes are per STAT, since the two axes mean the same
# thing in all seven: `g` is the parameter group the gradients were flattened
# over, `mb0`/`mbL` is which minibatch of the epoch was measured.
_TAG_STAT_NOTES = {
    "cossim_cross":
    "cos(g_T, g_CT) over the FULL subsets — the lower-noise descriptive number, NOT the "
    "criterion statistic: a full-size cosine has a larger expected same-distribution value "
    "than any n/2 one, so comparing it against the within-team null biases toward 'no "
    "conflict'. Use cossim_cross_half for that.",
    "cossim_cross_half":
    "THE criterion statistic (spec 2026-08-13 §4.2): cos(g_Ta, g_CTa), size-matched at n/2 "
    "rows to the within-team null so within − cross is an unbiased conflict estimate.",
    "cossim_within_t":
    "Within-team null for T: cos(g_Ta, g_Tb) across the env-parity halves, which are "
    "exchangeable — the same-distribution reference cossim_cross_half is read against.",
    "cossim_within_ct":
    "Within-team null for CT, same env-parity construction as the T half.",
    "gnorm_t":
    "‖g_T‖ of the subset policy-gradient. Scaled by the parked-row rescale at "
    "n_active < TEAM_SIZE (cosine is invariant to it; a norm is not).",
    "gnorm_ct":
    "‖g_CT‖, the CT twin of gnorm_t and subject to the same parked-row scaling.",
}
for _s in _TAG_STATS:
    for _m in _tag_members(_s):
        REGISTRY[_m] = _e("last", "dimensionless", (), _TAG_STAT_NOTES[_s] + " " + _TAG_NOTE)
for _lbl in TAG_MB_LABELS:
    REGISTRY[f"tag/cossim_vf/{_lbl}"] = _e(
        "last", "dimensionless", (),
        "Known-anticorrelated CONTROL over value_head params only (the pg graph never "
        "touches value_head). Biased UPWARD at n_active < TEAM_SIZE: a parked row's "
        "normalized return is -mean/std rather than 0, so it adds a common-mode residual to "
        "both team value gradients. " + _TAG_NOTE)

# ── self_play/* — train.py, next to the pool bookkeeping ──────────────────
REGISTRY.update({
    "self_play/pool_size":
    _e("last", "count", ()),
    "self_play/used_past":
    _e(
        "last", "flag", ("rung1_gate", "rung1a_smoke_read"),
        "0.0/1.0 float, not a bool: the persist filter keeps ints/floats only. "
        "rung1_gate EXCLUDES rows where this is 1.0 from its gate window."),
    "self_play/opponent_team":
    _e("last", "flag", ("rung1a_smoke_read", ), "1.0 = CT opponent, 0.0 = T opponent."),
})

# ── timing/* — the trainer.train wall-clock patch ─────────────────────────
REGISTRY.update({
    "timing/collect_ms":
    _e("last", "milliseconds", (), "Dropped by the §3 gate's row filter (wall-clock)."),
    "timing/update_ms":
    _e("last", "milliseconds", (), "Dropped by the §3 gate's row filter (wall-clock)."),
})

# ── eval/* — BaselineEvaluator via ScheduledEval ──────────────────────────
# The eight EVAL_KEYS are evaluate()'s output contract; ScheduledEval adds the
# two below. All ten are written into `logs` after mean_and_log, one value per
# logged row, and are absent unless --eval-interval is on.
_EVAL_UNITS = {
    "eval/win_vs_random": "fraction",
    "eval/win_vs_random_as_t": "fraction",
    "eval/win_vs_random_as_ct": "fraction",
    "eval/kills_per_episode_vs_random": "count",
    "eval/win_vs_oracle": "fraction",
    "eval/win_vs_oracle_as_t": "fraction",
    "eval/win_vs_oracle_as_ct": "fraction",
    "eval/kills_per_episode_vs_oracle": "count",
}
_EVAL_CONSUMERS = {
    "eval/win_vs_random_as_t": ("rung1_gate", ),
    "eval/win_vs_random_as_ct": ("rung1_gate", ),
    "eval/win_vs_oracle": ("rung1_gate", ),
}
for _k in EVAL_KEYS:
    REGISTRY[_k] = _e("last", _EVAL_UNITS[_k], _EVAL_CONSUMERS.get(_k, ()),
                      "FLAG-GATED on --eval-interval; the §3 gate runs with eval off.")
REGISTRY.update({
    "eval/epoch":
    _e(
        "last", "epochs", (),
        "The epoch the eval numbers were MEASURED at — may lag the row's own `epoch` by "
        "a few, because ScheduledEval buffers the result until a logged row arrives."),
    "eval/wall_s":
    _e("last", "seconds", ()),
})

# ── derived — names in the surface that no emitter writes ─────────────────
# rung1_gate's report columns, computed over the whole gate window W from the
# emitted keys named in `inputs`. Registering these is what stops a reader's
# column name from being mistaken for an emitted key (the naive "parse the GATES
# tuple" completeness test would have registered three of them as emitted).
_EPISODE_WEIGHTED = ("Episode-weighted over the gate window W: "
                     "sum(row_value * row_episodes) for both sides.")
REGISTRY.update({
    "kills_per_episode":
    _d("count", ("game/kills_per_episode", "environment/episodes"),
       "rung1_gate",
       notes="GATED column (> 0.5). " + _EPISODE_WEIGHTED +
       " Distinct from the emitted game/kills_per_episode."),
    "hit_per_facing":
    _d("ratio", ("game/shots_hit", "game/shots_facing_enemy", "environment/episodes"),
       "rung1_gate",
       notes="GATED column (> 0.45). " + _EPISODE_WEIGHTED),
    "facing_per_fired":
    _d("ratio", ("game/shots_facing_enemy", "game/shots_fired", "environment/episodes"),
       "rung1_gate",
       notes="GATED column (> 0.7). " + _EPISODE_WEIGHTED),
    "on_target_per_facing":
    _d("ratio", ("game/shots_on_target", "game/shots_facing_enemy", "environment/episodes"),
       "rung1_gate",
       notes="Report-only column. " + _EPISODE_WEIGHTED),
    "hit_per_on_target":
    _d("ratio", ("game/shots_hit", "game/shots_on_target", "environment/episodes"),
       "rung1_gate",
       notes="Report-only column. " + _EPISODE_WEIGHTED),
    "stance_blocked_per_facing":
    _d("ratio", ("game/shots_stance_blocked", "game/shots_facing_enemy", "environment/episodes"),
       "rung1_gate",
       notes="REPORT_EXTRA ratio column. " + _EPISODE_WEIGHTED),
    "los_per_fired":
    _d("ratio", ("game/shots_with_enemy_in_los", "game/shots_fired", "environment/episodes"),
       "rung1_gate",
       notes="REPORT_EXTRA ratio column. " + _EPISODE_WEIGHTED),
    "shots_fired":
    _d("count", ("game/shots_fired", "environment/episodes"),
       "rung1_gate",
       notes="Report-only column AND the pre-gate floor (< 10.0 fails the seed). Bare "
       "name — not the emitted game/shots_fired it is computed from."),
    "episodes":
    _d("count", ("environment/episodes", ),
       "rung1_gate",
       notes="Report-only column: the window's total episode count (zero → INVALID)."),
    "rows":
    _d("count", (),
       "rung1_gate",
       notes="Report-only column: how many rows the window has. Computed from len(W), "
       "not from any metrics key — hence no inputs."),
    "losses/approx_kl_p90":
    _d("dimensionless", ("losses/approx_kl", ),
       "rung1_gate",
       notes="A REPORT_EXTRA COLUMN NAME, never an emitted key: the p90 of "
       "losses/approx_kl over W. It is key-shaped and sits in the same tuple as real "
       "keys, so an extractor loosened to 'any slash-name in the script' would demand "
       "an emitter for it. Registered as derived instead."),
                                                                                                  # The one derived entry with no producer: nothing in this repo computes it.
    "environment/use_at_site_frac":
    _d("fraction", (),
       consumers=("compute_game_metrics", ),
       notes="READ but never written: compute_game_metrics presence-gates on it and "
       "re-keys it to actions/use_at_site_frac when the C env grows it. No emitter "
       "produces it today, so both `inputs` and `producer` are empty by fact, not by "
       "omission."),
})


def spec(key):
    """The `MetricSpec` for `key`, or None. Exact match only — a concrete member
    of a family has its own entry when the family is closed."""
    return REGISTRY.get(key)


def keys_of_kind(kind):
    """Sorted registry keys of one kind. Raises on an unknown kind rather than
    returning an empty tuple, which is how a typo'd filter goes unnoticed."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(KINDS)}")
    return tuple(sorted(k for k, v in REGISTRY.items() if v.kind == kind))


def family_members():
    """{family template: members} for every CLOSED family (open ones excluded)."""
    return {k: v.members for k, v in REGISTRY.items() if v.kind == "family" and v.members}
