"""Training-loop metrics (#144 seam 2), split out of the flat train.py.

WHAT: the pure ``logs``/policy readers — network-health weight norms, the aim-σ
emitter, the TCT head/trunk divergence probes, the scheduled fixed-baseline eval
wrapper, the game-metrics dashboard derivation, and the TAG metrics hand-off.
Moved here VERBATIM by the 2026-08-31 post-rung1a refactor: no renames, no
signature changes, no behaviour change. The flat ``train.py`` re-exported every name
below (see its ``__all__``) until #205 part 3 removed the re-exports: ``cs2rl.train``
exports nothing now, so import from this module.

WHY its own module: these are read-only derivations over a dict or a policy —
they own no state and mutate no trainer — so they are the cheapest part of the
loop to unit-test and the part most often edited when a metric is added.

SCOPE BOUNDARY (deliberate, do not "finish the job"): ``self_play_used_past_metric``
lives in ``cs2rl.train.selfplay``, beside the pool it reports on, and the
``logs["self_play/*"]`` assignments STAY in ``cs2rl.train.loop.train``. Those assignment
lines and their order are pinned by tests/train/test_kl_break_metrics.py inside
``inspect.getsource(cs2rl.train.loop)``, and they guard the key cs2rl/experiment/gate.py
reads; keeping the call sites beside the pool bookkeeping is the lower-risk spelling.
PufferLib's own ``self.mean_and_log()`` likewise stays out of this module — its
single call site is ``cs2rl.train.trainer.Cs2PuffeRL._log_and_checkpoint``.

IMPORT-LIGHTNESS INVARIANT: module scope stays torch/nav/env.c-free, for the
reason spelled out in tests/train/test_w1_modules.py's docstring (WHY property 3 is
load-bearing). Every torch import below is function-local ON PURPOSE.
"""

import time

import numpy as np

from cs2rl.policy import LOG_STD_MAX, LOG_STD_MIN


def format_train_status(epoch, ts_val, logs):
    sps = logs.get("SPS", 0.0)
    timeout = logs.get("environment/timed_out", 0.0)
    t_win = logs.get("environment/winner_t", 0.0)
    ct_win = logs.get("environment/winner_ct", 0.0)
    plant = logs.get("environment/bomb_planted", 0.0)
    kills_t = logs.get("environment/kills_t", 0.0)
    kills_ct = logs.get("environment/kills_ct", 0.0)
    round_len = logs.get("environment/round_length", 0.0)
    move_1 = logs.get("environment/action_move_1", 0.0)
    # Batch 3.5: per-axis aim log_std (clamped). Defaults to 0.0 if missing.
    # T7 acceptance gate 2 greps for aim_log_std_pitch= — keep this substring.
    aim_log_std_yaw = logs.get("policy/aim_log_std_yaw", 0.0)
    aim_log_std_pitch = logs.get("policy/aim_log_std_pitch", 0.0)
    return (f"Epoch {epoch} | SPS: {sps:.0f} | Timeout: {timeout:.3f} | "
            f"TWin: {t_win:.3f} | CTWin: {ct_win:.3f} | Plant: {plant:.3f} | "
            f"Kills(T/CT): {kills_t:.2f}/{kills_ct:.2f} | RoundLen: {round_len:.1f} | "
            f"Move1: {move_1:.1f} | TS: {ts_val:.3f} | "
            f"aim_log_std_yaw={aim_log_std_yaw:.4f} aim_log_std_pitch={aim_log_std_pitch:.4f}")


# ── SECTION: R0-I fixed-baseline evaluation hooks ─────────────────────────


def elimination_only_win_rates(logs):
    """R0-I: (win_rate_t, win_rate_ct) with timeouts removed from the CT side.

    WHY: cs2_rewards.h scores a timeout as a CT win (`winner_ct`), so on a
    bombsite-less duel map a CT that never engages "wins" every round and
    SelfPlayManager.win_threshold would pool-save a statue. `winner_t` is
    elimination-only on bombsites=[] maps. All three keys are window means
    over the same episode set, so the subtraction is exact; clamped at 0 for
    the float-noise case. Missing keys (first epoch) → 0.0, never KeyError.
    """
    wt = float(logs.get("environment/winner_t", 0.0))
    wct = max(
        0.0,
        float(logs.get("environment/winner_ct", 0.0)) -
        float(logs.get("environment/timed_out", 0.0)))
    return wt, wct


# ── SECTION: Network Health Monitoring ────────────────────────────────────


def compute_network_health(model, device):
    """Compute network health metrics for logging.

    Returns a dict with:
      - health/weight_norm_<name>: L2 norm of each named parameter
      - health/lstm_h_norm: norm of LSTM hidden state (TODO: requires trainer access)

    Alarm thresholds (informational, not enforced here):
      - dead neurons > 20% (not tracked — would require forward hooks)
      - effective rank < 30 (not tracked — expensive)
      - lstm_h_norm > 50
    """

    metrics = {}

    # Weight norms per named parameter
    for name, param in model.named_parameters():
        safe_name = name.replace(".", "_")
        metrics[f"health/weight_norm_{safe_name}"] = param.norm().item()

    # TODO: LSTM hidden state norm requires access to trainer's stored LSTM state,
    # which is not easily accessible from outside PufferLib's training loop.
    # Would need trainer.policy or similar. Skipping for now.

    return metrics


def log_aim_log_std(policy, logs):
    """Emit policy/aim_log_std_* into `logs` under BOTH architectures (spec §3.6).

    WHAT — the exact meaning of every emitted key, per architecture:

      legacy policy (one `aim_log_std`):
        policy/aim_log_std_{yaw,pitch}         the CLAMPED parameter
        policy/aim_log_std_{yaw,pitch}_raw     the UNCLAMPED parameter

      split policy (`aim_log_std_t` + `aim_log_std_ct`):
        policy/aim_log_std_{yaw,pitch}_{t,ct}      that team's CLAMPED copy
        policy/aim_log_std_{yaw,pitch}_{t,ct}_raw  that team's UNCLAMPED copy
        policy/aim_log_std_{yaw,pitch}         MEAN of the two CLAMPED copies
        policy/aim_log_std_{yaw,pitch}_raw     MAX of the two UNCLAMPED copies

    Under pin_pitch (aim_dim_mask[1] == 0) every `pitch` key above is omitted.

    WHY the raw twin (Rung 1a T1, spec 2026-08-30 §3): the clamped key is
    censored at the cap, so a σ that the optimizer has pushed past the cap —
    the state in which clamp zeroes its gradient and σ is dead — is
    indistinguishable from a σ sitting happily AT the cap. The gate reads
    `policy/aim_log_std_yaw_raw` for both of its σ questions: "did σ move ≥ 0.1
    from its init" (⇒ the continuous head receives gradient at all) and "is raw
    ≤ cap" (⇒ the movement measurement is still meaningful). health/
    weight_norm_aim_log_std already exposes a raw NORM, but only every 5 epochs
    and unsigned/aggregated — per-row, per-dim and signed is what the gate
    needs.

    WHY the legacy-named `_raw` key is a MAX under the split while its clamped
    twin stays a MEAN: the two keys answer different questions and must be
    aggregated differently. The clamped key reports the σ the policy actually
    used, and the mean of the two copies is the honest summary of that. The raw
    key exists solely to answer "has any σ overshot the cap and gone gradient-
    dead", and a mean HIDES exactly that: with cap = −2.9957, a T copy at
    cap + 0.5 = −2.4957 (dead) averaged with a healthy CT copy reads −3.2479,
    i.e. below the cap, so the pre-flight `raw ≤ cap` check passes on a frozen
    σ. Max is the aggregation that answers the overshoot question truthfully.

    PITFALL — what the split `_raw` key does NOT promise: for the *movement*
    question the max is only conservative in one direction. Two copies that
    both moved up, or any copy that moved up, show through; a single copy that
    moved only DOWN while the other sat at its init is invisible in the max
    (max == init ⇒ "no movement"). Read the per-team `_t_raw`/`_ct_raw` keys
    whenever per-copy movement is the question. This does not affect the Rung
    1a gate, which runs the legacy architecture, where the key is the exact
    unclamped parameter.

    WHY the legacy keys survive as a mean rather than being replaced: the T7
    acceptance gate greps the status line for `aim_log_std_pitch=` (see
    format_train_status), and every dashboard/analysis consumer reads those
    two names. Adding the per-team keys alongside is what exposes the actually
    interesting Batch 7 signal — whether the teams learn different aim noise.

    WHY this is a function and not three inline lines in the outer loop: the
    pre-Batch-7 code read policy.aim_log_std unconditionally, which raises
    AttributeError on a split policy before the run writes a single metrics
    row. Extracting it makes that path directly testable (spec §5 test 10,
    which re-review N3 flagged as untested).

    PITFALL: CLAMP EACH COPY, THEN AVERAGE — never average then clamp. The
    forward path clamps per copy (spec §3.2), so a mean-then-clamp here would
    report a σ the policy never used whenever one copy sits outside the band.

    Architecture is detected from the policy object (hasattr aim_log_std_t),
    never from config — config.json is rewritten every launch and lies after a
    flag-less resume (spec §3.4).
    """
    # cs2rl.train.loop imports torch lazily inside functions (module import stays cheap
    # for the CLI/help paths) — keep that convention here.
    import torch

    # R0-E.3: clamp to the RUN's bounds (policy.aim_log_std_min/max), not the module
    # constants — otherwise the log would report a σ the forward never used.
    floor = float(getattr(policy, "aim_log_std_min", LOG_STD_MIN))
    cap = float(getattr(policy, "aim_log_std_max", LOG_STD_MAX))
    # R0-E.2 (#131): when pitch is pinned (aim_dim_mask[1] == 0) the pitch σ
    # is a dead parameter — never sampled, never in log_prob_c, never
    # updated. SKIP its keys (not NaN: NaN survives json.dumps only as the
    # non-standard `NaN` token and would read as a live-but-broken signal on
    # a dashboard). Every consumer already tolerates absence:
    # format_train_status uses logs.get(..., 0.0); the T7 gate is 5v5-only.
    mask = getattr(policy, "aim_dim_mask", None)
    pitch_live = mask is None or float(mask[1]) != 0.0
    with torch.no_grad():
        if hasattr(policy, "aim_log_std_t"):
            raw_t = policy.aim_log_std_t.detach().cpu().numpy()
            raw_ct = policy.aim_log_std_ct.detach().cpu().numpy()
            ls_t = torch.clamp(policy.aim_log_std_t, floor, cap).cpu().numpy()
            ls_ct = torch.clamp(policy.aim_log_std_ct, floor, cap).cpu().numpy()
            logs["policy/aim_log_std_yaw_t"] = float(ls_t[0])
            logs["policy/aim_log_std_yaw_ct"] = float(ls_ct[0])
            logs["policy/aim_log_std_yaw_t_raw"] = float(raw_t[0])
            logs["policy/aim_log_std_yaw_ct_raw"] = float(raw_ct[0])
            if pitch_live:
                logs["policy/aim_log_std_pitch_t"] = float(ls_t[1])
                logs["policy/aim_log_std_pitch_ct"] = float(ls_ct[1])
                logs["policy/aim_log_std_pitch_t_raw"] = float(raw_t[1])
                logs["policy/aim_log_std_pitch_ct_raw"] = float(raw_ct[1])
            clamped = 0.5 * (ls_t + ls_ct)
            # MAX, deliberately NOT the mean that the clamped key uses: the raw
            # key's job is "did any copy overshoot the cap and go gradient-
            # dead", and averaging a dead copy with a healthy one reads as
            # healthy. See the docstring for the worked counterexample and for
            # the one thing max under-reports (a copy that moved only down).
            raw = np.maximum(raw_t, raw_ct)
        else:
            raw = policy.aim_log_std.detach().cpu().numpy()
            clamped = torch.clamp(policy.aim_log_std, floor, cap).cpu().numpy()
    logs["policy/aim_log_std_yaw"] = float(clamped[0])
    logs["policy/aim_log_std_yaw_raw"] = float(raw[0])
    if pitch_live:
        logs["policy/aim_log_std_pitch"] = float(clamped[1])
        logs["policy/aim_log_std_pitch_raw"] = float(raw[1])


def compute_head_divergence(policy):
    """split/head_l2_rel/<module> — how far the two team head copies have moved apart.

    Metric (spec §4 Q3):  ‖W_t − W_ct‖ / (0.5‖W_t‖ + 0.5‖W_ct‖)

    WHY relative and not raw L2: the optimizer runs weight_decay=1e-4
    (see the Adam construction in train()), so even a copy that receives zero
    gradient keeps moving. Raw L2 therefore has no achievable null. Normalising
    by the mean norm of the two copies makes "how different are the teams'
    heads" scale-free; the honest null is still a decay-aware control (zero-
    advantage steps), which is what tests/test_tct_split.py exercises, and the
    run readout reports the TRAJECTORY, not a binary.

    Returns {} for a legacy policy — the metric is undefined with one copy,
    and emitting a fake 0.0 would read as "the teams agree" to anyone
    plotting it.

    PITFALL: modules are grouped, not per-tensor — all 7 discrete heads
    contribute to one `action_heads` number. Per-head keys would be 9 series
    per epoch of mostly-identical curves; if a per-head breakdown is ever
    needed, add it as a separate function rather than widening this one.
    """
    import torch

    if not hasattr(policy, "aim_log_std_t"):
        return {}

    def _flat(obj):
        if isinstance(obj, torch.nn.Parameter):
            return obj.detach().reshape(-1)
        return torch.cat([p.detach().reshape(-1) for p in obj.parameters()])

    out = {}
    with torch.no_grad():
        for name, mod_t, mod_ct in (
            ("action_heads", policy.action_heads_t, policy.action_heads_ct),
            ("aim_mu", policy.aim_mu_t, policy.aim_mu_ct),
            ("aim_log_std", policy.aim_log_std_t, policy.aim_log_std_ct),
        ):
            w_t, w_ct = _flat(mod_t), _flat(mod_ct)
            denom = 0.5 * float(w_t.norm()) + 0.5 * float(w_ct.norm())
            if denom > 0.0:
                # NaN in either copy propagates through the ratio — visible,
                # never masked as "teams identical".
                val = float((w_t - w_ct).norm()) / denom
            else:
                # denom == 0.0 → both copies all-zero → genuinely identical.
                # denom NaN fails both comparisons → emit NaN, not a fake 0.0.
                val = 0.0 if denom == 0.0 else float("nan")
            out[f"split/head_l2_rel/{name}"] = val
    return out


def compute_trunk_divergence(policy):
    """split/trunk_l2_rel/<module> — how far the two team trunk copies have moved apart.

    WHAT: relative L2 between the T and CT copies of encoder and lstm.
    Metric is the same formula as compute_head_divergence (spec §4 Q3):
        ‖W_t − W_ct‖ / (0.5‖W_t‖ + 0.5‖W_ct‖)
    Keys: split/trunk_l2_rel/encoder, split/trunk_l2_rel/lstm.

    WHY relative and not raw L2: the optimizer runs weight_decay=1e-4, so
    even a copy that receives zero gradient keeps moving. Raw L2 has no
    achievable null. The ratio is scale-free; the honest null is still a
    decay-aware control. Gate is hasattr(policy, "encoder_t") — the live
    architecture, never config.json — matching split/trunk_active.

    Returns {} when there is no encoder_t. The metric is undefined with
    one copy, and emitting a fake 0.0 would read as "the teams agree".

    PITFALL: modules are grouped, not per-tensor — every Linear in the
    Sequential encoder and every LSTM weight (ih/hh/bias) contribute to
    one number. A per-layer series is a separate function if ever needed.
    T=1 + zero LSTM state leaves weight_hh unmoved; that does not make
    the encoder ratio 0 after a team-asymmetric step.
    """
    import torch

    if not hasattr(policy, "encoder_t"):
        return {}

    def _flat(obj):
        if isinstance(obj, torch.nn.Parameter):
            return obj.detach().reshape(-1)
        return torch.cat([p.detach().reshape(-1) for p in obj.parameters()])

    out = {}
    with torch.no_grad():
        for name, mod_t, mod_ct in (
            ("encoder", policy.encoder_t, policy.encoder_ct),
            ("lstm", policy.lstm_t, policy.lstm_ct),
        ):
            w_t, w_ct = _flat(mod_t), _flat(mod_ct)
            denom = 0.5 * float(w_t.norm()) + 0.5 * float(w_ct.norm())
            if denom > 0.0:
                val = float((w_t - w_ct).norm()) / denom
            else:
                val = 0.0 if denom == 0.0 else float("nan")
            out[f"split/trunk_l2_rel/{name}"] = val
    return out


# ── SECTION: R0-I fixed-baseline evaluation hooks ─────────────────────────
# The banner of this name also sits above elimination_only_win_rates, at the top of
# this file; restated here so the "Network Health Monitoring" banner above does not
# appear to cover this class.


class ScheduledEval:
    """Runs the fixed-baseline eval every `interval` epochs and delivers its
    keys on the next LOGGED metrics row.

    WHY a buffer: PuffeRL's train() throttles mean_and_log to once per 0.25 s
    (logs is None on the other epochs). A naive `if isinstance(logs, dict)
    and epoch % interval == 0` silently skips the eval whenever the eval epoch
    happens to be throttled — on a fast CPU box that is most epochs. So the
    eval decision is made on `trainer.epoch` alone, and the result waits in
    `pending` until a dict row comes through. `eval/epoch` stamps the epoch
    the numbers were measured at (may lag the row's epoch by a few).

    PITFALL: call after_train() on EVERY epoch, outside any `isinstance(logs,
    dict)` guard, or the buffer never drains.
    """

    def __init__(self, evaluator, interval, policy, device):
        if int(interval) <= 0:
            raise ValueError(f"ScheduledEval interval must be >= 1, got {interval}")
        self.evaluator = evaluator
        self.interval = int(interval)
        self.policy = policy
        self.device = device
        self.pending = {}

    def after_train(self, trainer, logs):
        if trainer.epoch % self.interval == 0:
            t0 = time.time()
            self.pending.update(self.evaluator.evaluate(self.policy, self.device))
            self.pending["eval/epoch"] = int(trainer.epoch)
            self.pending["eval/wall_s"] = time.time() - t0
        if isinstance(logs, dict) and self.pending:
            logs.update(self.pending)
            self.pending = {}

    def close(self):
        self.evaluator.env.close()


# ── SECTION: Game Metrics Dashboard ───────────────────────────────────────


def compute_game_metrics(logs):
    """Extract and normalize game metrics from the training logs dict.

    The C env exposes per-episode stats as ``environment/<key>`` entries in
    the logs dict returned by PufferLib's ``mean_and_log()``.  Values are
    already averaged over the collection window, so most just need re-keying
    and minor arithmetic.

    Always-present ``game/*`` keys (already in today's terminal info) are
    re-keyed with ``_get(..., default=0.0)``. ``game/plant_tick`` and
    ``game/win_by_*`` are presence-gated: emit them only when the source
    key already exists in ``logs``. A synthetic 0.0 would make old log
    dicts look new-format.

    Returns a flat dict with ``game/*`` and ``actions/*`` keys ready to be
    merged back into logs for W&B or stdout.
    """
    if not isinstance(logs, dict):
        return {}

    def _get(key, default=0.0):
        return logs.get(f"environment/{key}", logs.get(key, default))

    winner_t = _get("winner_t", 0.0)
    winner_ct = _get("winner_ct", 0.0)
    timed_out = _get("timed_out", 0.0)
    kills_t = _get("kills_t", 0.0)
    kills_ct = _get("kills_ct", 0.0)
    bomb_planted = _get("bomb_planted", 0.0)
    round_length = _get("round_length", 0.0)

    # win rates: already normalised per-episode by PufferLib's mean_and_log
    game_metrics = {
        "game/win_rate_t": winner_t,
        "game/win_rate_ct": winner_ct,
        "game/timeout_rate": timed_out,
        "game/kills_per_episode": kills_t + kills_ct,
        "game/bomb_plant_rate": bomb_planted,
        "game/avg_episode_length": round_length,
    }

    # Always-present splits/rewards: these keys already land in logs today
    # via _build_terminal_info. game/reward/win is NOT emitted (#128, R0-A):
    # C reward_win is the cross-team sum and nets ~0 by the zero-sum
    # identity; the one-sided game/reward/win_t|win_ct below carry the signal.
    game_metrics["game/defuse_rate"] = _get("bomb_defused", 0.0)
    game_metrics["game/kills_t"] = kills_t
    game_metrics["game/kills_ct"] = kills_ct
    for src, dst in (
        ("reward_kills", "game/reward/kills"),
        ("reward_deaths", "game/reward/deaths"),
        ("reward_bomb", "game/reward/bomb"),
        ("reward_pbrs", "game/reward/pbrs"),
        ("reward_shots", "game/reward/shots"),
        ("reward_survival", "game/reward/survival"),
        ("reward_inaction", "game/reward/inaction"),
    ):
        game_metrics[dst] = _get(src, 0.0)

    # R0-A: combat counters are window MEANS per episode (mean_and_log).
    for k in ("shots_fired", "shots_with_enemy_in_los", "shots_facing_enemy", "shots_on_target",
              "shots_hit", "shots_stance_blocked", "damage_dealt", "mutual_vis_pair_ticks",
              "agent_ticks_with_visible_enemy"):
        game_metrics[f"game/{k}"] = _get(k, 0.0)
    game_metrics["game/reward/win_t"] = _get("reward_win_t", 0.0)
    game_metrics["game/reward/win_ct"] = _get("reward_win_ct", 0.0)
    # Ratio of window means = conditional mean over episodes where a pair
    # coexisted. A max(·,1) guard would silently return the unconditional
    # mean — keep the explicit zero-valid branch.
    _med_sum = _get("min_enemy_distance_sum", 0.0)
    _med_valid = _get("min_enemy_distance_valid", 0.0)
    game_metrics["game/min_enemy_distance"] = (_med_sum / _med_valid) if _med_valid > 0 else 0.0
    game_metrics["game/min_enemy_distance_valid_frac"] = _med_valid

    # Presence-gate plant_tick / win_by_*: a synthetic 0.0 would make old
    # log dicts look new-format. Do not _get(..., default=0.0) these three.
    def _maybe(src, dst):
        if f"environment/{src}" in logs or src in logs:
            game_metrics[dst] = _get(src)

    _maybe("plant_tick", "game/plant_tick")
    _maybe("win_by_detonation", "game/win_by_detonation")
    _maybe("win_by_defuse", "game/win_by_defuse")

    # actions/use_at_site_frac — logged directly by the C env if available
    use_at_site = _get("use_at_site_frac", None)
    if use_at_site is not None:
        game_metrics["actions/use_at_site_frac"] = use_at_site

    return game_metrics


def _inject_tag_metrics(trainer, logs):
    """Move pending TAG metrics into this epoch's logs dict (spec §4.2).

    CALL-ORDER CONSTRAINT: must run AFTER dead_run_detector.check(...) in
    the outer loop — tag/* carries deliberate NaNs (zero-norm subsets,
    documented in tag_grad_cossim) and check() raises RuntimeError on any
    NaN in the metrics dict; injecting earlier aborts the run with exit
    code 3 on the first degenerate subset. Also never route these through
    the `losses` dict: its minibatch sums are divided by the executed-minibatch
    count (gh#90), its keys are prefixed losses/, and they lag environment/*
    by one epoch.

    logs=None (throttled epoch) is a no-op: the top-of-loop reset then
    DROPS the measurement — injecting it next epoch would mislabel its
    step/epoch (spec §4.2 drop semantics).
    """
    pending = trainer._tag_metrics
    if pending and isinstance(logs, dict):
        logs.update(pending)
