"""The policy network's factory and everything that reads or writes its outputs.

Owns: `build_policy` (the factory of `cs2rl.policy_net.Dust2Policy`), the hybrid-aim sampler
`_hybrid_sample_logits`, checkpoint loading (`load_policy_from_checkpoint`,
`load_state_dict_arch_checked`, the split-ness sniffers), the rollout-side helpers
(`init_policy_state`, `select_policy_actions_native`, `resolve_policy_mode`), the
log-std constants and the action-mask layout.

Module scope stays torch-free at run time: every runtime torch import is function-local
(the module-scope one sits under `if TYPE_CHECKING:`, for annotations only), and so is
`build_policy`'s import of `cs2rl.policy_net` (torch at its module scope), so eval, viz
and BC can import this module without paying for torch until they build a policy.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from cs2rl.env.factory import build_legacy_eval_env
from cs2rl.spec.action import ACTION_HEAD_SIZES

if TYPE_CHECKING:                      # annotations only; never at runtime
    import torch


def state_dict_is_split(state_dict):
    """True if this checkpoint was written by a T/CT split policy (spec §3.3).

    WHAT: presence of the `aim_log_std_t` parameter is the marker — it exists
    in exactly one architecture and nowhere else in the key space.

    WHY key inference rather than the config flag: config.json is rewritten
    from the launch's own flags on every launch that builds its trainer
    (`cs2rl.train.loop._write_config_json`; the `--dump-config` write in
    `cs2rl.train.__main__` is a separate early exit), so a flag-less
    crash-resume would stamp `tct_split_heads: false` over a split run's
    provenance. Deciding from the keys means resume, self-play snapshot
    loading and the eval/record loader all do the right thing with no flag at
    all — the flag governs only fresh construction and the legacy→split
    conversion direction.

    PITFALL: "aim_log_std_ct".endswith("aim_log_std_t") is False, so this does
    not accidentally fire on a CT-only key set; it is nonetheless deliberate
    that the marker is the T copy, since both are always written together.
    """
    return any(k.endswith("aim_log_std_t") for k in state_dict)


def state_dict_is_trunk_split(state_dict):
    """True if encoder/LSTM were written as per-team copies (spec §3.3).

    WHAT: presence of `encoder_t.0.weight` is the trunk-split marker — the T
    encoder first-layer weight exists in exactly that architecture. Both
    `encoder_t`/`encoder_ct` (and both LSTMs) are always written together.

    WHY key inference rather than the config flag: same as
    `state_dict_is_split` — `config.json` is rewritten on every launch that
    builds its trainer, so a flag-less crash-resume must recover trunk-ness
    from the keys. The heads
    helper stays the heads marker; loaders consult both bits independently.

    PITFALL: `"encoder_ct.0.weight".endswith("encoder_t.0.weight")` is False,
    so a CT-only key set does not fire. The `k ==` clause is the bare-key
    form every checkpoint this project writes; `endswith` covers a future
    wrapper prefix (e.g. `policy.encoder_t.0.weight`).
    """
    return any(k == "encoder_t.0.weight" or k.endswith("encoder_t.0.weight") for k in state_dict)


def load_state_dict_arch_checked(policy, state_dict, *, source):
    """load_state_dict with a loud architecture-mismatch error (spec §3.3).

    WHAT: compares the checkpoint's two architecture bits (heads via
    `state_dict_is_split`, trunk via `state_dict_is_trunk_split`) against
    the policy's (`policy.tct_split_heads`, `policy.tct_split_trunk`) and
    raises a message naming BOTH axes before loading anything. Never a
    silent partial load.

    WHY it exists even though every construction site infers: the sites that
    RECEIVE a pre-built policy and then load into it (train main's resume,
    load_policy_from_checkpoint, SelfPlayManager.load_past_policy) can be
    handed a mismatched pair by a caller that bypassed inference. A bare
    load_state_dict there raises a wall of missing/unexpected keys that names
    neither architecture — the operator's first hypothesis becomes "corrupt
    checkpoint", which is wrong and expensive.

    PITFALL: this does NOT convert. Legacy→split conversion is a deliberate
    act with a σ-re-init → heads convert → trunk convert ordering
    constraint, so it stays at the one call site that means it (the
    train-main warm split). An object without the flag attributes (every
    Dust2Policy has both) compares as legacy on that axis via getattr(..., False).
    """
    ckpt_heads = state_dict_is_split(state_dict)
    ckpt_trunk = state_dict_is_trunk_split(state_dict)
    # A missing flag attribute reads as legacy (off).
    policy_heads = bool(getattr(policy, "tct_split_heads", False))
    policy_trunk = bool(getattr(policy, "tct_split_trunk", False))
    if ckpt_heads != policy_heads or ckpt_trunk != policy_trunk:

        def _name_heads(flag):
            return "SPLIT (per-team T/CT policy heads)" if flag else "LEGACY (shared policy heads)"

        def _name_trunk(flag):
            return ("SPLIT (per-team T/CT encoder+LSTM)"
                    if flag else "LEGACY (shared encoder+LSTM)")

        raise ValueError(
            f"policy/checkpoint architecture mismatch loading {source}: the checkpoint is "
            f"heads={_name_heads(ckpt_heads)}, trunk={_name_trunk(ckpt_trunk)} but the policy is "
            f"heads={_name_heads(policy_heads)}, trunk={_name_trunk(policy_trunk)}. Rebuild the "
            f"policy with build_policy(..., tct_split_heads={ckpt_heads}, "
            f"tct_split_trunk={ckpt_trunk}) — loaders are supposed to infer both axes from the "
            f"checkpoint keys (state_dict_is_split / state_dict_is_trunk_split), see spec "
            f"2026-08-15 §3.3.")
    policy.load_state_dict(state_dict)


AGENT_IDS = tuple([f"t{i}" for i in range(5)] + [f"ct{i}" for i in range(5)])

# ── SECTION: Shared eval / record helpers ──────────────────────────────────


def load_policy_from_checkpoint(checkpoint_path, device, aim_log_std_max=None, pin_pitch=False):
    """Rebuild a policy from a bare state_dict checkpoint (eval / record / probe).

    R0-E (#131): ``aim_log_std_max`` / ``pin_pitch`` are RUN properties, not
    checkpoint state (non-persistent buffers on the policy), so the caller
    must pass the run's values — a checkpoint cannot tell you whether its
    run pinned pitch. Defaults reproduce the pre-R0-E policy.
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Policy] Loading checkpoint -> {checkpoint_path}")
    # weights_only=True: every checkpoint this project writes is a bare tensor
    # state_dict, and the other loaders (resume sniff, self-play pool) already
    # load with it — a checkpoint that fails here is untrusted or corrupt.
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Infer obs_dim from checkpoint to handle checkpoints trained with different obs sizes.
    # Trunk-split checkpoints have no shared encoder — the T copy is the marker
    # (same key state_dict_is_trunk_split uses). Both copies share obs_dim.
    if "encoder_t.0.weight" in state_dict:
        ckpt_obs_dim = state_dict["encoder_t.0.weight"].shape[1]
    else:
        ckpt_obs_dim = state_dict["encoder.0.weight"].shape[1]
    # W3 (#154): role eval_legacy — the DOCUMENTED bare-call defaults, which are
    # now EnvConfig()'s own field defaults: the eval_legacy builder passes a
    # bare EnvConfig(), and env/config.py declares those fields to be the
    # trained baseline, which is exactly the pre-Rung-0 env (full 5v5, pitch
    # live, crouch and jump enabled); test_defaults_equal_the_139a3a3_values in
    # tests/env/test_env_config.py pins them. Passing no knobs is the behaviour, not
    # an oversight; #143 tracks whether it should change, and the factory reduces
    # that future fix to one role's knob source.
    policy_env = build_legacy_eval_env()

    # Batch 3.5 (#24, Opus I3): defensive obs_dim consistency check.
    # The function rebuilds the policy with the *checkpoint's* obs_dim
    # (obs_dim_override=ckpt_obs_dim). For any cross-version checkpoint
    # (e.g., Batch-3 105-dim loaded against Batch-3.5 107-dim env), the
    # policy will silently mis-interpret obs slots after the insertion
    # point. Fail loud at load time instead.
    # Pitfall: compare DISK shape to LIVE shape (ckpt_obs_dim vs
    # env_obs_dim), not derived-to-derived (policy.obs_dim is set to
    # obs_dim_override and would be self-referentially equal).
    env_obs_dim = policy_env.single_observation_space.shape[0]
    if ckpt_obs_dim != env_obs_dim:
        policy_env.close()
        raise ValueError(
            f"checkpoint obs_dim={ckpt_obs_dim} ≠ env obs_dim={env_obs_dim}; "
            f"checkpoint is from a different obs schema. Retrain or use a matching env.")

    try:
        # Batch 7 / trunk split (spec §3.3): both architecture bits inferred
        # from the checkpoint keys, exactly like obs_dim above — this loader
        # gets no flag and needs none. A trunk-split file has no encoder.0.weight.
        policy = build_policy(policy_env,
                              device,
                              obs_dim_override=ckpt_obs_dim,
                              tct_split_heads=state_dict_is_split(state_dict),
                              tct_split_trunk=state_dict_is_trunk_split(state_dict),
                              aim_log_std_max=aim_log_std_max,
                              pin_pitch=pin_pitch)
    finally:
        policy_env.close()

    load_state_dict_arch_checked(policy, state_dict, source=str(checkpoint_path))
    policy.eval()
    return policy


def init_policy_state(policy, device):
    import torch

    if policy is None:
        return None

    return {
        "done": torch.zeros(len(AGENT_IDS), device=device),
        "lstm_h": torch.zeros(len(AGENT_IDS), policy.hidden_size, device=device),
        "lstm_c": torch.zeros(len(AGENT_IDS), policy.hidden_size, device=device),
    }


def select_policy_actions_native(policy, obs, device, policy_state, policy_mode):
    """Run policy in eval mode; return BOTH discrete and continuous actions.

    Returns
    -------
    (actions, cont) : (np.int32 (N, ACTION_DIM), np.float32 (N, AIM_DIM))
        actions : per-head argmax (greedy) or per-head sample (sample mode).
        cont    : continuous aim head output. Greedy uses μ directly (already
                  bounded by tanh*max_turn_speed); sample draws from
                  Normal(μ, exp(log_std)) clamped to ±max_turn_speed (matches
                  the rollout sampler's behaviour exactly).

    Why both: env.step now requires (actions, continuous_actions). Earlier
    (Batch 3) this helper returned discrete-only and the cont buffer was
    dropped silently — recording/eval paths effectively passed cont=zeros
    every tick, freezing aim at spawn (the "agents look forward in rerun"
    bug). Returning both lets callers feed the env exactly the actions the
    policy produced.
    """
    import torch

    if policy_mode == "random":
        raise ValueError("Random action selection should bypass select_policy_actions_native")

    obs_t = torch.as_tensor(obs, device=device)
    if hasattr(policy, "obs_dim") and obs_t.shape[-1] != policy.obs_dim:
        obs_t = obs_t[..., :policy.obs_dim]
    with torch.no_grad():
        logits, mu_aim, log_std_aim, _ = policy.forward_eval(obs_t, policy_state)
        if policy_mode == "sample":
            # 6-tuple return; we keep action + cont (logp/entropy unused here).
            act_t, cont_t, *_ = _hybrid_sample_logits(
                (logits, mu_aim, log_std_aim, None),
                max_turn_speed=policy.max_turn_speed.item(),
            )
        else:
            # Greedy: per-head argmax for discrete, μ directly for continuous.
            # μ is already tanh-squashed × max_turn_speed by the policy's forward
            # (the `torch.tanh(self.aim_mu...)` lines in `Dust2Policy._project_heads`),
            # so it's already bounded — no extra clamp needed.
            act_t = torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)
            cont_t = mu_aim

    return (act_t.cpu().numpy().astype(np.int32), cont_t.cpu().numpy().astype(np.float32))


def resolve_policy_mode(checkpoint_path, policy_mode):
    if checkpoint_path and policy_mode != "random":
        return "greedy" if policy_mode == "auto" else policy_mode
    return "random"


# ── SECTION: Policy ────────────────────────────────────────────────────────


def build_policy(vecenv,
                 device,
                 obs_dim_override=None,
                 tct_split_heads=False,
                 tct_split_trunk=False,
                 aim_log_std_max=None,
                 pin_pitch=False):
    """Build the run's `cs2rl.policy_net.Dust2Policy` from the env and these knobs, on `device`.

    Reads obs_dim (unless overridden) and max_turn_speed from the driver env,
    validates the σ cap and resolves the fresh σ init, then constructs the
    module on CPU and moves it to `device`. Importing `cs2rl.policy_net` here,
    not at module scope, keeps torch out of this module's import.

    aim_log_std_max / pin_pitch (R0-E.3 / R0-E.2, #131): per-RUN aim-head
    properties. The cap replaces LOG_STD_MAX at every σ clamp site; pin_pitch
    sets ``policy.aim_dim_mask`` to [1, 0] so the pitch dim drops out of
    log_prob_c / entropy_c. Both live as NON-persistent buffers/attrs so old
    checkpoints still load and a checkpoint never carries them — every loader
    (SelfPlayManager.load_past_policy, load_policy_from_checkpoint) must pass
    the run's values explicitly. Raises ValueError if the cap leaves the
    (LOG_STD_MIN, LOG_STD_MAX] band.

    tct_split_heads (Batch 7, spec 2026-08-13): when True the policy-head
    group — the 7 discrete action_heads, the aim_mu projection and the
    aim_log_std parameter — is duplicated per team (`_t` / `_ct` suffixes) and
    each row is routed to its own team's copy by the obs team bit obs[24].
    value_head stays SHARED. Default False builds the legacy head modules
    and executes the legacy head-forward lines verbatim, pinned by
    tests/test_tct_split.py::test_flag_off_builds_exactly_the_legacy_modules.

    tct_split_trunk (spec 2026-08-15): when True the trunk — encoder + LSTM —
    is replaced by per-team copies (`encoder_t`/`lstm_t`, `encoder_ct`/`lstm_ct`).
    Each team LSTM sees only its own encoder's activations; hidden and the
    rollout (h,c) blend on obs[24]. Default False builds the shared
    `encoder` / `lstm`.

    PITFALL: callers must not decide split-ness from config alone — every
    loader infers it from the checkpoint's keys (state_dict_is_split /
    state_dict_is_trunk_split), because config.json is rewritten on each
    launch that builds its trainer and a flag-less crash-resume would
    otherwise rebuild the wrong architecture (spec §3.3).
    """
    from cs2rl.policy_net import Dust2Policy

    # KNOWN LIMIT: pyrefly 1.2.0 types this getattr (untyped default) as Any | None; 1.3.2 does not.
    driver_env = getattr(vecenv, "driver_env", vecenv)
    obs_dim = (obs_dim_override
               if obs_dim_override is not None else driver_env.single_observation_space.shape[0])
    _cap = validate_aim_log_std_max(aim_log_std_max)
    # Rung 1a T1: the FRESH σ init follows the cap (see resolve_aim_log_std_init)
    # — LOG_STD_INIT at the 5v5 default cap, cap − 0.2 under a tight one, never
    # AT the cap where clamp would zero the gradient forever.
    _log_std_init = resolve_aim_log_std_init(_cap)
    return Dust2Policy(obs_dim,
                       driver_env._c_env.sd.contents.max_turn_speed,
                       aim_log_std_min=LOG_STD_MIN,
                       aim_log_std_max=_cap,
                       aim_log_std_init=_log_std_init,
                       pin_pitch=pin_pitch,
                       tct_split_heads=tct_split_heads,
                       tct_split_trunk=tct_split_trunk).to(device)


# ── SECTION: Hybrid-aim sampling ───────────────────────────────────────────
#
# `_hybrid_sample_logits` samples the 4-tuple Dust2Policy's forward and
# forward_eval emit (7 categorical heads + the Gaussian aim head) in place of
# pufferlib.pytorch.sample_logits. Joint factorised log-prob = sum of the
# categorical log-probs + the Normal log-prob (independence assumption per
# spec L8). Its update-time counterpart is `cs2rl.train.update._hybrid_ppo_loss`,
# and `cs2rl.train.trainer.HybridAimVecEnv` carries the continuous aim to the envs.


def _hybrid_sample_logits(
    policy_out,
    action=None,
    continuous_action=None,
    max_turn_speed=None,
    mask=None,
    aim_dim_mask=None
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Hybrid sampler for the 4-tuple Dust2Policy output.

    Replaces the four in-tree usages of
    ``pufferlib.pytorch.sample_logits(logits[, action=...])`` that previously
    assumed a 2-tuple policy contract. Pure function (no monkey-patching) so
    it can be unit-tested without spinning up a trainer.

    Inputs
    ------
    policy_out : 4-tuple
        (logits_list[7], mu_aim, log_std_aim, value) — the output of
        Dust2Policy.forward / forward_eval. The value slot is
        ignored here; the caller already has it from the original call.
    action : (B, ACTION_DIM=7) int64 tensor, or None
        If None, sample fresh from the categorical heads. If supplied
        (PPO update pass), evaluate log-prob under the new policy without
        re-sampling — this is the difference between rollout and update.
    continuous_action : (B, AIM_DIM=2) float32 tensor, or None
        If None, sample (Δyaw, Δpitch) from Normal(mu_aim, exp(log_std_aim))
        and clamp to ±max_turn_speed. If supplied, evaluate log-prob without
        re-sampling.
    max_turn_speed : float or None
        Hard clamp on sampled Δyaw. None means no clamp (only sane in the
        update-pass path where continuous_action is provided pre-clamped).
    mask : (B, ACTION_MASK_DIM) bool/int8 tensor, or None (F8)
        C-computed action masks (1 = valid; see cs2_env.h compute_masks).
        When given, invalid bins are excluded from sampling AND from the
        log-prob/entropy — the distribution IS the masked distribution, so
        the stored logprobs stay consistent with _hybrid_ppo_loss as long as
        the update pass receives the SAME mask (mb_masks). None = unmasked
        (legacy eval/record callers that have no mask plumbing).
    aim_dim_mask : (AIM_DIM,) float tensor, or None (R0-E.2, #131)
        Per-dimension weight on the Gaussian log-prob / entropy terms, applied
        BEFORE the sum over AIM_DIM. [1, 0] when pitch is pinned (the env
        ignores cont[:, 1], so its density must not enter the ratio). None ⇒
        all-ones ⇒ today's behaviour bit-for-bit. Sampling is NOT masked —
        the pinned dim is still drawn (and discarded by the env).

    Returns
    -------
    action, continuous_action, log_prob_d, log_prob_c, entropy_d, entropy_c
        action : (B, 7) int64
        continuous_action : (B, AIM_DIM=2) float32, ∈ [-max_turn_speed, max_turn_speed]
        log_prob_d : (B,) — discrete factor log-prob (sum over 7 categoricals).
            The PPO loss assembly in _hybrid_ppo_loss applies the clip to
            this half independently of log_prob_c (Fan et al. 2019 Eq 8).
            Callers that want the rollout-stored joint log_prob simply do
            `log_prob_d + log_prob_c` (joint factorised under independence,
            spec L8) — surfacing the halves directly here saves the rollout
            from reconstructing 7 Categorical + 1 Normal a second time.
        log_prob_c : (B,) — continuous factor log-prob (Normal sum-of-dims).
        entropy_d : (B,) — discrete entropy (sum of 7 categoricals).
        entropy_c : (B,) — Normal entropy 0.5·log(2πe·σ²). NEGATIVE for
            σ < 1/√(2πe) ≈ 0.242 — at σ_init=0.1 it is ≈ −0.886. This is
            mathematically correct; do NOT clip or assert entropy >= 0
            anywhere downstream. Callers that don't need entropy can ignore
            with `*_entropies` unpacking.

    WHY the halves rather than the joint sums: the rollout stores them
    separately (the trainer's logprobs_d / logprobs_c), and recovering them
    from sums meant building the 7 Categorical + 1 Normal a second time, which
    measured +437 ms/epoch on the i7-9750H smoke (16.0 vs 9.1 ms/step).
    """
    import torch
    import torch.nn.functional as F

    logits_list, mu_aim, log_std_aim, _value = policy_out

    # F8: mask BEFORE log_softmax so sampling, log-prob and entropy all see
    # the same (masked) distribution. Dead agents collapse to deterministic
    # per-head no-ops (entropy 0) instead of burning exploration samples.
    if mask is not None:
        logits_list = _apply_action_masks(logits_list, mask)

    # ── Discrete: 7 independent categorical heads — hand-rolled (Fix #2) ──
    # We avoid `torch.distributions.Categorical` because constructing 7 of them
    # per rollout step (×64 bptt × ~12 epochs/sec) accumulates measurable
    # Python-side overhead. The math is straightforward:
    #   sample(logits) ≡ multinomial(softmax(logits), 1)
    #   log_prob(a)    ≡ log_softmax(logits)[a]
    #   entropy()      ≡ -Σ p · log_softmax  where p = exp(log_softmax)
    # Bench at production batch=2560 measured 9.59 ms → 5.91 ms / step
    # (1.62× speedup, ~235 ms/epoch saved). Numerical equivalence vs
    # torch.distributions: |Δ| ≤ 1.91e-6 (different reduction order in
    # softmax; well within the fp32 tolerance PPO already runs at).
    log_probs_per_head = [F.log_softmax(lg, dim=-1) for lg in logits_list]
    if action is None:
        action = torch.stack(
            [torch.multinomial(lp.exp(), 1).squeeze(-1) for lp in log_probs_per_head],
            dim=-1,
        )
    # builtin sum starts from int 0, so a type checker reads it as int | Tensor;
    # summed over the 7 heads it is a Tensor. The casts keep that out of callers.
    log_prob_d = cast(
        torch.Tensor,
        sum(
            lp.gather(-1, action[..., i:i + 1]).squeeze(-1)
            for i, lp in enumerate(log_probs_per_head)))
    # Entropy: H = -Σ p log p. log_softmax already gives log p; multiply by
    # exp(log_softmax) = p. Single pass per head, no extra softmax call.
    entropy_d = cast(torch.Tensor, sum(-(lp.exp() * lp).sum(-1) for lp in log_probs_per_head))

    # ── Continuous: 1D Gaussian aim head — hand-rolled (Fix #2) ──
    # σ comes pre-clamped from forward()/forward_eval() (to the policy's
    # aim_log_std_min/aim_log_std_max), so
    # we don't re-clamp here — would silently mask a regression in the policy
    # if the clamp were removed upstream.
    # Analytic forms (replace torch.distributions.Normal):
    #   sample(μ, σ)   = μ + σ · randn_like(μ)         (vs rsample; no autograd
    #                                                    graph, PPO doesn't use
    #                                                    pathwise gradients)
    #   log_prob(x)    = -½((x-μ)/σ)² - log σ - ½ log 2π
    #   entropy()      = ½ + ½ log 2π + log σ
    sigma = torch.exp(log_std_aim).expand_as(mu_aim)
    if continuous_action is None:
        continuous_action = mu_aim + sigma * torch.randn_like(mu_aim)
        if max_turn_speed is not None:
            # Same clamp logic as Dust2Policy.get_action_and_value.
            # The C env (env_step, cs2_env.h) clamps silently with fminf/fmaxf;
            # storing the post-clamp value keeps the PPO ratio honest.
            continuous_action = torch.clamp(continuous_action, -max_turn_speed, max_turn_speed)
    diff = (continuous_action - mu_aim) / sigma
    # log_std_aim has shape (AIM_DIM,); expand_as(mu_aim) broadcasts to (B, AIM_DIM)
    # so .sum(-1) sums over AIM_DIM correctly.
    log_std_b = log_std_aim.expand_as(mu_aim)
    # R0-E.2: per-dimension weight (AIM_DIM,), ones ⇒ today's behaviour.
    # Applied BEFORE .sum(-1) so both log-prob and entropy exclude pinned dims.
    w = _aim_dim_weight(aim_dim_mask, mu_aim)
    log_prob_c = ((-0.5 * diff * diff - log_std_b - 0.5 * _LOG_2PI) * w).sum(-1)
    entropy_c = ((0.5 + 0.5 * _LOG_2PI + log_std_b) * w).sum(-1)

    return action, continuous_action, log_prob_d, log_prob_c, entropy_d, entropy_c


# Batch 3 (continuous aim H-PPO): state-independent log_std parameter
# for the Gaussian aim head. σ_init = 0.1 rad ≈ 5.7° matches mega-spec
# §9 lock and the H-PPO literature default. σ_min = 0.01 rad ≈ 0.6° —
# floors entropy without flooding the policy with noise; tanh+max_turn_speed
# clamp dominates the per-tick range regardless of σ. σ_max = 0.5 rad ≈ 28.6°
# — symmetric bound prevents explosion that would mask μ.
# Module-level and torch-free, so the CLI's cap check, the σ logs and the trainer's
# max_entropy calc (the SAC-α dual loop) read them without building a policy.
# build_policy hands LOG_STD_MIN to Dust2Policy as the floor of every σ clamp.
LOG_STD_INIT = math.log(0.1)
LOG_STD_MIN = math.log(0.01)
LOG_STD_MAX = math.log(0.5)

# Rung 1a T1 (spec 2026-08-30): how far BELOW the run's σ cap a fresh
# aim_log_std starts. `torch.clamp` back-propagates zero gradient strictly
# outside [min, max], so a parameter initialised AT the cap is gradient-dead
# from step 0 — that is exactly what killed the Rung 1 treatment arm (init
# log 0.1 under a log 0.05 cap: σ was a constant for 10M steps and the logged
# "log σ = −2.996" was the clamp, not a measurement). 0.2 in log-space ≈ an
# 18 % σ gap: wide enough that Adam needs many steps to walk into the clamp,
# small enough that the run still trains near its intended σ.
AIM_LOG_STD_INIT_MARGIN = 0.2
# The matching floor-side head-room the cap must leave. The init sits one
# margin below the cap, so a cap closer than 2× the margin to LOG_STD_MIN puts
# the init at/below the FLOOR, where the lower clamp kills the gradient just as
# dead as the upper one. Enforcing head-room on the cap (rather than clamping
# the init up with a max(LOG_STD_MIN + m, …) floor) is deliberate: a floor
# merely moves the dead zone from one end of the band to the other, silently.
AIM_LOG_STD_CAP_MIN_HEADROOM = 2 * AIM_LOG_STD_INIT_MARGIN


def resolve_aim_log_std_init(cap) -> float:
    """The log σ a FRESH aim head starts at under this run's cap (Rung 1a T1).

    WHAT: min(LOG_STD_INIT, cap − AIM_LOG_STD_INIT_MARGIN). At the 5v5 default
    cap (log 0.5) that is LOG_STD_INIT unchanged — every pre-Rung-1a run keeps
    its σ=0.1 start. Under a tight cap (Rung 1a's log 0.05) it is cap − 0.2,
    i.e. σ ≈ 0.041, strictly inside [LOG_STD_MIN, cap].

    WHY: `torch.clamp(x, lo, hi)` passes gradient only for lo <= x <= hi. An
    init at or above the cap is therefore a permanently frozen σ — the policy
    samples at exactly the cap forever and `policy/aim_log_std_yaw` reports the
    cap, which reads like a converged value rather than a dead parameter. This
    is the Rung 1 defect (spec 2026-08-30 §1(iv)).

    PITFALL: this is the FRESH-construction init only. A resume restores the
    checkpoint's σ verbatim, and the BC-frozen widener has its own rule
    (reinit_frozen_aim_log_std, min(AIM_LOG_STD_RESUME_INIT, cap) — that one
    may land exactly ON the cap, gh#91, untouched by T1). Callers that need the
    value in config.json must go through this helper, never re-derive it, so
    config and policy cannot drift.
    NOTE: whenever cap >= LOG_STD_INIT + AIM_LOG_STD_INIT_MARGIN (the 5v5
    default included) the init IS LOG_STD_INIT, so reinit_frozen_aim_log_std's
    "still exactly at LOG_STD_INIT ⇒ BC-frozen" signature matches a *fresh*
    policy. That was already true before T1 and stays harmless — the widener
    only ever runs on a checkpoint being resumed, never on a fresh build.
    """
    return min(LOG_STD_INIT, float(cap) - AIM_LOG_STD_INIT_MARGIN)


# Fix #2: precomputed log(2π) for the analytic Normal log-prob/entropy
# replacing torch.distributions.Normal in _hybrid_sample_logits.
_LOG_2PI = math.log(2.0 * math.pi)

# F8 (2026-07-06 adversarial review): per-head [start, end) column ranges of
# the flat (ACTION_MASK_DIM,) action-mask row, derived from ACTION_HEAD_SIZES
# exactly like the C side derives moff[] in compute_masks (cs2_env.h). Layout
# at time of writing: move 0-8, shoot 9-10, reload 11-12, weapon 13-15,
# use 16-17, crouch 18-19, jump 20-21.
_MASK_HEAD_SLICES = []
_off = 0
for _sz in ACTION_HEAD_SIZES:
    _MASK_HEAD_SLICES.append((_off, _off + _sz))
    _off += _sz
del _off, _sz


def _apply_action_masks(logits_list, mask):
    """Mask invalid action bins out of the per-head logits (F8).

    mask : (B, ACTION_MASK_DIM) bool/int8 tensor, 1 = valid — the C-computed
    masks from cs2_env.h compute_masks, sliced per head via _MASK_HEAD_SLICES.
    Invalid bins are filled with finfo.min/2, NOT -inf: after log_softmax the
    masked log-prob stays FINITE (≈ dtype-min/2, since any real logit is
    negligible against it), so entropy terms are exactly p·logp = 0·finite = 0
    instead of 0·(-inf) = NaN. exp(min/2 - lse) underflows to exactly 0, so
    multinomial can never draw a masked bin. The C side guarantees ≥1 valid
    bin per head per agent (dead agents get per-head no-ops), so the masked
    softmax is always well-defined — do NOT relax that invariant in C without
    revisiting this function.

    Returns a NEW list; input logits are not mutated (callers may hold them).
    """
    import torch

    masked = []
    for (lo, hi), lg in zip(_MASK_HEAD_SLICES, logits_list, strict=True):
        head_valid = mask[..., lo:hi] != 0
        fill = torch.finfo(lg.dtype).min / 2
        masked.append(lg.masked_fill(~head_valid, fill))
    return masked


def validate_aim_log_std_max(aim_log_std_max) -> float:
    """Resolve + range-check the run's aim σ cap (R0-E.3, #131).

    Returns the float cap (LOG_STD_MAX when None). Raises ValueError unless
    LOG_STD_MIN + 0.4 < cap <= LOG_STD_MAX, i.e. σ in (0.0149, 0.5].

    WHY a separate torch-free helper: build_policy() runs only once the env
    and torch are up, after `--dump-config` (the Modal runner's fingerprint
    step) has already exited, so a cap checked only there would let a bad
    --aim-log-std-max through the fingerprint. The CLI
    (`cs2rl.train.__main__`'s module-level `if __name__ == "__main__"` block)
    calls this after `parser.parse_args()` and the --seed check, above the
    --dump-config exit, so the fingerprint catches it.
    PITFALL (2026-08-30, rung1 sweep): the bound is INCLUSIVE at LOG_STD_MAX =
    log 0.5 = -0.693147..., so a hand-rounded "-0.6931" is > the cap by 5e-5
    and is REJECTED — pass -0.69315 (or omit the flag) for "σ cap 0.5".
    PITFALL (Rung 1a T1): the LOWER bound is no longer LOG_STD_MIN itself but
    LOG_STD_MIN + AIM_LOG_STD_CAP_MIN_HEADROOM — caps that narrow leave no room
    for the strictly-inside-the-band init (see resolve_aim_log_std_init) and
    would hand the run a gradient-dead σ. This NARROWS the accepted CLI range;
    σ caps below ~0.0149 rad (0.85°) have no experimental use (the recoil/
    hitbox scale alone is larger), so nothing legitimate is lost.
    """
    cap = float(LOG_STD_MAX if aim_log_std_max is None else aim_log_std_max)
    lo = LOG_STD_MIN + AIM_LOG_STD_CAP_MIN_HEADROOM
    if not (lo < cap <= LOG_STD_MAX):
        raise ValueError(
            f"aim_log_std_max={cap} must lie in ({lo}, {LOG_STD_MAX}] "
            f"(σ in ({math.exp(lo):.4f}, 0.5]). The lower bound is "
            f"LOG_STD_MIN + {AIM_LOG_STD_CAP_MIN_HEADROOM} rather than LOG_STD_MIN: the aim σ "
            f"is initialised {AIM_LOG_STD_INIT_MARGIN} below the cap so it starts strictly "
            f"inside the clamp band, and a cap this close to the σ floor would "
            f"put that init at or under LOG_STD_MIN={LOG_STD_MIN}, where clamp "
            f"back-propagates zero gradient and σ can never train.")
    return cap


def _aim_dim_weight(aim_dim_mask, mu_aim):
    """(AIM_DIM,) weight for the per-dim Gaussian terms (R0-E.2, #131).

    WHAT: ``aim_dim_mask`` moved to mu_aim's device/dtype, or all-ones when
    None. Shared by _hybrid_sample_logits and _hybrid_ppo_loss so the rollout
    and the update can never disagree on which dims are live — that
    disagreement would be an importance-ratio bug no single-site test sees.
    PITFALL: returns ones (not None) on the None path so callers can multiply
    unconditionally; the multiply by ones is exact in fp32.
    """
    import torch

    if aim_dim_mask is None:
        return torch.ones(mu_aim.shape[-1], device=mu_aim.device, dtype=mu_aim.dtype)
    return aim_dim_mask.to(device=mu_aim.device, dtype=mu_aim.dtype)
