"""The policy network and everything that reads or writes its outputs.

Owns: `build_policy` (the nested `Dust2Policy`), the hybrid-aim sampler
`_hybrid_sample_logits`, checkpoint loading (`load_policy_from_checkpoint`,
`load_state_dict_arch_checked`, the split-ness sniffers), the rollout-side helpers
(`init_policy_state`, `select_policy_actions_native`, `resolve_policy_mode`), the
log-std constants and the action-mask layout.

Module scope stays torch-free: every torch import is function-local, so eval, viz
and BC can import this module without paying for torch until they build a policy.
"""

import math
from pathlib import Path

import numpy as np

from cs2rl.env.factory import build_env_for
from cs2rl.spec.action import ACTION_HEAD_SIZES, AIM_DIM


def state_dict_is_split(state_dict):
    """True if this checkpoint was written by a T/CT split policy (spec §3.3).

    WHAT: presence of the `aim_log_std_t` parameter is the marker — it exists
    in exactly one architecture and nowhere else in the key space.

    WHY key inference rather than the config flag: config.json is rewritten
    unconditionally on every launch (the try-wrapped `config.json` write inside
    `cs2rl.train.loop.train()` — NOT the `--dump-config` early-exit write in
    `cs2rl.train.__main__`, which is conditional), so a flag-less
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
    `state_dict_is_split` — `config.json` is rewritten on every launch, so a
    flag-less crash-resume must recover trunk-ness from the keys. The heads
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
    train-main warm split). Policies built before the trunk attr exists
    compare as trunk-off via getattr(..., False).
    """
    ckpt_heads = state_dict_is_split(state_dict)
    ckpt_trunk = state_dict_is_trunk_split(state_dict)
    # Today's policies have no tct_split_trunk attr; treat missing as off.
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
    policy_env = build_env_for("eval_legacy")

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
            # μ is already tanh-squashed × max_turn_speed in HybridPolicy.forward
            # (the `torch.tanh(self.aim_mu...)` lines in build_policy's nested
            # class), so it's already bounded — no extra clamp needed.
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
    """Build the Dust2 recurrent policy.

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
    rollout (h,c) blend on obs[24]. Default False keeps today's
    `self.encoder` / `self.lstm` construction verbatim (legacy RNG pin).

    PITFALL: callers must not decide split-ness from config alone — every
    loader infers it from the checkpoint's keys (state_dict_is_split /
    state_dict_is_trunk_split), because config.json is rewritten on each
    launch and a flag-less crash-resume would otherwise rebuild the wrong
    architecture (spec §3.3).
    """
    import pufferlib.pytorch
    import torch
    import torch.nn as nn

    driver_env = getattr(vecenv, "driver_env", vecenv)
    obs_dim = (obs_dim_override
               if obs_dim_override is not None else driver_env.single_observation_space.shape[0])
    hidden = 256
    _cap = validate_aim_log_std_max(aim_log_std_max)
    # Rung 1a T1: the FRESH σ init follows the cap (see resolve_aim_log_std_init)
    # — LOG_STD_INIT at the 5v5 default cap, cap − 0.2 under a tight one, never
    # AT the cap where clamp would zero the gradient forever.
    _log_std_init = resolve_aim_log_std_init(_cap)

    class Dust2Policy(nn.Module):

        def __init__(self):
            super().__init__()
            self.hidden_size = hidden  # required by PufferLib LSTM logic
            self.obs_dim = obs_dim

            # Trunk-off keeps today's encoder/lstm construction verbatim so
            # the flag-off RNG stream (and LEGACY_PARAM_NAMES) stay pinned.
            # Trunk-on REPLACES those modules — do not keep a shared encoder
            # or lstm beside the copies.
            if not tct_split_trunk:
                self.encoder = nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden)),
                    nn.ReLU(),
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, hidden)),
                    nn.ReLU(),
                )
                self.lstm = nn.LSTM(hidden, hidden, batch_first=False)
                for name, p in self.lstm.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(p, 0)
                    elif "weight" in name:
                        nn.init.orthogonal_(p, gain=1.0)
            else:
                self.encoder_t = nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden)),
                    nn.ReLU(),
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, hidden)),
                    nn.ReLU(),
                )
                self.lstm_t = nn.LSTM(hidden, hidden, batch_first=False)
                for name, p in self.lstm_t.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(p, 0)
                    elif "weight" in name:
                        nn.init.orthogonal_(p, gain=1.0)
                # RNG hygiene (same as heads §3.7): CT construction is forked
                # so the subsequent heads draw stays at the same stream point
                # as flag-off. devices=[] forks the CPU generator only.
                with torch.random.fork_rng(devices=[]):
                    self.encoder_ct = nn.Sequential(
                        pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden)),
                        nn.ReLU(),
                        pufferlib.pytorch.layer_init(nn.Linear(hidden, hidden)),
                        nn.ReLU(),
                    )
                    self.lstm_ct = nn.LSTM(hidden, hidden, batch_first=False)
                    for name, p in self.lstm_ct.named_parameters():
                        if "bias" in name:
                            nn.init.constant_(p, 0)
                        elif "weight" in name:
                            nn.init.orthogonal_(p, gain=1.0)

            # Batch 7 (spec 2026-08-13 §3.1): plain bool, NOT a buffer — it
            # must never enter state_dict() or every existing checkpoint would
            # gain a key. Loaders read it to detect a policy/checkpoint
            # architecture mismatch (load_state_dict_arch_checked).
            # `tct_split_heads` here is build_policy's parameter, captured by
            # closure exactly like `obs_dim` and `hidden` above — the inner
            # class takes no new constructor argument. Same for tct_split_trunk.
            self.tct_split_heads = bool(tct_split_heads)
            self.tct_split_trunk = bool(tct_split_trunk)

            # Separate heads for MultiDiscrete(ACTION_HEAD_SIZES)
            #
            # Batch 3: continuous Gaussian aim head.
            # mu_aim → (B, AIM_DIM); tanh-squashed and scaled by max_turn_speed
            #   in forward(). State-DEPENDENT (per-step linear projection) so
            #   the policy can react to the current obs (visible enemies, yaw
            #   delta to target, etc.) when picking the mean Δyaw.
            # aim_log_std → (AIM_DIM,) — state-INDEPENDENT learnable parameter
            #   per Fan et al. IJCAI 2019 H-PPO baseline. Clamped in forward()
            #   to [LOG_STD_MIN, LOG_STD_MAX] so neither σ collapse (entropy
            #   loss → −∞) nor explosion (σ floods policy) is reachable.
            #   Rung 1a T1: the init is _log_std_init, not LOG_STD_INIT — it
            #   must start strictly INSIDE that clamp band or the parameter
            #   receives zero gradient for the whole run.
            # Pitfall: keep `std=0.01` on aim_mu init so the pre-tanh mean
            #   starts ~zero — otherwise the policy starts saturated and
            #   learning the Gaussian head is much slower.
            #
            # Batch 7 note on the deliberate duplication of these three
            # expressions across the two branches: the construction ORDER
            # (7 discrete heads → value_head → aim_mu → aim_log_std) is what
            # determines how many draws each layer takes from the global torch
            # RNG. Factoring the head group into a shared helper would move
            # value_head's draw and change every layer's init relative to the
            # legacy baseline at the same seed. Repetition here buys exact
            # RNG-stream parity between the flag-off and flag-on `_t` copies,
            # which is the whole point of spec §3.7.
            if not self.tct_split_heads:
                self.action_heads = nn.ModuleList([
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                    for n in ACTION_HEAD_SIZES
                ])
                self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden, 1), std=1.0)
                self.aim_mu = pufferlib.pytorch.layer_init(nn.Linear(hidden, AIM_DIM), std=0.01)
                self.aim_log_std = nn.Parameter(torch.full((AIM_DIM, ), _log_std_init))
            else:
                self.action_heads_t = nn.ModuleList([
                    pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                    for n in ACTION_HEAD_SIZES
                ])
                self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden, 1), std=1.0)
                self.aim_mu_t = pufferlib.pytorch.layer_init(nn.Linear(hidden, AIM_DIM), std=0.01)
                self.aim_log_std_t = nn.Parameter(torch.full((AIM_DIM, ), _log_std_init))
                # RNG hygiene (spec §3.7): the CT copy's construction is what
                # draws from the default stream, so it is forked — post-hoc
                # weight cloning would NOT restore stream parity. Without this
                # the flag-on run's every subsequent sample shifts relative to
                # the baseline at the same seed and "the split is the only
                # changed variable" is strictly false. devices=[] forks the CPU
                # generator only (construction is on CPU; .to(device) happens
                # after) and skips CUDA device enumeration.
                with torch.random.fork_rng(devices=[]):
                    self.action_heads_ct = nn.ModuleList([
                        pufferlib.pytorch.layer_init(nn.Linear(hidden, n), std=0.01)
                        for n in ACTION_HEAD_SIZES
                    ])
                    self.aim_mu_ct = pufferlib.pytorch.layer_init(nn.Linear(hidden, AIM_DIM),
                                                                  std=0.01)
                self.aim_log_std_ct = nn.Parameter(torch.full((AIM_DIM, ), _log_std_init))

            # max_turn_speed mirrors C sd->max_turn_speed (StaticData, π/4
            # default). Pulled from the vecenv's static-data block so the
            # policy stays bound to the env's actual cap even if it changes
            # at env construction time. Stored as a buffer (no grad, not a
            # learnable param, follows .to(device)). T5 carry-forward (I-1):
            # reuse the `driver_env` helper resolved at the top of build_policy instead of
            # an inline hasattr ladder — the helper already handles the
            # vecenv-vs-driver-env duality (test path passes a bare env;
            # production passes a Multiprocessing/Serial vecenv). One source
            # of truth for the "what is the env?" question.
            self.register_buffer(
                'max_turn_speed',
                torch.tensor(driver_env._c_env.sd.contents.max_turn_speed, dtype=torch.float32),
            )
            # R0-E.3/4 (#131): run properties, NOT checkpoint state
            # (persistent=False so old checkpoints load and new ones don't
            # carry them; SelfPlayManager re-applies them to past policies).
            # aim_log_std_max caps σ in every forward (replaces LOG_STD_MAX at
            # all clamp sites below); aim_dim_mask weights the per-dim
            # Gaussian log-prob/entropy terms ([1,0] when pitch is pinned).
            # PITFALL: sampling still draws BOTH dims (the env ignores dim 1
            # when pinned) — only the density is masked, so the stored
            # cont_action stays byte-identical to what the env consumed.
            self.aim_log_std_max = _cap
            self.register_buffer("aim_dim_mask",
                                 torch.tensor([1.0, 0.0] if pin_pitch else [1.0, 1.0]),
                                 persistent=False)

        @staticmethod
        def _blend(mask, out_t, out_ct):
            """Route a per-row output to its team's head copy (spec §3.2).

            mask is 0/1 with 1.0 == T, broadcastable over out_t's trailing
            dims. Branch-free (GPU-friendly) and autograd-exact: a T row's
            blend weight on the CT copy is literally 0, so it contributes zero
            gradient there — that is the routing correctness proof, pinned by
            test_pure_team_batch_leaves_other_copy_gradient_exactly_zero.

            PITFALL: the cast is load-bearing. A float32 mask multiplied into
            fp16 head outputs would silently promote them under any future
            autocast; casting to the output dtype keeps the arithmetic in the
            head's own precision.
            """
            m = mask.to(out_t.dtype)
            return m * out_t + (1.0 - m) * out_ct

        def _project_heads(self, hidden_out, mask):
            """Project aligned features into discrete logits and bounded aim parameters.

            Callers align the team mask with their flattened feature rows.
            Reuse the registered layers and clamp each team's log std before
            blending; the shared log std also expands to the aim batch shape.
            Legacy action sampling keeps its interleaved projection/sampling
            order in get_action_and_value.
            """
            if self.tct_split_heads:
                logits = [
                    self._blend(mask, ht(hidden_out), hct(hidden_out))
                    for ht, hct in zip(self.action_heads_t, self.action_heads_ct, strict=True)
                ]
                mu_aim = self._blend(mask,
                                     torch.tanh(self.aim_mu_t(hidden_out)) * self.max_turn_speed,
                                     torch.tanh(self.aim_mu_ct(hidden_out)) * self.max_turn_speed)
                log_std = self._blend(
                    mask,
                    torch.clamp(self.aim_log_std_t, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim),
                    torch.clamp(self.aim_log_std_ct, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim))
            else:
                logits = [head(hidden_out) for head in self.action_heads]
                mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
                log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN,
                                      self.aim_log_std_max).expand_as(mu_aim)
            return logits, mu_aim, log_std

        def get_value(self, x, lstm_state=None, done=None):
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            return self.value_head(hidden_out), lstm_state

        def get_action_and_value(self,
                                 x,
                                 lstm_state=None,
                                 done=None,
                                 action=None,
                                 continuous_action=None):
            """Hybrid sampler combining 7 categorical heads + 1 Gaussian aim head.

            Args:
                x: (B, OBS_DIM) observation batch.
                lstm_state: optional (h, c) tuple for the LSTM rollout.
                done: optional (B,) done-mask used to reset LSTM state.
                action: (B, ACTION_DIM=7) int64 — discrete actions; if None,
                    sample from the categorical heads.
                continuous_action: (B, AIM_DIM=2) float32 — (Δyaw, Δpitch) in
                    radians, already in [-max_turn_speed, +max_turn_speed];
                    if None, sample from the Normal head.

            Returns:
                (action, continuous_action, log_prob, entropy, value, lstm_state)
                log_prob and entropy aggregate across all 8 factors (7
                categorical + 1 Normal) — discrete factors are independent so
                their log-probs sum, and the Gaussian factor adds to the
                total. PPO loss assembly + the matching trainer side
                (rollout buffer for continuous_action, ratio computation)
                lands in task 5 via HybridAimVecEnv.
            """
            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            if self.tct_split_heads:
                # 2D input (B, obs): the team bit is a column. (The 3D
                # timestep trap lives in forward(), not here.)
                mask = x[:, 24:25]
                logits = [
                    self._blend(mask, ht(hidden_out), hct(hidden_out))
                    for ht, hct in zip(self.action_heads_t, self.action_heads_ct, strict=True)
                ]
            else:
                logits = [head(hidden_out) for head in self.action_heads]

            # Discrete sample / log-prob / entropy.
            dists = [torch.distributions.Categorical(logits=h) for h in logits]
            if action is None:
                action = torch.stack([d.sample() for d in dists], dim=-1)
            log_prob_d = sum(d.log_prob(action[..., i]) for i, d in enumerate(dists))
            entropy_d = sum(d.entropy() for d in dists)

            # Continuous (Normal) sample / log-prob / entropy. tanh+scale
            # bounds μ ∈ [-max_turn_speed, +max_turn_speed]; σ is clamped so
            # the Normal can't collapse or explode mid-training.
            if self.tct_split_heads:
                mu_aim = self._blend(mask,
                                     torch.tanh(self.aim_mu_t(hidden_out)) * self.max_turn_speed,
                                     torch.tanh(self.aim_mu_ct(hidden_out)) * self.max_turn_speed)
                # clamp EACH copy, then blend (spec §3.2) — identical result
                # for a 0/1 mask, but it matches the legacy clamp-at-use
                # semantics and keeps §3.6's per-team σ logs interpretable.
                log_std = self._blend(
                    mask,
                    torch.clamp(self.aim_log_std_t, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim),
                    torch.clamp(self.aim_log_std_ct, LOG_STD_MIN,
                                self.aim_log_std_max).expand_as(mu_aim))
            else:
                mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
                log_std = torch.clamp(self.aim_log_std, LOG_STD_MIN, self.aim_log_std_max)
            sigma = torch.exp(log_std).expand_as(mu_aim)
            aim_dist = torch.distributions.Normal(mu_aim, sigma)
            if continuous_action is None:
                # rsample preserves the reparameterised path through μ in case
                # the trainer ever uses pathwise gradients (PPO doesn't, but
                # cheap to keep this future-proof).
                continuous_action = aim_dist.rsample()
                # Re-clamp post-sample (T5 carry-forward I-2): σ exploration
                # can land outside the tanh band. The C env (env_step, cs2_env.h)
                # clamps |Δyaw| ≤ max_turn_speed silently with fminf/fmaxf —
                # NOT an assert. The Python-side clamp keeps the recorded
                # `continuous_action` byte-identical to what the env actually
                # consumed, which matters for PPO's importance ratio: if we
                # stored the unclamped sample and the env clipped it, the
                # ratio re-evaluation in _hybrid_ppo_loss would be wrong by
                # the clipping amount on every saturated step.
                continuous_action = torch.clamp(
                    continuous_action,
                    -self.max_turn_speed,
                    self.max_turn_speed,
                )
            # R0-E.2: per-dim weight applied BEFORE the sum so a pinned dim
            # contributes neither log-prob nor entropy (mirrors
            # _hybrid_sample_logits / _hybrid_ppo_loss).
            log_prob_c = (aim_dist.log_prob(continuous_action) * self.aim_dim_mask).sum(-1)
            # Closed-form Gaussian entropy: 0.5·log(2πe·σ²), summed across
            # AIM_DIM. .entropy() returns per-dim, so .sum(-1) is correct
            # for AIM_DIM=1 today and stays correct if AIM_DIM bumps to ≥2.
            entropy_c = (aim_dist.entropy() * self.aim_dim_mask).sum(-1)

            log_prob = log_prob_d + log_prob_c
            entropy = entropy_d + entropy_c
            value = self.value_head(hidden_out)
            return action, continuous_action, log_prob, entropy, value, lstm_state

        def forward_eval(self, x, state):
            # Batch 3: returns 4-tuple (logits, mu_aim, log_std, value) so
            # downstream samplers (eval loop / record / past-policy mixing)
            # can construct the full hybrid action. Existing 2-tuple
            # consumers break here — task 5 updates them.
            done = state.get("done")
            if done is None:
                done = x.new_zeros(x.shape[0])

            lstm_state = None
            if state.get("lstm_h") is not None and state.get("lstm_c") is not None:
                lstm_state = (state["lstm_h"], state["lstm_c"])

            hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
            if lstm_state is not None:
                state["lstm_h"], state["lstm_c"] = lstm_state

            mask = x[:, 24:25] if self.tct_split_heads else None
            logits, mu_aim, log_std = self._project_heads(hidden_out, mask)
            value = self.value_head(hidden_out)
            return logits, mu_aim, log_std, value

        def forward(self, x, state):
            # Training-path forward: time-batched BPTT (LSTM-BPTT fix).
            #
            # WHAT: same 4-tuple contract as forward_eval, but a 3D input
            #   (B=segments, T=bptt_horizon, OBS_DIM) is now unrolled through
            #   the LSTM along T — mirroring upstream PufferLib 3.0's
            #   models.LSTMWrapper.forward (encode flat → reshape seq-first →
            #   one nn.LSTM call → heads on the flat output). Pre-fix this
            #   flattened to (B*T, OBS) and ran the LSTM stateless per tick
            #   (seq-len 1, zero state), so the recurrent weights never saw
            #   through-time gradients and the PPO update recomputed
            #   logprobs/values under a DIFFERENT function than the rollout
            #   (forward_eval carries state tick-to-tick) — importance
            #   ratios ≠ 1 before the first gradient step.
            #
            # WHY zero initial state is CORRECT here (not an approximation):
            #   evaluate() zeroes trainer.lstm_h/c at its start, and with
            #   compute_batch_dims' segments == total_agents each agent row
            #   fills exactly ONE bptt_horizon segment per evaluate() call —
            #   so every stored segment really did start from zero state.
            #   PITFALL: if batch dims ever change so a row fills >1 segment
            #   per evaluate(), zero-init becomes wrong for the later
            #   segments and initial states must be stored at rollout time.
            #
            # state keys consumed (all optional; dict is NOT mutated):
            #   lstm_h / lstm_c — initial state override, (B, H) or (1, B, H).
            #     The trainer passes None → zero init (see above).
            #   terminals — (B, T) done flags from the rollout buffer;
            #     replicates forward_eval's (1-done)*state reset mid-segment
            #     (see _lstm_bptt). Omit for the ONNX / single-tick path.
            #
            # ONNX (task 6): a 2D (B, OBS_DIM) input takes T=1 through the
            # same code — one seq-len-1 LSTM call from zero state, identical
            # math to the pre-fix path — so the export stays single-pathway.
            if x.ndim == 3:
                B, TT = x.shape[0], x.shape[1]
            else:
                B, TT = x.shape[0], 1

            lstm_h = state.get("lstm_h") if isinstance(state, dict) else None
            lstm_c = state.get("lstm_c") if isinstance(state, dict) else None
            terminals = state.get("terminals") if isinstance(state, dict) else None
            H = self.hidden_size

            if not self.tct_split_trunk:
                h = self.encoder(x.reshape(B * TT, x.shape[-1]).float())
                # (T, B, H) seq-first
                h = h.reshape(B, TT, H).transpose(0, 1)
                if lstm_h is not None and lstm_c is not None:
                    hc = (lstm_h.reshape(1, B, H), lstm_c.reshape(1, B, H))
                else:
                    hc = (h.new_zeros(1, B, H), h.new_zeros(1, B, H))
                h = self._lstm_bptt(self.lstm, h, hc, terminals)
                # transpose back to (B, T, H) then flatten row-major so flat row
                # b*T + t lines up with mb_actions.reshape(-1, ...) in
                # _hybrid_ppo_loss — segment-major, time-minor. Changing this
                # ordering silently misaligns every logprob/advantage pairing.
                hidden_out = h.transpose(0, 1).reshape(B * TT, H)
            else:
                # Encoder is stateless: both copies see the same flat rows.
                # LSTM is not a head: each team LSTM sees ONLY its encoder's
                # activations. Never feed a mixed batch through one LSTM.
                x_flat = x.reshape(B * TT, x.shape[-1]).float()
                h_t = self.encoder_t(x_flat).reshape(B, TT, H).transpose(0, 1)
                h_ct = self.encoder_ct(x_flat).reshape(B, TT, H).transpose(0, 1)
                # zero-init BOTH team states when the trainer does not pass lstm_h/c
                if lstm_h is not None and lstm_c is not None:
                    hc_t = (lstm_h.reshape(1, B, H), lstm_c.reshape(1, B, H))
                    hc_ct = (lstm_h.reshape(1, B, H), lstm_c.reshape(1, B, H))
                else:
                    hc_t = (h_t.new_zeros(1, B, H), h_t.new_zeros(1, B, H))
                    hc_ct = (h_ct.new_zeros(1, B, H), h_ct.new_zeros(1, B, H))
                y_t = self._lstm_bptt(self.lstm_t, h_t, hc_t, terminals)
                y_ct = self._lstm_bptt(self.lstm_ct, h_ct, hc_ct, terminals)
                # PITFALL (spec §3.2): mask from 3D x with x[..., 24], never
                # x[:, 24] — that silently selects TIMESTEP 24.
                mask = x[..., 24].reshape(B * TT, 1)
                hidden_out = self._blend(mask,
                                         y_t.transpose(0, 1).reshape(B * TT, H),
                                         y_ct.transpose(0, 1).reshape(B * TT, H))

            mask = None
            if self.tct_split_heads:
                # PITFALL (spec §3.2 — the bug class this comment exists to
                # prevent): build the mask from the 3D x with x[..., 24].
                # Writing x[:, 24] on a (B, T, obs) input silently selects
                # TIMESTEP 24 instead of the team column. The reshape to
                # (B*TT, 1) is aligned with hidden_out's
                # h.transpose(0,1).reshape(B*TT, H) — both segment-major,
                # time-minor. Works unchanged for the 2D/ONNX path, where
                # TT == 1 and x[..., 24] is already the team column.
                mask = x[..., 24].reshape(B * TT, 1)
            logits, mu_aim, log_std = self._project_heads(hidden_out, mask)
            value = self.value_head(hidden_out)
            return logits, mu_aim, log_std, value

        def _lstm_bptt(self, lstm, h_seq, hc, terminals):
            """Run one LSTM over a full (T, B, H) segment with done-masking.

            WHAT: one `lstm(...)` call when the segment contains no episode
            boundaries (the common case — native PufferLib BPTT); otherwise
            the sequence is split at every tick where ANY row has a done and
            h/c are zero-masked per-row at those ticks before continuing.
            `lstm` is the module to run — flag-off forward passes
            `self.lstm`; trunk-on passes `self.lstm_t` / `self.lstm_ct`
            separately so each copy sees only its encoder's activations.

            WHY: the rollout (forward_eval → _forward_core) multiplies the
            carried state by (1 - done) BEFORE processing each tick, so a
            new episode starts memory-free. Training must replicate that
            reset or the recomputed logprobs at post-done ticks come from a
            different function than the rollout stored (biased PPO ratios)
            and gradients leak across episode boundaries. Upstream
            LSTMWrapper skips this (it never resets on done, rollout OR
            train, so it is self-consistent); we reset in rollout, hence we
            must also reset here. Passing the module in avoids copy-pasting
            this done-chunk loop per team.

            PITFALLS:
              * terminals[:, t] == 1 means "the obs at tick t is the FIRST
                obs of a new episode" (PufferLib autoreset delivers the done
                flag alongside the reset obs) — mask BEFORE consuming tick t,
                not after. Off-by-one here shifts every episode boundary.
              * The chunked split is exact, not an approximation: an LSTM
                over [t0, t1) then [t1, t2) with state carried equals one
                call over [t0, t2). Splits only cost extra kernel launches;
                a no-done minibatch stays a single cuDNN/oneDNN call.
              * .tolist() forces one device→host sync per minibatch —
                acceptable (the train loop already syncs via .item()s).
              * LSTM must not see the other team's rows via a shared module:
                the caller encodes per team, then calls this twice. Do not
                blend encoder outputs and run one LSTM.
            """
            if terminals is None:
                out, _ = lstm(h_seq, hc)
                return out
            TT, B, _H = h_seq.shape
            term = terminals.reshape(B, TT) > 0.5
            reset_ticks = torch.nonzero(term.any(dim=0)).flatten().tolist()
            if not reset_ticks:
                out, _ = lstm(h_seq, hc)
                return out
            outs = []
            h0, c0 = hc
            t0 = 0
            for t in reset_ticks:
                if t > t0:
                    out, (h0, c0) = lstm(h_seq[t0:t], (h0, c0))
                    outs.append(out)
                keep = (~term[:, t]).float().view(1, B, 1)
                h0 = h0 * keep
                c0 = c0 * keep
                t0 = t
            out, _ = lstm(h_seq[t0:], (h0, c0))
            outs.append(out)
            return torch.cat(outs, dim=0)

        def _forward_core(self, x, lstm_state, done):
            """Single-tick encode + LSTM for rollout / eval.

            WHAT: one seq-len-1 LSTM step. Trunk-off is today's
            `self.encoder` then `self.lstm(h.unsqueeze(0), ...)`. Trunk-on
            runs both team encoders+lstms the same way, blends hidden, and
            blends the returned `(h,c)` with `mask.view(1, B, 1)` so the
            trainer still stores one pair.

            WHY: `forward_eval` / `get_action_and_value` inherit routing
            from here. The trainer LSTM buffers stay one `(h,c)` per agent
            (do not change PufferLib's rollout state).

            PITFALLS:
              * Do not route this through `_lstm_bptt` — that helper is the
                training-path T-unroll. This must stay the per-tick
                `lstm(h.unsqueeze(0))` call.
              * LSTM must not see the other team's encoder activations:
                each copy is fed only its encoder's h. The incoming blended
                state is `(1-done)`-reset once, then fed to BOTH team LSTMs
                (unused output dropped by the 0/1 blend; used path is exact).
              * 2D mask is `x[:, 24:25]`. The 3D timestep-24 trap lives in
                `forward()`, not here.
            """
            if not self.tct_split_trunk:
                h = self.encoder(x.float())
                # lstm expects (seq, batch, features)
                if lstm_state is not None:
                    done = done.float()
                    h, lstm_state = self.lstm(
                        h.unsqueeze(0),
                        (
                            (1.0 - done).view(1, -1, 1) * lstm_state[0],
                            (1.0 - done).view(1, -1, 1) * lstm_state[1],
                        ),
                    )
                    h = h.squeeze(0)
                else:
                    h, lstm_state = self.lstm(h.unsqueeze(0))
                    h = h.squeeze(0)
                return h, lstm_state

            # 2D path: team bit is a column.
            mask = x[:, 24:25]
            h_t = self.encoder_t(x.float())
            h_ct = self.encoder_ct(x.float())
            if lstm_state is not None:
                done = done.float()
                reset_state = (
                    (1.0 - done).view(1, -1, 1) * lstm_state[0],
                    (1.0 - done).view(1, -1, 1) * lstm_state[1],
                )
                y_t, state_t = self.lstm_t(h_t.unsqueeze(0), reset_state)
                y_ct, state_ct = self.lstm_ct(h_ct.unsqueeze(0), reset_state)
            else:
                y_t, state_t = self.lstm_t(h_t.unsqueeze(0))
                y_ct, state_ct = self.lstm_ct(h_ct.unsqueeze(0))
            hidden_out = self._blend(mask, y_t.squeeze(0), y_ct.squeeze(0))
            m_state = mask.view(1, x.shape[0], 1)
            lstm_state = (
                self._blend(m_state, state_t[0], state_ct[0]),
                self._blend(m_state, state_t[1], state_ct[1]),
            )
            return hidden_out, lstm_state

    return Dust2Policy().to(device)


# ── SECTION: Batch 3 hybrid PPO helpers ────────────────────────────────────
#
# These three functions are the trainer-side complement of T4's HybridPolicy
# (mu_aim + log_std_aim Gaussian head bolted onto 7 categorical heads).
#
#   _hybrid_sample_logits  — the rollout-time replacement for
#                            pufferlib.pytorch.sample_logits(logits[, action]).
#                            Pure function; takes the 4-tuple emitted by
#                            HybridPolicy.forward / forward_eval and produces
#                            (action, continuous_action, log_prob, entropy).
#                            Joint factorised log-prob = sum of categorical
#                            log-probs + Normal log-prob (independence
#                            assumption per spec L8).
#
#   _hybrid_ppo_loss       — the PPO-update-time replacement, doing the
#                            forward pass + per-factor clipped policy loss
#                            (Fan et al. IJCAI 2019 H-PPO baseline). Returns
#                            split discrete/continuous ratios so the caller
#                            can keep KL/clipfrac diagnostics on the discrete
#                            half (back-compat with the existing log surface).
#
#   HybridAimVecEnv — wraps the backend send and Serial per-env step to
#                         forward continuous aim. Cs2PuffeRL._init_hybrid_aim
#                         owns the parallel rollout buffers used by train and
#                         evaluate. Both are ready before the first rollout.


def _hybrid_sample_logits(policy_out,
                          action=None,
                          continuous_action=None,
                          max_turn_speed=None,
                          mask=None,
                          aim_dim_mask=None):
    """Hybrid sampler for the 4-tuple HybridPolicy output (Batch 3 task 5).

    Replaces the four in-tree usages of
    ``pufferlib.pytorch.sample_logits(logits[, action=...])`` that previously
    assumed a 2-tuple policy contract. Pure function (no monkey-patching) so
    it can be unit-tested without spinning up a trainer.

    Inputs
    ------
    policy_out : 4-tuple
        (logits_list[7], mu_aim, log_std_aim, value) — the canonical T4
        output of HybridPolicy.forward / forward_eval. The value slot is
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

    Pre-Fix#1 (Batch 3 T5) this returned the SUMMED log_prob and SUMMED
    entropy as a 4-tuple, forcing the rollout caller to reconstruct
    distributions to recover the per-factor halves for self.logprobs_d /
    self.logprobs_c. That double-construction was measured at +437 ms/epoch
    on the i7-9750H smoke (16.0 ms/step actual vs 9.1 ms/step minimal).
    Surfacing the halves directly drops that cost to ~9.1 ms/step.
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
    log_prob_d = sum(
        lp.gather(-1, action[..., i:i + 1]).squeeze(-1) for i, lp in enumerate(log_probs_per_head))
    # Entropy: H = -Σ p log p. log_softmax already gives log p; multiply by
    # exp(log_softmax) = p. Single pass per head, no extra softmax call.
    entropy_d = sum(-(lp.exp() * lp).sum(-1) for lp in log_probs_per_head)

    # ── Continuous: 1D Gaussian aim head — hand-rolled (Fix #2) ──
    # σ comes pre-clamped from forward()/forward_eval() (LOG_STD_MIN/MAX), so
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
            # Same clamp logic as HybridPolicy.get_action_and_value (T4).
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
# Module-level so tests can `from cs2rl.policy import LOG_STD_MIN` without poking
# at the inner Dust2Policy class. Used in build_policy() forward paths and
# in the max_entropy calc that drives the SAC-α dual loop.
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

    WHY a separate torch-free helper: build_policy() only runs after the env
    and torch are up, so a bad --aim-log-std-max used to surface ~30 s into a
    launch AND slip past `--dump-config` (the Modal/run_rung1 fingerprint
    step). main() now calls this right after parse_args(), above the
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
