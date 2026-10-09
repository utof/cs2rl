"""The recurrent policy network every run trains: `Dust2Policy`, an `nn.Module`.

Owns the module: its parameters and buffers, the order they are constructed in, and
its forward paths (`forward_eval` for the rollout, `forward` for the BPTT update,
`get_action_and_value`). `cs2rl.policy.build_policy` is its factory: it reads the
observation size and max_turn_speed from the env, validates the run's σ cap and
resolves the σ init, builds this module and moves it to the device. The constructor
takes those resolved values as they are and checks none of them.

PITFALL: construction ORDER is the global torch RNG stream, and the parameter and
buffer names are the checkpoint format (`cs2rl.policy.state_dict_is_split` and
`state_dict_is_trunk_split` read them). Reordering or renaming a module changes every
seeded run's initial weights or orphans every saved checkpoint.

WHY a module of its own: an `nn.Module` subclass needs torch at module scope, and
`cs2rl.policy`'s module scope stays torch-free (tests/train/test_w1_modules.py), so
`build_policy` imports this module inside its body.

WHY the σ floor is a constructor argument: this module sits below `cs2rl.policy` in
pyproject.toml's `cs2rl layers` contract, and the acyclic-siblings contract counts
function-local imports too, so it cannot import `LOG_STD_MIN` from `cs2rl.policy`.
"""

import pufferlib.pytorch
import torch
import torch.nn as nn

from cs2rl.spec.action import ACTION_HEAD_SIZES, AIM_DIM


def _make_trunk(obs_dim, hidden_size):
    """One encoder (Linear-ReLU-Linear-ReLU) and one LSTM, initialised in that order.

    The order is the global RNG stream: the two encoder layer_init draws, the LSTM's
    default parameter draw, then orthogonal weights over it (biases zeroed). Every
    trunk, shared or per team, is built by this one function, so the copies cannot
    drift apart in shape or init.
    """
    encoder = nn.Sequential(
        pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden_size)),
        nn.ReLU(),
        pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
    )
    lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
    for name, p in lstm.named_parameters():
        if "bias" in name:
            nn.init.constant_(p, 0)
        elif "weight" in name:
            nn.init.orthogonal_(p, gain=1.0)
    return encoder, lstm


def _make_action_heads(hidden_size):
    """One Linear per discrete head (ACTION_HEAD_SIZES), std 0.01 so logits start near 0."""
    return nn.ModuleList([
        pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in ACTION_HEAD_SIZES
    ])


def _make_aim_mu(hidden_size):
    """The aim-mean projection. std 0.01 keeps the pre-tanh mean near 0, unsaturated."""
    return pufferlib.pytorch.layer_init(nn.Linear(hidden_size, AIM_DIM), std=0.01)


class Dust2Policy(nn.Module):
    """Encoder + LSTM trunk, 7 categorical action heads, a Gaussian aim head, a value head.

    Arguments are the resolved run values; see `cs2rl.policy.build_policy` for what
    each flag builds and how loaders choose them. `max_turn_speed` is the env's cap in
    radians per tick (the C StaticData field). `aim_log_std_min` / `aim_log_std_max`
    bound every σ clamp; `aim_log_std_init` is the fresh σ parameter value and must lie
    strictly inside that band (`cs2rl.policy.resolve_aim_log_std_init`).
    """

    # Registered with register_buffer in __init__. Declared here so a type checker
    # reads them as tensors: nn.Module.__getattr__ types them as Tensor | Module.
    max_turn_speed: torch.Tensor
    aim_dim_mask: torch.Tensor

    def __init__(self,
                 obs_dim: int,
                 max_turn_speed: float,
                 *,
                 aim_log_std_min: float,
                 aim_log_std_max: float,
                 aim_log_std_init: float,
                 pin_pitch: bool = False,
                 tct_split_heads: bool = False,
                 tct_split_trunk: bool = False,
                 hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size                 # required by PufferLib LSTM logic
        self.obs_dim = obs_dim

        # Trunk-off builds the shared encoder/lstm (their names are
        # tests/test_tct_split.py's LEGACY_PARAM_NAMES). Trunk-on REPLACES
        # those modules — do not keep a shared encoder or lstm beside the copies.
        if not tct_split_trunk:
            self.encoder, self.lstm = _make_trunk(obs_dim, hidden_size)
        else:
            self.encoder_t, self.lstm_t = _make_trunk(obs_dim, hidden_size)
            # RNG hygiene (same as heads §3.7): CT construction is forked
            # so the subsequent heads draw stays at the same stream point
            # as flag-off. devices=[] forks the CPU generator only.
            with torch.random.fork_rng(devices=[]):
                self.encoder_ct, self.lstm_ct = _make_trunk(obs_dim, hidden_size)

        # Plain bools, NOT buffers (spec 2026-08-13 §3.1): they must never
        # enter state_dict() or every existing checkpoint would gain a key.
        # Loaders read them to detect a policy/checkpoint architecture
        # mismatch (load_state_dict_arch_checked).
        self.tct_split_heads = bool(tct_split_heads)
        self.tct_split_trunk = bool(tct_split_trunk)

        # Separate heads for MultiDiscrete(ACTION_HEAD_SIZES), and a
        # continuous Gaussian aim head:
        # mu_aim → (B, AIM_DIM); tanh-squashed and scaled by max_turn_speed
        #   in forward(). State-DEPENDENT (per-step linear projection) so
        #   the policy can react to the current obs (visible enemies, yaw
        #   delta to target, etc.) when picking the mean Δyaw.
        # aim_log_std → (AIM_DIM,) — state-INDEPENDENT learnable parameter
        #   per Fan et al. IJCAI 2019 H-PPO baseline. Clamped in forward()
        #   to [aim_log_std_min, aim_log_std_max] so neither σ collapse (entropy
        #   loss → −∞) nor explosion (σ floods policy) is reachable.
        #   The init is aim_log_std_init, not LOG_STD_INIT: it must start
        #   strictly INSIDE that clamp band or the parameter receives zero
        #   gradient for the whole run.
        # Pitfall: keep `std=0.01` on aim_mu init so the pre-tanh mean
        #   starts ~zero — otherwise the policy starts saturated and
        #   learning the Gaussian head is much slower.
        #
        # The head group is spelled out in each branch, not built by one
        # helper per team: construction ORDER (7 discrete heads → value_head
        # → aim_mu → aim_log_std) fixes which draws of the global torch RNG
        # each layer takes, and the one shared value_head sits between the T
        # heads and the T aim_mu. A per-team group helper would move
        # value_head's draw and change every later layer's init at the same
        # seed. Spelled out, the flag-on `_t` copies draw exactly the flag-off
        # stream (spec §3.7), pinned by tests/test_tct_split.py::
        # test_split_copies_draw_the_flag_off_rng_stream.
        if not self.tct_split_heads:
            self.action_heads = _make_action_heads(hidden_size)
            self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)
            self.aim_mu = _make_aim_mu(hidden_size)
            self.aim_log_std = nn.Parameter(torch.full((AIM_DIM, ), aim_log_std_init))
        else:
            self.action_heads_t = _make_action_heads(hidden_size)
            self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)
            self.aim_mu_t = _make_aim_mu(hidden_size)
            self.aim_log_std_t = nn.Parameter(torch.full((AIM_DIM, ), aim_log_std_init))
            # RNG hygiene (spec §3.7): the CT copy's construction is what
            # draws from the default stream, so it is forked — post-hoc
            # weight cloning would NOT restore stream parity. Without this
            # the flag-on run's every subsequent sample shifts relative to
            # the baseline at the same seed and "the split is the only
            # changed variable" is strictly false. devices=[] forks the CPU
            # generator only (construction is on CPU; .to(device) happens
            # after) and skips CUDA device enumeration.
            with torch.random.fork_rng(devices=[]):
                self.action_heads_ct = _make_action_heads(hidden_size)
                self.aim_mu_ct = _make_aim_mu(hidden_size)
            self.aim_log_std_ct = nn.Parameter(torch.full((AIM_DIM, ), aim_log_std_init))

        # max_turn_speed mirrors C sd->max_turn_speed (StaticData, π/4
        # default). build_policy reads it from the env's static-data block so
        # the policy stays bound to the env's actual cap even if it changes
        # at env construction time. Stored as a buffer (no grad, not a
        # learnable param, follows .to(device)).
        self.register_buffer(
            'max_turn_speed',
            torch.tensor(max_turn_speed, dtype=torch.float32),
        )
        # R0-E.3/4 (#131): run properties, NOT checkpoint state (plain
        # floats, and a persistent=False buffer, so old checkpoints load and
        # new ones don't carry them; SelfPlayManager re-applies them to past
        # policies). aim_log_std_min/max bound σ at every clamp site below;
        # aim_dim_mask weights the per-dim Gaussian log-prob/entropy terms
        # ([1,0] when pitch is pinned).
        # PITFALL: sampling still draws BOTH dims (the env ignores dim 1
        # when pinned) — only the density is masked, so the stored
        # cont_action stays byte-identical to what the env consumed.
        self.aim_log_std_min = aim_log_std_min
        self.aim_log_std_max = aim_log_std_max
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
                torch.clamp(self.aim_log_std_t, self.aim_log_std_min,
                            self.aim_log_std_max).expand_as(mu_aim),
                torch.clamp(self.aim_log_std_ct, self.aim_log_std_min,
                            self.aim_log_std_max).expand_as(mu_aim))
        else:
            logits = [head(hidden_out) for head in self.action_heads]
            mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
            log_std = torch.clamp(self.aim_log_std, self.aim_log_std_min,
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
            total.

        No production path calls this: the trainer samples through
        forward_eval + `cs2rl.policy._hybrid_sample_logits`. Tests use it.
        """
        hidden_out, lstm_state = self._forward_core(x, lstm_state, done)
        # 2D input (B, obs): the team bit is a column. (The 3D
        # timestep trap lives in forward() and _bptt_trunk(), not here.)
        mask = x[:, 24:25] if self.tct_split_heads else None
        if self.tct_split_heads:
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
                torch.clamp(self.aim_log_std_t, self.aim_log_std_min,
                            self.aim_log_std_max).expand_as(mu_aim),
                torch.clamp(self.aim_log_std_ct, self.aim_log_std_min,
                            self.aim_log_std_max).expand_as(mu_aim))
        else:
            mu_aim = torch.tanh(self.aim_mu(hidden_out)) * self.max_turn_speed
            log_std = torch.clamp(self.aim_log_std, self.aim_log_std_min, self.aim_log_std_max)
        sigma = torch.exp(log_std).expand_as(mu_aim)
        aim_dist = torch.distributions.Normal(mu_aim, sigma)
        if continuous_action is None:
            # rsample preserves the reparameterised path through μ in case
            # the trainer ever uses pathwise gradients (PPO doesn't, but
            # cheap to keep this future-proof).
            continuous_action = aim_dist.rsample()
            # Re-clamp post-sample: σ exploration
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
        # Closed-form Gaussian entropy, 0.5·log(2πe·σ²) per dim: .entropy()
        # returns one value per aim dim, aim_dim_mask zeroes a pinned one,
        # and .sum(-1) adds the AIM_DIM dims.
        entropy_c = (aim_dist.entropy() * self.aim_dim_mask).sum(-1)

        log_prob = log_prob_d + log_prob_c
        entropy = entropy_d + entropy_c
        value = self.value_head(hidden_out)
        return action, continuous_action, log_prob, entropy, value, lstm_state

    def forward_eval(self, x, state):
        """Rollout forward, one tick: (logits, mu_aim, log_std, value).

        `state` holds optional done / lstm_h / lstm_c; the new (h, c) is written
        back into it. The 4-tuple is what the samplers (rollout, eval, record,
        past-policy mixing) build the full hybrid action from.
        """
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
        """Training-path forward, time-batched BPTT: (logits, mu_aim, log_std, value).

        WHAT: the same 4-tuple as forward_eval, but a 3D input
          (B=segments, T=bptt_horizon, OBS_DIM) is unrolled through the LSTM
          along T, as upstream PufferLib 3.0's models.LSTMWrapper.forward does
          (encode flat → reshape seq-first → one nn.LSTM call → heads on the
          flat output). That is what lets the recurrent weights see
          through-time gradients, and it recomputes logprobs/values under the
          same function as the rollout (forward_eval carries state tick to
          tick), so on the first minibatch every row the current policy
          sampled has an importance ratio of 1 up to float rounding. Rows
          it did not sample store other logprobs (a past-policy opponent's,
          or 0 for a scripted `--opponent` noop/walker team), so their ratios need not be 1.

        WHY zero initial state is CORRECT here (not an approximation):
          evaluate() zeroes trainer.lstm_h/c at its start, and with
          compute_batch_dims' segments == total_agents each agent row
          fills exactly ONE bptt_horizon segment per evaluate() call —
          so every stored segment really did start from zero state.
          PITFALL: if batch dims ever change so a row fills >1 segment
          per evaluate(), zero-init becomes wrong for the later
          segments and initial states must be stored at rollout time.

        state keys consumed (all optional; dict is NOT mutated):
          lstm_h / lstm_c — initial state override, (B, H) or (1, B, H).
            The trainer passes None → zero init (see above).
          terminals — (B, T) done flags from the rollout buffer;
            replicates forward_eval's (1-done)*state reset mid-segment
            (see _lstm_bptt). Omit for a single-tick input.

        A 2D (B, OBS_DIM) input runs as T=1: one seq-len-1 LSTM call from
        zero state (BC's stateless `policy(x, {})` path in train_bc.py).
        """
        if x.ndim == 3:
            B, TT = x.shape[0], x.shape[1]
        else:
            B, TT = x.shape[0], 1

        lstm_h = state.get("lstm_h") if isinstance(state, dict) else None
        lstm_c = state.get("lstm_c") if isinstance(state, dict) else None
        terminals = state.get("terminals") if isinstance(state, dict) else None
        hidden_out = self._bptt_trunk(x, B, TT, lstm_h, lstm_c, terminals)

        mask = None
        if self.tct_split_heads:
            # PITFALL (spec §3.2 — the bug class this comment exists to
            # prevent): build the mask from the 3D x with x[..., 24].
            # Writing x[:, 24] on a (B, T, obs) input silently selects
            # TIMESTEP 24 instead of the team column. The reshape to
            # (B*TT, 1) is aligned with hidden_out's
            # h.transpose(0,1).reshape(B*TT, H) — both segment-major,
            # time-minor. Works unchanged for a 2D (B, OBS_DIM) input, where
            # TT == 1 and x[..., 24] is already the team column.
            mask = x[..., 24].reshape(B * TT, 1)
        logits, mu_aim, log_std = self._project_heads(hidden_out, mask)
        value = self.value_head(hidden_out)
        return logits, mu_aim, log_std, value

    def _bptt_trunk(self, x, B, TT, lstm_h, lstm_c, terminals):
        """Encode (B, T, obs) rows and unroll the trunk LSTM(s) along T: (B*T, H) out.

        `forward`'s trunk. Initial state is `lstm_h`/`lstm_c` when both are given, else
        zeros; `terminals` resets it mid-segment (see `_lstm_bptt`). Trunk-split runs
        each team's encoder and LSTM separately and blends their outputs on obs[24].
        """
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
            return h.transpose(0, 1).reshape(B * TT, H)
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
        return self._blend(mask,
                           y_t.transpose(0, 1).reshape(B * TT, H),
                           y_ct.transpose(0, 1).reshape(B * TT, H))

    def _lstm_bptt(self, lstm, h_seq, hc, terminals):
        """Run one LSTM over a full (T, B, H) segment with done-masking.

        WHAT: one `lstm(...)` call when the segment contains no episode
        boundaries (the common case — native PufferLib BPTT); otherwise
        the sequence is split at every tick where ANY row has a done and
        h/c are zero-masked per-row at those ticks before continuing.
        `lstm` is the module to run — `_bptt_trunk` passes `self.lstm`
        when the trunk is shared, and `self.lstm_t` / `self.lstm_ct`
        separately when it is split, so each copy sees only its
        encoder's activations.

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

        WHAT: one seq-len-1 LSTM step. Trunk-off is
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
            `forward()` and `_bptt_trunk()`, not here.
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
