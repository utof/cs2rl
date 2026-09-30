"""`python -m cs2rl.train --eval`: a checkpoint over many seeds, printed.
"""

from collections import Counter

import numpy as np

from cs2rl.env.factory import build_env_for
from cs2rl.policy import (
    AGENT_IDS,
    init_policy_state,
    load_policy_from_checkpoint,
    resolve_policy_mode,
    select_policy_actions_native,
)
from cs2rl.spec.action import ACTION_HEAD_NAMES, ACTION_HEAD_SIZES


def extract_env_info(infos):
    if isinstance(infos, list):
        for info in infos:
            if info:
                return info
        return {}
    for aid in AGENT_IDS:
        info = infos.get(aid)
        if info:
            return info
    return {}


def format_histogram_line(label, counts):
    total = int(np.sum(counts))
    if total <= 0:
        return f"{label}: []"

    parts = []
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        pct = 100.0 * float(count) / total
        parts.append(f"{idx}={count} ({pct:.1f}%)")
    return f"{label}: [{', '.join(parts)}]"


# ── SECTION: Checkpoint evaluation ─────────────────────────────────────────


def evaluate_checkpoint(checkpoint_path=None,
                        device="cpu",
                        start_seed=0,
                        num_episodes=50,
                        policy_mode="auto"):
    policy = None
    policy_mode = resolve_policy_mode(checkpoint_path, policy_mode)
    if policy_mode != "random":
        policy = load_policy_from_checkpoint(checkpoint_path, device)

    metrics = Counter()
    action_hist = [np.zeros(size, dtype=np.int64) for size in ACTION_HEAD_SIZES]
    joint_hist = Counter()

    for episode_idx in range(num_episodes):
        seed = start_seed + episode_idx
        # W3 (#154): the SECOND eval_legacy site, and the only one that passes a
        # seed. That difference is the whole reason the role's builder takes an
        # UNSET sentinel rather than seed=None — env.c.cs2_env.make_env's own
        # default is 0 (not train.py's own make_env, which takes no seed), so spelling the other site's absent seed as None would have changed
        # the env it builds, invisibly to static_data_scalars().
        env = build_env_for("eval_legacy", seed=seed)
        obs, _ = env.reset(seed=seed)
        policy_state = init_policy_state(policy, device)

        done = False
        step_count = 0
        # R0-G: the env's INSTANCE round_time (a --round-time-ticks run may be
        # far shorter than nav.ROUND_TIME); ×2 is the runaway guard only.
        while not done and step_count < env.round_time * 2:
            if policy_mode == "random":
                actions = np.asarray(env.action_space.sample(), dtype=np.int32)
                # See record_episode comment: random has no continuous head;
                # pass zeros. Eval metrics under random policy reflect "no aim
                # input" which is the prior behaviour anyway.
                cont = np.zeros((actions.shape[0], 2), dtype=np.float32)
            else:
                actions, cont = select_policy_actions_native(policy, obs, device, policy_state,
                                                             policy_mode)

            for action in actions:
                for head_idx, action_value in enumerate(action):
                    action_hist[head_idx][int(action_value)] += 1
                joint_hist[tuple(int(v) for v in action)] += 1

            obs, rewards, terms, truncs, infos = env.step(actions, cont)
            step_count += 1

            step_info = extract_env_info(infos)
            metrics["bomb_planted"] += int(step_info.get("bomb_planted", 0))
            metrics["bomb_defused"] += int(step_info.get("bomb_defused", 0))
            metrics["kills_t"] += int(step_info.get("kills_t", 0))
            metrics["kills_ct"] += int(step_info.get("kills_ct", 0))
            metrics["blocked_moves_t"] += int(step_info.get("blocked_moves_t", 0))
            metrics["blocked_moves_ct"] += int(step_info.get("blocked_moves_ct", 0))

            done = bool(np.all(terms))
            if policy_state is not None:
                policy_state["done"] = policy_state["done"].new_tensor(
                    np.logical_or(terms, truncs).astype(np.float32))

            if done:
                metrics["episodes"] += 1
                metrics["winner_t"] += int(step_info.get("winner_t", 0))
                metrics["winner_ct"] += int(step_info.get("winner_ct", 0))
                metrics["timed_out"] += int(step_info.get("timed_out", 0))
                metrics["alive_t_end"] += int(step_info.get("alive_t_end", 0))
                metrics["alive_ct_end"] += int(step_info.get("alive_ct_end", 0))
                metrics["round_length"] += int(step_info.get("round_length", step_count))

    episodes = max(1, metrics["episodes"])
    total_actions = sum(joint_hist.values())

    print(f"[Eval] checkpoint={checkpoint_path or 'None'} policy={policy_mode} "
          f"episodes={metrics['episodes']} seeds={start_seed}..{start_seed + num_episodes - 1}")
    print(f"[Eval] timeout_rate={metrics['timed_out'] / episodes:.3f} "
          f"t_win_rate={metrics['winner_t'] / episodes:.3f} "
          f"ct_win_rate={metrics['winner_ct'] / episodes:.3f}")
    print(f"[Eval] plant_rate={metrics['bomb_planted'] / episodes:.3f} "
          f"defuse_rate={metrics['bomb_defused'] / episodes:.3f} "
          f"kills_t_per_round={metrics['kills_t'] / episodes:.3f} "
          f"kills_ct_per_round={metrics['kills_ct'] / episodes:.3f}")
    print(f"[Eval] avg_round_length={metrics['round_length'] / episodes:.1f} "
          f"avg_alive_t_end={metrics['alive_t_end'] / episodes:.2f} "
          f"avg_alive_ct_end={metrics['alive_ct_end'] / episodes:.2f}")
    print(f"[Eval] blocked_moves_t_per_round={metrics['blocked_moves_t'] / episodes:.2f} "
          f"blocked_moves_ct_per_round={metrics['blocked_moves_ct'] / episodes:.2f}")

    for head_name, counts in zip(ACTION_HEAD_NAMES, action_hist, strict=True):
        print(f"[Eval] {format_histogram_line(head_name, counts)}")

    top_joint = joint_hist.most_common(5)
    if total_actions > 0 and top_joint:
        parts = []
        for action, count in top_joint:
            parts.append(f"{list(action)}={count} ({100.0 * count / total_actions:.1f}%)")
        print(f"[Eval] top_actions: {', '.join(parts)}")
