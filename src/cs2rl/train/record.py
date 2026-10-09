"""`python -m cs2rl.train --record`: one episode to a rerun replay (`cs2rl.viz.render`).

The episode is written by `cs2rl.viz.render.ReplayRecorder`, the recorder
`python -m cs2rl.experiment.replay` uses for a rung's rounds: one file layout, one viewer
command (`rerun FILE`).

`cs2rl.viz.render` is imported inside `record_episode`: it loads rerun at module scope.
"""

from pathlib import Path

import numpy as np

from cs2rl.policy import (
    init_policy_state,
    load_policy_from_checkpoint,
    resolve_policy_mode,
    select_policy_actions_native,
)
from cs2rl.spec.paths import RECORDINGS_DIR

# ── SECTION: Record Episode ────────────────────────────────────────────────


def record_episode(
        checkpoint_path=None,
        device="cpu",
        seed=0,
        policy_mode="auto",
        save_path=str(RECORDINGS_DIR / "latest.rrd"),
        map_data=None,
):
    from cs2rl.env.c.cs2_env import make_env as make_c_env
    from cs2rl.eval.baselines import StateReader
    from cs2rl.viz.render import ReplayRecorder, view_command

    save_path = Path(save_path)
    print(f"[Record] Recording one episode -> {save_path}")
    env = make_c_env(seed=seed, auto_reset=False, map_data=map_data)
    # The env's own MapData: for dust2 a second make_cs2_map call would build the vis matrix.
    recorder = ReplayRecorder(save_path, env.map_data)
    reader = StateReader(env)

    obs, _ = env.reset(seed=seed)

    policy = None
    policy_mode = resolve_policy_mode(checkpoint_path, policy_mode)
    if policy_mode != "random":
        policy = load_policy_from_checkpoint(checkpoint_path, device)

    policy_state = init_policy_state(policy, device)

    done = False
    step_count = 0
    st = reader.read().snapshot()
    with recorder:
        while not done and step_count < env.round_time * 2:
            if policy_mode == "random":
                actions = np.asarray(env.action_space.sample(), dtype=np.int32)
                # Random policy doesn't have a continuous head; pass zeros. This
                # leaves agents with pitch=0 / Δyaw=0 every tick, which is fine
                # for "random eval baseline" but obviously no aim variation.
                cont = np.zeros((actions.shape[0], 2), dtype=np.float32)
            else:
                # Returns (actions, cont) — the policy's actual continuous-aim
                # output. Without this, recordings/eval used cont=zeros and the
                # rerun replay showed agents stuck at spawn facing (the "look
                # forward" bug). AIM_DIM=2 is hardcoded against cs2_types.h;
                # if AIM_DIM ever changes the binding-side shape check will
                # raise before we ever silently miscount.
                actions, cont = select_policy_actions_native(policy, obs, device, policy_state,
                                                             policy_mode)

            obs, rewards, terms, truncs, infos = env.step(actions, cont)
            step_count += 1
            st_prev, st = st, reader.read().snapshot()
            recorder(step_count, st_prev, st, rewards=rewards)
            done = bool(np.all(terms))
            if policy_state is not None:
                policy_state["done"] = policy_state["done"].new_tensor(
                    np.logical_or(terms, truncs).astype(np.float32))

    print(f"[Record] Episode complete ({step_count} ticks). Saved to {save_path}")
    print(f"[Record] view: {view_command(save_path)}")
