from sample_factory.algo.utils.context import global_env_registry
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_non_batched
from sample_factory.algo.utils.shared_buffers import (
    alloc_policy_output_tensors,
    alloc_trajectory_tensors,
)
from sample_factory.utils.attr_dict import AttrDict

from train import _CS2EnvFactory


def main():
    cfg = AttrDict(
        {
            "env": "cs2-dust2",
            "env_gpu_actions": False,
            "env_gpu_observations": True,
            "num_workers": 4,
            "worker_num_splits": 2,
            "num_envs_per_worker": 4,
            "serial_mode": False,
            "async_rl": True,
            "num_policies": 1,
            "rollout": 64,
            "num_batches_to_accumulate": 2,
            "batched_sampling": False,
            "device": "gpu",
            "restart_behavior": "resume",
            "env_frameskip": 1,
        }
    )

    global_env_registry()["cs2-dust2"] = _CS2EnvFactory()
    env = make_env_func_non_batched(cfg, env_config=None)
    env_info = extract_env_info(env, cfg)
    traj = alloc_trajectory_tensors(env_info, 1, cfg.rollout, 512, "cpu", False)
    policy_tensors, output_names, output_sizes = alloc_policy_output_tensors(
        cfg, env_info, 512, "cpu", False
    )

    print("action_space:", env_info.action_space)
    print("action_splits:", env_info.action_splits)
    print("all_discrete:", env_info.all_discrete)
    print("traj.actions:", traj["actions"].shape, traj["actions"].dtype)
    print("traj.action_logits:", traj["action_logits"].shape, traj["action_logits"].dtype)
    print("traj.log_prob_actions:", traj["log_prob_actions"].shape, traj["log_prob_actions"].dtype)
    print("policy_output_names:", output_names)
    print("policy_output_sizes:", output_sizes)
    print("policy_tensors:", policy_tensors.shape, policy_tensors.dtype)


if __name__ == "__main__":
    main()
