import os, glob, random
from typing import List, Optional, Union
import numpy as np
import torch

from agents.agent_factory import create_agent
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.io_handler import ReadJsonIOHandler
from base.runner import run_episode

base_dir = os.path.dirname(os.path.abspath(__file__))
trained_model_output_path = os.path.join(base_dir, "..", "checkpoints")
os.makedirs(trained_model_output_path, exist_ok=True)

def as_cfg_list(paths: Union[str, List[str]]) -> List[str]:
    if isinstance(paths, list):
        return paths
    # allow glob or single file
    if any(ch in paths for ch in ["*", "?", "["]):
        return sorted(glob.glob(paths))
    if os.path.isdir(paths):
        return sorted(glob.glob(os.path.join(paths, "*.json")))
    return [paths]

def scan_caps(cfg_files: List[str]) -> tuple[int, int]:
    N_MAX, M_MAX = 0, 0
    for p in cfg_files:
        io = ReadJsonIOHandler(p)
        m = io.get_input()
        N_MAX = max(N_MAX, m.get_num_jobs())
        M_MAX = max(M_MAX, m.get_num_machines())
    return N_MAX, M_MAX

def set_seed(seed: Optional[int] = None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(cfg_path: str, N_MAX: int, M_MAX: int, verbose: bool = False) -> SimpleSplitSchedulingEnv:
    model_obj = ReadJsonIOHandler(cfg_path).get_input()
    return SimpleSplitSchedulingEnv(model_obj, verbose=verbose, n_max=N_MAX, m_max=M_MAX)

def train_dqn(
    model_config_paths: Union[str, List[str]],
    model_save_path: str = "qnet.pt",
    episodes: int = 5000,
    seed: int = 20250609,
    eval_k: int = 30,  # evaluate on K random cfgs
):
    set_seed(seed)

    cfg_files = as_cfg_list(model_config_paths)
    list_datasets = random.sample(cfg_files, episodes)
    # print(f" list datasets: {list_datasets}")
    assert len(cfg_files) > 0, "No config files found."
    N_MAX, M_MAX = scan_caps(cfg_files)

    # Build a probe env to get fixed dims
    probe_env = make_env(cfg_files[0], N_MAX, M_MAX, verbose=False)
    state_dim, action_dim = probe_env.state_dim(), probe_env.action_dim()
    print(f"[train_dqn] Fixed dims with padding: state_dim={state_dim} action_dim={action_dim} "
          f"(N_MAX={N_MAX}, M_MAX={M_MAX}; cfgs={len(cfg_files)}), training set: {len(list_datasets)} files")

    # Create shared agent with fixed dims
    agent = create_agent("dqn", state_dim, action_dim)

    best_eval_reward = float("-inf")
    train_rewards: List[float] = []
    eval_rewards: List[float] = []

    for ep in range(1):
        for cfg_path in list_datasets:
        # for ep in range(episodes):
            # ---- sample one cfg for training episode ----
            env = make_env(cfg_path, N_MAX, M_MAX, verbose=False)

            if hasattr(agent, "q_net"):
                agent.q_net.train()

            # Train (epsilon-greedy, learning enabled)
            train_result = run_episode(env, agent, train=True)
            tr = float(train_result["total_reward"])
            train_rewards.append(tr)
            # print(f"Ep {ep:04d} | Train cfg={os.path.basename(cfg_path)} | reward: {tr:.2f}")

            # Decay epsilon (if exists)
            if hasattr(agent, "epsilon"):
                agent.epsilon = max(agent.epsilon * 0.99, 0.05)

            # ---- Eval on K random cfgs (greedy, no learning) ----
            if hasattr(agent, "q_net"):
                agent.q_net.eval()
            if hasattr(agent, "epsilon"):
                old_eps = agent.epsilon
                agent.epsilon = 0.0

            sample_cfgs = random.sample(list_datasets, k=min(eval_k, len(list_datasets)))
            eval_sum = 0.0
            with torch.no_grad():
                for p in sample_cfgs:
                    eval_env = make_env(p, N_MAX, M_MAX, verbose=False)
                    eval_result = run_episode(eval_env, agent, train=False)
                    eval_sum += float(eval_result["total_reward"])

            er = eval_sum / len(sample_cfgs)
            eval_rewards.append(er)
            # print(f"          Eval@{len(sample_cfgs)} cfgs avg reward: {er:.2f}")

            # ---- Model selection ----
            if er > best_eval_reward:
                best_eval_reward = er
                torch.save({
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "N_MAX": N_MAX,
                    "M_MAX": M_MAX,
                    "model_state_dict": agent.q_net.state_dict(),
                }, model_save_path)
                # print(f"📌 Saved BETTER model at Ep {ep}: eval_avg={best_eval_reward:.2f} -> {model_save_path}")

            # restore epsilon
            if hasattr(agent, "epsilon"):
                agent.epsilon = old_eps if 'old_eps' in locals() else agent.epsilon

            # target update (if implemented)
            if hasattr(agent, "update_target_net"):
                agent.update_target_net()

    print(f"Training complete. Best EVAL(avg) reward: {best_eval_reward:.2f}. Model saved to {model_save_path}")
    return {"train_rewards": train_rewards, "eval_rewards": eval_rewards}
