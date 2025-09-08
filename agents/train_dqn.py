import os
from typing import List, Optional
import random
import numpy as np
import torch

from agents.agent_factory import create_agent
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.io_handler import ReadJsonIOHandler
from base.runner import run_episode

# ---------------- helpers ----------------
def set_seed(seed: Optional[int] = None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------- training -----------------
def train_dqn(
    model_config_path: str,
    model_save_path: str = "qnet.pt",
    episodes: int = 500,
    seed: Optional[int] = None,
):
    """
    Train DQN + chọn model theo eval greedy (ε=0) sau mỗi episode.
    """
    set_seed(seed)

    io = ReadJsonIOHandler(model_config_path)
    model_obj = io.get_input()

    # Train env
    # Set verbose = False khi test xong
    env = SimpleSplitSchedulingEnv(model_obj, verbose=True)

    state_dim = env.state_dim()
    action_dim = env.action_dim()
    print(f"[train_dqn] state_dim={state_dim} action_dim={action_dim}")

    agent = create_agent("dqn", state_dim, action_dim)

    best_eval_reward = float("-inf")
    train_rewards: List[float] = []
    eval_rewards: List[float] = []

    for ep in range(episodes):
        # ---- Train episode (epsilon-greedy, có học) ----
        train_result = run_episode(env, agent)
        tr = float(train_result["total_reward"])
        train_rewards.append(tr)
        print(f"Ep {ep:03d} | Train reward: {tr:.2f}")

        # Decay epsilon
        if hasattr(agent, "epsilon"):
            agent.epsilon = max(agent.epsilon * 0.99, 0.05)

        # ---- Eval greedy (ε=0, không học) ----
        eval_env = SimpleSplitSchedulingEnv(model_obj, True)
        eval_agent = create_agent("dqn", state_dim, action_dim)
        eval_agent.q_net.load_state_dict(agent.q_net.state_dict())
        eval_agent.q_net.eval()
        if hasattr(eval_agent, "epsilon"):
            eval_agent.epsilon = 0.0

        eval_result = run_episode(eval_env, eval_agent, train=False)
        er = float(eval_result["total_reward"])
        eval_rewards.append(er)
        print(f"          Eval  reward: {er:.2f}")

        # ---- Model selection theo eval ----
        if er > best_eval_reward:
            best_eval_reward = er
            torch.save(agent.q_net.state_dict(), model_save_path)
            print(f"📌 Saved BETTER (eval) model at Ep {ep}: eval={best_eval_reward:.2f} -> {model_save_path}")

        # Hard update target (nếu agent có)
        if hasattr(agent, "update_target_net"):
            agent.update_target_net()

    print(f"Training complete. Best EVAL reward: {best_eval_reward:.2f}. Model saved to {model_save_path}")
    return {"train_rewards": train_rewards, "eval_rewards": eval_rewards}

if __name__ == "__main__":
    cfg_path = os.path.join("configs", "splittable_jobs.json")
    stats = train_dqn(cfg_path, model_save_path="qnet.pt", episodes=500, seed=42)
