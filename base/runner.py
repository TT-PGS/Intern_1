from typing import Any, Dict
import numpy as np
import torch
from base.agent_base import AgentBase
from envs.simple_split_env import SimpleSplitSchedulingEnv

def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype)

def run_episode(env: SimpleSplitSchedulingEnv, agent: AgentBase, train: bool = True) -> Dict[str, Any]:
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        mask = env.valid_action_mask()

        if hasattr(agent, "select_action"):
            action = agent.select_action(state, mask)
        else:
            raise RuntimeError("Agent must implement select_action function.")

        next_state, reward, done, truncated, info = env.step(action)

        if train and hasattr(agent, "add_transition"):
            agent.add_transition(
                to_tensor(state, torch.float32),
                torch.as_tensor(action, dtype=torch.long),
                float(reward),
                to_tensor(next_state, torch.float32),
                bool(done or truncated),
            )
        if train and hasattr(agent, "update"):
            agent.update()

        state = next_state
        total_reward += reward

    if train and hasattr(agent, "update_target_net"):
        agent.update_target_net()

    return {"total_reward": float(total_reward), "job_assignments": getattr(env, "job_assignments", {})}
