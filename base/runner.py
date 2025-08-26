# base/runner.py
from typing import Any, Dict
import numpy as np
import torch
from base.agent_base import AgentBase
from envs.simple_split_env import SimpleSplitSchedulingEnv

def _to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype)

def run_episode(env: SimpleSplitSchedulingEnv, agent: AgentBase, train: bool = True) -> Dict[str, Any]:
    state = env.reset()
    done = False
    total_reward = 0.0

    # step cap: Using in testing version, to avoid endless loops
    # can remove in future versions if needed
    max_steps = getattr(env, "max_steps", env.num_jobs * env.num_machines * 50)
    steps = 0

    while not done and steps < max_steps:
        mask = env.valid_action_mask()

        # chọn action index
        if hasattr(agent, "select_action"):
            try:
                action = agent.select_action(state, mask)
            except RuntimeError as e:
                print(f"RuntimeError: {e} in select_action.")
        else:
            raise RuntimeError("Agent must implement select_action function.")

        next_state, reward, done, info = env.step(action)

        if train and hasattr(agent, "add_transition"):
            agent.add_transition(
                _to_tensor(state, torch.float32),
                torch.as_tensor(action, dtype=torch.long),
                float(reward),
                _to_tensor(next_state, torch.float32),
                bool(done),
            )
        if train and hasattr(agent, "update"):
            agent.update()

        state = next_state
        total_reward += reward
        steps += 1

    if train and hasattr(agent, "update_target_net"):
        agent.update_target_net()

    return {"total_reward": total_reward, "job_assignments": getattr(env, "job_assignments", {})}
