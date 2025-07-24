from typing import Any
from base.agent_base import AgentBase
import numpy as np

def run_episode(env, agent: AgentBase, io_handler) -> dict:
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    return {
        "total_reward": total_reward,
        "job_assignments": env.job_assignments
    }
