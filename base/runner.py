from typing import Any
from base.agent_base import AgentBase
import numpy as np
import torch

def run_episode(env, agent: AgentBase, io_handler) -> dict:
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if hasattr(agent, "select_action") and agent.__class__.__name__ == "DQNAgent":
            valid_actions = env.valid_action_vectors()
            valid_actions_tensor = [torch.FloatTensor(a).unsqueeze(0) for a in valid_actions]
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = agent.select_action(state_tensor, valid_actions_tensor)
            action_vec = action_tensor.squeeze(0).numpy()
            job_id = int(action_vec[0])
            machine_id = int(action_vec[1])
            action = job_id * env.num_machines + machine_id
        else:
            action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action)

        if hasattr(agent, "add_transition"):
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            if isinstance(action, (int, np.integer)):
                job_id = int(action) // env.num_machines
                machine_id = int(action) % env.num_machines
                action_tensor = torch.tensor([job_id, machine_id], dtype=torch.float32)
            else:
                action_tensor = torch.FloatTensor(action_vec[:2])

            agent.add_transition(state_tensor, action_tensor, reward, next_state_tensor, done)
            agent.update()

        state = next_state
        total_reward += reward

    if hasattr(agent, "update_target_net"):
        agent.update_target_net()

    return {
        "total_reward": total_reward,
        "job_assignments": env.job_assignments
    }
