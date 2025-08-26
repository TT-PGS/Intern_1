from .dqn_agent import DQNAgent

def create_agent(agent_name: str, state_dim: int, action_dim: int, **kwargs):
    print(f"Creating agent: {agent_name} with state_dim={state_dim}, action_dim={action_dim}, kwargs={kwargs}")
    if agent_name == "dqn":
        return DQNAgent(state_dim, action_dim, **kwargs)
    # ... other agents
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")