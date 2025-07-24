import random
import numpy as np
from base.agent_base import AgentBase

class RandomAgent(AgentBase):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray) -> int:
        return random.randint(0, self.action_dim - 1)

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        # Random agent không học gì cả
        pass
