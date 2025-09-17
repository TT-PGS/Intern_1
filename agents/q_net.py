import torch
import torch.nn as nn

class QNet(nn.Module):
    """
    DQN head với 2 hidden layers, input chỉ là state.
    - forward(state) -> Q(s,·)  shape (B, action_dim)
    - forward(state, action_idx) -> Q(s,a)  shape (B,)
    """
    def __init__(self, state_dim: int, action_dim: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )

    def forward(self, state: torch.Tensor, action_idx: torch.Tensor | None = None):
        # state: (B, state_dim)
        q_all = self.net(state)  # (B, action_dim)
        if action_idx is None:
            return q_all
        # action_idx: (B,)
        return q_all.gather(1, action_idx.long().view(-1, 1)).squeeze(1)
