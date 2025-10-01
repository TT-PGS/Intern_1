import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(h2, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(h2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x)
        v = self.value_stream(z)                    # (B,1)
        a = self.adv_stream(z)                      # (B,A)
        q = v + (a - a.mean(dim=1, keepdim=True))   # combine value + advantage
        return q