import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_factor=1.5):
        super().__init__()
        input_dim = state_dim + action_dim
        h1 = int(hidden_factor * input_dim)
        h2 = int(input_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            # nn.BatchNorm1d(h1),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(h1, h2),
            # nn.BatchNorm1d(h2),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(h2, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)