import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from logs.log import write_log

from .q_net import QNet
from .replay_buffer import ReplayBuffer
from base.agent_base import AgentBase

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class DQNAgent(AgentBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: str = "hard",
        # tau: float = 0.005,
    ):
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = int(batch_size)
        self.target_update = target_update
        # self.tau = float(tau)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-net & target-net (kiến trúc 2 hidden layers, input chỉ state)
        self.q_net = QNet(state_dim, action_dim).to(self.device)
        self.target_net = QNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    # --------------------------------------------------------------------- #
    #  Action selection: epsilon-greedy on Q(s,·) with optional action mask #
    # --------------------------------------------------------------------- #
    def select_action(self, state, mask) -> int:
        """
        state: np.ndarray or torch.Tensor shape (S,) or (1,S)
        mask: Optional[np.ndarray] shape (A,), 1 = valid, 0 = invalid
        returns: action index int in [0, action_dim-1]
        """
        if mask is not None:
            mask = mask.astype(bool)
        else:
            print("there is cases when mask was None, invalid because action space is huge. Dangerous!!!")

        if np.random.rand() < float(self.epsilon):
            valid_idx = np.flatnonzero(mask)
            if len(valid_idx) == 0:
                print(f"valid_idx: {valid_idx}")
                return int(np.random.randint(self.action_dim))
            action = int(np.random.choice(valid_idx))

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q_net(s).squeeze(0).detach().cpu().numpy()

        if mask is not None:
            q[~mask] = -1e9  # chặn invalid action
        action = int(q.argmax())
        return action

    # ------------------------------------------------------------- #
    #                 Replay & one-step online update               #
    # ------------------------------------------------------------- #
    def add_transition(self, state, action, reward, next_state, done):
        """
        state, next_state: torch.Tensor float shape (S,)
        action: LongTensor scalar OR python int (action index)
        reward: float
        done: bool
        """
        # normalize action to LongTensor scalar
        if torch.is_tensor(action):
            action_t = action.detach().to(dtype=torch.long)
            if action_t.dim() > 0:
                action_t = action_t.view(-1)[0]
        else:
            action_t = torch.as_tensor(int(action), dtype=torch.long)

        self.replay_buffer.add(state.detach(), action_t.detach(), float(reward), next_state.detach(), bool(done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Build batches
        state_batch = torch.stack(batch.state).to(self.device)                 # (B,S)
        next_state_batch = torch.stack(batch.next_state).to(self.device)       # (B,S)
        action_batch = torch.stack(batch.action).to(self.device).view(-1)      # (B,)
        reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).view(-1)  # (B,)
        done_batch = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).view(-1)      # (B,)

        # Q(s,a)
        q_all = self.q_net(state_batch)                                        # (B,A)
        q_sa = q_all.gather(1, action_batch.unsqueeze(1)).squeeze(1)           # (B,)

        # target: r + γ * (1-done) * max_a' Q_target(s',a')
        with torch.no_grad():
            q_next_all = self.target_net(next_state_batch)                     # (B,A)
            q_next_max = q_next_all.max(dim=1).values                          # (B,)
            target = reward_batch + self.gamma * (1.0 - done_batch) * q_next_max

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # soft update (optional)
        if self.target_update == "soft":
            with torch.no_grad():
                for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                    tp.data.lerp_(p.data, self.tau)

    def update_target_net(self):
        if self.target_update == "hard":
            self.target_net.load_state_dict(self.q_net.state_dict())
