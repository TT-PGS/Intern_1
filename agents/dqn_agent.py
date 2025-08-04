import torch
import torch.nn.functional as F
import random
from .q_net import QNet
from .replay_buffer import ReplayBuffer
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0,
                 buffer_size=10000, batch_size=64):
        self.q_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state_tensor, valid_actions_tensor):
        if random.random() < self.epsilon:
            return random.choice(valid_actions_tensor)
        else:
            with torch.no_grad():
                q_values = [self.q_net(state_tensor.to(self.device), a.to(self.device)).item()
                            for a in valid_actions_tensor]
            best_idx = int(torch.tensor(q_values).argmax().item())
            return valid_actions_tensor[best_idx]

    def add_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Tính Q-target
        q_targets = []
        for s_ in next_state_batch:
            with torch.no_grad():
                valid_actions = self._generate_valid_actions(s_)
                if not valid_actions:
                    q_targets.append(0.0)
                else:
                    q_vals = [self.target_net(s_.unsqueeze(0), a.to(self.device)).item() for a in valid_actions]
                    q_targets.append(max(q_vals))
        q_targets = torch.tensor(q_targets, dtype=torch.float32).unsqueeze(1).to(self.device)
        q_targets = reward_batch + self.gamma * q_targets * (1 - done_batch)

        # print("state_batch.shape =", state_batch.shape)
        # print("action_batch.shape =", action_batch.shape)
        # Tính Q-value hiện tại
        q_values = self.q_net(state_batch, action_batch)

        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _generate_valid_actions(self, state_tensor):
        # ⚠️ Tạm thời placeholder – cần gọi từ môi trường thật
        # Gợi ý: có thể truyền thêm env reference nếu cần
        return []
