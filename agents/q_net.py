import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class PerActionQNet(nn.Module):
    """
    Tính Q(s,(j,m)) cho mọi (j,m) trong một forward.
    - Đầu vào tối thiểu: job_vec (N,), mac_vec (M,), global_vec (d_g,)
      -> encode thành job_feat (N, d_j), mac_feat (M, d_m)
    """
    def __init__(self, N_max, M_max, d_j=16, d_m=16, d_g=8, hidden=64):
        super().__init__()
        self.N_max = N_max
        self.M_max = M_max
        self.d_j, self.d_m, self.d_g = d_j, d_m, d_g

        # Encoders (ví dụ tối giản: Linear + ReLU)
        self.job_enc = nn.Sequential(
            nn.Linear(1, d_j), nn.ReLU()
        )
        self.mac_enc = nn.Sequential(
            nn.Linear(1, d_m), nn.ReLU()
        )
        self.glo_enc = nn.Sequential(
            nn.Linear(d_g, d_g), nn.ReLU()
        )

        # Head dự đoán Q cho từng (j,m)
        d_tot = d_j + d_m + d_g
        self.head = nn.Sequential(
            nn.Linear(d_tot, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # ra Q cho 1 cặp
        )

    def forward(self, job_vec, mac_vec, global_vec):
        """
        job_vec: (N,) float32  -- ví dụ: remaining_time đã pad tới N_max
        mac_vec: (M,) float32  -- ví dụ: machine_finish đã pad tới M_max
        global_vec: (d_g,) float32 -- feature toàn cục (nếu chưa có, có thể dùng zeros)
        Trả về:
          Q_all: (N*M,)
        """
        N, M = job_vec.shape[0], mac_vec.shape[0]
        assert N == self.N_max and M == self.M_max, "Cần pad về (N_max, M_max)"

        # (N,1) -> (N,d_j)
        jf = self.job_enc(job_vec.view(N, 1).float())
        # (M,1) -> (M,d_m)
        mf = self.mac_enc(mac_vec.view(M, 1).float())
        # (d_g,) -> (d_g,)  (giữ nguyên chiều, đã qua MLP nhỏ)
        gf = self.glo_enc(global_vec.view(1, -1).float()).squeeze(0)  # (d_g,)

        # Broadcast thành (N,M, *)
        J = jf.unsqueeze(1).expand(N, M, self.d_j)       # (N,M,d_j)
        M_ = mf.unsqueeze(0).expand(N, M, self.d_m)      # (N,M,d_m)
        G = gf.view(1, 1, -1).expand(N, M, self.d_g)     # (N,M,d_g)

        X = torch.cat([J, M_, G], dim=-1)                # (N,M,d_tot)
        Q_grid = self.head(X).squeeze(-1)                # (N,M)
        return Q_grid.reshape(N * M)                     # (A,)

