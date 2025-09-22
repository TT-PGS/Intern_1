import gymnasium as gym
import numpy as np
from typing import Tuple, Optional

from base.model import SchedulingModel
from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
    assign_job_to_machine,
)

class SimpleSplitSchedulingEnv(gym.Env):
    """
    Fixed-dim env via padding:
      - Observation: [jobs_remaining_pad (N_MAX), machines_finish_pad (M_MAX)]
      - Action: Discrete(N_MAX * M_MAX), decode to (job, machine) with M_MAX
      - Mask invalid (pad or infeasible) actions to 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        model: SchedulingModel,
        mode: str = "assign",
        verbose: bool = False,
        n_max: int = 50,
        m_max: int = 4,
        w_max: int = 24
    ):
        super().__init__()

        # --- Real instance data
        self.model = model
        self.real_num_jobs = model.get_num_jobs()
        self.real_num_machines = model.get_num_machines()
        self.processing_times = model.get_processing_times()
        self.time_windows = model.get_time_windows()
        self.split_min = model.get_split_min()

        # --- Fixed caps (padding targets)
        self.N_MAX = n_max if n_max is not None else self.real_num_jobs
        self.M_MAX = m_max if m_max is not None else self.real_num_machines
        self.W_MAX = w_max
        assert self.N_MAX >= self.real_num_jobs and self.M_MAX >= self.real_num_machines, \
            f"N_MAX {self.N_MAX}/M_MAX {self.M_MAX} must cover the real instance size {self.real_num_jobs} and {self.real_num_machines}."

        # --- Train/eval controls
        self.verbose = verbose
        self.mode = mode

        # --- Dynamics
        self.current_makespan = 0
        self.done = False
        self.jobs_remaining = []        # length = real_num_jobs
        self.machines_finish = []       # length = real_num_machines
        self.job_assignments = {}

        # --- RL API: fixed spaces
        obs_dim = self.N_MAX + self.M_MAX * (2 * self.W_MAX)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.N_MAX * self.M_MAX)

        # --- step cap (dùng kích thước THỰC để không kéo dài vô hạn)
        self.max_steps = self.real_num_jobs * max(1, self.real_num_machines) * 50

    # ----------------- Helpers -----------------

    def _decode_machine_windows_from_state(self, state_np: np.ndarray, m: int):
        """Trả list [(start, end)] dài W_MAX (pad 0 nếu thiếu)."""
        base = self.N_MAX + m * (2 * self.W_MAX)
        arr = state_np[base: base + 2 * self.W_MAX]
        return [(float(arr[2*i]), float(arr[2*i+1])) for i in range(self.W_MAX)]

    def _earliest_start_len(self, windows, split_min=1):
        """Chọn (start, length) sớm nhất với length >= split_min; nếu không có thì (inf, inf)."""
        best_s, best_len = float('inf'), float('inf')
        for s, e in windows:
            length = max(0.0, e - s)
            if length >= split_min:
                if (s < best_s) or (s == best_s and length < best_len):
                    best_s, best_len = s, length
        return best_s, best_len

    def suggest_action_earliest(self, state, mask, split_min=1):
        """
        Trả về action index (int) theo tiêu chí: machine có window sớm nhất.
        - mask: (A,) với A = N_MAX * M_MAX, 1=valid, 0=invalid
        - action map: flat = job * M_MAX + machine
        - tie-break: window ngắn hơn; tie-break nữa: job remaining lớn hơn
        """
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)

        valid_idx = np.flatnonzero(mask_arr)
        if valid_idx.size == 0:
            return None

        # jobs_remaining nằm ở đầu state
        jobs_remaining = state_np[: self.N_MAX]

        # precompute rank cho từng machine
        machine_rank = []
        for m in range(self.M_MAX):
            windows = self._decode_machine_windows_from_state(state_np, m)
            est, wlen = self._earliest_start_len(windows, split_min=split_min)
            machine_rank.append((est, wlen))

        best_key = None
        best_action = None

        for flat in valid_idx:
            j = flat // self.M_MAX
            m = flat %  self.M_MAX
            est, wlen = machine_rank[m]
            if np.isinf(est):  # không có window đủ dài
                continue

            jr = float(jobs_remaining[j]) if j < len(jobs_remaining) else 0.0
            key = (est, wlen, -jr, flat)  # sớm hơn, ngắn hơn, job lớn hơn

            if (best_key is None) or (key < best_key):
                best_key = key
                best_action = int(flat)

        return best_action

    def _encode_machine_timewindows(self, windows_for_machine):
        """
        Encode một máy thành vector cố định độ dài = 2 * W_MAX.
        Mỗi timewindow lấy (start, end). Nếu nhiều hơn W_MAX -> cắt bớt.
        Nếu ít hơn -> pad (0, 0).
        """
        encoded = []
        # Lấy tối đa W_MAX cửa sổ
        for w in windows_for_machine[: self.W_MAX]:
            if len(w) >= 2:
                # Trường hợp [start, end] hoặc [*, start, end]: lấy 2 phần tử cuối
                start, end = w[-2], w[-1]
            else:
                # Phòng hờ dữ liệu xấu
                start, end = 0, 0
            encoded.extend([start, end])

        # Pad thêm (0,0) nếu thiếu
        missing = self.W_MAX - min(len(windows_for_machine), self.W_MAX)
        if missing > 0:
            encoded.extend([0, 0] * missing)

        # Đảm bảo độ dài đúng
        if len(encoded) != 2 * self.W_MAX:
            encoded = (encoded + [0] * (2 * self.W_MAX - len(encoded)))[: 2 * self.W_MAX]
        return encoded


    def _pad_observation(self) -> np.ndarray:
        """
        Build observation vector với shape:
        obs_dim = N_MAX + M_MAX * (2 * W_MAX)

        - First N_MAX: jobs_remaining (pad 0)
        - Then M_MAX blocks, mỗi block dài 2*W_MAX ứng với (start, end) của các timewindow của 1 máy (pad (0,0))
        """
        # --- Jobs block (N_MAX) ---
        jobs_vector = list(self.jobs_remaining) + [0] * (self.N_MAX - self.real_num_jobs)
        jobs_vector = jobs_vector[: self.N_MAX]  # đề phòng lệch

        # --- Machines block (M_MAX * 2*W_MAX) ---
        machine_blocks = []
        # encode các máy hiện có
        for machine_id in range(self.real_num_machines):
            machine_blocks.extend(self._encode_machine_timewindows(self.time_windows[machine_id]))

        # pad thêm các máy trống nếu < M_MAX
        zeros_per_machine = 2 * self.W_MAX
        missing_machines = self.M_MAX - self.real_num_machines
        if missing_machines > 0:
            machine_blocks.extend([0] * (missing_machines * zeros_per_machine))

        # cắt/chắn để đúng kích thước
        machine_blocks = machine_blocks[: self.M_MAX * zeros_per_machine]

        obs = np.array(jobs_vector + machine_blocks, dtype=np.float32)

        # --- Sanity check chiều ---
        expected_len = self.N_MAX + self.M_MAX * (2 * self.W_MAX)
        if obs.shape[0] != expected_len:
            # ép chiều cho chắc (pad/truncate)
            if obs.shape[0] < expected_len:
                obs = np.pad(obs, (0, expected_len - obs.shape[0]), mode='constant')
            else:
                obs = obs[:expected_len]
        return obs

    def state_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def action_dim(self) -> int:
        return int(self.action_space.n)

    def encode_action(self, job_id: int, machine_id: int) -> int:
        return job_id * self.M_MAX + machine_id

    def decode_action(self, a: int) -> tuple[int, int]:
        return (a // self.M_MAX, a % self.M_MAX)

    # ----------------- Gym API -----------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.jobs_remaining = self.processing_times.copy()
        self.machines_finish = [0] * self.real_num_machines
        self.job_assignments = {}
        self.current_makespan = 0
        self.done = False
        return self._pad_observation(), {}

    def render(self, mode="human"):
        print(
            f"Jobs Remaining (real): {self.jobs_remaining}, "
            f"Machines (real finish): {self.machines_finish}, "
            f"Assignments: {self.job_assignments}"
        )

    # ----------------- Masking -----------------
    def valid_action_mask(self) -> np.ndarray:
        """
        Return a (N_MAX * M_MAX) uint8 vector: 1 = valid, 0 = invalid.
        Valid iff:
          - job_id < real_num_jobs AND job unfinished
          - machine_id < real_num_machines
          - DP assign feasible (has chunks)
        Everything outside real ranges (padding) is masked out.
        """
        mask = np.zeros(self.N_MAX * self.M_MAX, dtype=np.uint8)

        for job_id in range(self.real_num_jobs):
            if self.jobs_remaining[job_id] <= 0:
                continue
            remaining_time = self.jobs_remaining[job_id]
            for machine_id in range(self.real_num_machines):
                windows = self.time_windows[machine_id]
                finish_time, chunks = assign_job_to_machine(
                    total_processing_time=remaining_time,
                    windows_list=windows,
                    split_min=self.split_min
                )
                if finish_time is not None and chunks is not None:
                    idx = self.encode_action(job_id, machine_id)
                    mask[idx] = 1

        # (padding region stays 0)
        return mask

    # ----------------- Windows update -----------------
    def _update_windows(self, machine_id: int, start: int, end: int):
        new_windows = []
        for w_start, w_end in self.time_windows[machine_id]:
            if end <= w_start or start >= w_end:
                new_windows.append((w_start, w_end))
            elif start == w_start and end == w_end:
                # consumed fully
                continue
            elif start == w_start:
                new_windows.append((end, w_end))
            else:
                # we assume DP never returns "middle split" violating constraints
                raise ValueError("Invalid DP: assignment breaks a window in the middle.")
        self.time_windows[machine_id] = new_windows

    # ----------------- Transition -----------------
    def step(self, action: int):
        job_id, machine_id = self.decode_action(action)

        # Invalid if action points to padding or finished job
        if (
            job_id >= self.real_num_jobs
            or machine_id >= self.real_num_machines
            or self.jobs_remaining[job_id] <= 0
        ):
            # small penalty, keep episode running
            return self._pad_observation(), -10, self.done, False, {}

        remaining_time = self.jobs_remaining[job_id]
        windows = self.time_windows[machine_id]

        if self.mode == "timespan":
            finish_time, chunks = solve_min_timespan_cfg(remaining_time, windows, self.split_min)
        elif self.mode == "leftover":
            finish_time, chunks = solve_feasible_leftover_rule_cfg(remaining_time, windows, self.split_min)
        else:
            finish_time, chunks = assign_job_to_machine(remaining_time, windows, self.split_min)

        if finish_time is None or chunks is None:
            # infeasible now → small penalty
            return self._pad_observation(), -15, self.done, False, {}

        # apply chunks
        for (_w, start, end) in chunks:
            self.job_assignments.setdefault(job_id, {}).setdefault(machine_id, []).append((start, end))
            self._update_windows(machine_id, start, end)

        # mark job done; update machine finish
        self.jobs_remaining[job_id] = 0
        if machine_id < self.real_num_machines:
            self.machines_finish[machine_id] = max(self.machines_finish[machine_id], finish_time)

        # reward = -Δmakespan
        new_makespan = max(self.machines_finish) if self.machines_finish else 0
        delta = new_makespan - self.current_makespan
        self.current_makespan = new_makespan
        reward = -float(delta)

        # done?
        if all(t <= 0 for t in self.jobs_remaining):
            self.done = True
            return self._pad_observation(), reward, True, False, {}

        return self._pad_observation(), reward, False, False, {}
