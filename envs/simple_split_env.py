import gym
import numpy as np
from typing import Tuple
from base.model import SchedulingModel
from split_job.dp_single_job import solve_min_timespan_cfg, solve_feasible_leftover_rule_cfg

class SimpleSplitSchedulingEnv(gym.Env):
    def __init__(self, model: SchedulingModel, mode: str = "timespan", verbose: bool = False):
        super(SimpleSplitSchedulingEnv, self).__init__()
        if isinstance(model, dict):
            tw = {int(k): [[int(a), int(b)] for (a, b) in v] for k, v in model["time_windows"].items()}
            model = SchedulingModel(
                num_jobs=int(model["num_jobs"]),
                num_machines=int(model["num_machines"]),
                processing_times=[int(x) for x in model["processing_times"]],
                split_min=int(model["split_min"]),
                time_windows=tw,
            )
        self.model = model
        self.num_jobs = model.num_jobs
        self.num_machines = model.num_machines
        self.processing_times = model.processing_times
        self.time_windows = model.time_windows
        self.split_min = model.split_min
        self.current_makespan = 0

        self.jobs_remaining = []
        self.machines = []
        self.job_assignments = {}
        self.done = False
        self.verbose = verbose
        self.mode = mode

        self.action_space = gym.spaces.Discrete(self.num_jobs * self.num_machines)
        obs_dim = self.num_jobs + self.num_machines
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.jobs_remaining = self.processing_times.copy()
        self.machines = [0] * self.num_machines
        self.job_assignments = {}
        self.done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array(self.jobs_remaining + self.machines, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Jobs Remaining: {self.jobs_remaining}, Machines: {self.machines}, Assignments: {self.job_assignments}")

    def action_space_sample(self) -> int:
        return self.action_space.sample()

    # thêm (nếu chưa có)
    def state_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def action_dim(self) -> int:
        return int(self.action_space.n)

    def encode_action(self, j: int, m: int) -> int:
        return j * self.num_machines + m

    def decode_action(self, a: int) -> tuple[int, int]:
        return (a // self.num_machines, a % self.num_machines)

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_jobs * self.num_machines, dtype=np.uint8)

        for job_id in range(self.num_jobs):
            if self.jobs_remaining[job_id] <= 0:
                continue
            for machine_id in range(self.num_machines):
                windows = self.time_windows[machine_id]
                remaining_time = self.jobs_remaining[job_id]
                result = solve_min_timespan_cfg(
                    total_processing_time=remaining_time,
                    windows_list=windows,
                    split_min=self.split_min
                )
                if result[0] is not None and result[1] is not None:
                    idx = job_id * self.num_machines + machine_id
                    mask[idx] = 1
        return mask

    def _update_windows(self, machine_id: int, start: int, end: int):
        new_windows = []
        for w_start, w_end in self.time_windows[machine_id]:
            if end <= w_start or start >= w_end:
                new_windows.append((w_start, w_end))
            elif start == w_start and end == w_end:
                continue  # dùng hết window → bỏ
            elif start == w_start:
                new_windows.append((end, w_end))  # còn lại phía sau
            else:
                raise ValueError("❌ Invalid DP: tried to assign in the middle of window, which shouldn't happen.")
        self.time_windows[machine_id] = new_windows

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        job_id = action // self.num_machines
        machine_id = action % self.num_machines

        # Nếu job đã xong → phạt nhẹ
        if self.jobs_remaining[job_id] <= 0:
            return self._get_state(), -0.1, self.done, {}

        # Lấy thời lượng còn lại và time windows của máy
        remaining_time = self.jobs_remaining[job_id]
        windows = self.time_windows[machine_id]

        # Gọi DP để chia job
        result = solve_min_timespan_cfg(
                total_processing_time=remaining_time,
                windows_list=windows,
                split_min=self.split_min
        )

        # if self.mode == "timespan":
        #     result = solve_min_timespan_cfg(
        #         total_processing_time=remaining_time,
        #         windows_list=windows,
        #         split_min=self.split_min
        #     )
        # else:
        #     result = solve_feasible_leftover_rule_cfg(
        #         total_processing_time=remaining_time,
        #         windows_list=windows,
        #         split_min=self.split_min
        #     )

        if result[0] is None or result[1] is None:
            # Không chia được → phạt nhẹ
            return self._get_state(), -0.1, self.done, {}

        finish_time, chunks = result
        # Cập nhật môi trường theo chunks
        for window_index, start, end in chunks:
            self.job_assignments.setdefault(job_id, {}).setdefault(machine_id, []).append((start, end))
            self._update_windows(machine_id, start, end)

        self.jobs_remaining[job_id] = 0
        self.machines[machine_id] = max(self.machines[machine_id], finish_time)

        # Tính reward: -Δ makespan
        new_makespan = max(time for time in self.machines)
        delta = new_makespan - self.current_makespan
        self.current_makespan = new_makespan
        reward = -delta

        # Kiểm tra kết thúc
        if all(t <= 0 for t in self.jobs_remaining):
            self.done = True
            return self._get_state(), reward, True, {}

        return self._get_state(), reward, False, {}