import gym
import numpy as np
from typing import Tuple

from base.model import SchedulingModel

class SimpleSplitSchedulingEnv(gym.Env):
    def __init__(self, model: SchedulingModel):
        super(SimpleSplitSchedulingEnv, self).__init__()
        self.model = model
        self.num_jobs = model.num_jobs
        self.num_machines = model.num_machines
        self.processing_times = model.processing_times
        self.split_min = model.split_min
        self.time_windows = model.time_windows  # Dict[int, List[List[int]]]

        self.jobs_remaining = []
        self.machines = []  # Current end times of each machine
        self.job_assignments = {}  # job_id -> machine_id
        self.done = False

        self.action_space = gym.spaces.Discrete(self.num_jobs * self.num_machines)
        obs_dim = self.num_jobs + self.num_machines
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.jobs_remaining = self.processing_times.copy()
        self.machines = [0] * self.num_machines
        self.job_assignments = {}
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        job_id = action // self.num_machines
        machine_id = action % self.num_machines
        print(f"++++++++++++ Action {action} taken: Job {job_id} on Machine {machine_id}")

        if self.done or self.jobs_remaining[job_id] <= 0:
            return self._get_state(), -1.0, self.done, {}

        if job_id not in self.job_assignments:
            self.job_assignments[job_id] = {}

        # Get valid time windows for this machine
        available_windows = self.time_windows[machine_id]
        print(f"Available time windows for Machine {machine_id}: {available_windows}")
        job_remaining = self.jobs_remaining[job_id]
        print(f"Job {job_id} remaining time: {job_remaining}")
        scheduled = False

        # for window_start, window_end in available_windows:
        for window in available_windows:
            print(f"Checking window: {window}")
            window_start, window_end = window
            window_length = window_end - window_start
            print(f"Windows start: {window_start}, end: {window_end}, length: {window_length}")
            if window_length < self.split_min:
                print(f"Window length {window_length} is less than split_min {self.split_min}, skipping")
                continue

            new_job_remaining = job_remaining - window_length
            print(f"New job remaining after scheduling of job {job_id}: {new_job_remaining}")

            if new_job_remaining > 0:
                self.jobs_remaining[job_id] = new_job_remaining
                job_remaining = new_job_remaining
                window[0] = window[1]
                if machine_id not in self.job_assignments[job_id]:
                    self.job_assignments[job_id].setdefault(machine_id, []).append((window_start, window_end))
                else:
                    self.job_assignments[job_id][machine_id].append((window_start, window_end))
            else:
                print(f"Job {job_id} scheduled on Machine {machine_id} from {window_start} to {window_start + job_remaining}")
                self.jobs_remaining[job_id] = 0
                new_window_start = window_start + job_remaining
                job_remaining = 0
                window[0] = new_window_start

                if machine_id not in self.job_assignments[job_id]:
                    self.job_assignments[job_id].setdefault(machine_id, []).append((window_start, new_window_start))
                else:
                    self.job_assignments[job_id][machine_id].append((window_start, new_window_start))

                scheduled = True
                break

        if not scheduled:
            return self._get_state(), -5.0, self.done, {}

        if all(j <= 0 for j in self.jobs_remaining):
            self.done = True
            makespan = max(self.machines)
            reward = -makespan
        else:
            reward = 0.0

        return self._get_state(), reward, self.done, {}

    def _get_state(self) -> np.ndarray:
        return np.array(self.jobs_remaining + self.machines, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Jobs Remaining: {self.jobs_remaining}, Machines: {self.machines}, Assignments: {self.job_assignments}")

    def action_space_sample(self) -> int:
        return self.action_space.sample()