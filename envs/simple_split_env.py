# envs/simple_split_env.py

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
        print(f"Action {action} taken: Job {job_id} on Machine {machine_id}")

        if self.done or self.jobs_remaining[job_id] <= 0:
            return self._get_state(), -1.0, self.done, {}

        job_duration = self.jobs_remaining[job_id]
        start_time = max(self.machines[machine_id], self._earliest_available(machine_id))
        end_time = start_time + job_duration

        self.job_assignments.setdefault(job_id, {})
        self.job_assignments[job_id].setdefault(machine_id, []).append((start_time, end_time))

        self.jobs_remaining[job_id] = 0
        self.machines[machine_id] = end_time

        if all(j <= 0 for j in self.jobs_remaining):
            self.done = True
            makespan = max(self.machines)
            reward = -makespan
        else:
            reward = 0.0

        return self._get_state(), reward, self.done, {}

    def _earliest_available(self, machine_id: int) -> int:
        windows = self.time_windows[machine_id]
        for window_start, window_end in windows:
            if window_end - window_start > 0:
                return window_start
        return 0

    def _get_state(self) -> np.ndarray:
        return np.array(self.jobs_remaining + self.machines, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Jobs Remaining: {self.jobs_remaining}, Machines: {self.machines}, Assignments: {self.job_assignments}")

    def action_space_sample(self) -> int:
        return self.action_space.sample()

    def state_dim(self) -> int:
        return self.num_jobs + self.num_machines

    def action_dim(self) -> int:
        return 2  # job_id and machine_id as vector form for DQN agent

    def valid_action_vectors(self):
        actions = []
        for job_id, remaining in enumerate(self.jobs_remaining):
            if remaining <= 0:
                continue
            for machine_id in range(self.num_machines):
                actions.append([job_id, machine_id])
        return actions