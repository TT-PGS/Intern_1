# class SchedulingModel:
#     def __init__(self, num_jobs, num_machines, max_job_size):
#         self.num_jobs = num_jobs
#         self.num_machines = num_machines
#         self.max_job_size = max_job_size

#     def describe(self):
#         return f"{self.num_jobs} jobs, {self.num_machines} machines, max size {self.max_job_size}"

from typing import List, Dict

class SchedulingModel:
    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        processing_times: List[int],
        split_min: int,
        time_windows: Dict[int, List[List[int]]]  # machine_id → [(start, end), ...]
    ):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = processing_times  # pi
        self.split_min = split_min
        self.time_windows = time_windows          # bj,t và bj,t+1

    def describe(self):
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "processing_times": self.processing_times,
            "split_min": self.split_min,
            "time_windows": self.time_windows
        }
