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

    def get_job_size(self, job_id: int) -> int:
        return self.processing_times[job_id]

    def get_machine_windows(self, machine_id: int) -> list:
        return self.time_windows.get(machine_id, [])

    def describe(self):
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "processing_times": self.processing_times,
            "split_min": self.split_min,
            "time_windows": self.time_windows
        }
