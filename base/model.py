from typing import List, Dict

class SchedulingModel:
    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        processing_times: List[int],
        split_min: int,
        time_windows: Dict[int, List[List[int]]],
        total_capacity_all_windows: int,
        lower_bound: int,
        total_processing_time: int,
        feasible: bool,
    ):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = processing_times
        self.split_min = split_min
        self.time_windows = time_windows
        self.total_capacity_all_windows = total_capacity_all_windows
        self.lower_bound = lower_bound
        self.total_processing_time = total_processing_time
        self.feasible = feasible

    def get_total_capacity_all_windows(self) -> int:
        return self.total_capacity_all_windows

    def get_lower_bound(self) -> int:
        return self.lower_bound

    def get_total_processing_time(self) -> int:
        return self.total_processing_time

    def is_feasible(self) -> bool:
        return self.feasible

    def get_num_jobs(self) -> int:
        return self.num_jobs
    
    def get_num_machines(self) -> int:
        return self.num_machines

    def get_processing_times(self) -> List[int]:
        return self.processing_times

    def get_split_min(self) -> int:
        return self.split_min

    def get_job_size(self, job_id: int) -> int:
        return self.processing_times[job_id]

    def get_machine_windows(self, machine_id: int) -> List[List[int]]:
        return self.time_windows.get(machine_id, [])

    def get_time_windows(self) -> Dict[int, List[List[int]]]:
        return self.time_windows

    def describe(self):
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "processing_times": self.processing_times,
            "split_min": self.split_min,
            "time_windows": self.time_windows,
            "total_capacity_all_windows": self.total_capacity_all_windows,
            "lower_bound": self.lower_bound,
            "total_processing_time": self.total_processing_time,
            "feasible": self.feasible
        }
