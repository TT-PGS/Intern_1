from typing import List, Dict, Optional, Tuple

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
        # --- normalized (optional) ---
        processing_times_norm: Optional[List[float]] = None,
        split_min_norm: Optional[float] = None,
        time_windows_norm: Optional[Dict[int, List[List[float]]]] = None,
        total_capacity_all_windows_norm: Optional[float] = None,
        lower_bound_norm: Optional[float] = None,
        total_processing_time_norm: Optional[float] = None,
        # --- scales (optional, tiện debug) ---
        job_scale: Optional[float] = None,
        time_scale: Optional[float] = None,
    ):
        # --- raw ---
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = processing_times
        self.split_min = split_min
        self.time_windows = time_windows
        self.total_capacity_all_windows = total_capacity_all_windows
        self.lower_bound = lower_bound
        self.total_processing_time = total_processing_time
        self.feasible = feasible

        # --- norm (optional) ---
        self.processing_times_norm = processing_times_norm
        self.split_min_norm = split_min_norm
        self.time_windows_norm = time_windows_norm
        self.total_capacity_all_windows_norm = total_capacity_all_windows_norm
        self.lower_bound_norm = lower_bound_norm
        self.total_processing_time_norm = total_processing_time_norm

        # --- scales (optional) ---
        self.job_scale = job_scale
        self.time_scale = time_scale

        # --- minimal validation ---
        if len(self.processing_times) != self.num_jobs:
            raise ValueError("processing_times length must equal num_jobs")
        if self.processing_times and self.split_min > max(self.processing_times):
            # không raise cứng nếu bài toán cho phép, nhưng warn hợp lý
            pass
        # validate time_windows structure
        for m, wins in self.time_windows.items():
            for it in wins:
                if not (isinstance(it, list) and len(it) == 2):
                    raise ValueError(f"time_windows[{m}] item must be [start, end]")
                if not (isinstance(it[0], (int, float)) and isinstance(it[1], (int, float))):
                    raise ValueError(f"time_windows[{m}] values must be numeric")
                if it[0] >= it[1]:
                    raise ValueError(f"time_windows[{m}] window start must be < end")

    # --- raw getters ---
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

    def get_processing_times(self, norm: bool = False) -> List[float]:
        if norm and self.processing_times_norm is not None:
            return self.processing_times_norm
        return self.processing_times

    def get_split_min(self, norm: bool = False):
        if norm and self.split_min_norm is not None:
            return self.split_min_norm
        return self.split_min

    def get_job_size(self, job_id: int, norm: bool = False):
        if norm and self.processing_times_norm is not None:
            return self.processing_times_norm[job_id]
        return self.processing_times[job_id]

    def get_machine_windows(self, machine_id: int, norm: bool = False):
        if norm and self.time_windows_norm is not None:
            return self.time_windows_norm.get(machine_id, [])
        return self.time_windows.get(machine_id, [])

    def get_time_windows(self, norm: bool = False):
        if norm and self.time_windows_norm is not None:
            return self.time_windows_norm
        return self.time_windows

    # tiện lấy scales (có thể None)
    def get_scales(self) -> Tuple[Optional[float], Optional[float]]:
        return self.job_scale, self.time_scale

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
            "feasible": self.feasible,
            "processing_times_norm": self.processing_times_norm,
            "split_min_norm": self.split_min_norm,
            "time_windows_norm": self.time_windows_norm,
            "total_capacity_all_windows_norm": self.total_capacity_all_windows_norm,
            "lower_bound_norm": self.lower_bound_norm,
            "total_processing_time_norm": self.total_processing_time_norm,
            "job_scale": self.job_scale,
            "time_scale": self.time_scale,
        }
