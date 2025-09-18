import json
import os
from typing import List, Tuple, Optional, Dict, Any
from base.model import SchedulingModel
from base.io_handler import ReadJsonIOHandler

def assign_job_to_machine(
    total_processing_time: int,
    windows_list: List[List[int]],
    split_min: int
) -> Tuple[Optional[int], Optional[List[Tuple[int, int, int]]]]:
    """Assign job_id to machine_id following the paper

    Parameters
    ----------
    total_processing_time : int
        processing time of job
    windows_list : List[List[int]]
        windows of machine
    split_min : int
        split min of job

    Returns
    -------
    Tuple[Optional[int], Optional[List[Tuple[int, int, int]]]]
        (best_timespan, best_plan)
        best_timespan: None if not feasible
        best_plan: None if not feasible
    """
    job_remaing = total_processing_time
    plan = []
    for i, (start, end) in enumerate(windows_list):
        if job_remaing <= 0:
            break
        cap = end - start
        if job_remaing < 2*split_min:
            if job_remaing <= cap:
                plan.append((i, start, start + job_remaing))
                job_remaing = 0
                break
            else:
                continue
        
        if cap < split_min:
            continue
        if job_remaing <= cap - split_min:
            plan.append((i, start, start + job_remaing))
            job_remaing = 0
            break
        elif job_remaing > cap - split_min:
            if job_remaing <= cap:
                plan.append((i, start, start + job_remaing - split_min))
                job_remaing = split_min
                continue
            elif job_remaing > cap:
                if job_remaing - cap >= split_min:
                    plan.append((i, start, end))
                    job_remaing -= cap
                    continue
                elif job_remaing - cap < split_min:
                    if cap >= 2*split_min:
                        plan.append((i, start, end - split_min))
                        job_remaing = job_remaing - (cap - split_min)
                        continue
                    elif cap < 2*split_min:
                        continue
    
    if job_remaing > 0:
        #TODO: should get log here
        return None, None
    else:
        timespan = max(end for (_, _, end) in plan) - min(start for (_, start, _) in plan)
        return timespan, plan
    
def solve_min_timespan_cfg(
    total_processing_time: int,
    windows_list: List[List[int]],
    split_min: int
) -> Tuple[Optional[int], Optional[List[Tuple[int, int, int]]]]:
    """
    Tìm cách phân bổ một job có tổng thời gian `total_processing_time`
    vào các time windows cho trước sao cho **timespan = finish - start_first** là nhỏ nhất.

    Parameters
    ----------
    total_processing_time : int
        Tổng thời gian job cần xử lý (số đơn vị).
    windows_list : List[List[int]]
        Danh sách các time windows dạng [[start, end], ...], đã sắp theo start.
    split_min : int
        Kích thước nhỏ nhất của một mảnh (sub-job) khi chia.

    Returns
    -------
    best_timespan : Optional[int]
        Giá trị timespan nhỏ nhất tìm được, hoặc None nếu không khả thi.
    best_plan : Optional[List[Tuple[int, int, int]]]
        Danh sách các mảnh được gán [(window_index, start, end), ...],
        hoặc None nếu không khả thi.
    """
    num_windows = len(windows_list)
    capacities = [end - start for start, end in windows_list]
    starts = [start for start, _ in windows_list]

    if total_processing_time <= 0 or split_min <= 0 or sum(capacities) < total_processing_time:
        return None, None

    INF = 10**15
    dp_finish = [[INF] * (total_processing_time + 1) for _ in range(num_windows + 1)]
    parent = [[(-1, -1)] * (total_processing_time + 1) for _ in range(num_windows + 1)]

    # Base case: cần xử lý 0 đơn vị ⇒ hoàn tất ngay
    for idx in range(num_windows + 1):
        dp_finish[idx][0] = 0
        parent[idx][0] = (-1, 0)

    # Điền bảng DP từ window cuối về đầu
    for window_idx in range(num_windows - 1, -1, -1):
        cap = capacities[window_idx]
        start_time = starts[window_idx]
        for remain_time in range(total_processing_time + 1):
            best_finish = dp_finish[window_idx + 1][remain_time]
            best_choice = (0, remain_time)

            max_k = min(cap, remain_time)
            if max_k >= split_min:
                for k in range(split_min, max_k + 1):
                    if remain_time - k == 0:
                        candidate_finish = start_time + k
                        if candidate_finish < best_finish:
                            best_finish, best_choice = candidate_finish, (k, 0)
                    else:
                        next_finish = dp_finish[window_idx + 1][remain_time - k]
                        if next_finish < INF and next_finish < best_finish:
                            best_finish, best_choice = next_finish, (k, remain_time - k)

            dp_finish[window_idx][remain_time] = best_finish
            parent[window_idx][remain_time] = best_choice

    # Tìm best timespan khi chọn window đầu tiên để bắt đầu
    best_timespan = INF
    best_plan = None
    for first_win in range(num_windows):
        cap_first = capacities[first_win]
        start_first = starts[first_win]
        max_k = min(cap_first, total_processing_time)
        if max_k < split_min:
            continue
        for k in range(split_min, max_k + 1):
            if total_processing_time - k == 0:
                finish = start_first + k
            else:
                finish = dp_finish[first_win + 1][total_processing_time - k]
            if finish >= INF:
                continue
            timespan = finish - start_first
            if timespan < best_timespan:
                plan = [(first_win, start_first, start_first + k)]
                idx, remain = first_win + 1, total_processing_time - k
                while remain > 0 and idx <= num_windows:
                    used_k, next_remain = parent[idx][remain]
                    if used_k > 0:
                        s_i = starts[idx]
                        plan.append((idx, s_i, s_i + used_k))
                    remain = next_remain
                    idx += 1
                best_timespan, best_plan = timespan, plan
    return (best_timespan, best_plan) if best_plan else (None, None)


def solve_feasible_leftover_rule_cfg(
    total_processing_time: int,
    windows_list: List[List[int]],
    split_min: int
) -> Tuple[bool, Optional[List[Tuple[int, int, int]]]]:
    """
    Kiểm tra khả thi khi phân bổ một job vào các windows,
    với luật leftover: nếu chỉ dùng một phần window thì phần dư phải >= split_min.

    Chính sách chọn k:
        - Ưu tiên dùng hết window (k = cap).
        - Nếu partial: duyệt từ lớn xuống nhỏ để ưu tiên mảnh lớn, ít mảnh.

    Parameters
    ----------
    total_processing_time : int
        Tổng thời gian job cần xử lý.
    windows_list : List[List[int]]
        Danh sách windows [[start, end], ...].
    split_min : int
        Kích thước mảnh tối thiểu.

    Returns
    -------
    feasible : bool
        True nếu tìm được cách gán hợp lệ, False nếu không.
    plan : Optional[List[Tuple[int, int, int]]]
        Kế hoạch [(window_index, start, end), ...] nếu khả thi.
    """
    # Chuẩn hóa windows: bỏ windows rỗng và sort theo start
    windows_sorted = sorted(
        [[a, b] for a, b in windows_list if b > a],
        key=lambda ab: (ab[0], ab[1])
    )
    num_windows = len(windows_sorted)
    if num_windows == 0:
        return (0, []) if total_processing_time == 0 else (None, None)

    capacities = [b - a for a, b in windows_sorted]
    starts = [a for a, _ in windows_sorted]

    if total_processing_time <= 0 or split_min <= 0 or sum(capacities) < total_processing_time:
        return None, None

    choices: List[List[int]] = []
    for cap in capacities:
        opts = [0]
        if cap >= split_min:
            opts.append(cap)
            max_partial = cap - split_min
            if max_partial >= split_min:
                for k in range(max_partial, split_min - 1, -1):
                    if k not in opts:
                        opts.append(k)
        choices.append(opts)

    T = total_processing_time
    reach = [[False] * (T + 1) for _ in range(num_windows + 1)]
    parent = [[(-1, -1)] * (T + 1) for _ in range(num_windows + 1)]

    reach[num_windows][0] = True
    parent[num_windows][0] = (-1, 0)

    for win_idx in range(num_windows - 1, -1, -1):
        for remain in range(0, T + 1):
            if reach[win_idx + 1][remain]:
                reach[win_idx][remain] = True
                parent[win_idx][remain] = (0, remain)
            for k in choices[win_idx]:
                if k == 0:
                    continue
                if k <= remain and reach[win_idx + 1][remain - k]:
                    reach[win_idx][remain] = True
                    parent[win_idx][remain] = (k, remain - k)
                    break

    if not reach[0][T]:
        return None, None

    alloc: List[Tuple[int, int, int]] = []
    win_idx, remain = 0, T
    while win_idx < num_windows and remain > 0:
        k, rem = parent[win_idx][remain]
        if k > 0:
            start_i = starts[win_idx]
            alloc.append((win_idx, start_i, start_i + k))
        remain = rem
        win_idx += 1

    finish_time = max(end for (_, _, end) in alloc) if alloc else 0
    return finish_time, alloc

def print_list(prefix: str, lst: list) -> None:
    print(prefix)
    for item in lst:
        print(f"{item}\n")

if __name__ == "__main__":
    # config_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "splittable_jobs.json")
    config_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "input_10_2_4_1.json")
    config = ReadJsonIOHandler(config_path).get_input()

    total_processing_time_job_0 = config.get_job_size(0)
    windows0 = config.get_time_windows()[0]

    print("split_min:", config.get_split_min())

    print("Machine 0 windows:", windows0)
    print("Job size:", total_processing_time_job_0)
    print("Min timespan:", solve_min_timespan_cfg(total_processing_time_job_0, windows0, config.get_split_min()))
    print("Feasible leftover rule:", solve_feasible_leftover_rule_cfg(total_processing_time_job_0, windows0, config.get_split_min()))
    print("Timespan:", assign_job_to_machine(total_processing_time_job_0, windows0, config.get_split_min()))    

    # total_processing_time_job_1 = config.get_job_size(1)
    # windows1 = config.get_time_windows()["1"]
    # print("Machine 1 windows:", windows1)
    # print("Job size:", total_processing_time_job_1)
    # print("Min timespan:", solve_min_timespan_cfg(total_processing_time_job_1, windows1, config.get_split_min()))
    # print("Feasible leftover rule:", solve_feasible_leftover_rule_cfg(total_processing_time_job_1, windows1, config.get_split_min()))
