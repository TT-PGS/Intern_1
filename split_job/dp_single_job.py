import json
import os
from typing import List, Tuple, Optional, Dict, Any
from typing import List, Tuple, Optional

def get_windows_from_config(config: Dict[str, Any], machine_id: str) -> List[List[int]]:
    """Lấy danh sách time windows [[start,end],...] của 1 máy từ config."""
    return config["model"]["time_windows"].get(machine_id, [])

def get_all_windows_from_config(config: Dict[str, Any]) -> List[List[List[int]]]:
    """Lấy danh sách time windows [[start,end],...] của tất cả các máy từ config."""
    return [get_windows_from_config(config, str(m)) for m in range(config["model"]["num_machines"])]

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
        return (total_processing_time == 0, []) if total_processing_time == 0 else (False, None)

    capacities = [b - a for a, b in windows_sorted]
    starts = [a for a, _ in windows_sorted]

    if total_processing_time <= 0 or split_min <= 0 or sum(capacities) < total_processing_time:
        return False, None

    # Các lựa chọn k cho từng window
    choices: List[List[int]] = []
    for cap in capacities:
        opts: List[int] = [0]
        if cap >= split_min:
            opts.append(cap)  # full
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

    # DP kiểm tra khả thi
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
                    break  # ưu tiên k đầu tiên trong choices

    if not reach[0][T]:
        return False, None

    # Truy vết để dựng plan
    alloc: List[Tuple[int, int, int]] = []
    win_idx, remain = 0, T
    while win_idx < num_windows and remain > 0:
        k, rem = parent[win_idx][remain]
        if k > 0:
            start_i = starts[win_idx]
            alloc.append((win_idx, start_i, start_i + k))
        remain = rem
        win_idx += 1

    return True, alloc


if __name__ == "__main__":
    # Xác định đường dẫn file JSON config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "splittable_jobs.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    split_min = config["model"]["split_min"]

    total_processing_time_job_0 = config["model"]["processing_times"][0]
    windows0 = get_windows_from_config(config, "0")

    print("Machine 0 windows:", windows0)
    print("Min timespan:", solve_min_timespan_cfg(total_processing_time_job_0, windows0, split_min))
    print("Feasible leftover rule:", solve_feasible_leftover_rule_cfg(total_processing_time_job_0, windows0, split_min))

    total_processing_time_job_1 = config["model"]["processing_times"][1]
    windows1 = get_windows_from_config(config, "1")
    print("Machine 1 windows:", windows1)
    print("Min timespan:", solve_min_timespan_cfg(total_processing_time_job_1, windows1, split_min))
    print("Feasible leftover rule:", solve_feasible_leftover_rule_cfg(total_processing_time_job_1, windows1, split_min))
