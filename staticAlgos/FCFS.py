"""
FCFS scheduler (reworked):
- Nhận danh sách các job, sắp xếp theo arrival_time tăng dần, nếu bằng nhau (hoặc không có) thì theo job_id tăng dần.
- Gán job -> machine theo luật FCFS đơn giản (máy có cửa sổ khả dụng sớm nhất; nếu hoà thì machine_id thấp hơn trước).
- Sau khi đã CHỌN máy, mới gọi solver trong dp_single_job (theo mode 'timespan' hoặc 'leftover') để chia/schedule các mảnh vào windows của máy đó.
- In kết quả và tổng timespan cuối cùng.

Chạy:
  python3 staticAlgos/FCFS.py --mode timespan
  python3 staticAlgos/FCFS.py --mode leftover
  # tuỳ chọn:
  python3 staticAlgos/FCFS.py --config ./configs/splittable_jobs.json
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

# Import các solver & tiện ích từ module DP single-job
from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
    get_windows_from_config,
)

# Thêm project root vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------
# Helpers
# -----------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_jobs_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chuẩn hóa danh sách job từ config.
    - Bắt buộc: model.processing_times = [p0, p1, ...]
    - Tuỳ chọn: model.arrival_times = [t0, t1, ...] (nếu thiếu thì mặc định 0)
    Trả về: list[{job_id, ptime, arrival}]
    """
    model = config["model"]
    pts = model["processing_times"]
    arrivals = model.get("arrival_times", [0] * len(pts))
    if len(arrivals) != len(pts):
        # Nếu mảng arrival thiếu, pad về 0 cho an toàn
        arrivals = (arrivals + [0] * len(pts))[:len(pts)]

    jobs = []
    for j, p in enumerate(pts):
        jobs.append({"job_id": j, "ptime": int(p), "arrival": int(arrivals[j])})
    return jobs

def sort_jobs_fcfs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    FCFS: sort theo arrival_time tăng dần rồi job_id tăng dần
    """
    return sorted(jobs, key=lambda x: (x["arrival"], x["job_id"]))

def clone_windows_map(config: Dict[str, Any]) -> Dict[str, List[List[int]]]:
    """
    Lấy bản sao windows theo từng máy, dạng {machine_id(str): [[start,end], ...]}.
    """
    machine_ids = sorted(config["model"]["time_windows"].keys(), key=lambda s: int(s))
    return {mid: get_windows_from_config(config, mid) for mid in machine_ids}

def earliest_available_time_of_machine(windows: List[List[int]]) -> int:
    """
    Thời điểm khả dụng sớm nhất của một máy (đầu mối window đầu tiên).
    - Nếu không còn window nào, trả về +inf (không thể nhận thêm).
    """
    if not windows:
        return 10**15
    # windows đã là các [start, end]; lấy start nhỏ nhất
    return min(w[0] for w in windows)

def pick_machine_fcfs(windows_map: Dict[str, List[List[int]]]) -> Optional[str]:
    """
    Chọn máy có thời điểm khả dụng sớm nhất.
    Tie-break: machine_id (int) nhỏ hơn trước.
    Trả về machine_id (str) hoặc None nếu tất cả đã hết cửa sổ.
    """
    best_mid = None
    best_time = 10**15
    for mid, wins in windows_map.items():
        t0 = earliest_available_time_of_machine(wins)
        if t0 < best_time or (t0 == best_time and (best_mid is None or int(mid) < int(best_mid))):
            best_mid = mid
            best_time = t0
    if best_time >= 10**15:
        return None
    return best_mid

def apply_allocation_to_windows(
    windows_list: List[List[int]],
    plan: List[Tuple[int, int, int]]
) -> List[List[int]]:
    """
    Cập nhật lại windows sau khi gán các đoạn (start->end) trong 'plan'.
    Giả định mỗi mảnh bắt đầu đúng tại 'start' của một window (phù hợp solver hiện tại).
    """
    windows = sorted([[a, b] for a, b in windows_list], key=lambda ab: (ab[0], ab[1]))
    for _, start, end in plan:
        found = False
        for i, (a, b) in enumerate(windows):
            if a == start and b >= end:
                new_start = end
                if new_start < b:
                    windows[i][0] = new_start  # thu hẹp từ trái
                else:
                    windows.pop(i)             # dùng hết window
                found = True
                break
        if not found:
            raise RuntimeError(
                f"Không thể áp allocation: không có window bắt đầu tại {start} để trừ tới {end}."
            )
    return windows

def assign_on_machine_by_mode(
    mode: str,
    total_processing_time: int,
    machine_windows: List[List[int]],
    split_min: int
) -> Tuple[bool, Optional[List[Tuple[int,int,int]]], Optional[int], Optional[int]]:
    """
    Gọi solver single-job tương ứng với 'mode' TRÊN MỘT MÁY đã chọn.
    Trả về: (feasible, plan, timespan, finish_time)
    - Với mode = 'timespan': timespan != None, finish_time được tính từ plan
    - Với mode = 'leftover': timespan = None; finish_time lấy max end trong plan
    """
    if mode == "timespan":
        timespan, plan = solve_min_timespan_cfg(total_processing_time, machine_windows, split_min)
        if timespan is None or plan is None:
            return False, None, None, None
        start_first = plan[0][1] if plan else 0
        finish = start_first + timespan
        return True, plan, timespan, finish

    elif mode == "leftover":
        feasible, plan = solve_feasible_leftover_rule_cfg(total_processing_time, machine_windows, split_min)
        if not feasible or plan is None:
            return False, None, None, None
        finish = max(p[2] for p in plan) if plan else 0
        return True, plan, None, finish

    else:
        raise ValueError("mode phải là 'timespan' hoặc 'leftover'")

def compute_overall_timespan(assignments: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Tính: (overall_start, overall_finish, overall_timespan)
    - overall_start: min start của tất cả plan
    - overall_finish: max end của tất cả plan
    - overall_timespan: overall_finish - overall_start (0 nếu không có mảnh nào)
    """
    starts, ends = [], []
    for a in assignments:
        plan = a.get("plan") or []
        for _, s, e in plan:
            starts.append(s)
            ends.append(e)
    if not starts or not ends:
        return 0, 0, 0
    overall_start = min(starts)
    overall_finish = max(ends)
    return overall_start, overall_finish, (overall_finish - overall_start)

# -----------------------
# FCFS chính
# -----------------------

def schedule_fcfs(config: Dict[str, Any], mode: str = "timespan") -> Dict[str, Any]:
    """
    Quy trình:
    1) Chuẩn hóa & sort job theo FCFS (arrival tăng dần, job_id tăng dần, ưu tiên thấp trước).
    2) Khởi tạo windows_map cho từng máy.
    3) Với mỗi job theo thứ tự FCFS:
       - Chọn MỘT máy theo luật FCFS đơn giản (máy có thời điểm khả dụng sớm nhất; tie-break machine_id).
       - Gọi solver dp_single_job tương ứng 'mode' TRÊN máy đã chọn để split và assign job.
       - Nếu feasible → cập nhật windows của MÁY đó, lưu assignment.
         Nếu không feasible (máy hết chỗ/không nhét được) → đánh dấu unassigned (machine=None).
    4) Tính tổng timespan cuối: overall_timespan (trên tất cả mảnh đã gán).
    """
    model = config["model"]
    split_min = int(model["split_min"])
    jobs = sort_jobs_fcfs(normalize_jobs_from_config(config))
    windows_map = clone_windows_map(config)

    assignments: List[Dict[str, Any]] = []
    machine_ids_sorted = sorted(windows_map.keys(), key=lambda s: int(s))  # để in thôi

    for job in jobs:
        job_id = job["job_id"]
        ptime = job["ptime"]

        # Chọn máy theo FCFS (không dùng DP để so sánh máy)
        chosen_mid = pick_machine_fcfs(windows_map)

        if chosen_mid is None:
            # Không còn máy nào có cửa sổ → fail
            assignments.append({
                "job_id": job_id,
                "arrival": job["arrival"],
                "machine_id": None,
                "plan": None,
                "timespan": None,
                "finish": None,
                "num_fragments": 0,
                "note": "No available windows on any machine"
            })
            continue

        # Sau khi CHỌN máy, mới gọi solver đơn-job để gán
        feasible, plan, timespan, finish = assign_on_machine_by_mode(
            mode=mode,
            total_processing_time=ptime,
            machine_windows=windows_map[chosen_mid],
            split_min=split_min
        )

        if not feasible:
            # Máy đã chọn không nhét được job này (theo rule đã định) → đánh dấu fail
            assignments.append({
                "job_id": job_id,
                "arrival": job["arrival"],
                "machine_id": chosen_mid,
                "plan": None,
                "timespan": None,
                "finish": None,
                "num_fragments": 0,
                "note": "Chosen machine cannot fit this job under current windows"
            })
            continue

        # Cập nhật windows của máy đã chọn theo plan trả về
        windows_map[chosen_mid] = apply_allocation_to_windows(windows_map[chosen_mid], plan)

        assignments.append({
            "job_id": job_id,
            "arrival": job["arrival"],
            "machine_id": chosen_mid,
            "plan": plan,
            "timespan": timespan,
            "finish": finish,
            "num_fragments": len(plan),
        })

    overall_start, overall_finish, overall_timespan = compute_overall_timespan(assignments)

    return {
        "mode": mode,
        "assignments": assignments,
        "final_windows": {mid: windows_map[mid] for mid in machine_ids_sorted},
        "overall_start": overall_start,
        "overall_finish": overall_finish,
        "overall_timespan": overall_timespan,
    }

# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="FCFS scheduler (assign machine first, then call single-job DP solver)")
    parser.add_argument("--mode", choices=["timespan", "leftover"], default="timespan",
                        help="Chọn solver: 'timespan' (min timespan) hoặc 'leftover' (feasible + leftover rule).")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "splittable_jobs.json"),
                        help="Đường dẫn file JSON config (mặc định: ../configs/splittable_jobs.json).")
    args = parser.parse_args()

    # Chuẩn hoá đường dẫn
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")

    config = load_config(config_path)
    result = schedule_fcfs(config, mode=args.mode)

    print(f"Mode: {result['mode']}")
    print(f"Overall timespan: {result['overall_timespan']} (start={result['overall_start']}, finish={result['overall_finish']})")
    print("Assignments:")
    for a in result["assignments"]:
        print(f"  Job {a['job_id']} (arrival={a['arrival']}) -> Machine {a['machine_id']}")
        if a["machine_id"] is not None:
            print(f"    plan: {a['plan']}")
            print(f"    timespan: {a['timespan']}, finish: {a['finish']}, fragments: {a['num_fragments']}")
        else:
            print(f"    plan: None (reason: {a.get('note','unassigned')})")

    print("Final windows per machine:")
    for mid in sorted(result["final_windows"].keys(), key=lambda s: int(s)):
        print(f"  Machine {mid}: {result['final_windows'][mid]}")

if __name__ == "__main__":
    main()
