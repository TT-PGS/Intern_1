"""
FCFS scheduler: schedule các job theo thứ tự đến; nếu cùng arrival, job_id nhỏ trước.
Việc gán job vào machine được thực hiện bởi dp_single_job.py (Dynamic Programming).

Chạy:
  python3 -m staticAlgos.FCFS --mode timespan --config ./configs/splittable_jobs.json
  python3 -m staticAlgos.FCFS --mode leftover --config ./configs/splittable_jobs.json
"""

import os, sys, json, argparse
from typing import Any, Dict, List, Tuple, Optional

# Thêm project root vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Gọi lại các hàm đã có
from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
    get_all_windows_from_config,
)

# ----------------------- util -----------------------

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _ensure_config(x: Dict[str, Any]) -> Dict[str, Any]:
    return x if "model" in x else {"model": x}

def _jobs_processing_times(cfg: Dict[str, Any]) -> List[int]:
    return cfg["model"]["processing_times"]

def _split_min(cfg: Dict[str, Any]) -> int:
    return cfg["model"]["split_min"]

def _arrival_order(num_jobs: int) -> List[int]:
    # FCFS: nếu tất cả arrival=0 thì về mặc định là [0..n-1]
    return list(range(num_jobs))

def _apply_plan_to_windows(windows: List[List[int]], plan: List[Tuple[int,int,int]]) -> List[List[int]]:
    """
    windows: list các [start, end]
    plan: list các (win_idx, seg_start, seg_end) với thời gian tuyệt đối (cùng hệ quy chiếu với windows)

    Trả về windows mới sau khi trừ đi các đoạn [seg_start, seg_end].
    """
    # Đưa về dạng các đoạn cấm (để trừ)
    cuts = [(seg_start, seg_end) for (_w, seg_start, seg_end) in plan]
    cuts.sort()

    new_windows = []
    for (w_start, w_end) in windows:
        segs = [(w_start, w_end)]
        for (c_start, c_end) in cuts:
            next_segs = []
            for (s, e) in segs:
                if c_end <= s or c_start >= e:
                    # không giao
                    next_segs.append((s, e))
                else:
                    # có giao, cắt
                    if s < c_start:
                        next_segs.append((s, c_start))
                    if c_end < e:
                        next_segs.append((c_end, e))
            segs = next_segs
        # lọc các đoạn rỗng/âm
        for (s, e) in segs:
            if e > s:
                new_windows.append([s, e])
    # Gộp/chuẩn hoá (optional)
    new_windows.sort()
    merged = []
    for s, e in new_windows:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged

def _evaluate_assignment(mode: str,
                         ptime: int,
                         windows: List[List[int]],
                         split_min: int) -> Tuple[Optional[int], Optional[List[Tuple[int,int,int]]], Optional[int], Optional[int]]:
    """
    Trả về: (finish, plan, timespan, fragments)
    - finish: thời điểm kết thúc (max end trong plan)
    - timespan: tổng thời gian xử lý (== ptime) hoặc min_timespan tùy hàm DP trả về
    - fragments: số mảnh
    Lưu ý: solve_* trả về (objective, plan). Với 'timespan' ta dùng solve_min_timespan_cfg.
           Với 'leftover' ta gọi solve_feasible_leftover_rule_cfg (đồng nhất cách xử lý).
    """
    if mode == "timespan":
        obj, plan = solve_min_timespan_cfg(ptime, windows, split_min)
        if obj is None or not plan:
            return None, None, None, None
        finish = max(seg[2] for seg in plan)  # seg = (win_idx, start, end)
        timespan = obj
        fragments = len(plan)
        return finish, plan, timespan, fragments
    elif mode == "leftover":
        obj, plan = solve_feasible_leftover_rule_cfg(ptime, windows, split_min)
        if obj is None or not plan:
            return None, None, None, None
        finish = max(seg[2] for seg in plan)
        # Với leftover mode, obj có thể là bool/0-1 hoặc thời lượng còn dư; ta vẫn đánh giá theo finish trước.
        timespan = ptime  # placeholder (hoặc đặt = sum(end-start) theo plan)
        fragments = len(plan)
        return finish, plan, timespan, fragments
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ----------------------- core FCFS -----------------------

def schedule_fcfs(config: Dict[str, Any], mode: str = "timespan") -> Dict[str, Any]:
    cfg = _ensure_config(config)
    p_times = _jobs_processing_times(cfg)
    n_jobs = len(p_times)
    split_min = _split_min(cfg)

    # Khởi tạo windows per machine từ config (dạng [[start,end], ...])
    machine_windows = get_all_windows_from_config(cfg)  # List[List[List[int]]], mỗi máy là list các [start,end]
    n_machines = len(machine_windows)

    # FCFS theo order đến
    order = _arrival_order(n_jobs)

    assignments = []  # list dict per job
    for j in order:
        p = p_times[j]

        # Dò tất cả máy → lấy best theo (finish, timespan, fragments, machine_id)
        best = None  # (key_tuple, m_best, plan_best)
        for m in range(n_machines):
            finish, plan, timespan, fragments = _evaluate_assignment(mode, p, machine_windows[m], split_min)
            if finish is None:
                continue
            key = (finish, timespan if timespan is not None else 10**15, fragments if fragments is not None else 10**9, m)
            if best is None or key < best[0]:
                best = (key, m, plan, finish, timespan, fragments)

        if best is None:
            # Không gán được job j
            assignments.append({
                "job": j,
                "machine": None,
                "plan": [],
                "timespan": None,
                "finish": None,
                "fragments": 0,
            })
            continue

        _key, m_best, plan_best, finish, timespan, fragments = best

        # Commit vào machine m_best: trừ các đoạn đã dùng khỏi windows
        machine_windows[m_best] = _apply_plan_to_windows(machine_windows[m_best], plan_best)

        assignments.append({
            "job": j,
            "machine": m_best,
            "plan": plan_best,
            "timespan": timespan,
            "finish": finish,
            "fragments": fragments,
        })

    # Makespan = max finish over all jobs (có thể None)
    finishes = [a["finish"] for a in assignments if a["finish"] is not None]
    if finishes:
        start_time = 0  # theo dataset, bắt đầu từ 0
        makespan = max(finishes) - start_time
        overall_start = start_time
        overall_finish = max(finishes)
    else:
        makespan, overall_start, overall_finish = None, None, None

    return {
        "mode": mode,
        "makespan": makespan,
        "overall": {"start": overall_start, "finish": overall_finish},
        "assignments": assignments,
        "final_windows": machine_windows,
    }

# ----------------------- CLI -----------------------

def _print_result(res: Dict[str, Any]):
    mode = res["mode"]
    mk = res["makespan"]
    s = res["overall"]["start"]
    f = res["overall"]["finish"]
    print(f"Mode: {mode}")
    print(f"Overall timespan: {mk} (start={s}, finish={f})")
    print("Assignments:")
    # In theo job order đã xử lý
    for a in res["assignments"]:
        j = a["job"]
        m = a["machine"]
        plan = a["plan"]
        ts = a["timespan"]
        fin = a["finish"]
        fr = a["fragments"]
        if m is None:
            print(f"  Job {j}: UNASSIGNED")
        else:
            print(f"  Job {j} (arrival=0) -> Machine {m}")
            print(f"    plan: {plan}")
            print(f"    timespan: {ts}, finish: {fin}, fragments: {fr}")

    print("Final windows per machine:")
    for mi, wins in enumerate(res["final_windows"]):
        compact = [[s, e] for (s, e) in wins]
        print(f"  Machine {mi}: {compact}")

def main():
    parser = argparse.ArgumentParser(description="FCFS with best-fit across machines (DP per job).")
    parser.add_argument("--config", type=str, default="./configs/splittable_jobs.json")
    parser.add_argument("--mode", type=str, default="timespan", choices=["timespan", "leftover"])
    args = parser.parse_args()

    cfg = _load_config(args.config)
    res = schedule_fcfs(cfg, mode=args.mode)
    _print_result(res)

if __name__ == "__main__":
    main()
