"""
Simulated Annealing for Splittable Job Scheduling (DP per job).

- Tối ưu: thứ tự phục vụ job (FCFS theo thứ tự này).
- Decode lịch: với mỗi job, chạy DP trên TẤT CẢ máy, chọn best-fit theo tuple
    (finish, timespan, fragments, machine_id), commit và trừ cửa sổ máy.
- Cost: makespan thực = max end over all segments của mọi job.
- In kết quả: finish đúng (= max end của job), horizon = makespan thực, Gantt theo máy.

Chạy:
    python3 -m staticAlgos.SA --config ./datasets/splittable_jobs.json --mode timespan
    # Tuỳ chọn:
    --Tmax 500 --Tthreshold 1 --alpha 0.99 --moves_per_T 50 --seed 42 --verbose --out path.json
"""

import os
import sys
import math
import random
import argparse
import json
from typing import Any, Dict, List, Tuple, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from copy import deepcopy
from base.model import SchedulingModel
from base.io_handler import ReadJsonIOHandler
from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
    assign_job_to_machine
)

# -----------------------------------------------------------------------------

def _apply_plan_to_windows(windows_of_machine: List[List[int]],
                        plan: List[Tuple[int, int, int]]) -> List[List[int]]:
    """Trừ các đoạn trong plan khỏi list windows [[s,e], ...]"""
    cuts = [(start_time, end_time) for (window_id, start_time, end_time) in plan]
    cuts.sort()

    new_windows = []
    for window_start, window_end in windows_of_machine:
        segment = [(window_start, window_end)]
        for (c_start, c_end) in cuts:
            nxt = []
            for start, end in segment:
                if c_end <= start or c_start >= end:
                    nxt.append((start, end))
                else:
                    if start < c_start:
                        nxt.append((start, c_start))
                    if c_end < end:
                        nxt.append((c_end, end))
            segment = nxt
        for start, end in segment:
            if end > start:
                new_windows.append([start, end])

    new_windows.sort()
    merged = []
    for start, end in new_windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

def _evaluate_job_on_machine(
    mode: str,
    processing_time: int,
    windows_of_machine: List[List[int]],
    split_min: int,
) -> Tuple[Optional[int], Optional[List[Tuple[int,int,int]]], Optional[int], Optional[int]]:
    """
    Trả về: (finish, plan, timespan, fragments)
    - finish = max end của plan (None nếu không khả thi)
    - timespan: obj từ DP (timespan tối thiểu ở mode='timespan'), dùng để tie-break
    - fragments: số mảnh
    """
    if mode == "timespan":
        obj, plan = solve_min_timespan_cfg(processing_time, windows_of_machine, split_min)
    elif mode == "leftover":
        obj, plan = solve_feasible_leftover_rule_cfg(processing_time, windows_of_machine, split_min)
    elif mode == "assign":
        obj, plan = assign_job_to_machine(processing_time, windows_of_machine, split_min)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if obj is None or not plan:
        return None, None, None, None

    finish = max(seg[2] for seg in plan)
    timespan = obj if mode == "timespan" else processing_time
    fragments = len(plan)
    return finish, plan, timespan, fragments

def _decode_order_best_fit(
    cfg: SchedulingModel,
    mode: str,
    order: List[int],
) -> Dict[str, Any]:
    """
    Từ một thứ tự job (order), dựng lịch bằng cách:
        - Lặp job theo order,
        - Trên MỌI máy: chạy DP, chọn best theo (finish, timespan, fragments, machine_id),
        - Commit, trừ cửa sổ máy,
        - Lưu assignment.
    Trả về dict: { 'assignments': [...], 'final_windows': ..., 'makespan': int }
    """
    processing_times = cfg.get_processing_times()
    split_min = cfg.get_split_min()
    time_windows = cfg.get_time_windows()
    time_windows_copy = deepcopy(time_windows)  # deep copy
    number_of_machine = cfg.get_num_machines()

    assignments = []  # per job in given order (nhưng vẫn lưu tên job)
    for job in order:
        processing_time_job = processing_times[job]
        best = None
        for machine in range(number_of_machine):
            finish, plan, timespan, fragments = _evaluate_job_on_machine(
                mode, processing_time_job, time_windows_copy[machine], split_min
            )
            if finish is None:
                continue
            key = (finish, timespan if timespan is not None else 10**15,
                    fragments if fragments is not None else 10**9, machine)
            if best is None or key < best[0]:
                best = (key, machine, plan, finish, timespan, fragments)

        if best is None:
            # Không gán được (giữ thông tin để debug)
            assignments.append({
                "job": job,
                "machine": None,
                "segments": [],
                "timespan": None,
                "finish": None,
                "fragments": 0,
            })
            continue

        _key, machine_best, plan_best, finish, timespan, fragments = best
        time_windows_copy[machine_best] = _apply_plan_to_windows(time_windows_copy[machine_best], plan_best)
        assignments.append({
            "job": job,
            "machine": machine_best,
            "segments": plan_best,   # (win_idx, start, end)
            "timespan": timespan,
            "finish": finish,
            "fragments": fragments,
        })

    makespan = compute_makespan(assignments)
    return {
        "assignments": assignments,
        "final_windows": time_windows_copy,
        "makespan": makespan,
    }

# =============================================================================
# Correct metrics & printing
# =============================================================================

def compute_job_finish(segments: List[Tuple[int,int,int]]) -> Optional[int]:
    return max(seg[2] for seg in segments) if segments else None

def compute_makespan(assignments: List[Dict[str, Any]]) -> int:
    ends = []
    for a in assignments:
        for (_w, s, e) in a.get("segments", []):
            ends.append(e)
    return max(ends) if ends else 0

def print_schedule(sa_result: Dict[str, Any], title: str = "SA RESULT") -> None:
    ass = sa_result["assignments"]
    mk = sa_result["makespan"]

    print(f"\n=== {title} ===")
    print(f"Best makespan: {mk}")

    print("\nBest schedule (job -> machine, finish_time, allocation):")
    for a in sorted(ass, key=lambda x: x["job"]):
        j = a["job"]; m = a["machine"]; segs = a["segments"]
        fin = compute_job_finish(segs)
        print(f"  Job {j}: Machine {m}, finish={fin}, segments={segs}")

    # Gantt ASCII theo máy (in danh sách các khoảng, tránh vẽ mk ký tự)
    print("\nGantt chart (ASCII):")
    by_m = {}
    for a in ass:
        if a["machine"] is None:
            continue
        by_m.setdefault(a["machine"], []).extend([(a["job"], s, e) for (_w, s, e) in a["segments"]])

    for m in sorted(by_m.keys()):
        intervals = sorted(by_m[m], key=lambda x: (x[1], x[2]))
        labels = " ".join([f"[J{j}:{s}-{e}]" for (j, s, e) in intervals])
        print(f"M{m:02d} | {labels}")
    print(f"\n(horizon = {mk})")

# =============================================================================
# Simulated Annealing
# =============================================================================

def _neighbor_swap(order: List[int]) -> List[int]:
    """Đổi chỗ 2 job ngẫu nhiên."""
    n = len(order)
    if n < 2:
        return order[:]
    i, j = random.sample(range(n), 2)
    if i > j:
        i, j = j, i
    new_order = order[:]
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order

def schedule_sa_config(
    cfg: SchedulingModel,
    mode: str = "assign",
    Tmax: float = 500.0,
    Tthreshold: float = 1.0,
    alpha: float = 0.99,
    moves_per_T: int = 50,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Chạy SA trên không gian thứ tự job.
    - State: order (list job id)
    - Decode: _decode_order_best_fit(cfg, mode, order)
    - Cost: makespan của lịch decode
    """
    if seed is not None:
        random.seed(seed)

    number_of_jobs = cfg.get_num_jobs()

    cur_order = list(range(number_of_jobs))
    cur_decoded = _decode_order_best_fit(cfg, mode, cur_order)
    cur_cost = float(cur_decoded["makespan"])

    best_order = cur_order[:]
    best_decoded = cur_decoded
    best_cost = cur_cost

    T = float(Tmax)
    while T > Tthreshold:
        for _ in range(moves_per_T):
            cand_order = _neighbor_swap(cur_order)
            cand_decoded = _decode_order_best_fit(cfg, mode, cand_order)
            cand_cost = float(cand_decoded["makespan"])
            delta = cand_cost - cur_cost

            if delta <= 0:
                accept = True
            else:
                # Metropolis
                prob = math.exp(-delta / max(T, 1e-9))
                accept = (random.random() < prob)

            if accept:
                cur_order = cand_order
                cur_decoded = cand_decoded
                cur_cost = cand_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_order = cur_order[:]
                    best_decoded = cur_decoded
                    if verbose:
                        print(f"[SA] improved: cost={best_cost:.3f} at T={T:.3f}")

        T *= alpha
        if verbose:
            print(f"[SA] cool down: T={T:.4f}, best={best_cost:.3f}")
    return {
        "best_makespan": int(best_cost),
        "best_order": best_order,
        "assignments": best_decoded["assignments"],
        "final_windows": best_decoded["final_windows"],
    }

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simulated Annealing for Splittable Job Scheduling (DP per job).")
    parser.add_argument("--config", type=str, default="./datasets/splittable_jobs.json", help="Path to JSON config.")
    parser.add_argument("--mode", type=str, default="timespan", choices=["timespan", "leftover"], help="DP mode.")
    parser.add_argument("--Tmax", type=float, default=500.0)
    parser.add_argument("--Tthreshold", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--moves_per_T", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=str, default="", help="Optional path to write JSON result.")
    args = parser.parse_args()

    cfg = ReadJsonIOHandler(args.config).get_input()
    result = schedule_sa_config(
        cfg,
        mode=args.mode,
        Tmax=args.Tmax,
        Tthreshold=args.Tthreshold,
        alpha=args.alpha,
        moves_per_T=args.moves_per_T,
        seed=args.seed,
        verbose=args.verbose,
    )
    printable = {
        "assignments": result["assignments"],
        "makespan": compute_makespan(result["assignments"]),
    }
    print("\n=== SA RESULT ===")
    print(f"Mode: {args.mode}")
    print(f"Best makespan: {printable['makespan']}")
    print(f"Best order: {result['best_order']}\n")
    print("Best schedule (job -> machine, finish_time, allocation):")
    for a in sorted(result["assignments"], key=lambda x: x["job"]):
        j = a["job"]; m = a["machine"]; segs = a["segments"]
        fin = compute_job_finish(segs)
        print(f"  Job {j}: Machine {m}, finish={fin}, segments={segs}")

    # Gantt
    print("\nGantt chart (ASCII):")
    mk = printable["makespan"]
    by_m = {}
    for a in result["assignments"]:
        if a["machine"] is None:
            continue
        by_m.setdefault(a["machine"], []).extend([(a["job"], s, e) for (_w, s, e) in a["segments"]])
    for m in sorted(by_m.keys()):
        intervals = sorted(by_m[m], key=lambda x: (x[1], x[2]))
        labels = " ".join([f"[J{j}:{s}-{e}]" for (j, s, e) in intervals])
        print(f"M{m:02d} | {labels}")
    print(f"\n(horizon = {mk})")

    if args.out:
        payload = {
            "mode": args.mode,
            "best_order": result["best_order"],
            "makespan": printable["makespan"],
            "assignments": result["assignments"],
            "final_windows": result["final_windows"],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved result to: {args.out}")

if __name__ == "__main__":
    main()
