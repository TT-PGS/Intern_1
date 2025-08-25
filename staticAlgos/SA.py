# staticAlgos/SA.py
"""
Simulated Annealing for Splittable Job Scheduling (DP per job).

- Tối ưu: thứ tự phục vụ job (FCFS theo thứ tự này).
- Decode lịch: với mỗi job, chạy DP trên TẤT CẢ máy, chọn best-fit theo tuple
  (finish, timespan, fragments, machine_id), commit và trừ cửa sổ máy.
- Cost: makespan thực = max end over all segments của mọi job.
- In kết quả: finish đúng (= max end của job), horizon = makespan thực, Gantt theo máy.

Chạy:
  python3 -m staticAlgos.SA --config ./configs/splittable_jobs.json --mode timespan
  # Tuỳ chọn:
  --Tmax 500 --Tthreshold 1 --alpha 0.99 --moves_per_T 50 --seed 42 --verbose --out path.json
"""

import os
import sys
import json
import math
import random
import argparse
from typing import Any, Dict, List, Tuple, Optional

# ---- sys.path: add project root ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---- DP funcs ----
from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
    get_all_windows_from_config
)

# =============================================================================
# Utilities
# =============================================================================

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _ensure_config(x: Dict[str, Any]) -> Dict[str, Any]:
    return x if "model" in x else {"model": x}

def _processing_times(cfg: Dict[str, Any]) -> List[int]:
    return cfg["model"]["processing_times"]

def _split_min(cfg: Dict[str, Any]) -> int:
    return cfg["model"]["split_min"]

# -----------------------------------------------------------------------------

def _clone_windows(wins: List[List[List[int]]]) -> List[List[List[int]]]:
    # deep-ish copy [[ [s,e], ... ] for each machine]
    return [ [ [a, b] for (a, b) in mach ] for mach in wins ]

def _apply_plan_to_windows(windows: List[List[int]],
                           plan: List[Tuple[int, int, int]]) -> List[List[int]]:
    """Trừ các đoạn trong plan khỏi list windows [[s,e], ...] (cùng hệ quy chiếu thời gian)."""
    cuts = [(s, e) for (_w, s, e) in plan]
    cuts.sort()

    new_windows = []
    for (w_start, w_end) in windows:
        segs = [(w_start, w_end)]
        for (c_start, c_end) in cuts:
            nxt = []
            for (s, e) in segs:
                if c_end <= s or c_start >= e:
                    nxt.append((s, e))
                else:
                    if s < c_start:
                        nxt.append((s, c_start))
                    if c_end < e:
                        nxt.append((c_end, e))
            segs = nxt
        for (s, e) in segs:
            if e > s:
                new_windows.append([s, e])

    new_windows.sort()
    merged = []
    for s, e in new_windows:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged

def _evaluate_job_on_machine(
    mode: str,
    ptime: int,
    windows: List[List[int]],
    split_min: int,
) -> Tuple[Optional[int], Optional[List[Tuple[int,int,int]]], Optional[int], Optional[int]]:
    """
    Trả về: (finish, plan, timespan, fragments)
    - finish = max end của plan (None nếu không khả thi)
    - timespan: obj từ DP (timespan tối thiểu ở mode='timespan'), dùng để tie-break
    - fragments: số mảnh
    """
    if mode == "timespan":
        obj, plan = solve_min_timespan_cfg(ptime, windows, split_min)
    elif mode == "leftover":
        obj, plan = solve_feasible_leftover_rule_cfg(ptime, windows, split_min)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if obj is None or not plan:
        return None, None, None, None

    finish = max(seg[2] for seg in plan)
    timespan = obj if mode == "timespan" else ptime
    fragments = len(plan)
    return finish, plan, timespan, fragments

def _decode_order_best_fit(
    cfg: Dict[str, Any],
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
    p_times = _processing_times(cfg)
    split_min = _split_min(cfg)
    mach_wins0 = get_all_windows_from_config(cfg)  # List[machine][ [s,e], ... ]
    mach_wins = _clone_windows(mach_wins0)
    n_machines = len(mach_wins)

    assignments = []  # per job in given order (nhưng vẫn lưu tên job)
    for j in order:
        p = p_times[j]
        best = None
        for m in range(n_machines):
            finish, plan, timespan, fragments = _evaluate_job_on_machine(
                mode, p, mach_wins[m], split_min
            )
            if finish is None:
                continue
            key = (finish, timespan if timespan is not None else 10**15,
                   fragments if fragments is not None else 10**9, m)
            if best is None or key < best[0]:
                best = (key, m, plan, finish, timespan, fragments)

        if best is None:
            # Không gán được (giữ thông tin để debug)
            assignments.append({
                "job": j,
                "machine": None,
                "segments": [],
                "timespan": None,
                "finish": None,
                "fragments": 0,
            })
            continue

        _key, m_best, plan_best, finish, timespan, fragments = best
        mach_wins[m_best] = _apply_plan_to_windows(mach_wins[m_best], plan_best)
        assignments.append({
            "job": j,
            "machine": m_best,
            "segments": plan_best,   # (win_idx, start, end)
            "timespan": timespan,
            "finish": finish,
            "fragments": fragments,
        })

    makespan = compute_makespan(assignments)
    return {
        "assignments": assignments,
        "final_windows": mach_wins,
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
    cfg: Dict[str, Any],
    mode: str = "timespan",
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

    cfg = _ensure_config(cfg)
    p_times = _processing_times(cfg)
    n_jobs = len(p_times)
    # Khởi tạo: order mặc định 0..n-1
    cur_order = list(range(n_jobs))
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

    # Kết quả
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
    parser.add_argument("--config", type=str, default="./configs/splittable_jobs.json", help="Path to JSON config.")
    parser.add_argument("--mode", type=str, default="timespan", choices=["timespan", "leftover"], help="DP mode.")
    parser.add_argument("--Tmax", type=float, default=500.0)
    parser.add_argument("--Tthreshold", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--moves_per_T", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=str, default="", help="Optional path to write JSON result.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
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

    # In kết quả chuẩn
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
