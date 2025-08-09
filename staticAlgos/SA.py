# staticAlgos/SA.py
import os
import json
import math
import random
import argparse
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

from split_job.dp_single_job import (
    solve_min_timespan_cfg,              # (finish_time | None, allocation | None)
    solve_feasible_leftover_rule_cfg    # (ok: bool, allocation | None)
)

# ============================== Helpers ==============================

def _gantt_from_detail(detail: dict) -> tuple[dict[int, list[tuple[int,int,int]]], int]:
    """
    Trích xuất các đoạn (start,end) theo từng máy từ detail['allocations'].
    Trả về:
      - segments_by_machine: {machine_id: [(start, end, job_id), ...]}
      - horizon: max(last_finish_per_machine)
    """
    allocations = detail.get("allocations", {})
    segments_by_machine: dict[int, list[tuple[int,int,int]]] = {}

    for job_id, info in allocations.items():
        machine_id = info["machine"]
        for (window_index, segment_start, segment_end) in info["allocation"]:
            segments_by_machine.setdefault(machine_id, []).append(
                (segment_start, segment_end, job_id)
            )

    for m in segments_by_machine:
        segments_by_machine[m].sort(key=lambda seg: (seg[0], seg[1]))

    horizon = max(detail.get("last_finish", [0]))
    return segments_by_machine, horizon


def _render_gantt_ascii(
    segments_by_machine: dict[int, list[tuple[int,int,int]]],
    horizon: int,
    max_width: int = 120,
) -> str:
    """
    Vẽ Gantt chart ASCII. Nếu horizon quá lớn, tự scale để vừa max_width.
    Ký hiệu:
      - '.': idle (trống)
      - '0'..'9': job_id % 10 (nếu job lớn hơn 9 thì lặp lại chữ số)
    """
    if horizon <= 0:
        return "(empty schedule)"

    scale = max(1, math.ceil(horizon / max_width))
    width = horizon // scale + 1  # số ký tự cho mỗi dòng

    lines = []
    # từng máy một
    for machine_id in sorted(segments_by_machine.keys()):
        row = ['.'] * width
        for start, end, job_id in segments_by_machine[machine_id]:
            # tô kín [start, end)
            for t in range(start, end):
                idx = t // scale
                if 0 <= idx < width:
                    row[idx] = str(job_id % 10)
        lines.append(f"M{machine_id:02d} | " + ''.join(row) + " |")

    # thêm thước đo (scale)
    scale_note = f"(scale = {scale} time-unit/char, horizon = {horizon})"
    ruler = "     " + ''.join(['|' if (i % 10 == 0) else ' ' for i in range(width)])
    index_line = "     " + ''.join([str((i*scale) % 10) for i in range(width)])

    return "\n".join(lines + ["", scale_note, ruler, index_line])

def _load_config(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_config(config_like: Dict[str, Any]) -> Dict[str, Any]:
    return config_like if "model" in config_like else {"model": config_like}

def _get_num_jobs(config: Dict[str, Any]) -> int:
    model = config["model"]
    if "num_jobs" in model:
        return int(model["num_jobs"])
    if "processing_times" in model:
        return len(model["processing_times"])
    raise ValueError("Model must include 'num_jobs' or 'processing_times'.")

def _get_processing_times(config: Dict[str, Any]) -> List[int]:
    model = config["model"]
    if "processing_times" not in model:
        raise ValueError("config['model']['processing_times'] is required.")
    return list(model["processing_times"])

def _get_split_min(config: Dict[str, Any]) -> int:
    model = config["model"]
    split_min = model.get("split_min", None)
    if split_min is None:
        raise ValueError("config['model']['split_min'] is required.")
    return int(split_min)

def _apply_allocation_to_windows(
    windows: List[List[int]],
    allocation: List[Tuple[int, int, int]],
) -> List[List[int]]:
    """
    Trừ các đoạn đã dùng (allocation) khỏi 'windows'.
    allocation phần tử: (window_index, segment_start, segment_end)
    """
    allocated_segments_by_window: Dict[int, List[Tuple[int, int]]] = {}
    for window_index, segment_start, segment_end in allocation:
        allocated_segments_by_window.setdefault(window_index, []).append((segment_start, segment_end))

    for window_index in allocated_segments_by_window:
        allocated_segments_by_window[window_index].sort(key=lambda seg: seg[0])

    remaining_windows: List[List[int]] = []
    for idx, (window_start, window_end) in enumerate(windows):
        if idx not in allocated_segments_by_window:
            remaining_windows.append([window_start, window_end])
            continue

        cuts_for_window = allocated_segments_by_window[idx]
        current_segments = [[window_start, window_end]]

        for cut_start, cut_end in cuts_for_window:
            next_segments: List[List[int]] = []
            for avail_start, avail_end in current_segments:
                # không giao
                if cut_end <= avail_start or cut_start >= avail_end:
                    next_segments.append([avail_start, avail_end])
                else:
                    # cắt trái
                    if cut_start > avail_start:
                        next_segments.append([avail_start, cut_start])
                    # cắt phải
                    if cut_end < avail_end:
                        next_segments.append([cut_end, avail_end])
            current_segments = next_segments

        for seg_start, seg_end in current_segments:
            if seg_start < seg_end:
                remaining_windows.append([seg_start, seg_end])

    # gộp các đoạn còn lại nếu liền nhau
    remaining_windows.sort(key=lambda seg: (seg[0], seg[1]))
    merged: List[List[int]] = []
    for seg_start, seg_end in remaining_windows:
        if not merged or seg_start > merged[-1][1]:
            merged.append([seg_start, seg_end])
        else:
            merged[-1][1] = max(merged[-1][1], seg_end)
    return merged

def _swap_neighbor(job_order: List[int]) -> List[int]:
    num_jobs = len(job_order)
    if num_jobs < 2:
        return job_order[:]
    idx_a, idx_b = random.sample(range(num_jobs), 2)
    new_order = job_order[:]
    new_order[idx_a], new_order[idx_b] = new_order[idx_b], new_order[idx_a]
    return new_order

def _finish_time_from_allocation(allocation: List[Tuple[int, int, int]]) -> Optional[int]:
    """Tính finish_time = max(segment_end) từ allocation."""
    if not allocation:
        return None
    return max(segment_end for (_, _, segment_end) in allocation)

# =================== Lập lịch 1 hoán vị bằng DP ======================

def _schedule_with_order(
    config: Dict[str, Any],
    job_order: List[int],
    mode: str,
) -> Tuple[int, Dict[str, Any]]:
    """
    Gán tuần tự các job theo 'job_order' lên máy qua DP đơn-job.
    - mode == "timespan": solve_min_timespan_cfg -> (finish_time, allocation)
    - mode == "leftover": solve_feasible_leftover_rule_cfg -> (ok, allocation) + suy finish_time
    Trả về (makespan, chi_tiet)
    """
    split_min = _get_split_min(config)
    processing_times = _get_processing_times(config)
    windows_per_machine = [
        deepcopy(config["model"]["time_windows"][str(machine_id)])
        for machine_id in range(config["model"]["num_machines"])
        ]

    num_machines = len(windows_per_machine)
    last_finish_per_machine = [0] * num_machines
    allocation_by_job: Dict[int, Dict[str, Any]] = {}

    for job_id in job_order:
        best_machine_index: Optional[int] = None
        best_finish_time: Optional[int] = None
        best_allocation: Optional[List[Tuple[int, int, int]]] = None

        for machine_index in range(num_machines):
            machine_windows = windows_per_machine[machine_index]

            if mode == "timespan":
                finish_time, allocation = solve_min_timespan_cfg(
                    processing_times[job_id], machine_windows, split_min
                )
                feasible = (finish_time is not None) and (allocation is not None)
                candidate_finish_time = finish_time
            elif mode == "leftover":
                feasible, allocation = solve_feasible_leftover_rule_cfg(
                    processing_times[job_id], machine_windows, split_min
                )
                candidate_finish_time = _finish_time_from_allocation(allocation) if feasible and allocation else None
            else:
                raise ValueError("mode must be 'timespan' or 'leftover'.")

            if not feasible or candidate_finish_time is None:
                continue

            if (best_finish_time is None) or (candidate_finish_time < best_finish_time):
                best_finish_time = candidate_finish_time
                best_machine_index = machine_index
                best_allocation = allocation

        if best_machine_index is None:
            # Không gán được job này lên bất kỳ máy nào
            return (10**15, {
                "feasible": False,
                "reason": f"Job {job_id} infeasible under current windows.",
                "order": job_order
            })

        # Cập nhật windows còn lại cho máy được chọn
        windows_per_machine[best_machine_index] = _apply_allocation_to_windows(
            windows_per_machine[best_machine_index], best_allocation  # type: ignore[arg-type]
        )
        # Cập nhật thời điểm kết thúc
        last_finish_per_machine[best_machine_index] = max(
            last_finish_per_machine[best_machine_index],
            best_finish_time  # type: ignore[arg-type]
        )

        # Ghi trace cho job
        allocation_by_job[job_id] = {
            "machine": best_machine_index,
            "finish_time": best_finish_time,
            "allocation": best_allocation,
        }

    makespan = max(last_finish_per_machine)
    return makespan, {
        "feasible": True,
        "order": job_order[:],
        "last_finish": last_finish_per_machine,
        "allocations": allocation_by_job,
    }

# ============================ SA chính ===============================

def schedule_sa_config(
    config: Dict[str, Any],
    mode: str = "timespan",
    *,
    Tmax: float = 500.0,
    Tthreshold: float = 1.0,
    alpha: float = 0.99,
    moves_per_T: int = 50,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    config = _ensure_config(config)
    num_jobs = _get_num_jobs(config)

    current_order = list(range(num_jobs))
    random.shuffle(current_order)

    current_makespan, current_detail = _schedule_with_order(config, current_order, mode)
    best_makespan, best_detail = current_makespan, current_detail
    best_order = current_order[:]

    if verbose:
        print(f"[SA] init: order={current_order}, makespan={current_makespan}")

    temperature = float(Tmax)
    while temperature >= Tthreshold:
        for _ in range(moves_per_T):
            candidate_order = _swap_neighbor(current_order)
            candidate_makespan, _ = _schedule_with_order(config, candidate_order, mode)

            delta = candidate_makespan - current_makespan
            accept = (delta < 0) or (random.random() < math.exp(-delta / temperature if temperature > 0 else -1e9))

            if accept:
                current_order = candidate_order
                current_makespan = candidate_makespan

                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_order = current_order[:]
                    # Tính lại chi tiết cho best (để lưu allocations chuẩn)
                    best_makespan, best_detail = _schedule_with_order(config, best_order, mode)
                    if verbose:
                        print(f"[SA] improved: T={temperature:.2f}, makespan={best_makespan}, order={best_order}")

        temperature *= alpha

    return {
        "algo": "SA",
        "params": {
            "Tmax": Tmax,
            "Tthreshold": Tthreshold,
            "alpha": alpha,
            "moves_per_T": moves_per_T,
            "seed": seed,
            "mode": mode,
        },
        "best_makespan": best_makespan,
        "best_order": best_order,
        "detail": best_detail,  # allocations theo job + last_finish per machine
    }

# ============================== CLI =================================

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

    config = _load_config(args.config)
    result = schedule_sa_config(
        config,
        mode=args.mode,
        Tmax=args.Tmax,
        Tthreshold=args.Tthreshold,
        alpha=args.alpha,
        moves_per_T=args.moves_per_T,
        seed=args.seed,
        verbose=args.verbose,
    )

    print("\n=== SA RESULT ===")
    print("Mode:", result["params"]["mode"])
    print("Best makespan:", result["best_makespan"])
    print("Best order:", result["best_order"])

    # In lịch chi tiết (job -> machine, finish_time, segments) ###
    if "detail" in result and result["detail"].get("feasible", False):
        print("\nBest schedule (job -> machine, finish_time, allocation):")
        allocations = result["detail"]["allocations"]
        for job_id in result["best_order"]:
            info = allocations[job_id]
            print(
                f"  Job {job_id}: Machine {info['machine']}, "
                f"finish={info['finish_time']}, "
                f"segments={info['allocation']}"
            )

        # ### Vẽ Gantt chart ASCII ###
        segments_by_machine, horizon = _gantt_from_detail(result["detail"])
        print("\nGantt chart (ASCII):")
        print(_render_gantt_ascii(segments_by_machine, horizon))
    else:
        print("\n(No feasible schedule detail to render)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote result to: {args.out}")

if __name__ == "__main__":
    main()
