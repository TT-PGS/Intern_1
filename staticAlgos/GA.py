import os
import sys
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from copy import deepcopy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base.model import SchedulingModel
from base.io_handler import ReadJsonIOHandler

from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
    assign_job_to_machine
)

# =============================================
# GA representation & utilities
# =============================================
@dataclass
class Chromosome:
    """Chromosome encodes:
    - A permutation of job indices (order in which jobs are considered)
    - A machine assignment list (machine id for each job index)
    """
    order: List[int]
    machines: List[int]

@dataclass
class GAParams:
    pop_size: int = 100
    generations: int = 500
    tournament_k: int = 5
    crossover_rate: float = 0.8
    mutation_rate_perm: float = 0.1   # probability to mutate permutation (swap)
    mutation_rate_mach: float = 0   # probability per gene to mutate machine id
    elite: int = 2
    seed: Optional[int] = None

# =============================================
# Helpers for capacity & segment normalization
# =============================================

def _subtract_segments_from_windows(windows: List[List[int]],
                                    segs: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Trừ các đoạn [s,e) trong segs khỏi danh sách windows [[ws,we),...].
    Trả về windows còn lại (không chồng lấn, đã sắp xếp).
    """
    if not segs:
        return windows[:]
    segs_sorted = sorted(segs, key=lambda x: (x[0], x[1]))
    cur_windows = sorted(windows, key=lambda w: (w[0], w[1]))
    out: List[List[int]] = []
    i = j = 0
    while i < len(cur_windows):
        ws, we = cur_windows[i]
        wstart = ws
        wend = we
        while j < len(segs_sorted) and segs_sorted[j][1] <= wstart:
            j += 1
        k = j
        consumed = False
        while k < len(segs_sorted) and segs_sorted[k][0] < wend:
            cs, ce = max(wstart, segs_sorted[k][0]), min(wend, segs_sorted[k][1])
            if cs < ce:
                consumed = True
                if wstart < cs:
                    out.append([wstart, cs])
                wstart = ce
                if wstart >= wend:
                    break
            k += 1
        if not consumed:
            out.append([wstart, wend])
        elif wstart < wend:
            out.append([wstart, wend])
        i += 1
    # gộp nếu dính nhau
    out.sort(key=lambda w: (w[0], w[1]))
    merged: List[List[int]] = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    # Loại các cửa sổ rỗng
    return [[s, e] for s, e in merged if e > s]


def _normalize_segs(segs_raw: List[Tuple[int, ...]]) -> List[Tuple[int, int]]:
    """
    Chuẩn hoá segs từ solver về list[(start, end)].
    Chấp nhận mỗi phần tử có dạng (s, e) hoặc (x, s, e).
    """
    norm: List[Tuple[int, int]] = []
    for t in segs_raw:
        if len(t) == 2:
            s, e = t
        elif len(t) == 3:
            _, s, e = t
        else:
            raise ValueError(f"Unexpected segment tuple shape: {t}")
        norm.append((s, e))
    return norm


def _total_capacity_windows(wins: List[List[int]]) -> int:
    return sum(max(0, we - ws) for (ws, we) in wins)


# =============================================
# GA operators
# =============================================

def tournament_selection(pop: List[Chromosome], fits: List[float], k: int) -> Chromosome:
    idxs = random.sample(range(len(pop)), k)
    idxs.sort(key=lambda i: fits[i])  # lower fitness is better (makespan)
    return pop[idxs[0]]


def pmx_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    n = len(p1)
    if n < 2:
        return p1[:], p2[:]
    c1, c2 = p1[:], p2[:]
    a = random.randint(0, n - 2)
    b = random.randint(a + 1, n - 1)

    # Build mapping for segment [a, b]
    mapping1 = {p2[i]: p1[i] for i in range(a, b + 1)}
    mapping2 = {p1[i]: p2[i] for i in range(a, b + 1)}

    # Copy segment
    c1[a:b + 1] = p2[a:b + 1]
    c2[a:b + 1] = p1[a:b + 1]

    # Resolve outside segment via mapping (PMX)
    def resolve(child: List[int], mapping: Dict[int, int], a: int, b: int):
        for i in range(0, a):
            while child[i] in mapping:
                child[i] = mapping[child[i]]
        for i in range(b + 1, len(child)):
            while child[i] in mapping:
                child[i] = mapping[child[i]]

    resolve(c1, mapping1, a, b)
    resolve(c2, mapping2, a, b)
    return c1, c2


def uniform_crossover_list(m1: List[int], m2: List[int]) -> Tuple[List[int], List[int]]:
    assert len(m1) == len(m2)
    c1, c2 = m1[:], m2[:]
    for i in range(len(m1)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def mutate_permutation(order: List[int], rate: float) -> None:
    if random.random() < rate and len(order) >= 2:
        i, j = random.sample(range(len(order)), 2)
        order[i], order[j] = order[j], order[i]


def mutate_machines(machs: List[int], num_machines: int, rate: float) -> None:
    for i in range(len(machs)):
        if random.random() < rate:
            machs[i] = random.randint(0, num_machines - 1)

# =============================================
# Single-job assignment (forward/backward) + Job-wise packing per machine
# =============================================

def _assign_single_job(need: int,
                        windows: List[List[int]],
                        split_min: int,
                        mode: str,
                        direction: str = "forward") -> Tuple[Optional[int], List[Tuple[int, int]], bool]:
    """
    Gán một job vào windows theo 1 trong 2 chiến lược:
        - forward (earliest-fit): dùng solver trực tiếp
        - backward (latest-fit): ánh xạ thời gian t -> -t để tái sử dụng solver
    Trả (finish_time, segs[(s,e)], ok)
    """
    wins_sorted = [w[:] for w in sorted(windows, key=lambda w: (w[0], w[1]))]

    if direction == "backward":
        # ánh xạ t -> -t và đảo thứ tự để pack từ cuối
        wins_rev = [[-w[1], -w[0]] for w in wins_sorted][::-1]
        if mode == "leftover":
            ok, _ = solve_feasible_leftover_rule_cfg(need, wins_rev, split_min)
            if not ok:
                return None, [], False
            fin_r, segs_r = solve_min_timespan_cfg(need, wins_rev, split_min)
        elif mode == "timespan":
            fin_r, segs_r = solve_min_timespan_cfg(need, wins_rev, split_min)
        elif mode == "assign":
            fin_r, segs_r = assign_job_to_machine(need, wins_rev, split_min)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if fin_r is None or not segs_r:
            return None, [], False
        segs_fwd: List[Tuple[int, int]] = []
        for t in segs_r:
            if len(t) == 2:
                s_r, e_r = t
            else:
                _, s_r, e_r = t
            s, e = -e_r, -s_r
            segs_fwd.append((s, e))
        segs_fwd.sort()
        fin = max(e for (_, e) in segs_fwd)
        return fin, segs_fwd, True

    # direction == "forward"
    if mode == "leftover":
        ok, _ = solve_feasible_leftover_rule_cfg(need, wins_sorted, split_min)
        if not ok:
            return None, [], False
        fin, segs_raw = solve_min_timespan_cfg(need, wins_sorted, split_min)
    elif mode == "timespan":
        fin, segs_raw = solve_min_timespan_cfg(need, wins_sorted, split_min)
    elif mode == "assign":
        fin, segs_raw = assign_job_to_machine(need, wins_sorted, split_min)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if fin is None or not segs_raw:
        return None, [], False

    segs = _normalize_segs(segs_raw)
    return fin, segs, True

def _machine_total_capacity(time_windows: Dict[int, List[List[int]]]) -> Dict[int, int]:
    return {m: sum(max(0, we - ws) for (ws, we) in sorted(wins)) for m, wins in time_windows.items()}

def _repair_machines(order: List[int],
                        machines: List[int],
                        processing_times: List[int],
                        time_windows: Dict[int, List[List[int]]]) -> List[int]:
    """
    Sửa vector machines (gán máy) theo hướng capacity-aware:
        - Tính tổng tải theo máy (sum p_j các job đang gán lên máy)
        - Nếu máy nào vượt capacity, chuyển bớt job sang máy còn dư nhiều nhất
        - Chọn job chuyển: job có p_j lớn nhất ở máy đang quá tải, duyệt cho tới khi hết quá tải
    Trả về bản sao machines đã sửa (không mutate tham số gốc).
    """
    num_jobs = len(order)
    mach_fixed = machines[:]  # copy

    # capacity mỗi máy
    cap = _machine_total_capacity(time_windows)

    # tổng tải hiện tại theo máy
    load: Dict[int, int] = {m: 0 for m in cap.keys()}
    jobs_on_m: Dict[int, List[Tuple[int,int]]] = {m: [] for m in cap.keys()}  # (pos, p_j)
    for pos, j in enumerate(order):
        m = mach_fixed[pos]
        pj = processing_times[j]
        load[m] += pj
        jobs_on_m[m].append((pos, pj))

    # máy có thể không có trong dict nếu key windows là list 0..M-1
    # => đảm bảo đầy đủ key
    for m in range(max(cap.keys())+1):
        load.setdefault(m, 0)
        jobs_on_m.setdefault(m, [])

    # lặp tới khi tất cả máy không quá tải hoặc không chuyển được nữa
    changed = True
    while changed:
        changed = False
        # tìm máy quá tải nhất
        over_machines = [(m, load[m] - cap.get(m, 0)) for m in load if load[m] > cap.get(m, 0)]
        if not over_machines:
            break
        # sắp xếp giảm dần theo mức vượt
        over_machines.sort(key=lambda x: x[1], reverse=True)

        for m_over, excess in over_machines:
            if excess <= 0:
                continue
            # ứng viên nhận: máy còn dư capacity nhiều nhất
            under = [(m2, cap.get(m2, 0) - load[m2]) for m2 in load if cap.get(m2, 0) - load[m2] > 0 and m2 != m_over]
            if not under:
                # không có máy nhận -> hết cách sửa
                continue
            under.sort(key=lambda x: x[1], reverse=True)
            receiver = under[0][0]

            # chọn job lớn nhất trên m_over để giảm nhanh
            if not jobs_on_m[m_over]:
                continue
            jobs_on_m[m_over].sort(key=lambda t: t[1], reverse=True)  # by pj desc
            pos_move, pj_move = jobs_on_m[m_over][0]

            # chuyển
            mach_fixed[pos_move] = receiver
            jobs_on_m[m_over].pop(0)
            jobs_on_m[receiver].append((pos_move, pj_move))
            load[m_over] -= pj_move
            load[receiver] += pj_move
            changed = True

    return mach_fixed


def _assign_jobs_on_machine(jobs_order: List[int],
                            processing_times: List[int],
                            windows: List[List[int]],
                            split_min: int,
                            mode: str = "assign",
                            debug: bool = False) -> Tuple[Optional[int], List[Tuple[int, int, int]], bool]:
    """
    Gán tuần tự các job theo jobs_order vào 'windows' của 1 máy.
    Trả:
        - finish_time (max end) hoặc None nếu infeasible,
        - segments [(job, start, end), ...] cho đúng job,
        - feasible (bool)
    """
    # Guard: tổng capacity >= tổng nhu cầu
    total_need = sum(processing_times[j] for j in jobs_order)
    if _total_capacity_windows(windows) < total_need:
        # if debug:
        #     print("[DBG] Machine infeasible: total capacity < total need")
        return None, [], False

    def try_direction(direction: str) -> Tuple[Optional[int], List[Tuple[int, int, int]], bool]:
        win_cur = [w[:] for w in sorted(windows, key=lambda w: (w[0], w[1]))]
        all_segments: List[Tuple[int, int, int]] = []
        finish_time = 0

        for idx, j in enumerate(jobs_order):
            need = processing_times[j]
            if need == 0:
                continue

            fin, segs, ok = _assign_single_job(need, win_cur, split_min, mode, direction=direction)
            if not ok or fin is None:
                if debug:
                    print(f"[DBG] Fail {direction} at job {j}: solver returned None/empty")
                return None, [], False

            # ép mỗi mảnh >= split_min
            for (s, e) in segs:
                if e - s < split_min:
                    if debug:
                        print(f"[DBG] Fail {direction} at job {j}: chunk {e - s} < split_min")
                    return None, [], False

            # ghi lại và trừ capacity
            for (s, e) in segs:
                all_segments.append((j, s, e))
            finish_time = max(finish_time, max(e for (_, e) in segs))
            win_cur = _subtract_segments_from_windows(win_cur, segs)

            # Guard: capacity còn lại phải >= nhu cầu còn lại
            remain_need = sum(processing_times[jobs_order[t]] for t in range(idx + 1, len(jobs_order)))
            if _total_capacity_windows(win_cur) < remain_need:
                if debug:
                    print(f"[DBG] Early prune {direction}: remaining capacity insufficient after job {j}")
                return None, [], False

        return finish_time, all_segments, True

    # Thử earliest trước, nếu fail thì thử latest
    fin, segs, ok = try_direction("forward")
    if ok:
        return fin, segs, True
    fin, segs, ok = try_direction("backward")
    if ok:
        return fin, segs, True
    return None, [], False


# =============================================
# Evaluator (job-wise) — returns segments ready to use
# =============================================

def _evaluate_chromosome(
    chrom: Chromosome,
    model: SchedulingModel,
    mode: str = "assign",
    penalty: float = 10**9,
) -> Tuple[float, Dict[int, int], Dict[int, List[int]], Dict[int, List[Tuple[int, int, int]]]]:
    """
    Đánh giá theo từng job trên mỗi máy (tuân thủ split_min):
        - Lấy thứ tự job theo chrom.order và máy theo chrom.machines,
        - Gom job theo máy (giữ nguyên thứ tự phục vụ),
        - Duyệt từng máy: cho từng job gọi solver -> lấy segs, trừ capacity, cập nhật finish,
        - Nếu có job infeasible -> trả penalty.

    Trả:
        fitness, machine_finish_times, per_machine_job_order, per_machine_segments
    """
    num_machines = model.get_num_machines()
    processing_times = model.get_processing_times()
    split_min = model.get_split_min()
    tw_raw = model.get_time_windows()
    # chuẩn hoá key
    time_windows = {int(k): v for k, v in tw_raw.items()} if isinstance(tw_raw, dict) else tw_raw

    # gom job theo máy theo thứ tự phục vụ
    per_machine_jobs: Dict[int, List[int]] = {m: [] for m in range(num_machines)}
    for pos, j in enumerate(chrom.order):
        m = int(chrom.machines[pos])
        per_machine_jobs[m].append(j)

    machine_finish: Dict[int, int] = {}
    per_machine_segments: Dict[int, List[Tuple[int, int, int]]] = {m: [] for m in range(num_machines)}

    for m in range(num_machines):
        jobs_m = per_machine_jobs[m]
        if not jobs_m:
            machine_finish[m] = 0
            continue
        wins = time_windows[m]
        if not wins:
            return float(penalty), {}, {}, {}
        # debug có thể bật thủ công tại đây nếu cần
        need_total = sum(processing_times[j] for j in jobs_m)
        cap_total  = _total_capacity_windows(wins)
        # if cap_total < need_total:
            # print(f"[DBG] M{m}: total capacity {cap_total} < total need {need_total}  (jobs={jobs_m})")

        fin, segs, ok = _assign_jobs_on_machine(jobs_m, processing_times, wins, split_min, mode, debug=False)
        if not ok or fin is None:
            # print(f"[DBG] M{m}: assignment failed (mode={mode}). See earlier debug lines for the failing job/chunk.")
            return float(penalty), {}, {}, {}
        machine_finish[m] = int(fin)
        per_machine_segments[m] = segs  # segs đã gắn job id

    makespan = max(machine_finish.values()) if machine_finish else 0
    return float(makespan), machine_finish, per_machine_jobs, per_machine_segments


# =============================================
# GA main loop
# =============================================

def run_ga(config: SchedulingModel, params: GAParams, mode: str = "assign") -> Dict[str, Any]:
    if params.seed is not None:
        random.seed(params.seed)

    model = deepcopy(config)

    num_jobs = model.get_num_jobs()
    num_machines = model.get_num_machines()
    processing_times = model.get_processing_times()
    time_windows = model.get_time_windows()

    if isinstance(time_windows, dict):
        time_windows = {int(k): v for k, v in time_windows.items()}

    # --- initialize population ---
    def random_chrom() -> Chromosome:
        order = list(range(num_jobs))
        random.shuffle(order)
        machines = [random.randint(0, num_machines - 1) for _ in range(num_jobs)]
        return Chromosome(order, machines)

    population: List[Chromosome] = [random_chrom() for _ in range(params.pop_size)]

    # >>> REPAIR SAU KHỞI TẠO
    population = [
        Chromosome(
            order=chrom.order[:],
            machines=_repair_machines(chrom.order, chrom.machines, processing_times, time_windows)
        )
        for chrom in population
    ]

    # Evaluate initial population
    fitness: List[float] = []
    meta_cache: List[Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, List[Tuple[int, int, int]]]]] = []
    for chrom in population:
        f, mf, pmj, pms = _evaluate_chromosome(chrom, model, mode)
        fitness.append(f)
        meta_cache.append((mf, pmj, pms))

    # --- evolution ---
    for _ in range(params.generations):
        # Elitism
        ranked = sorted(zip(population, fitness, meta_cache), key=lambda x: x[1])
        new_pop: List[Chromosome] = [ranked[i][0] for i in range(min(params.elite, len(ranked)))]

        # Generate offspring
        while len(new_pop) < params.pop_size:
            p1 = tournament_selection(population, fitness, params.tournament_k)
            p2 = tournament_selection(population, fitness, params.tournament_k)

            c1 = Chromosome(p1.order[:], p1.machines[:])
            c2 = Chromosome(p2.order[:], p2.machines[:])

            if random.random() < params.crossover_rate:
                # PMX on permutation
                c1.order, c2.order = pmx_crossover(p1.order, p2.order)
                # Uniform crossover on machine genes
                c1.machines, c2.machines = uniform_crossover_list(p1.machines, p2.machines)

            # Mutations
            mutate_permutation(c1.order, params.mutation_rate_perm)
            mutate_permutation(c2.order, params.mutation_rate_perm)
            mutate_machines(c1.machines, num_machines, params.mutation_rate_mach)
            mutate_machines(c2.machines, num_machines, params.mutation_rate_mach)

            # >>> REPAIR TRƯỚC KHI THÊM VÀO POP
            c1.machines = _repair_machines(c1.order, c1.machines, processing_times, time_windows)
            c2.machines = _repair_machines(c2.order, c2.machines, processing_times, time_windows)

            new_pop.append(c1)
            if len(new_pop) < params.pop_size:
                new_pop.append(c2)

        # Replace
        population = new_pop

        # Re-evaluate
        fitness = []
        meta_cache = []
        for chrom in population:
            f, mf, pmj, pms = _evaluate_chromosome(chrom, model, mode)
            fitness.append(f)
            meta_cache.append((mf, pmj, pms))

    # --- choose the best ---
    best_idx = min(range(len(population)), key=lambda i: fitness[i])
    best = population[best_idx]
    best_fit = fitness[best_idx]
    best_machine_finish, best_per_m_jobs, best_per_m_segments = meta_cache[best_idx]

    def _keys_to_int(d):
        try:
            return {int(k): v for k, v in d.items()}
        except Exception:
            return d

    best_per_m_jobs = _keys_to_int(best_per_m_jobs)
    best_machine_finish = _keys_to_int(best_machine_finish)
    best_per_m_segments = {int(k): v for k, v in best_per_m_segments.items()}

    per_machine_segments = best_per_m_segments  # đã đúng split_min & khả thi

    result = {
        "makespan": int(best_fit),
        "order": best.order,
        "machines": best.machines,
        "machine_finish_times": best_machine_finish,
        "per_machine_jobs": best_per_m_jobs,
        "per_machine_segments": per_machine_segments,
    }
    return result

# =============================================
# ASCII Gantt chart (simple)
# =============================================

def _ascii_gantt(per_machine_segments: Dict[int, List[Tuple[int, int, int]]],
                    time_windows: Dict[int, List[List[int]]],
                    width_limit: int = 120) -> str:
    """Draw a minimal ASCII Gantt. Each machine on one row.
    We compute the horizon from its windows. If the horizon is too large, we scale down.
    """
    lines: List[str] = []
    # Compute global horizon to decide scaling (per machine horizon varies, we scale per machine)
    for m in sorted(time_windows.keys()):
        windows = sorted(time_windows[m], key=lambda w: (w[0], w[1]))
        if not windows:
            lines.append(f"M{m}: <no windows>")
            continue
        start0 = windows[0][0]
        endL = max(w[1] for w in windows)
        horizon = max(1, endL - start0)
        scale = 1
        if horizon > width_limit:
            scale = max(1, horizon // width_limit + (1 if horizon % width_limit else 0))

        # Build empty canvas
        length = max(1, horizon // scale + (1 if horizon % scale else 0))
        row = [" "] * length

        # Mark windows as '.' (capacity)
        for (ws, we) in windows:
            a = max(0, (ws - start0) // scale)
            b = max(a, min(length, (we - start0 + scale - 1) // scale))
            for x in range(a, b):
                row[x] = "." if row[x] == " " else row[x]

        # Draw job segments as letters A,B,C,... for job ids
        for (j, s, e) in per_machine_segments.get(m, []):
            a = max(0, (s - start0) // scale)
            b = max(a, min(length, (e - start0 + scale - 1) // scale))
            ch = chr(ord('A') + (j % 26))
            for x in range(a, b):
                row[x] = ch

        lines.append(f"M{m} |" + "".join(row) + f"|  (scale={scale}:1, origin={start0})")
    return "\n".join(lines)

# =============================================
# Pretty printing result
# =============================================

def print_summary(result: Dict[str, Any], model: SchedulingModel) -> None:
    print("==== GA Result ====")
    print(f"Makespan (timespan): {result['makespan']}")
    print(f"Best order: {result['order']}")
    print(f"Machine genes: {result['machines']}")
    print("Machine finish times:")
    for m, ft in sorted(result["machine_finish_times"].items()):
        print(f"  - M{m}: {ft}")
    print("Jobs per machine (in service order):")
    for m, jobs in sorted(result["per_machine_jobs"].items()):
        print(f"  - M{m}: {jobs}")

    # List segments
    print("\nSegments per machine (job, start, end):")
    for m, segs in sorted(result["per_machine_segments"].items()):
        print(f"  M{m}:")
        if not segs:
            print("    <idle>")
        else:
            for (j, s, e) in segs:
                print(f"    job {j}: [{s}, {e})")

    # Gantt (ASCII)
    print("\nASCII Gantt (simple):")
    print(_ascii_gantt(result["per_machine_segments"], model.get_time_windows()))

# =============================================
# CLI
# =============================================

def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Splittable Job Scheduling (uses DP per machine).")
    parser.add_argument("--config", type=str, default="./datasets/splittable_jobs.json", help="Path to JSON config.")
    parser.add_argument("--mode", type=str, default="timespan", choices=["timespan", "leftover"], help="Objective: minimize timespan or just check feasibility then rank by timespan")
    parser.add_argument("--pop_size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--tournament_k", type=int, default=5)
    parser.add_argument("--crossover_rate", type=float, default=0.9)
    parser.add_argument("--mutation_rate_perm", type=float, default=0.2)
    parser.add_argument("--mutation_rate_mach", type=float, default=0.05)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = ReadJsonIOHandler(args.config).get_input()
    params = GAParams(
        pop_size=args.pop_size,
        generations=args.generations,
        tournament_k=args.tournament_k,
        crossover_rate=args.crossover_rate,
        mutation_rate_perm=args.mutation_rate_perm,
        mutation_rate_mach=args.mutation_rate_mach,
        elite=args.elite,
        seed=args.seed,
    )

    result = run_ga(cfg, params, mode=args.mode)
    print_summary(result, cfg)

if __name__ == "__main__":
    main()
