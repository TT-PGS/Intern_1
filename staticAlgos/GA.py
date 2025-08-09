import os
import sys
import json
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# ---------------------------------------------
# Project import path (project root one level up)
# ---------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Now we can import the DP helpers
from split_job.dp_single_job import (
    solve_min_timespan_cfg,
    solve_feasible_leftover_rule_cfg,
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
    pop_size: int = 40
    generations: int = 150
    tournament_k: int = 5
    crossover_rate: float = 0.9
    mutation_rate_perm: float = 0.2   # probability to mutate permutation (swap)
    mutation_rate_mach: float = 0.05  # probability per gene to mutate machine id
    elite: int = 2
    seed: Optional[int] = None

# =============================================
# Config loading
# =============================================

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Normalize time_windows keys: "0" -> 0, etc.
    model = cfg.get("model", {})
    tw = model.get("time_windows", {})
    if isinstance(tw, dict):
        normalized = {}
        for k, v in tw.items():
            try:
                normalized[int(k)] = v
            except ValueError:
                raise ValueError(f"time_windows key '{k}' is not an int-like key")
        model["time_windows"] = normalized
        cfg["model"] = model
    return cfg

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
    c1[a:b+1] = p2[a:b+1]
    c2[a:b+1] = p1[a:b+1]

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
# Scheduling evaluation using DP per MACHINE (aggregate, then split to jobs)
# =============================================

def _sum_proc_time_for_machine(order: List[int], machines: List[int], processing_times: List[int], m: int) -> Tuple[int, List[int]]:
    """Return (total_pt, jobs_on_m_in_order). The order among jobs is induced by global 'order'."""
    jobs_on_m: List[int] = [j for j in order if machines[j] == m]
    total = sum(processing_times[j] for j in jobs_on_m)
    return total, jobs_on_m


def _evaluate_chromosome(
    chrom: Chromosome,
    model: Dict[str, Any],
    mode: str = "timespan",
    penalty: float = 10**9,
) -> Tuple[float, Dict[int, int], Dict[int, List[int]]]:
    """Evaluate chromosome.

    Returns:
        fitness (float): makespan or large penalty if infeasible
        machine_finish_times (dict m -> finish_time)
        per_machine_job_order (dict m -> [job ids in order])
    """
    num_machines = int(model["num_machines"])
    processing_times: List[int] = list(model["processing_times"])
    split_min = int(model["split_min"])
    time_windows: Dict[int, List[List[int]]] = model["time_windows"]

    machine_finish: Dict[int, int] = {}
    per_m_jobs: Dict[int, List[int]] = {}

    for m in range(num_machines):
        total_pt, jobs_on_m = _sum_proc_time_for_machine(
            chrom.order, chrom.machines, processing_times, m
        )
        per_m_jobs[m] = jobs_on_m
        if total_pt == 0:
            machine_finish[m] = 0
            continue

        windows_m = time_windows.get(m, [])
        if not windows_m:
            # No capacity for any work on this machine
            return penalty, {}, {}

        # Check feasibility (if requested) then compute finish time
        if mode == "leftover":
            feas, _ = solve_feasible_leftover_rule_cfg(total_pt, windows_m, split_min)
            if not feas:
                return penalty, {}, {}
            # If feasible, also get the finish time via min_timespan solver (for comparison)
            finish, _ = solve_min_timespan_cfg(total_pt, windows_m, split_min)
        else:
            finish, _ = solve_min_timespan_cfg(total_pt, windows_m, split_min)

        if finish is None:
            return penalty, {}, {}
        machine_finish[m] = int(finish)

    makespan = max(machine_finish.values()) if machine_finish else 0
    return float(makespan), machine_finish, per_m_jobs

# =============================================
# Reconstruct per-job segments by greedy packing across machine windows
# (We rely on DP for finish time; packing is solely for visualization/report)
# =============================================

def _pack_jobs_into_windows(
    jobs_order: List[int],
    processing_times: List[int],
    windows: List[List[int]],
) -> List[Tuple[int, int, int]]:
    """Greedy earliest-fit packing of jobs into absolute windows (start,end).

    Returns list of (job_id, start, end) segments for this machine.
    This is *only* for visualization; we do not use it to compute the objective.
    """
    # Sort windows by start time
    windows_sorted = sorted(windows, key=lambda w: (w[0], w[1]))
    # For each window keep a cursor of how much used inside that window
    cursors = [w[0] for w in windows_sorted]  # current time pointer inside window

    segments: List[Tuple[int, int, int]] = []

    for j in jobs_order:
        remain = int(processing_times[j])
        for idx, (ws, we) in enumerate(windows_sorted):
            if remain <= 0:
                break
            cur = cursors[idx]
            if cur >= we:
                continue  # window full
            avail = we - cur
            if avail <= 0:
                continue
            use = min(avail, remain)
            seg_start = cur
            seg_end = cur + use
            segments.append((j, seg_start, seg_end))
            cursors[idx] = seg_end
            remain -= use
        # If remain > 0 here, windows were insufficient; this should not happen if DP said feasible
    return segments

# =============================================
# GA main loop
# =============================================

def run_ga(config: Dict[str, Any], params: GAParams, mode: str = "timespan") -> Dict[str, Any]:
    if params.seed is not None:
        random.seed(params.seed)

    model = config["model"]
    num_jobs = int(model["num_jobs"])
    num_machines = int(model["num_machines"])
    processing_times: List[int] = list(model["processing_times"])
    time_windows: Dict[int, List[List[int]]] = model["time_windows"]

    # --- initialize population ---
    def random_chrom() -> Chromosome:
        order = list(range(num_jobs))
        random.shuffle(order)
        machines = [random.randint(0, num_machines - 1) for _ in range(num_jobs)]
        return Chromosome(order, machines)

    population: List[Chromosome] = [random_chrom() for _ in range(params.pop_size)]

    # Evaluate initial population
    fitness: List[float] = []
    meta_cache: List[Tuple[Dict[int, int], Dict[int, List[int]]]] = []
    for chrom in population:
        f, mf, pmj = _evaluate_chromosome(chrom, model, mode)
        fitness.append(f)
        meta_cache.append((mf, pmj))

    # --- evolution ---
    for gen in range(params.generations):
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

            new_pop.append(c1)
            if len(new_pop) < params.pop_size:
                new_pop.append(c2)

        # Replace
        population = new_pop

        # Re-evaluate
        fitness = []
        meta_cache = []
        for chrom in population:
            f, mf, pmj = _evaluate_chromosome(chrom, model, mode)
            fitness.append(f)
            meta_cache.append((mf, pmj))

    # --- choose the best ---
    best_idx = min(range(len(population)), key=lambda i: fitness[i])
    best = population[best_idx]
    best_fit = fitness[best_idx]
    best_machine_finish, best_per_m_jobs = meta_cache[best_idx]

    # Reconstruct segments for report (visualization only)
    per_machine_segments: Dict[int, List[Tuple[int, int, int]]] = {}
    for m in range(num_machines):
        jobs_on_m = best_per_m_jobs.get(m, [])
        if not jobs_on_m:
            per_machine_segments[m] = []
            continue
        windows_m = time_windows.get(m, [])
        segs = _pack_jobs_into_windows(jobs_on_m, processing_times, windows_m)
        per_machine_segments[m] = segs

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

def print_summary(result: Dict[str, Any], model: Dict[str, Any]) -> None:
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
    print(_ascii_gantt(result["per_machine_segments"], model["time_windows"]))

# =============================================
# CLI
# =============================================

def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Splittable Job Scheduling (uses DP per machine).")
    parser.add_argument("--config", type=str, default="./configs/splittable_jobs.json", help="Path to JSON config.")
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

    cfg = _load_config(args.config)
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
    print_summary(result, cfg["model"])

if __name__ == "__main__":
    main()
