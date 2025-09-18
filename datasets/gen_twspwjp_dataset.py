"""
Generate TWSPwJP dataset for training.
The specs of the dataset are following:
- Number of jobs (n: integer): [10,20,30,40,50]
- Number of machines (k: integer): [2,3,4]
- Minimum split size (spm: integer): [3,4,5]
- Instances per (n,k,spm) combination: 10
- Processing times (integer): discrete uniform in [spm, 24]
- Machine time windows:
    - Each machine has 3-6 non-overlapping windows.
    - Each window length is discrete uniform in [spm, 12].
    - Gaps between windows are discrete uniform in [0, 3].
    - Ensure total capacity across all machines >= total processing time of all jobs (with 5-15% slack).
- Random seed: start on 20250609 (for reproducibility)

The dataset is saved as JSON files in the specified output directory. Each instance file is named as:
    twspwjp_n{n}_k{k}_spm{spm}_i{index}.json
where:
    n is the number of jobs in integer
    k is the number of machines in integer
    spm is the minimum split size in integer
    index is the instance index for the given (n,k,spm) combination.

Each instance JSON has the structure:
{
    "agent": "dqn",
    "model": {
        "num_jobs": n,
        "num_machines": k,
        "processing_times": [...],  # list of n processing times
        "split_min": spm,
        "time_windows": {           # dict of k machines with their time windows
            "0": [[start1, end1], [start2, end2], ...],
            "1": [[start1, end1], [start2, end2], ...],
            ...
            "k-1": [[start1, end1], [start2, end2], ...]
        }
    }
}
"""
import argparse, json, math, os, random
from pathlib import Path
from typing import List, Dict, Tuple

def d_unif(rng: random.Random, lo: int, hi: int) -> int:
    """Discrete uniform in [lo, hi]."""
    return rng.randint(lo, hi)

def gen_processing_times(rng: random.Random, n: int, spm: int) -> List[int]:
    return [d_unif(rng, spm, 24) for _ in range(n)]

def gen_machine_windows(
    rng: random.Random,
    capacity_needed: int,
    spm: int,
    min_wins: int = 20,
    max_wins: int = 24,
    gap_lo: int = 0,
    gap_hi: int = 3
) -> List[List[int]]:
    """
    Sinh danh sách windows không chồng lấn: [[start, end], ...], đã sort.
    - Mỗi window có length ~ U[spm,12]
    - Gaps giữa windows ~ U[gap_lo, gap_hi]
    - Đảm bảo tổng capacity >= capacity_needed (nếu thiếu, thêm 1 window cuối cho đủ)
    """
    windows = []
    t = 0
    total_cap = 0

    # Số window ban đầu
    num_wins = d_unif(rng, min_wins, max_wins)
    for _ in range(num_wins):
        length = d_unif(rng, spm, 12)
        start = t
        end = start + length
        windows.append([start, end])
        total_cap += (end - start)
        t = end + d_unif(rng, gap_lo, gap_hi)

    # Nếu chưa đủ capacity thì nối thêm 1 window cuối
    if total_cap < capacity_needed:
        need = capacity_needed - total_cap
        start = t
        end = start + need
        windows.append([start, end])
        total_cap += (end - start)

    return total_cap, windows

def ensure_capacity_all_machines(
    rng: random.Random,
    k: int,
    total_work: int,
    spm: int
) -> Dict[str, List[List[int]]]:
    """
    Phân bổ nhu cầu capacity tổng thể vào k máy.
    - Ước lượng nhu cầu cho mỗi máy theo tỉ lệ (ngẫu nhiên nhẹ),
        rồi sinh windows cho từng máy bảo đảm đủ phần của nó.
    """
    # Chia “quota” cho từng máy bằng random simplex
    weights = [rng.random() for _ in range(k)]
    sw = sum(weights) if sum(weights) > 0 else 1.0
    weights = [w / sw for w in weights]

    # Thêm một chút slack để tránh thiếu (5–15%)
    slack_factor = 1.0 + (0.05 + 0.10 * rng.random())
    total_target = math.ceil(total_work * slack_factor)

    time_windows: Dict[str, List[List[int]]] = {}
    allocated = 0
    total_cap = 0
    for m in range(k):
        # Máy cuối nhận phần còn lại để đảm bảo tổng
        if m == k - 1:
            cap_m = max(spm, total_target - allocated)
        else:
            cap_m = max(spm, int(round(total_target * weights[m])))
            allocated += cap_m

        cap, wins = gen_machine_windows(rng, cap_m, spm)
        time_windows[str(m)] = wins
        total_cap += cap

    return total_cap, time_windows

def gen_instance(
    base_seed: int,
    n: int,
    k: int,
    spm: int,
    index: int
) -> dict:
    """
    Sinh 1 instance với seed xác định từ (base_seed, n, k, spm, index).
    """
    # Seed ổn định theo tổ hợp tham số
    seed = (base_seed * 1_000_003) ^ (n * 7) ^ (k * 13) ^ (spm * 17) ^ (index * 97)
    rng = random.Random(seed)

    processing_times = gen_processing_times(rng, n, spm)
    total_work = sum(processing_times)
    lower_bound = total_work / k

    total_cap, time_windows = ensure_capacity_all_machines(rng, k, total_work, spm)

    feasible = total_cap >= total_work
    return {
        "agent": "dqn",
        "model": {
            "num_jobs": n,
            "num_machines": k,
            "total_capacity_all_windows": total_cap,
            "total_processing_time": total_work,
            "feasible": feasible,
            "lower_bound": lower_bound,
            "processing_times": processing_times,
            "split_min": spm,
            "time_windows": time_windows
        }
    }

def main():
    p = argparse.ArgumentParser(
        description="Generate TWSPwJP dataset for DQN training."
    )
    p.add_argument("--ns", type=str, default="10,20,30,40,50",
                    help="Comma-separated n values (num_jobs).")
    p.add_argument("--ks", type=str, default="2,3,4",
                    help="Comma-separated k values (num_machines).")
    p.add_argument("--spms", type=str, default="3,4,5",
                    help="Comma-separated split_min values.")
    p.add_argument("--instances-per-combo", type=int, default=10,
                    help="Instances per (n,k,spm) combination.")
    p.add_argument("--seed", type=int, default=20250609,
                    help="Base seed for reproducibility.")
    p.add_argument("--outdir", type=str, default="./",
                    help="Output directory.")
    args = p.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    spms = [int(x) for x in args.spms.split(",") if x.strip()]
    per = args.instances_per_combo

    outdir = Path(args.outdir).joinpath(str(args.seed))
    outdir.mkdir(parents=True, exist_ok=True)

    total = 0
    for n in ns:
        for k in ks:
            for spm in spms:
                for i in range(per):
                    inst = gen_instance(args.seed, n, k, spm, i)
                    fname = f"twspwjp_Jobs_{n}_Machines_{k}_Splitmin_{spm}_Index_{i}.json"
                    fpath = outdir.joinpath(fname)
                    with open(fpath, "w") as f:
                        json.dump(inst, f, indent=2)
                    total += 1

    print(f"Generated {total} instances to {outdir.resolve()}")

if __name__ == "__main__":
    main()
