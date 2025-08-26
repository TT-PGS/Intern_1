import os
import json
import argparse
import torch

# ==== Input - Output management ====
from base.io_handler import ReadJsonIOHandler

# ==== DRL core ====
from agents.agent_factory import create_agent
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.runner import run_episode
from agents.train_dqn import train_dqn

# ==== Static algorithms ====
from staticAlgos.FCFS import schedule_fcfs
from staticAlgos.SA import schedule_sa_config
from staticAlgos.GA import run_ga, GAParams

# ---------------------------
# Helpers
# ---------------------------

def compute_makespan_from_assignments(assignments):
    max_end = 0
    for a in assignments:
        for (_w, s, e) in a.get("segments", []):
            if e > max_end:
                max_end = e
    return max_end

def print_assignments(assignments):
    """
    In lịch theo dạng: Job -> Machine, segments=(w, start, end).
    Hỗ trợ đầu ra đồng nhất cho FCFS/SA/GA (đã chuẩn hóa thành 'assignments').
    """
    if not assignments:
        print("Empty schedule")
        return
    print("Best schedule (job -> machine, finish_time, allocation):")
    # sort theo job id để dễ đọc
    for a in sorted(assignments, key=lambda x: x["job"]):
        j = a["job"]
        m = a["machine"]
        segs = a.get("segments", [])
        finish = max((e for (_w, s, e) in segs), default=None)
        print(f"  Job {j}: Machine {m}, finish={finish}, segments={segs}")

def maybe_write_out(out_path: str, payload: dict):
    if not out_path:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved result to: {out_path}")

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified entrypoint: RL (random/DQN) + static (FCFS/SA/GA)")
    # Common
    parser.add_argument("--mode", type=str, default=None,
                        help="one of: dqn, fcfs, sa, ga. If omitted, use env SCHEDULER_MODE or default 'dqn'.")
    parser.add_argument("--config", type=str, default="./configs/splittable_jobs.json",
                        help="Path to JSON config containing 'model'")
    parser.add_argument("--out", type=str, default="",
                        help="Optional path to write JSON result (assignments, makespan, etc.)")
    parser.add_argument("--split-mode", type=str, default="timespan", choices=["timespan", "leftover"],
                        help="DP mode for splitting jobs when assigning to machines")

    # RL options
    parser.add_argument("--dqn-episodes", type=int, default=10, help="DQN training episodes")
    parser.add_argument("--dqn-model", type=str, default="qnet.pt", help="Path to save/load DQN weights")

    # SA options
    parser.add_argument("--sa-Tmax", type=float, default=500.0)
    parser.add_argument("--sa-Tthreshold", type=float, default=1.0)
    parser.add_argument("--sa-alpha", type=float, default=0.99)
    parser.add_argument("--sa-moves-per-T", type=int, default=50)
    parser.add_argument("--sa-seed", type=int, default=None)
    parser.add_argument("--sa-verbose", action="store_true")

    # GA options
    parser.add_argument("--ga-pop", type=int, default=40)
    parser.add_argument("--ga-gen", type=int, default=200)
    parser.add_argument("--ga-cx", type=float, default=0.9)
    parser.add_argument("--ga-mut", type=float, default=0.2)
    parser.add_argument("--ga-tk", type=int, default=5)
    parser.add_argument("--ga-seed", type=int, default=None)
    parser.add_argument("--ga-verbose", action="store_true")
    parser.add_argument("--env-verbose", action="store_true",
                    help="In chi tiết từng action trong env.step() (debug).")

    args = parser.parse_args()

    # Load config
    io_json_input_file_path = args.config
    cfg = ReadJsonIOHandler(io_json_input_file_path).get_input()

    # Mode resolution
    mode = (args.mode or os.environ.get("SCHEDULER_MODE") or "dqn").lower()
    if mode not in ["dqn", "fcfs", "sa", "ga"]:
        raise ValueError(f"Unknown mode: {mode}. Use one of: dqn, fcfs, sa, ga.")

    # ---------------- RL branch ----------------
    if mode == "dqn":
        env = SimpleSplitSchedulingEnv(cfg)
        agent_name = mode

        model_path = args.dqn_model
        print("\nTraining DQN agent...")
        train_dqn(io_json_input_file_path, model_save_path=model_path, episodes=args.dqn_episodes)

        print("\nLoading trained model and running final evaluation...")
        agent = create_agent(agent_name, env.state_dim(), env.action_dim())
        agent.q_net.load_state_dict(torch.load(model_path))
        agent.q_net.eval()
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0  # disable exploration for eval

        io = ReadJsonIOHandler(io_json_input_file_path)
        result = run_episode(env, agent)
        io.show_output(result)
        return

    # ---------------- Static algorithms ----------------
    if mode == "fcfs":
        print("\nRunning FCFS (best-fit across machines, DP per job)...")
        res = schedule_fcfs(cfg, mode=args.split_mode)
        print(f"Mode: {args.split_mode}")
        print(f"Overall timespan: {res['makespan']} (start={res['overall']['start']}, finish={res['overall']['finish']})\n")
        print_assignments(res["assignments"])

        payload = {
            "algorithm": "fcfs",
            "mode": args.split_mode,
            "makespan": res["makespan"],
            "assignments": res["assignments"],
            "final_windows": res["final_windows"],
        }
        maybe_write_out(args.out, payload)
        return

    if mode == "sa":
        print("\nRunning SA ...")
        res = schedule_sa_config(
            cfg,
            mode=args.split_mode,
            Tmax=args.sa_Tmax,
            Tthreshold=args.sa_Tthreshold,
            alpha=args.sa_alpha,
            moves_per_T=args.sa_moves_per_T,
            seed=args.sa_seed,
            verbose=args.sa_verbose,
        )

        mk = res["best_makespan"]
        order = res["best_order"]
        assignments = res["assignments"]
        if mk is None or mk == 0:
            mk = compute_makespan_from_assignments(assignments)

        print(f"Mode: {args.split_mode}")
        print(f"Best makespan: {mk}")
        print(f"Best order: {order}\n")
        print_assignments(assignments)

        payload = {
            "algorithm": "sa",
            "mode": args.split_mode,
            "best_order": order,
            "makespan": mk,
            "assignments": assignments,
            "final_windows": res.get("final_windows", []),
        }
        maybe_write_out(args.out, payload)
        return

    if mode == "ga":
        print("\nRunning GA ...")
        params = GAParams(
            pop_size=args.ga_pop,
            generations=args.ga_gen,
            tournament_k=args.ga_tk,
            crossover_rate=args.ga_cx,
            mutation_rate_perm=args.ga_mut,
            # mutation_rate_mach giữ default=0.05 trong GAParams (TODO:có thể thêm tham số riêng sau)
            # elite giữ default=2
            seed=args.ga_seed,
        )

        ga_res = run_ga(cfg, params, mode=args.split_mode)

        print(f"Mode: {args.split_mode}")
        print(f"Makespan: {ga_res['makespan']}")
        print(f"Best order: {ga_res['order']}")
        print(f"Machine genes: {ga_res['machines']}")

        print("Machine finish times:")
        for m, ft in sorted(ga_res["machine_finish_times"].items()):
            print(f"  - M{m}: {ft}")

        print("Jobs per machine (in service order):")
        for m, jobs in sorted(ga_res["per_machine_jobs"].items()):
            print(f"  - M{m}: {jobs}")

        print("\nSegments per machine (job, start, end):")
        for m, segs in sorted(ga_res["per_machine_segments"].items()):
            print(f"  M{m}:")
            for (j, s, e) in segs:
                print(f"    job {j}: [{s}, {e})")

        assignments = []
        for m, segs in ga_res["per_machine_segments"].items():
            by_job = {}
            for (j, s, e) in segs:
                by_job.setdefault(j, []).append((0, s, e))  # win_idx=0 (không cần trong viz)
            for j, seg_list in by_job.items():
                finish = max(e for (_w, s, e) in seg_list)
                assignments.append({
                    "job": j,
                    "machine": m,
                    "segments": seg_list,
                    "finish": finish,
                    "fragments": len(seg_list),
                })

        if assignments:
            print("\nBest schedule (normalized assignments):")
            print_assignments(assignments)

        payload = {
            "algorithm": "ga",
            "mode": args.split_mode,
            "best_genome": {"order": ga_res["order"], "machine_genes": ga_res["machines"]},
            "makespan": ga_res["makespan"],
            "assignments": assignments,  # để viz/validator dùng ngay
            "final_windows": [],         # GA.py không trả; nếu cần, có thể thêm từ model
            "machine_finish_times": ga_res["machine_finish_times"],
            "per_machine_jobs": ga_res["per_machine_jobs"],
            "per_machine_segments": ga_res["per_machine_segments"],
        }
        maybe_write_out(args.out, payload)
        return


if __name__ == "__main__":
    main()
