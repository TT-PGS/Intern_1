import os, glob
import json
import argparse
import torch
import time
from copy import deepcopy

from logs.log import log_info

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

def write_output_results(out_path: str, payload: dict):
    if not out_path:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved result to: {out_path}")

# ---------------------------
# Main
# ---------------------------

@log_info
def main():
    parser = argparse.ArgumentParser(description="Unified entrypoint: RL (random/DQN) + static (FCFS/SA/GA)")
    # Common
    parser.add_argument("--mode", type=str, default=None,
                        help="one of: dqn, fcfs, sa, ga. If omitted, use env SCHEDULER_MODE or default 'dqn'.")
    parser.add_argument("--config", type=str, default="./datasets/splittable_jobs.json",
                        help="Path to JSON config containing 'model'")
    parser.add_argument("--out", type=str, default="",
                        help="Optional path to write JSON result (assignments, makespan, etc.)")
    parser.add_argument("--split-mode", type=str, default="assign", choices=["timespan", "leftover", "assign"],
                        help="DP mode for splitting jobs when assigning to machines")

    # RL options
    parser.add_argument("--dqn-episodes", type=int, default=10, help="DQN training episodes")
    parser.add_argument("--dqn-model", type=str, default="qnet.pt", help="Path to save/load DQN weights")
    parser.add_argument("--dqn-train", action="store_true", help="Enable DQN training")

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

    cfg = None
    # Load config - common for all modes, except all_1/all_2
    if args.mode not in ["all_1", "all_2"]:
        io_json_input_file_path = args.config
        cfg = ReadJsonIOHandler(io_json_input_file_path).get_input()

    # Mode resolution
    mode = (args.mode).lower()
    if mode not in ["dqn", "fcfs", "sa", "ga", "all_1", "all_2"]:
        raise ValueError(f"Unknown mode: {mode}. Use one of: dqn, fcfs, sa, ga, all_1, all_2.")

    # ---------------- RL branch ----------------
    if mode == "dqn":
        env = SimpleSplitSchedulingEnv(cfg)
        agent_name = mode

        model_file_path = args.dqn_model
        enable_training = args.dqn_train

        if enable_training or not os.path.isfile(model_file_path):
            print("\nTraining DQN agent...")
            train_dqn(io_json_input_file_path, model_save_path=model_file_path, episodes=args.dqn_episodes)

        print("\nLoading trained model and running final evaluation...")
        agent = create_agent(agent_name, env.state_dim(), env.action_dim())
        ckpt = torch.load(model_file_path, map_location="cpu")
        assert ckpt["state_dim"] == env.state_dim(), f"state_dim mismatch: {ckpt['state_dim']} vs {env.state_dim()}"
        assert ckpt["action_dim"] == env.action_dim(), f"action_dim mismatch: {ckpt['action_dim']} vs {env.action_dim()}"
        agent.q_net.load_state_dict(ckpt["model_state_dict"])
        agent.q_net.eval()
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0  # disable exploration for eval

        io = ReadJsonIOHandler(io_json_input_file_path)
        result = run_episode(env, agent, train=False)
        io.show_output(result)

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
        write_output_results(args.out, payload)

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
        write_output_results(args.out, payload)

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
        write_output_results(args.out, payload)
    
    if mode == "all_1":
        """
            Run all algorithms in sequence: FCFS, SA, GA with all datasets.
            Note: DQN is excluded due to its training time and variability.
            This mode is useful for benchmarking and comparing static algorithms.
            And getting a comprehensive overview of performance across different methods
            As well as generating results for reports or analysis.
            Then those results can be used as best-known solutions for DQN and any RDL methods.
        """
        split_mode = "assign"
        datasets_folder = "datasets/"
        output_folder = "metrics/"
        sa_config = "configs/SA_configs.json"
        ga_config = "configs/GA_configs.json"

        with open(sa_config) as f:
            sa_params = json.load(f)

        with open(ga_config) as f:
            ga_params = json.load(f)

        for file_name in glob.glob(os.path.join(datasets_folder, "**", "*.json"), recursive=True):
            '''
            Create subfolder in output_folder based on the dataset's seed (parent folder name).
            This organizes results by dataset for easier analysis.
            E.g., datasets/20250616/42.json -> metrics/20250616/42.json
            '''
            seed_of_input = file_name.split("/")[-2]
            output_subfolder = os.path.join(output_folder, seed_of_input)
            os.makedirs(output_subfolder, exist_ok=True)
            output_file_path = os.path.join(output_subfolder, os.path.basename(file_name))

            '''
                Run FCFS, SA, GA in sequence for the current dataset.
                Collect results and write to a single JSON file in output_subfolder.
            '''
            print(f"\n\n=== Processing dataset: {file_name} ===")
            io_json_input_file_path = file_name
            cfg = ReadJsonIOHandler(io_json_input_file_path).get_input()
            cfg_fcfs = deepcopy(cfg)
            cfg_sa = deepcopy(cfg)
            cfg_ga = deepcopy(cfg)
            lower_bound = cfg.get_lower_bound()

            '''================ FCFS ================'''
            start_fcfs = time.time()
            res_fcfs = schedule_fcfs(cfg_fcfs, mode=split_mode)
            end_fcfs = time.time()
            processing_time_fcfs = end_fcfs - start_fcfs
            percentage_gap_fcfs = (res_fcfs["makespan"] - lower_bound) * 100 / lower_bound

            '''================ SA ================'''
            start_sa = time.time()
            res_sa = schedule_sa_config(
                cfg_sa,
                mode=split_mode,
                Tmax=sa_params["sa"].get("Tmax"),
                Tthreshold=sa_params["sa"].get("Tthreshold"),
                alpha=sa_params["sa"].get("alpha"),
                moves_per_T=sa_params["sa"].get("moves_per_T"),
                seed=seed_of_input,
                verbose=sa_params["sa"].get("verbose"),
            )
            end_sa = time.time()
            processing_time_sa = end_sa - start_sa
            percentage_gap_sa = (res_sa["best_makespan"] - lower_bound) * 100 / lower_bound

            '''================ GA ================'''
            params = GAParams(
                pop_size=ga_params["ga"].get("pop_size"),
                generations=ga_params["ga"].get("generations"),
                tournament_k=ga_params["ga"].get("tournament_k"),
                crossover_rate=ga_params["ga"].get("crossover_rate"),
                mutation_rate_perm=ga_params["ga"].get("mutation_rate_perm"),
                seed=seed_of_input,
            )
            start_ga = time.time()
            res_ga = run_ga(cfg_ga, params, mode=split_mode)
            end_ga = time.time()
            processing_time_ga = end_ga - start_ga
            percentage_gap_ga = (res_ga["makespan"] - lower_bound) * 100 / lower_bound

            payload = {
                "dataset": os.path.basename(file_name),
                "lower_bound": lower_bound,
                "split_mode": split_mode,
                "fcfs_results": {
                    "makespan": res_fcfs["makespan"],
                    "assignments": res_fcfs["assignments"],
                    "final_windows": res_fcfs["final_windows"],
                    "processing_time_milliseconds": processing_time_fcfs * 1000,
                    "percentage_gap": percentage_gap_fcfs,
                },
                "sa_results": {
                    "best_order": res_sa.get("best_order"),
                    "makespan": res_sa.get("best_makespan"),
                    "assignments": res_sa.get("assignments"),
                    "final_windows": res_sa.get("final_windows"),
                    "processing_time_milliseconds": processing_time_sa * 1000,
                    "percentage_gap": percentage_gap_sa,
                },
                "ga_results": {
                    "best_genome": {
                        "order": res_ga.get("order"),
                        "machine_genes": res_ga.get("machines"),
                    },
                    "makespan": res_ga.get("makespan"),
                    "assignments": res_ga.get("assignments"),
                    "machine_finish_times": res_ga.get("machine_finish_times"),
                    "per_machine_jobs": res_ga.get("per_machine_jobs"),
                    "per_machine_segments": res_ga.get("per_machine_segments"),
                    "processing_time_milliseconds": processing_time_ga * 1000,
                    "percentage_gap": percentage_gap_ga,
                },
            }

            write_output_results(output_file_path, payload)

    if mode == "all_2":
        """
            Run all RDL algorithms in sequence: DQN (for now) with all datasets.
            Note: DQN training time may be significant, so consider this when using this mode.
            Ensure that DQN training parameters are set appropriately for each dataset.
            And the trained models are saved with distinct names to avoid overwriting.
            This mode is useful for evaluating DQN performance across different datasets.
            As well as generating results for reports or analysis.
        """
        pass


if __name__ == "__main__":
    main()
