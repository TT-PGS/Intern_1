import os
import torch

from agents.agent_factory import create_agent
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.runner import run_episode
from base.io_handler import SimpleIOHandler, ReadJsonIOHandler
from agents.train_dqn import train_dqn

# ---- static algos ----
from staticAlgos.FCFS import schedule_fcfs
from staticAlgos.SA import solve_with_SA
from staticAlgos.GA import run_ga


def print_schedule(schedule):
    if not schedule:
        print("⚠️  Empty schedule")
        return
    print("📋 Schedule:")
    for sj in sorted(schedule, key=lambda x: (x["machine_id"], x["start"], x["job_id"])):
        print(
            f'  {sj["subjob_id"]}  | M{sj["machine_id"]} | '
            f'[{sj["start"]}, {sj["end"]})'
        )


if __name__ == "__main__":
    # ---- I/O config ----
    io_json_input_file_name = "splittable_jobs.json"
    io_input_folder = "./configs"
    io_json_input_file_path = os.path.join(io_input_folder, io_json_input_file_name)
    io = ReadJsonIOHandler(io_json_input_file_path)
    config = io.get_input()  # JSON dict, key "model" chứa tham số

    # ---- chọn chế độ chạy ----
    # one of: ["random", "dqn", "fcfs", "sa", "ga"]
    mode = os.environ.get("SCHEDULER_MODE", "dqn").lower()

    if mode in ["random", "dqn"]:
        # ====== RL pipeline (giữ nguyên) ======
        env = SimpleSplitSchedulingEnv(config["model"])
        agent_name = mode  # "random" hoặc "dqn"

        if agent_name == "random":
            agent = create_agent(agent_name, None, env.action_space.n)
        else:
            model_path = "qnet.pt"
            print("\n🚀 Training DQN agent...")
            train_dqn(io_json_input_file_path, model_save_path=model_path, episodes=10)

            print("\n📦 Loading trained model and running final evaluation...")
            agent = create_agent(agent_name, env.state_dim(), env.action_dim())
            agent.q_net.load_state_dict(torch.load(model_path))
            agent.q_net.eval()
            if hasattr(agent, "epsilon"):
                agent.epsilon = 0.0  # tắt randomness khi đánh giá

        result = run_episode(env, agent, io)
        io.show_output(result)

    elif mode == "fcfs":
        # ====== FCFS tĩnh ======
        # job_order mặc định theo id tăng dần; hoặc bạn có thể truyền list khác
        schedule, makespan = schedule_fcfs(config, job_order=None)
        print("\n✅ FCFS done.")
        print_schedule(schedule)
        print(f"🏁 Makespan: {makespan}")

    elif mode == "sa":
        # ====== Simulated Annealing (tham số theo bài báo) ======
        print("\n🔥 Running SA (Tmax=500, Tthreshold=1, alpha=0.99)...")
        result = solve_with_SA(
            config,
            Tmax=500.0,
            Tthreshold=1.0,
            alpha=0.99,
            # iters_per_temp=None -> mặc định = số job
            # random_seed=42,
            # progress_cb=lambda s: print(f'[SA] step={s["step"]} T={s["T"]:.2f} best={s["best_makespan"]}')
        )
        print("\n✅ SA done.")
        print(f"Best perm: {result['best_perm']}")
        print_schedule(result["best_schedule"])
        print(f"🏁 Best makespan: {result['best_makespan']}")

    elif mode == "ga":
        # ====== Genetic Algorithm (tham số theo bài báo) ======
        print("\n🧬 Running GA (pop=100, gens=500, pmx=0.8, swap=0.1)...")
        best_perm, schedule, ms = run_ga(
            config,
            pop_size=100,
            generations=500,
            tourn_size=5,
            cx_prob=0.8,
            mut_prob=0.1,
            seed=42,  # optional: tái lập kết quả
        )
        print("\n✅ GA done.")
        print(f"Best perm: {best_perm}")
        print_schedule(schedule)
        print(f"🏁 Best makespan: {ms}")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use one of: random, dqn, fcfs, sa, ga.")
