# Intern_1

## Project for Intern 1

This repository aims to create a modular framework, not just a one-off solution, for scheduling problems using RL and DRL.  
By separating concerns into independent modules and interfaces, we make it easier to manage, extend, and experiment with various RL-based strategies for splittable task scheduling.  
In addition, the scheduling model itself is configurable and adaptable, allowing future modifications in problem definition or constraints.

---

```text
Intern_1/
│
├── base/                           # 💡 Các interface nền tảng, cấu trúc lõi
│   ├── agent_base.py               # AgentBase: giao diện chung cho agent
│   ├── io_handler.py               # IOHandler: xử lý input/output (load file json, ...)
│   ├── model.py                    # SchedulingModel: mô tả bài toán
│   └── runner.py                   # run_episode(): vòng lặp tương tác agent-env
│
├── agents/                         # 🤖 Các agent cụ thể kế thừa từ AgentBase
│   ├── dqn_agent.py                # 🤖 Định nghĩa DQNAgent
│   ├── q_net.py                    # 🧠 Định nghĩa mạng nơ-ron Q-Network
│   ├── replay_buffer.py            # 📦 Replay buffer (experience replay)
│   ├── train_dqn.py                # 🎓 Huấn luyện DQNAgent qua nhiều tập
│   └── agent_factory.py            # AgentFactory để load agent qua tên (hiện tại: "dqn")
│
├── envs/                           # 🌍 Môi trường RL cụ thể (theo chuẩn gym)
│   └── simple_split_env.py         # SimpleSplitSchedulingEnv
│
├── datasets/                        # 📁 Chứa các tập samples
│   └── gen_twspwjp_dataset.py       # file gen các tập samples (thoả các điều kiện của testcase trong bài báo)
│
├── configs/                        # ⚙️ Chứa các config
│   ├── SA_config.json              # Config của SA
│   └── GA_config.json              # Config của GA
│
├── metrics/                        # 📈 Chứa kết quả chạy của các giải thuật trong datasets và các phân tích đánh giá
│   └── convert_results_to_csv.py   # tạo file csv từ results
│
├── logger/                         # 📊 Module ghi dữ liệu training dưới dạng số liệu - Hiện tại chưa có #TODO
│   ├── structured_logger.py        # Log dạng JSON-line, Prometheus-style, Influx, etc.
│   ├── exporters/
│   │   ├── json_exporter.py        # Ghi JSON theo thời gian thực
│   │   ├── clickhouse_exporter.py  # Gửi log vào ClickHouse qua HTTP
│   │   └── prometheus_exporter.py  # (option) nếu push gateway
│   └── schema/
│       └── reward_schema.json      # Chuẩn schema log (dễ mapping sau)
│
├── logs/                           # 📁 Tự sinh log (file json hoặc gửi thẳng DB) - Hiện tại chưa có #TODO
│   └── 2025-07-22_12-30-00/
│       ├── reward_log.jsonl        # JSON line từng episode
│       └── metadata.yaml
│
├── main.py                         # 🎮 Điểm bắt đầu: chọn config, agent, chạy
└── requirements.txt                # Python dependencies
```

# How to run

## Run cmd

source /home/a/PGS/Intern_1/MinSplit/bin/activate

#### 1. For running DQN (not success yet)

python -m main --mode dqn --config <input .json file> --dqn-episodes 10 --dqn-model qnet.pt

#### 2. For running GA

python -m main --mode ga --config <input .json file> --split-mode assign --ga-pop 40 --ga-gen 200 --ga-cx 0.9 --ga-mut 0.2 --ga-tk 5 --ga-seed 42 --ga-verbose --out ./results_ga.json

#### 3. For running SA

python -m main --mode sa --config <input .json file> --split-mode assign --sa-Tmax 500 --sa-Tthreshold 1 --sa-alpha 0.99 --sa-moves-per-T 50 --sa-seed 42 --out ./results_sa.json

#### 4. For running FCFS

python -m main --mode fcfs --config <input .json file> --split-mode assign --out ./results_fcfs.json
