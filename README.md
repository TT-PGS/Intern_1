# Intern_1

project for Intern 1

This repository aims to create a modular framework, not just a one-off solution, for scheduling problems using RL and DRL.  
By separating concerns into independent modules and interfaces, we make it easier to manage, extend, and experiment with various RL-based strategies for splittable task scheduling.  
In addition, the scheduling model itself is configurable and adaptable, allowing future modifications in problem definition or constraints.  

----------------------------------------------------------

```text
Intern_1/
│
├── base/                           # 💡 Các interface nền tảng, cấu trúc lõi
│   ├── agent_base.py               # AgentBase: giao diện chung cho agent
│   ├── io_handler.py               # IOHandler: input/output xử lý
│   ├── model.py                    # SchedulingModel: mô tả bài toán
│   ├── runner.py                   # run_episode(): vòng lặp tương tác agent-env
│
├── agents/                         # 🤖 Các agent cụ thể kế thừa từ AgentBase
│   ├── random_agent.py             # Agent chọn action ngẫu nhiên
│   ├── q_learning_agent.py         # Q-Learning Agent
│   └── agent_factory.py            # AgentFactory để load agent qua tên
│
├── envs/                           # 🌍 Môi trường RL cụ thể (có thể theo chuẩn gym)
│   └── simple_split_env.py         # SimpleSplitSchedulingEnv
│
├── configs/                        # ⚙️ Các file config YAML/JSON để cấu hình model
│   └── default_model.yaml          # Config cho số job, máy, size...
│
├── metrics/                        # 📈 Đánh giá kết quả (makespan, fairness...)
│   └── evaluator.py                # Tính metrics từ lịch/schedule
│
├── logger/                         # 📊 Module ghi dữ liệu training dưới dạng số liệu
│   ├── structured_logger.py        # Log dạng JSON-line, Prometheus-style, Influx, etc.
│   ├── exporters/
│   │   ├── json_exporter.py        # Ghi JSON theo thời gian thực
│   │   ├── clickhouse_exporter.py  # Gửi log vào ClickHouse qua HTTP
│   │   └── prometheus_exporter.py  # (option) nếu push gateway
│   └── schema/
│       └── reward_schema.json      # Chuẩn schema log (dễ mapping sau)
│
├── logs/                           # 📁 Tự sinh log (file json hoặc gửi thẳng DB)
│   └── 2025-07-22_12-30-00/
│       ├── reward_log.jsonl        # JSON line từng episode
│       └── metadata.yaml
│
├── main.py                         # 🎮 Điểm bắt đầu: chọn config, agent, chạy
└── requirements.txt                # Python dependencies
```

----------------------------------------------------------

                    ┌────────────────────────────┐
                    │      SchedulingModel       │  ◀── Có thể mở rộng mô hình (số job, máy...)
                    └────────────┬───────────────┘
                                 │
                           tạo environment
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │      env.reset()           │  ◀── Trả về trạng thái ban đầu
                    └────────────┬───────────────┘
                                 │
                          state: np.ndarray
                                 │
                                 ▼
                ┌────────────────────────────────────┐
                │         IOHandler.get_input()      │  ◀── Có thể override: xử lý state đầu vào
                └────────────┬───────────────────────┘
                                 │
                         state_input: np.ndarray
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │        AgentBase.select_action()       │  ◀── Thay thế agent ở đây: Random, Q, PPO...
              └────────────────┬───────────────────────┘
                               │
                           action: int
                               │
                               ▼
              ┌────────────────────────────────────┐
              │         IOHandler.show_output()    │  ◀── Có thể override: print/log/GUI
              └────────────────┬───────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    env.step(action)    │  ◀── Môi trường cập nhật state & trả reward
                  └──────┬────────┬────────┘
                         │        │
             next_state: np.ndarray   reward: float
                         │
                         ▼
       ┌─────────────────────────────────────────────────────┐
       │     AgentBase.update(state, action, reward, ...)    │  ◀── Cho agent học từ kết quả hành động
       └─────────────────────────────────────────────────────┘
                         │
                         ▼
                    Loop tới bước đầu nếu !done

--------------------------------------------------------------

| Mục        | Class / Function          | Mục đích                                            |
| ---------- | ------------------------- | --------------------------------------------------- |
| 🧠 Agent   | `AgentBase`               | Thay đổi chính sách hành động: random, Q-table, PPO |
| 🧾 Input   | `IOHandler.get_input()`   | Tiền xử lý input, ví dụ scale, encode,...           |
| 📤 Output  | `IOHandler.show_output()` | Thay vì `print()`, có thể log ra file, vẽ lên GUI   |
| 🧩 Mô hình | `SchedulingModel`         | Dễ thay đổi nếu có deadline, job ưu tiên,...        |
| 🔁 Loop    | `run_episode()`           | Dễ tích hợp training loop, logging, stop condition  |
