# Roadmap

## 🎯 MỤC TIÊU 1: Xây dựng framework RL scheduling đơn giản

### 🔹 Giai đoạn A.1 – Mô hình bài toán & môi trường

**✔ Files cần hoàn thiện:**

* `base/model.py`: class `SchedulingModel`
* `envs/simple_split_env.py`: môi trường `SimpleSplitSchedulingEnv`

**🎯 Yêu cầu:**

* Mô hình có thể config dễ dàng (`num_jobs`, `num_machines`, `max_job_size`)
* Môi trường trả state, nhận action, tính reward

**🧪 Cách test:**

* Tạo `test_env.py`: reset + step + in ra reward/makespan

---

### 🔹 Giai đoạn A.2 – Interface Agent + Agent Random

**✔ Files cần hoàn thiện:**

* `base/agent_base.py`
* `agents/random_agent.py`

**🎯 Yêu cầu:**

* Agent tuân theo `AgentBase`
* Agent random dùng `action_dim` để chọn action

**🧪 Cách test:**

* Dùng `main.py` chạy env + random agent → thấy reward in ra là đủ

---

### 🔹 Giai đoạn A.3 – I/O handler + run loop

**✔ Files cần hoàn thiện:**

* `base/io_handler.py`
* `base/runner.py`

**🎯 Yêu cầu:**

* `get_input()` và `show_output()` dễ override
* `run_episode()` thực hiện RL loop: reset, step, agent.select/update, return total reward

**🧪 Cách test:**

* Gọi `run_episode(env, agent, io)` từ `main.py`, in ra tổng reward

---

### 🔹 Giai đoạn A.4 – Agent factory + config

**✔ Files cần hoàn thiện:**

* `agents/agent_factory.py`
* `configs/default_model.yaml`

**🎯 Yêu cầu:**

* Agent được load từ config (ví dụ `agent: random`)
* Config dễ chỉnh số job, số máy

**🧪 Cách test:**

* Sửa `main.py` gọi qua config, đổi agent dễ dàng

---

### 🔹 Giai đoạn A.5 – Log đơn giản

**✔ Files cần viết:**

* `logger/structured_logger.py` (chưa cần exporters)
* Ghi JSON/line mỗi episode

**🧪 Cách test:**

* Sau mỗi `run_episode()`, log ra `logs/YYYYMMDD-HHMM/train.jsonl`

---

## ✅ Kết quả Mục tiêu 1

* Chạy `main.py` → gọi được đúng env, agent, log reward theo tập
* Output rõ ràng, agent có thể dễ dàng thay

---

## 🎯 MỤC TIÊU 2: Tích hợp Deep Q-Learning (DQN)

### 🔹 Giai đoạn B.1 – Hiện thực agent DQN

**✔ Files:**

* `agents/dqn_agent.py` (mới)
* Dùng `torch`, `replay buffer`, `target network`

**🎯 Yêu cầu:**

* Network chọn action theo Q-value
* Update Q theo Bellman (target Q)
* Có epsilon-greedy

---

### 🔹 Giai đoạn B.2 – Tích hợp agent vào factory

* Sửa `agent_factory.py` để thêm `"dqn": DQNAgent(...)`

---

### 🔹 Giai đoạn B.3 – Tuning & log

* Ghi log reward qua nhiều episode
* Có thể lưu mô hình (checkpoint)

---

## 🎯 MỤC TIÊU 3: Tích hợp PPO (policy-based)

### 🔹 Giai đoạn C.1 – Dùng `stable-baselines3` (hoặc viết PPO riêng)

* Viết `agents/ppo_agent.py`: wrapper của SB3 PPO agent

**🎯 Yêu cầu:**

* Dùng chuẩn `gym.Env` để tương thích với SB3
* Huấn luyện qua `model.learn()`, predict action

---

### 🔹 Giai đoạn C.2 – So sánh kết quả

* Ghi log reward → Grafana/ClickHouse
* So sánh reward/makespan giữa DQN vs PPO

---

## ✅ Tổng kết: Các mốc đánh giá

| Mục tiêu   | Thành phẩm rõ ràng              | Đo lường kết quả              |
| ---------- | ------------------------------- | ----------------------------- |
| Mục tiêu 1 | `main.py` chạy random agent     | Có reward, makespan, log JSON |
| Mục tiêu 2 | `DQNAgent` tích hợp và training | Log reward tăng dần theo tập  |
| Mục tiêu 3 | `PPOAgent` tích hợp             | So sánh PPO vs DQN bằng log   |
