from base.model import SchedulingModel
from envs.simple_split_env import SimpleSplitSchedulingEnv
from agents.random_agent import RandomAgent

# Step 1: Khởi tạo model
model = SchedulingModel(
    num_jobs=2,
    num_machines=2,
    processing_times=[5, 7],
    split_min=2,
    time_windows={
        0: [[0, 5], [6, 10]],
        1: [[1, 4], [5, 9]]
    }
)

# Step 2: Khởi tạo môi trường
env = SimpleSplitSchedulingEnv(model)
state = env.reset()

# Step 3: Tạo random agent
agent = RandomAgent(action_dim=env.action_space.n)

# Step 4: Chạy vòng lặp đơn giản
total_reward = 0
done = False

step_count = 0
max_steps = 20

while not done and step_count < max_steps:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    state = next_state
    total_reward += reward
    step_count += 1
    env.render()

print("✅ Tổng reward:", total_reward)
