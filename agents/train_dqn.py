import os
import torch
from agents.agent_factory import create_agent
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.io_handler import ReadJsonIOHandler
from base.runner import run_episode
import matplotlib.pyplot as plt


def train_dqn(model_config_path, model_save_path="qnet.pt", episodes=500):
    io = ReadJsonIOHandler(model_config_path)
    model = io.get_input()
    env = SimpleSplitSchedulingEnv(model)

    agent = create_agent("dqn", env.state_dim(), env.action_dim())
    rewards = []

    best_reward = float("-inf")

    for ep in range(episodes):
        result = run_episode(env, agent, io)
        rewards.append(result["total_reward"])
        print(f"Ep {ep:03d} | Reward: {result['total_reward']:.2f}")

        if hasattr(agent, "epsilon"):
            agent.epsilon = max(agent.epsilon * 0.995, 0.05)
        
        if result["total_reward"] > best_reward:
            best_reward = result["total_reward"]
            torch.save(agent.q_net.state_dict(), model_save_path)
            print(f"📌 Saved better model at Ep {ep} with reward {best_reward:.2f} to {model_save_path}")

    print(f"Training complete. Best reward: {best_reward:.2f}. Model saved to {model_save_path}")
    return rewards


if __name__ == "__main__":
    config_path = os.path.join("configs", "splittable_jobs.json")
    rewards = train_dqn(config_path, model_save_path="qnet.pt", episodes=500)

    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("DQN Training Reward Over Time")
    # plt.grid(True)
    # plt.show()