import os
import torch

from agents.agent_factory import create_agent
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.runner import run_episode
from base.io_handler import SimpleIOHandler, ReadJsonIOHandler
from agents.train_dqn import train_dqn

if __name__ == "__main__":
    # io = SimpleIOHandler()
    # model = io.get_input()
    # env = SimpleSplitSchedulingEnv(model)
    # agent = RandomAgent(env.action_space.n)

    # result = run_episode(env, agent, io)
    # io.show_output(result)

    io_json_input_file_name = "splittable_jobs.json"
    io_input_folder = "./configs"
    io_json_input_file_path = os.path.join(io_input_folder, io_json_input_file_name)
    io = ReadJsonIOHandler(io_json_input_file_path)
    model = io.get_input()

    env = SimpleSplitSchedulingEnv(model)
    # agent_name is in ["random", "dqn"]
    agent_name = "dqn"

    if agent_name == "random":
        agent = create_agent(agent_name, None, env.action_space.n)
    else:
        # agent = create_agent(agent_name, env.state_dim(), env.action_dim(),
        #                     buffer_size=10000, batch_size=64)
        model_path = "qnet.pt"
        # Step 1: Train the model
        print("\n🚀 Training DQN agent...")
        train_dqn(io_json_input_file_path, model_save_path=model_path, episodes=10)

        # Step 2: Load model and run inference
        print("\n📦 Loading trained model and running final evaluation...")
        agent = create_agent(agent_name, env.state_dim(), env.action_dim())

        agent.q_net.load_state_dict(torch.load(model_path))
        agent.q_net.eval()  # Set to evaluation mode

        # Đảm bảo không dùng epsilon-greedy nữa (tắt randomness)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0  # Luôn chọn hành động tốt nhất (argmax)

    result = run_episode(env, agent, io)
    io.show_output(result)
