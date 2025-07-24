from agents.random_agent import RandomAgent
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.runner import run_episode
from base.io_handler import SimpleIOHandler

if __name__ == "__main__":
    io = SimpleIOHandler()
    model = io.get_input()
    env = SimpleSplitSchedulingEnv(model)
    agent = RandomAgent(env.action_space.n)

    result = run_episode(env, agent, io)
    io.show_output(result)
