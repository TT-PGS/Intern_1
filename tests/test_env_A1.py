# import pytest
# import numpy as np
# from envs.simple_split_env import SimpleSplitSchedulingEnv
# from base.model import SchedulingModel

# def test_env_reset_returns_correct_state():
#     model = SchedulingModel(num_jobs=3, num_machines=2, max_job_size=10)
#     env = SimpleSplitSchedulingEnv(model)
#     state = env.reset()

#     assert isinstance(state, np.ndarray)
#     assert state.shape == (5,)  # 3 jobs + 2 machines
#     assert all(state[:3] == 10)  # job sizes
#     assert all(state[3:] == 0)   # machine times

# def test_env_step_progresses_state():
#     model = SchedulingModel(3, 2, 10)
#     env = SimpleSplitSchedulingEnv(model)
#     env.reset()
#     action = 0  # job_id=0, machine_id=0

#     next_state, reward, done, _ = env.step(action)
#     assert isinstance(next_state, np.ndarray)
#     assert isinstance(reward, float)
#     assert isinstance(done, bool)
#     assert next_state[0] == 8  # job 0 giảm 2 đơn vị
#     assert next_state[3] == 2  # máy 0 tốn 2 đơn vị

# def test_env_done_when_all_jobs_completed():
#     model = SchedulingModel(1, 1, 3)
#     env = SimpleSplitSchedulingEnv(model)
#     env.reset()

#     state, reward, done, _ = env.step(0)  # chạy job 0 trên máy 0
#     assert not done
#     state, reward, done, _ = env.step(0)  # chạy phần còn lại
#     assert done
#     assert reward == -3.0  # makespan = 4 (2+2 do split 2 lần)

import pytest
import numpy as np
from envs.simple_split_env import SimpleSplitSchedulingEnv
from base.model import SchedulingModel

def create_test_model():
    processing_times = [4, 6]  # job 0 = 4 units, job 1 = 6 units
    split_min = 2
    time_windows = {
        0: [(0, 5), (6, 10)],      # máy 0 có 2 windows
        1: [(1, 4), (5, 9)]        # máy 1 có 2 windows
    }
    return SchedulingModel(
        num_jobs=2,
        num_machines=2,
        processing_times=processing_times,
        split_min=split_min,
        time_windows=time_windows
    )

def test_env_reset_returns_correct_state():
    model = create_test_model()
    env = SimpleSplitSchedulingEnv(model)
    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)  # 2 jobs + 2 machines
    assert all(state[:2] == model.processing_times)
    assert all(state[2:] == 0)

def test_env_step_progresses_state():
    model = create_test_model()
    env = SimpleSplitSchedulingEnv(model)
    env.reset()
    action = 0  # job_id = 0, machine_id = 0

    next_state, reward, done, _ = env.step(action)
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert next_state[0] < model.processing_times[0]  # job 0 should reduce
    assert next_state[2] > 0  # machine 0 time should increase

def test_job_assignment_consistency():
    model = create_test_model()
    env = SimpleSplitSchedulingEnv(model)
    env.reset()

    # Assign job 0 to machine 0
    env.step(0)  # job 0, machine 0
    # Try assigning job 0 to machine 1 – should be penalized
    _, reward, _, _ = env.step(1)  # job 0, machine 1
    assert reward < 0
