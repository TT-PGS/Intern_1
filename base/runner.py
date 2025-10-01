from typing import Any, Dict
import time
import numpy as np
import torch
from base.agent_base import AgentBase
from envs.simple_split_env import SimpleSplitSchedulingEnv
from logs.log import write_log

def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype)

def convert_to_norm(state_raw, model, target_dim: int = None, step: int = None):
    job_scale = model.job_scale or max(1.0, max(model.processing_times))
    time_scale = model.time_scale or max(1.0, model.total_processing_time)

    state_norm = np.array(state_raw, dtype=np.float32)

    n_jobs = model.get_num_jobs()
    state_norm[:n_jobs] = state_norm[:n_jobs] / job_scale
    state_norm[n_jobs:] = state_norm[n_jobs:] / time_scale

    if target_dim is not None:
        if len(state_norm) < target_dim:
            state_norm = np.pad(state_norm, (0, target_dim - len(state_norm)), mode="constant")
        elif len(state_norm) > target_dim:
            state_norm = state_norm[:target_dim]

    if step is not None and step < 3:
        from logs.log import write_log
        raw_jobs = state_raw[:n_jobs]
        norm_jobs = state_norm[:n_jobs]
        write_log(f"[norm_debug] step={step} jobs_raw={raw_jobs} jobs_norm={norm_jobs}", "runner.log")
        # cũng log một số windows
        raw_win = state_raw[n_jobs:n_jobs+6]   # 3 windows đầu (start,end)
        norm_win = state_norm[n_jobs:n_jobs+6]
        write_log(f"[norm_debug] step={step} win_raw={raw_win} win_norm={norm_win}", "runner.log")

    return state_norm

def run_episode(
    env: SimpleSplitSchedulingEnv,
    agent: AgentBase,
    train: bool = True,
    max_steps: int = None,
    max_seconds: float = 30.0,
    stall_patience: int = 100,
    log_every: int = 100
) -> Dict[str, Any]:
    t0 = time.time()
    state_raw, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    steps = 0
    invalid_action_count = 0
    infeasible_count = 0
    mask_zero_count = 0

    # max_steps mặc định: tỉ lệ theo kích thước bài toán
    if max_steps is None:
        max_steps = int(10 * env.N_MAX * max(1, getattr(env, "W_MAX", 1)))

    # theo dõi tiến độ để phát hiện "stall"
    jobs_left = float(np.sum(env.jobs_remaining[:env.real_num_jobs]))
    stalled = 0

    while not (done or truncated):
        if (time.time() - t0) > max_seconds:
            truncated = True
            write_log(f"[runner] timeout after {steps} steps, elapsed={time.time()-t0:.1f}s", "runner.log")
            break
        
        state_norm = convert_to_norm(state_raw, env.model, target_dim=env.state_dim(), step=steps)

        # ---- Mask hành động ----
        if hasattr(env, "valid_action_mask"):
            mask = env.valid_action_mask()
        else:
            mask = env.build_action_mask()
        mask = np.asarray(mask, dtype=bool)

        if mask.size == 0 or not mask.any():
            mask_zero_count += 1
            total_reward += -5.0  # phạt nhẹ
            truncated = True
            write_log(f"[runner] mask_zero at step={steps}", "runner.log")
            break

        # ---- Chọn action (với fallback an toàn) ----
        try:
            # action = agent.select_action(state, mask)
            action = agent.select_action(state_norm, mask)
        except Exception as e:
            write_log(f"[runner] select_action error: {e}. Fallback -> random valid.", "runner.log")
            valid_idx = np.flatnonzero(mask)
            action = int(np.random.choice(valid_idx))

        # agent có thể trả về action bị mask (do bug) -> sửa lại
        action = int(action)
        if not mask[action]:
            invalid_action_count += 1
            valid_idx = np.flatnonzero(mask)
            action = int(np.random.choice(valid_idx))

        # ---- Bước môi trường ----
        next_state, reward, done, trunc_flag, info = env.step(action)
        next_state_norm = convert_to_norm(next_state, env.model, target_dim=env.state_dim(), step=steps)
        truncated = truncated or bool(trunc_flag)
        total_reward += float(reward)

        # đếm lỗi từ env để chẩn đoán
        if isinstance(info, dict) and info.get("infeasible"):
            infeasible_count += 1

        # ---- Học (nếu train) ----
        if train and hasattr(agent, "add_transition"):
            agent.add_transition(
                to_tensor(state_norm, torch.float32),
                torch.as_tensor(action, dtype=torch.long),
                float(reward),
                to_tensor(next_state_norm, torch.float32),
                bool(done or truncated),
            )
        if train and hasattr(agent, "update"):
            agent.update()

        state_raw = next_state
        steps += 1

        # ---- Watchdog "stall" (không tiến bộ) ----
        new_jobs_left = float(np.sum(env.jobs_remaining[:env.real_num_jobs]))
        if new_jobs_left < jobs_left - 1e-6:
            stalled = 0
            jobs_left = new_jobs_left
        else:
            stalled += 1
            if stalled >= stall_patience:
                truncated = True
                write_log(f"[runner] stalled {stalled} steps, jobs_left={jobs_left}, stop.", "runner.log")
                break

        # ---- Watchdog bước ----
        if steps >= max_steps:
            truncated = True
            write_log(f"[runner] reached max_steps={max_steps}", "runner.log")
            break

        # ---- Log định kỳ ----
        if (steps % log_every) == 0:
            write_log(
                f"[runner] steps={steps} R={total_reward:.2f} jobs_left={jobs_left:.0f} "
                f"mask_sum={int(mask.sum())} invalid={invalid_action_count} inf={infeasible_count} "
                f"elapsed={time.time()-t0:.1f}s",
                "runner.log"
            )

    if train and hasattr(agent, "update_target_net"):
        agent.update_target_net()

    return {
        "total_reward": float(total_reward),
        "steps": steps,
        "elapsed_s": float(time.time() - t0),
        "jobs_left": float(jobs_left),
        "invalid": int(invalid_action_count),
        "infeasible": int(infeasible_count),
        "mask_zero": int(mask_zero_count),
        "done": bool(done),
        "truncated": bool(truncated),
        "job_assignments": getattr(env, "job_assignments", {}),
    }
