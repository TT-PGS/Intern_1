"""
This file will check the envs for CUDA lib, which will be used in training DRL methods
"""
import os
import sys
import time
import importlib

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def safe_import(name):
    try:
        mod = importlib.import_module(name)
        print(f"[OK] import {name} -> {getattr(mod, '__version__', 'no __version__')}")
        return mod
    except Exception as e:
        print(f"[FAIL] import {name}: {e}")
        sys.exit(1)

def main():
    section("1) PyTorch check")
    torch = safe_import("torch")
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda version (build):", torch.version.cuda)
        print("device name:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
    # small matmul on device to ensure kernels OK
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn((2048, 2048), device=device)
    b = torch.randn((2048, 2048), device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"matmul(2048x2048) on {device}: {time.time()-t0:.3f}s, mean={c.mean().item():.6f}")

    section("2) Gym check (CartPole-v1)")
    gym = safe_import("gym")
    # Gym 0.26+ API: reset() -> (obs, info), step() -> (obs, reward, terminated, truncated, info)
    env = gym.make("CartPole-v1")
    try:
        obs, info = env.reset(seed=42)
    except TypeError:
        # fallback if old API
        obs = env.reset(seed=42)
        info = {}
    print("env.observation_space:", env.observation_space)
    print("env.action_space:", env.action_space)

    # run a few random steps
    total_r = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, r, terminated, truncated, info = step_out
            done = terminated or truncated
        else:  # older API
            obs, r, done, info = step_out
        total_r += r
        if done:
            try:
                obs, info = env.reset()
            except TypeError:
                obs = env.reset()
    env.close()
    print("CartPole 10 random steps total reward:", total_r)

    section("3) Stable-Baselines3 (PPO quick smoke test)")
    sb3 = safe_import("stable_baselines3")
    from stable_baselines3 import PPO

    # small / quick config to avoid heavy training
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=128, n_epochs=1,
                learning_rate=3e-4, verbose=0, device="auto")  # device="auto" -> cuda if available
    print("SB3 model device:", next(model.policy.parameters()).device)

    # do a tiny learn to ensure forward/backward works
    t0 = time.time()
    model.learn(total_timesteps=1000, progress_bar=False)
    dt = time.time() - t0
    print(f"SB3 PPO.learn(1000) finished in {dt:.2f}s")

    # quick predict
    obs, _ = env.reset(seed=123)
    action, _ = model.predict(obs, deterministic=True)
    print("Predict one action OK:", action)
    env.close()

    section("ALL CHECKS PASSED ✅")

if __name__ == "__main__":
    main()
