"""Unified training script for the dual-scene highway RL project.

Examples
--------
    python -m src.train --scene compound --algo ppo --use-ttc --n-envs 8
    python -m src.train --scene roundabout  --algo ppo --use-ttc
    python -m src.train --scene intersection --algo dqn --use-ttc --noise 0.1
    python -m src.train --scene roundabout  --algo ppo --no-ttc   # baseline ablation

Outputs (under --log-dir): <tag>.zip model, <tag>_reward.png curve, and
periodic snapshots in <tag>_ckpt/.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: no GUI window needed
import matplotlib.pyplot as plt
import pandas as pd
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from . import configs
from .custom_envs import make_env

def parse_args():
    p = argparse.ArgumentParser(description="Train a highway RL agent.")
    p.add_argument("--scene", choices=["roundabout", "intersection", "compound"],
                   default="roundabout")
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--use-ttc", dest="use_ttc", action="store_true",
                   help="enable TTC safety reward (default)")
    p.add_argument("--no-ttc", dest="use_ttc", action="store_false",
                   help="disable TTC reward (baseline ablation)")
    p.set_defaults(use_ttc=True)
    p.add_argument("--crash-terminate", dest="crash_terminate",
                   action="store_true",
                   help="(A/B) end the episode when an agent crashes instead of "
                        "rebooting it in place, so a collision costs the whole "
                        "remaining return. Default OFF keeps the loop-forever "
                        "design; the trained policy is evaluated with reboot.")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Gaussian observation-noise std (sensor robustness)")
    p.add_argument("--n-envs", type=int, default=8,
                   help="parallel envs for PPO via SubprocVecEnv (compound is "
                        "CPU-bound and runs ~2 FPS on one env; 8 parallel envs "
                        "roughly multiply throughput. Ignored for DQN, which is "
                        "single-env).")
    p.add_argument("--timesteps", type=int, default=100000,
                   help="total training timesteps (default: 100000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="torch device for training (auto=prefer GPU if present)")
    p.add_argument("--suffix", type=str, default="",
                   help="appended to the output tag, so V1/V2/V3 experiments "
                        "write separate models, curves and checkpoints instead of "
                        "overwriting each other (e.g. --suffix _v3)")
    p.add_argument("--log-dir", default="results")
    return p.parse_args()


class _StepCheckpoint(BaseCallback):
    """Save one checkpoint every `save_freq` TRUE timesteps.

    NB: SB3's CheckpointCallback triggers on ``self.n_calls``, i.e. every
    ``on_step`` invocation. With a VecEnv each invocation steps ALL n_envs
    environments at once, so ``save_freq=20000`` would actually fire every
    20000 * n_envs timesteps (8x too sparse). Keying on ``model.num_timesteps``
    keeps the interval exact regardless of n_envs.
    """

    def __init__(self, save_freq: int, save_path: str, prefix: str,
                 verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.prefix = prefix
        self._last_bucket = 0

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        bucket = n // self.save_freq
        if bucket > self._last_bucket:
            self._last_bucket = bucket
            self.model.save(
                os.path.join(self.save_path, f"{self.prefix}_"
                                              f"{bucket * self.save_freq}_steps"))
        return True


class _ProgressCallback(BaseCallback):
    """Dependency-free training progress display.

    ``model.num_timesteps`` already counts all vectorised workers, so the
    displayed total is the aggregate training budget on Windows and Ubuntu.
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self._last_percent = -1

    def _on_training_start(self) -> None:
        print(f"[train] progress: [------------------------------] 0/"
              f"{self.total_timesteps} steps (0%)",
              end="", flush=True)

    def _on_step(self) -> bool:
        current = min(int(self.model.num_timesteps), self.total_timesteps)
        percent = min(100, int(current * 100 / self.total_timesteps))
        # Print at most once per percentage point.  This keeps the integrated
        # terminal readable even when eight environments step rapidly.
        if percent != self._last_percent:
            filled = int(percent * 30 / 100)
            bar = "#" * filled + "-" * (30 - filled)
            print(f"\r[train] progress: [{bar}] {current}/"
                  f"{self.total_timesteps} steps ({percent}%)",
                  end="", flush=True)
            self._last_percent = percent
        return True

    def _on_training_end(self) -> None:
        print()


def _make_env_factory(scene: str, noise: float, seed: int):
    """Build a fresh, seeded env for one SubprocVecEnv worker."""
    def _factory():
        # A worker that lets numpy/torch grab all cores just thrashes the other
        # workers. Pin BLAS + torch to 1 thread each so the CPU budget is shared
        # cleanly (these env vars are inherited by the spawned subprocesses).
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        return make_env(scene, noise_std=noise, seed=seed, render_mode=None)
    return _factory


def _build_env(scene: str, noise: float, seed: int, n_envs: int, monitor_path: str):
    """Return (env, n_envs). PPO with n_envs>1 -> SubprocVecEnv + VecMonitor."""
    if n_envs > 1:
        venv = SubprocVecEnv(
            [_make_env_factory(scene, noise, seed + i) for i in range(n_envs)])
        return VecMonitor(venv, filename=monitor_path), n_envs
    env = Monitor(make_env(scene, noise_std=noise, seed=seed, render_mode=None),
                  filename=monitor_path)
    return env, 1


def _build_model(algo: str, env, seed: int, n_envs: int, device: str = "auto"):
    # A small 2x64 net is too weak for a 150-dim obs / 25-action joint space; a
    # wider net + a bit of entropy (ent_coef) reliably learn negotiation.
    policy_kwargs = dict(net_arch=[256, 256])
    if algo == "ppo":
        # n_steps is per-env; keep ~2048 transitions per update so the batch and
        # minibatch count stay fixed whatever n_envs is. For the default n_envs=8
        # this is exactly 256 (the value the vec config was tuned around).
        n_steps = max(256, 2048 // max(1, n_envs))
        return PPO("MlpPolicy", env, verbose=1, seed=seed,
                   n_steps=n_steps, batch_size=256, n_epochs=5,
                   learning_rate=2e-4, gamma=0.99, gae_lambda=0.95,
                   # ent_coef 0.02: the joint action space is 25 discrete and the
                   # useful manoeuvres are a small fraction; a little entropy keeps
                   # the policy from collapsing onto "both agents just slow down".
                   ent_coef=0.02,
                   # vf_coef kept at 0.5 (not lowered): the critic is already weak
                   # (explained_variance went negative at times); thinning it further
                   # would hurt more than it helps.
                   vf_coef=0.5, max_grad_norm=0.5,
                   policy_kwargs=policy_kwargs, device=device)
    return DQN("MlpPolicy", env, verbose=1, seed=seed,
               learning_rate=1e-4, buffer_size=100_000,
               learning_starts=2000, exploration_fraction=0.15,
               exploration_final_eps=0.05,
               policy_kwargs=policy_kwargs, device=device)


def main():
    args = parse_args()
    configs.USE_TTC = args.use_ttc  # toggle TTC reward before env construction
    configs.CRASH_TERMINATE = args.crash_terminate  # A/B: crash ends the episode

    # Pin compute threads BEFORE building the VecEnv so each spawned worker
    # inherits 1 thread/worker (see _make_env_factory comment).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    os.makedirs(args.log_dir, exist_ok=True)
    tag = f"{args.scene}_{args.algo}" + ("_ttc" if args.use_ttc else "_base")
    if args.suffix:
        tag += args.suffix
    if args.crash_terminate:
        tag += "_crashterm"
    if args.noise > 0:
        tag += f"_noise{args.noise}"

    n_envs = args.n_envs if args.algo == "ppo" else 1
    monitor_path = os.path.join(args.log_dir, tag + "_monitor.csv")
    env, n_envs = _build_env(args.scene, args.noise, args.seed, n_envs,
                             monitor_path)
    print(f"[train] scene={args.scene} algo={args.algo} n_envs={n_envs}")

    model = _build_model(args.algo, env, args.seed, n_envs, args.device)

    # Snapshots every 20k TRUE timesteps so a long run that is killed mid-way is
    # not lost. Deliberately NOT auto-resumed: re-launching always starts a fresh
    # model at step 0 -- checkpoints are a safety net only (recover via evaluate.py).
    ckpt_dir = os.path.join(args.log_dir, tag + "_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = _StepCheckpoint(save_freq=20000, save_path=ckpt_dir, prefix=tag)
    print(f"[train] checkpoint every 20k true timesteps -> {ckpt_dir}")

    print(f"[train] learning {args.timesteps} steps across {n_envs} envs "
          f"(target {args.timesteps} total)...")
    # Always show progress through the dependency-free callback.
    progress_cb = _ProgressCallback(args.timesteps)
    print("[train] progress bar: ON (built-in callback)")
    model.learn(total_timesteps=args.timesteps, progress_bar=False,
                callback=[ckpt_cb, progress_cb])

    model_path = os.path.join(args.log_dir, tag)
    model.save(model_path)
    print(f"[train] saved model -> {model_path}.zip")

    # Plot smoothed episode-reward curve from the monitor log.
    if os.path.exists(monitor_path):
        df = pd.read_csv(monitor_path, comment='#')
        if "r" in df.columns and len(df) > 1:
            df["r_smooth"] = df["r"].rolling(50, min_periods=1).mean()
            plt.figure(figsize=(7, 4))
            plt.plot(df["r_smooth"])
            plt.xlabel("episode")
            plt.ylabel("episode reward (smoothed)")
            plt.title(tag)
            plt.tight_layout()
            curve_path = os.path.join(args.log_dir, tag + "_reward.png")
            plt.savefig(curve_path)
            print(f"[train] saved curve -> {curve_path}")

    env.close()


if __name__ == "__main__":
    main()
