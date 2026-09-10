"""Bounded runtime self-check for the portable training/evaluation stack.

This is intentionally short: it builds the real compound environment with
eight workers, performs one PPO update, saves/reloads a temporary model, and
then exits.  It never touches ``results/`` or the production checkpoint.
Run from the project root after installing requirements and setting
``PYTHONPATH=me5418_env_vendor``.
"""

from __future__ import annotations

import os
import tempfile

import torch
from stable_baselines3 import PPO

from . import configs
from .train import _ProgressCallback, _build_env


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    configs.USE_TTC = True

    env = None
    try:
        env, n_envs = _build_env(
            "compound", noise=0.0, seed=0, n_envs=8,
            monitor_path=None,
        )
        model = PPO(
            "MlpPolicy", env, verbose=0, seed=0,
            n_steps=4, batch_size=32, n_epochs=1,
            policy_kwargs=dict(net_arch=[256, 256]), device="cpu",
        )
        model.learn(
            total_timesteps=32,
            progress_bar=False,
            callback=_ProgressCallback(32),
        )
        with tempfile.TemporaryDirectory(prefix="me5418_runtime_smoke_") as tmp:
            checkpoint = os.path.join(tmp, "smoke_model")
            model.save(checkpoint)
            PPO.load(checkpoint, env=env, device="cpu")
        print("[runtime-smoke] PASS: 8 envs, PPO update, save/reload")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
