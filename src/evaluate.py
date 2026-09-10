"""Evaluate a trained policy: collision rate, mean return, optional video.

Examples
--------
    python -m src.evaluate --scene roundabout --algo ppo \
        --model results/roundabout_ppo_ttc.zip --episodes 100
    python -m src.evaluate --scene intersection --algo dqn \
        --model results/intersection_dqn_ttc.zip --noise 0.1 --render
    python -m src.evaluate --scene roundabout --algo ppo \
        --model results/roundabout_ppo_ttc.zip --video eval.mp4
    python -m src.evaluate --scene compound --algo ppo \
        --model results/compound_ppo_ttc.zip --episodes 1 --video compound.mp4

--video streams physical substep frames to FFmpeg and retains the best episode.
It does not change the policy frequency or retain all RGB frames in memory.
Playback speed is controlled by --playback-speed. With the default 60 FPS and
15 Hz physics, 2.5x playback produces 24 output frames per simulated second.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_variable, '1')

import numpy as np

from stable_baselines3 import PPO, DQN
import torch

from . import configs
from .configs import VIDEO_FPS, VIDEO_FRAME_REPEAT
from .custom_envs import make_env


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained highway agent.")
    p.add_argument("--scene", required=True,
                   choices=["roundabout", "intersection", "compound"])
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--model", required=True,
                   help="path to model, with or without the .zip extension")
    p.add_argument("--use-ttc", dest="use_ttc", action="store_true")
    p.add_argument("--no-ttc", dest="use_ttc", action="store_false")
    p.set_defaults(use_ttc=True)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--duration", type=float, default=None,
                   help="optional episode duration in simulated seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true",
                   help="show the pygame window while evaluating")
    p.add_argument("--video", default=None,
                   help="optional path to save an mp4 (needs ffmpeg)")
    p.add_argument("--fps", type=float, default=float(VIDEO_FPS),
                   help="output frames per second for the recorded --video (default: 60)")
    p.add_argument("--playback-speed", type=float, default=2.5,
                   help="video playback speed relative to simulation time (default: 2.5x)")
    p.add_argument("--slowdown", type=int, default=VIDEO_FRAME_REPEAT,
                   help=("(legacy) per-frame repeat used by the old recorder; "
                         "the new recorder renders every internal sub-step instead "
                         "so this is ignored when --video is set)"))
    p.add_argument("--no-panoramic", action="store_true",
                   help="follow the ego vehicle instead of using the fixed "
                        "map-wide camera (compound scene only)")
    p.add_argument("--end-on-crash", action="store_true",
                   help="(display) end the episode when a car crashes instead of "
                        "rebooting it in place, so the video shows the real "
                        "collision instead of a teleport/refresh. Maps to "
                        "CRASH_TERMINATE; training still uses reboot by default.")
    return p.parse_args()


def _has_arrived(env) -> bool:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "has_arrived"):
        return False
    # Multi-agent compound: any controlled agent arriving (or, for the loop map,
    # reaching the far end of an arm). Single-agent scenes use `vehicle`.
    vehicles = getattr(unwrapped, "controlled_vehicles", None) or [unwrapped.vehicle]
    try:
        return any(bool(unwrapped.has_arrived(v)) for v in vehicles)
    except Exception:
        return False


def _episode_crashes(env) -> tuple[bool, int]:
    """Return (had_crash, n_events) for THIS episode.

    The compound scene clears an agent's ``crashed`` flag inside step() (it is
    rebooted in place), so reading ``v.crashed`` at episode end reports ~0
    collisions regardless of how many actually happened. The env now keeps a
    live ``_crash_count``; prefer it. Other scenes leave ``crashed`` set until
    episode end, so fall back to reading the flag there.
    """
    unwrapped = env.unwrapped
    n = getattr(unwrapped, "_crash_count", None)
    if n is not None:
        return n > 0, n
    vehicles = getattr(unwrapped, "controlled_vehicles", None) or [unwrapped.vehicle]
    had = any(bool(getattr(v, "crashed", False)) for v in vehicles)
    return had, (1 if had else 0)


def _step_with_frames(env, action, frames):
    """Record physics frames without changing the policy or traffic cadence."""
    raw = env.unwrapped
    original = raw._automatic_rendering
    def capture():
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())
    raw._automatic_rendering = capture
    try:
        result = env.step(action)
        capture()
        return result
    finally:
        raw._automatic_rendering = original


class _FrameWriter:
    """Append-only FFmpeg writer with streaming temporal resampling.

    The environment supplies one physical frame for each physics sub-step.
    ``fps / (physical_hz * playback_speed)`` determines how many output frames
    each physical frame contributes on average. This gives exact long-run
    playback timing without changing simulation or policy cadence.
    """
    def __init__(self, path, fps, physical_hz, playback_speed):
        self.path, self.fps = path, fps
        self.output_ratio = fps / (physical_hz * playback_speed)
        self._resample_credit = 0.0
        self.frame_count = 0
        self.writer = None

    def _append_one(self, frame):
        if self.writer is None:
            import imageio_ffmpeg
            height, width = frame.shape[:2]
            self.writer = imageio_ffmpeg.write_frames(
                str(self.path), (width, height), fps=self.fps,
                codec='libx264', pix_fmt_in='rgb24',
                output_params=['-crf', '20', '-preset', 'fast'])
            self.writer.send(None)
        self.writer.send(np.ascontiguousarray(frame))
        self.frame_count += 1

    def append(self, frame):
        self._resample_credit += self.output_ratio
        while self._resample_credit >= 1.0:
            self._append_one(frame)
            self._resample_credit -= 1.0

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class _Progress:
    """tqdm progress bar with a dependency-free terminal fallback."""
    def __init__(self, total, description):
        self._fallback = False
        try:
            from tqdm.auto import tqdm
        except ImportError:
            self._fallback = True
            self._total = total
            self._description = description
            self._value = 0
            self._last_print = -1
            print(f"[evaluate] {description}: 0" +
                  (f"/{total} steps" if total is not None else " steps"), end="", flush=True)
        else:
            self._bar = tqdm(total=total, desc=description, unit="step", dynamic_ncols=True)

    def update(self, n=1):
        if not self._fallback:
            self._bar.update(n)
            return
        self._value += n
        # Keep the fallback readable for long episodes without flooding the terminal.
        interval = max(1, (self._total or 100) // 100)
        if self._value == self._total or self._value - self._last_print >= interval:
            suffix = f"/{self._total}" if self._total is not None else ""
            print(f"\r[evaluate] {self._description}: {self._value}{suffix} steps",
                  end="", flush=True)
            self._last_print = self._value

    def close(self):
        if not self._fallback:
            self._bar.close()
        else:
            print()


def main():
    args = parse_args()
    configs.USE_TTC = args.use_ttc
    configs.CRASH_TERMINATE = args.end_on_crash  # display: no in-place reboot
    torch.set_num_threads(1)

    env = make_env(args.scene, noise_std=args.noise, seed=args.seed,
                   render_mode="rgb_array" if args.video else
                               ("human" if args.render else None))
    if args.no_panoramic:
        try:
            env.unwrapped.config["panoramic_view"] = False
        except Exception:  # pragma: no cover
            pass
    if args.duration is not None:
        if args.duration <= 0:
            raise ValueError('--duration must be positive')
        env.unwrapped.config['duration'] = args.duration

    Model = PPO if args.algo == "ppo" else DQN
    # Stable-Baselines3 appends ``.zip`` when it is missing.  Passing an
    # already suffixed path to some SB3 versions can therefore produce
    # ``.zip.zip``.  Normalize once so both CLI conventions work reliably.
    model_path = Path(args.model)
    if model_path.suffix.lower() != ".zip":
        model_path = Path(f"{model_path}.zip")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Pass --model with or without the .zip extension."
        )
    model = Model.load(str(model_path.with_suffix("")), env=env)

    simulation_frequency = env.unwrapped.config['simulation_frequency']
    policy_frequency = env.unwrapped.config['policy_frequency']
    if args.fps <= 0:
        raise ValueError('--fps must be positive')
    if args.playback_speed <= 0:
        raise ValueError('--playback-speed must be positive')

    crashes = 0
    crash_events = 0
    arrived = 0
    returns = []
    # Unified collision taxonomy (BB / CB / CC) -- see custom_envs.COLLISION_TYPES.
    # "How many times was an agent rebooted" is NOT a collision count: it also
    # fires for the stall watchdog and for dead-end resets. From now on every
    # table in the report uses these three numbers.
    collision_classes = {"BB": 0, "CB": 0, "CC": 0}
    # Keep completed temporary videos, rather than thousands of RGB arrays.
    best_video, best_idx, best_return = None, 0, -float("inf")
    best_frame_count = 0
    best_sim_steps = 0
    video_output = Path(args.video).resolve() if args.video else None
    if video_output:
        video_output.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if args.video:
            with tempfile.NamedTemporaryFile(prefix='compound_episode_', suffix='.mp4',
                    dir=video_output.parent, delete=False) as handle:
                temporary_video = Path(handle.name)
            cur_frames = _FrameWriter(temporary_video, args.fps,
                                      simulation_frequency, args.playback_speed)
        # A finite progress total is useful even when --duration is omitted:
        # use the environment's configured episode duration in that case.
        configured_duration = env.unwrapped.config.get("duration")
        progress_duration = (args.duration if args.duration is not None
                             else configured_duration)
        total_steps = (int(np.ceil(progress_duration * policy_frequency))
                       if progress_duration is not None else None)
        progress = _Progress(total_steps, f"episode {ep + 1}/{args.episodes}")
        done = False
        G = 0.0
        episode_steps = 0
        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if args.video:
                    obs, r, terminated, truncated, _ = _step_with_frames(env, action, cur_frames)
                else:
                    obs, r, terminated, truncated, _ = env.step(action)
                G += r
                done = terminated or truncated
                episode_steps += 1
                progress.update()
        finally:
            progress.close()
            if args.video:
                # Close FFmpeg before inspecting/replacing the temporary file,
                # including when model prediction or env.step raises.
                cur_frames.close()
        returns.append(G)
        if args.video:
            if G > best_return and cur_frames.frame_count:
                if best_video is not None:
                    best_video.unlink()
                best_video, best_return, best_idx = temporary_video, G, ep
                best_frame_count = cur_frames.frame_count
                best_sim_steps = episode_steps
            else:
                temporary_video.unlink()
        had_crash, n_events = _episode_crashes(env)
        if had_crash:
            crashes += 1
        crash_events += n_events
        for kind, n in getattr(env.unwrapped, "_collision_classes", {}).items():
            collision_classes[kind] = collision_classes.get(kind, 0) + n
        if _has_arrived(env):
            arrived += 1

    returns = np.asarray(returns)
    print(f"episodes        : {args.episodes}")
    print(f"collision_rate  : {crashes / args.episodes:.3f} "
          f"(fraction of episodes with >=1 crash)")
    print(f"collision_events: {crash_events / args.episodes:.3f} per episode "
          f"(real count; >0 even when an agent is rebooted in-place)")
    if args.scene == "compound":
        print("collision split : "
              + "  ".join(f"{k}={collision_classes[k] / args.episodes:.2f}/ep"
                          for k in ("BB", "CB", "CC"))
              + "   (BB=background<->background, CB=agent<->background, "
                "CC=agent<->agent)")
        if collision_classes["BB"]:
            print("                  WARNING: background traffic is colliding with "
                  "itself -> the ENVIRONMENT is unstable, agent numbers are not "
                  "interpretable. Run src/_test_background_traffic.py.")
    if args.scene in ("intersection", "compound"):
        # For the compound looping map "arrival" is rare by design (cars keep
        # circulating), so this is informational only.
        print(f"arrival_rate    : {arrived / args.episodes:.3f}")
    print(f"mean_return     : {returns.mean():.3f} +/- {returns.std():.3f}")

    if best_video is not None:
        best_video.replace(video_output)
        print(f"[evaluate] best episode {best_idx} (return {best_return:.3f}) -> {video_output}")
        print(f"[evaluate] {best_frame_count} output frames, {best_sim_steps} sim steps, "
              f"{best_frame_count / args.fps:.1f}s of video "
              f"({args.playback_speed:g}x real time, {args.fps:g} FPS, "
              f"temporal resampling from {simulation_frequency:g} Hz physics)")

    env.close()


if __name__ == "__main__":
    main()
