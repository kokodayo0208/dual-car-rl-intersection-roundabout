# ME5418 Final Project — Safe & Robust Unsignalized-Intersection Negotiation via RL

A self-contained workspace for a group project on **learning safe, robust
negotiation behaviour for unsignalized intersections** (roundabout + cross
intersection) with deep reinforcement learning. built on a vendored simulation core derived from HighwayEnv (MIT, Edouard Leurent / Farama Foundation), bundled under `me5418_env_vendor/`.

The current version (2026-09-09) runs **two RL cars** on one connected
intersection + roundabout map, controlled by a single shared PPO policy, with a
TTC safety layer and a conflict-reservation mechanism.

## Demo

Two RL cars (green + orange, one shared policy) negotiating the compound
cross-intersection + roundabout map among ~22 obstacle vehicles
(3× real time, fixed top-down camera):

<video src="https://github.com/kokodayo0208/dual-car-rl-intersection-roundabout/raw/main/results/compound_safe8.mp4" controls muted loop
       style="max-width:100%;">
  Your browser does not support embedded video —
  <a href="results/compound_safe8.mp4">download the demo (compound_safe8.mp4)</a>.
</video>

---

## Quick start

### 1. Environment

Install the dependencies and select the resulting interpreter as described in
[SETUP.md](SETUP.md). The setup uses only relative project paths and works on
Windows and Ubuntu.

### 2. Models

`results/` holds **only the latest generation**:

| File | What it is |
|---|---|
| `results/compound_ppo_ttc_safe8.zip` | Latest PPO model (8-env training, 2026-09-09 23:44) |
| `results/compound_ppo_ttc_safe8_monitor.csv` | Training reward data |
| `results/compound_ppo_ttc_safe8_reward.png` | Smoothed reward curve |
| `results/compound_safe8.mp4` | Demo video of the latest model |

The previously **validated V3** model was archived at
`outofuse/versions/2026-09-06_v3_original/results/compound_ppo_ttc_v3.zip` —
the `outofuse/` folder was moved to the **Windows Recycle Bin** on 2026-09-10;
restore it from there if the V3 model is ever needed.

### 3. Run

```powershell
# Train from scratch (100k steps, 8 parallel envs)
& $py -m src.train --scene compound --algo ppo --use-ttc --device cpu `
    --n-envs 8 --timesteps 100000 --suffix _safe8 --log-dir results

# Evaluate the latest model (5 episodes, stop on crash)
& $py -m src.evaluate --scene compound --algo ppo `
    --model results/compound_ppo_ttc_safe8.zip --episodes 5 --end-on-crash

# 120 s of simulation -> ~60 s of video (30 fps = 2x real time)
& $py -m src.evaluate --scene compound --algo ppo `
    --model results/compound_ppo_ttc_safe8.zip --episodes 1 --duration 120 `
    --seed 0 --fps 30 --video results\eval_latest.mp4
```

**In VS Code**, just pick **"1) train: compound + PPO (100k, 8 envs)"** or
**"2) evaluate: latest model (+video)"** from the Run and Debug dropdown
(`Ctrl+Shift+D`) and press the green play button. The configurations live in
`.vscode/launch.json`.

> ⚠️ The **safe8** model has a demo video but has **not** yet passed the
> five-seed × 1000-step safety validation. Do not describe it as a validated
> zero-collision model until that test is run and its output saved alongside it.
> The validated results belong to the archived **V3** model.

---

## Workspace layout

```
highway_rl_project/
├── src/                    # ACTIVE code — this is what you run
│   ├── configs.py          # shared obs/action config + TTC / noise knobs
│   ├── custom_envs.py      # MyRoundabout, MyIntersection (TTC reward),
│   │                       #   NoiseWrapper, MultiAgentRoundabout,
│   │                       #   CompoundEnv + JointAgentWrapper (compound map),
│   │                       #   registration + make_env
│   ├── compound_road.py    # compound map: cross intersection + roundabout in ONE network
│   ├── train.py            # unified training (PPO / DQN, scene, ttc, noise)
│   └── evaluate.py         # collision rate / mean return / video
├── me5418_env_vendor/     # vendored HighwayEnv source — required
├── results/                # latest model + video (safe8)
│   ├── compound_ppo_ttc_safe8.zip / _monitor.csv / _reward.png
│   └── compound_safe8.mp4
├── requirements_experiment.txt   # pip dependency list
└── README.md               # this file

(Old version snapshots / archives that used to live in `outofuse/` were moved
to the Windows Recycle Bin on 2026-09-10 and are no longer part of the repo.)
```

## Python version

Use **Python 3.13.14** to match the verified Windows environment. `env_core`
(vendored `1.12.2.dev0`) declares `requires-python = ">= 3.10"`, while the
exact runtime pins are in `requirements.txt`.
compatibility with `stable-baselines3` (>=2.3), `gymnasium` (>=1.0), PyTorch,
and `pygame-ce`. Ubuntu portability is documented but has not been physically
tested on this Windows host. The portable `requirements.txt` and the
commands in `SETUP.md` keep the environment reproducible without relying on a
machine-specific conda file.

## Setup on Ubuntu

Follow the complete virtual-environment and dependency commands in
[SETUP.md](SETUP.md). The former `outofuse/envs` one-click setup scripts were
historical only; they went to the Recycle Bin together with `outofuse/`.

## Run (full command reference)

```bash
# Train (PPO) on the roundabout with the TTC safety reward
python -m src.train --scene roundabout --algo ppo --use-ttc

# Train (DQN) on the intersection with TTC reward + 0.1 sensor noise
python -m src.train --scene intersection --algo dqn --use-ttc --noise 0.1

# Baseline ablation: same but WITHOUT the TTC reward
python -m src.train --scene roundabout --algo ppo --no-ttc

# Evaluate a trained policy (collision rate, mean return)
python -m src.evaluate --scene roundabout --algo ppo \
    --model results/roundabout_ppo_ttc.zip --episodes 100

# Evaluate with the pygame window open (needs a display)
python -m src.evaluate --scene intersection --algo dqn \
    --model results/intersection_dqn_ttc.zip --render

# Multi-agent compound map — trains two RL vehicles with a single shared PPO
python -m src.train --scene compound --algo ppo --use-ttc \
    --device cuda --timesteps 20000

# Evaluate it (cars keep looping, so `--episodes 1 --video` gives a >=10 s mp4)
python -m src.evaluate --scene compound --algo ppo \
    --model results/compound_ppo_ttc_safe8.zip --episodes 1 --video compound.mp4
#   --slowdown 10   -> 3x real time (default)
#   --slowdown 20   -> 1.5x real time (frame-by-frame study)
#   --no-panoramic  -> follow the ego vehicle instead of the fixed camera
```

Trained models and a smoothed reward curve are written under `results/`.
`--video` needs `imageio-ffmpeg` (listed in `requirements.txt`); it records every episode, then keeps only the
highest-return episode's recording.

### Why two RL cars + the panoramic, slowed-down video?

The compound map was built to make **four behaviours** visible at once:

- **Multi-lane avoidance** — the roundabout ring keeps both lanes (r=20 inner /
  r=24 outer) and the bridge has `BRIDGE_LANES=2`, so cars genuinely overtake.
- **Roundabout yielding** — lane `priority` gives circulating traffic
  (`PRIORITY_RING=5`) right of way over entering traffic (`PRIORITY_RB_ENTRY=0`).
- **Intersection yielding** — same priority scheme at the crossroads.
- **Dual-car game** — two RL cars (green + orange) share ONE policy and negotiate
  both junctions; obstacle traffic is a *population* kept at ~22 cars, routed
  across the map so both junctions stay busy.

The video is slowed to **3× real time** (`VIDEO_FRAME_REPEAT=10`) and shot from a
**fixed top-down camera** spanning the whole map, so you can follow a single car
through the junction.

## Three differentiation experiments

1. **TTC-based safety reward** — `--use-ttc` vs `--no-ttc`.
   A penalty fires whenever time-to-collision to the nearest vehicle drops
   below `TTC_THRESHOLD` (see `src/configs.py`). Compare collision rates.
2. **Perception-noise robustness** — sweep `--noise 0.0 0.05 0.1 0.2` and
   measure how much the collision rate degrades (uses `NoiseWrapper`).
3. **Multi-agent game equilibrium** — the **compound** map (`--scene compound`).
   It is a single network containing BOTH a cross intersection (at the origin)
   and a roundabout (to the east), bridged into one closed loop. Two RL vehicles
   are controlled by **one shared policy** (via `JointAgentWrapper`, which
   flattens the two per-agent observations to `Box(150,)` and encodes the joint
   5x5 action as `Discrete(25)`). The agents keep looping through both
   topologies, avoid obstacle vehicles that continuously enter / leave the map,
   and receive the TTC safety reward.

## Validation evidence (archived)

The safety-validation evidence (`FINAL_VALIDATION_REPORT.md`, merged log,
long-test/replay JSON, background characterisation, source-hash audit,
validation video + frames) was generated against an earlier profile
(`SAFE_RESERVATION_V1`) and references unit tests that are no longer in `src/`.
It is **not** tied to the current main code, so it was archived with the
`outofuse/` folder (moved to the **Windows Recycle Bin** on 2026-09-10).

The previously validated V3 model itself was in the same archive
(`outofuse/versions/2026-09-06_v3_original/results/compound_ppo_ttc_v3.zip`)
and is likewise only recoverable from the Recycle Bin now.

## Notes

- The scenes are forced to a **shared** observation (`vehicles_count=15`,
  5-D kinematics, flattened to 1-D for `stable-baselines3` `MlpPolicy`) and a
  **shared** 5-action discrete space, so one network can be compared across
  topologies. The compound scene stacks two agents into a `Box(150,)` +
  `Discrete(25)` joint space via `JointAgentWrapper`.
- `configs.py` is the single place to tune `USE_TTC`, `TTC_REWARD`,
  `TTC_THRESHOLD`, the shared spaces, and compound-specific constants
  (`N_AGENTS`, `JOINT_OBS_DIM`, `JOINT_ACTION_DIM`, `COMPOUND_DURATION`,
  `COMPOUND_SPAWN_PROB`, `COMPOUND_TARGET_VEHICLES`, `VIDEO_FRAME_REPEAT`).
- **Windows / Ubuntu:** the active pipeline uses the vendored environment and
  relative paths. Follow `SETUP.md` to create a local **Python 3.12 or 3.13**
  environment (3.13 is the verified reference; the pinned `numpy==2.5.3` /
  `pandas==3.0.5` / `matplotlib==3.11.1` require >= 3.12 / 3.11, so 3.10 will
  NOT install). No managed-user path is required.
- **Compound deadlock fix:** the stock `RegulatedRoad` predicts conflicts in a
  straight line, which deadlocked the curved ring. `compound_road.py` provides
  `CompoundRoad` (curve-aware conflict prediction) —   `CompoundEnv._make_road`
  uses it. The standalone re-verify script (`diagnostic_compound.py`) went to
  the Recycle Bin with the `outofuse/` archive
  (expect mean obstacle speed ≫ 5 m/s and ~0 crashed-obstacle steps).

## Former `outofuse/` archive

Nothing that was in `outofuse/` (per-version snapshots, docs, setup helpers,
the V3 model and the validation evidence) is needed to train, evaluate or
reproduce results with the current code. The whole folder was moved to the
**Windows Recycle Bin on 2026-09-10** — restore it from there if ever needed,
and note it will be gone permanently once the Recycle Bin is emptied.
