# Portable setup and VS Code workflow

The active project is the directory containing this file. Run commands from
that directory. The simulation core is included in `me5418_env_vendor/`, so no
GitHub or internet checkout of HighwayEnv is required.

### Scenario layout inside `me5418_env_vendor/` (2026-09-10)

- `env_using/` — scenarios used by the main code: `intersection_env.py`,
  `roundabout_env.py`. Import them as `from env_using.intersection_env import
  IntersectionEnv` (PYTHONPATH still points at `me5418_env_vendor/`).
- Since 2026-09-10, `src/__init__.py` auto-appends `me5418_env_vendor/` to
  `sys.path`, so setting PYTHONPATH is **optional** — a plain
  `python -m src.train ...` works out of the box after `pip install -r
  requirements.txt`. The exported variables below are kept for clarity and
  for tools that spawn subprocesses.
- `env_standby/` — spare scenarios not used by the project (exit, highway,
  lane-keeping, merge, parking, racetrack, random-road, two-way, u-turn) plus
  `road_generation_engine/`. Still importable as `env_standby.<module>`
  (`random_road_env` additionally requires the `noise` package, already in
  `requirements.txt`).
- `env_core/` — core package only (road/vehicle/common); its gymnasium
  string-ID registry was removed and env classes are imported directly.
- Original untouched copy: was at `outofuse/me5418_env_vendor_backup_2026-09-10/`
  — the whole `outofuse/` folder (old version archives + refactor backups) was
  moved to the Windows Recycle Bin on 2026-09-10 and is no longer in the repo.
  Restore from the Recycle Bin if ever needed.

## Ubuntu

The reference environment is Python 3.13.14 on Windows. Ubuntu instructions
are provided for a portable checkout, but have not been physically tested on
this Windows host. The most reliable route is Conda, since Ubuntu releases do
not all provide Python 3.13 as a system package.

```bash
cd highway_rl_project
conda create -n me5418 python=3.13
conda activate me5418
python -m pip install --upgrade pip
# Install the CPU build before the remaining packages, avoiding a CUDA download.
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/me5418_env_vendor"
export SDL_VIDEODRIVER=dummy
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m src.train --scene compound --algo ppo --use-ttc --n-envs 8 --timesteps 100000 --device cpu --suffix _safe8 --log-dir results
python -m src.evaluate --scene compound --algo ppo --model results/compound_ppo_ttc_safe8.zip --episodes 1 --duration 120 --end-on-crash --fps 60 --playback-speed 2.5 --video results/compound_safe8.mp4
```

If Python 3.13 is already installed on the target computer, the Conda steps
can instead be replaced with `python3.13 -m venv .venv` and
`source .venv/bin/activate`. If pip must build the `noise` source package,
install a compiler first: `sudo apt-get install -y build-essential`.

`imageio-ffmpeg` supplies the video encoder. A system `ffmpeg` package is
optional. On a server without a display, keep `SDL_VIDEODRIVER=dummy` and do
not pass `--render`.

## Windows

```powershell
Set-Location -LiteralPath 'D:\ME5418（9.8version）\highway_rl_project'
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\me5418_env_vendor).Path
$env:SDL_VIDEODRIVER = 'dummy'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

## VS Code Run and Debug

Open the repository root, `highway_rl_project`. In VS Code select the Python
interpreter created above (`.venv/bin/python` on Ubuntu or
`.venv\\Scripts\\python.exe` on Windows), then press `Ctrl+Shift+D`.

Choose `1) train: compound + PPO (100k, 8 envs)` to train 100,000 total steps
with eight parallel environments. Choose `2) evaluate: safe8 model (+video)`
to evaluate `results/compound_ppo_ttc_safe8.zip`, show the progress bar, and
write a 60 FPS, 2.5x video to `results/compound_safe8.mp4`.

The launch files use `${command:python.interpreterPath}` and relative paths;
they contain no machine-specific username or drive letter. The Python and
debugpy extensions are required by VS Code. If a configuration is missing
after changing the workspace folder, run `Developer: Reload Window`.

## Headless smoke check

```bash
PYTHONPATH="$PWD/me5418_env_vendor" SDL_VIDEODRIVER=dummy python -m src.evaluate --help
PYTHONPATH="$PWD/me5418_env_vendor" SDL_VIDEODRIVER=dummy python -m src.train --help
```

For a bounded runtime check after dependencies are installed, run
`python -m src._runtime_smoke`. It starts eight environment workers, performs
one PPO update for 32 total training steps, saves and reloads a temporary
model, then removes that temporary directory. It never overwrites a production
model or writes into `results/`.

The help checks do not train or create a model. A copied checkout still needs
the Python dependencies installed with `requirements.txt`; `.venv` itself is
machine-specific and is intentionally ignored by Git.

## GitHub upload

Open `highway_rl_project` in VS Code. In the Source Control view, choose
**Initialize Repository**, inspect the changed-file list, then choose
**Publish to GitHub** and select the repository visibility yourself. This
publishes the project source, the local `me5418_env_vendor/` dependency, and
the `.vscode/` run configurations. Do not commit personal access tokens,
virtual environments, cache files, or newly generated large experiment output.

Before publishing, review `.gitignore` and intentionally decide whether the
small `results/compound_ppo_ttc_safe8.zip` checkpoint is needed for others to
run the evaluation configuration. The project can always be trained from
source without it. This document does not create a remote repository or run
any GitHub command on your behalf.
