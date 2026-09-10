# results/

**Only the latest generation** lives here: latest model, latest reward plot,
and latest demo video. The older models used to be archived under
`../outofuse/versions/`, but that folder was moved to the **Windows Recycle
Bin** on 2026-09-10 (restore from there if ever needed).

## Latest model

| File | What it is |
|---|---|
| `compound_ppo_ttc_safe8.zip` | Latest PPO model — 8-env training, 2026-09-09 23:44 |
| `compound_ppo_ttc_safe8_monitor.csv` | Training reward data |
| `compound_ppo_ttc_safe8_reward.png` | Smoothed reward curve |
| `compound_safe8.mp4` | Demo video of the latest model |

This is also the **live output directory** for new training/evaluation runs
(`src/train.py --log-dir` defaults to `results`).

## validation/ (archived)

The safety-validation evidence was generated against an earlier profile
(`SAFE_RESERVATION_V1`) and is **not** tied to the current main code. It used to
be archived at `../outofuse/validation_archive/results_validation/`, which went
to the Recycle Bin with the rest of `outofuse/`. The `results/`
folder now holds only the latest model artifacts (safe8) plus this README.

## Caveat

Do **not** describe the `safe8` model as a validated zero-collision model: it has
a demo video but has not yet passed the five-seed x 1000-step safety workflow.

## Older model

The previously validated V3 model was archived at
`../outofuse/versions/2026-09-06_v3_original/results/compound_ppo_ttc_v3.zip` —
recoverable only from the Windows Recycle Bin (moved there 2026-09-10).
