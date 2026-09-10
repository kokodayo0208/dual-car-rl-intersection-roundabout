"""Spare scenario environments — NOT used by the project's main code.

These are the official course-package scenario environments that the
current experiment (ME5418 compound intersection/roundabout project) does
not use. They import ``env_core`` via absolute imports, so they still
work as long as ``me5418_env_vendor`` is on PYTHONPATH, e.g.::

    PYTHONPATH=me5418_env_vendor python -c "from env_standby.merge_env import MergeEnv"

Contents:
- exit_env.py            exit / connected-lane exit scenarios
- env_core.py          the classic highway scenario (HighwayEnv / Fast)
- lane_keeping_env.py    lane-keeping scenario
- merge_env.py           highway merge scenario
- parking_env.py         parking scenario
- racetrack_env.py       racetrack scenario
- random_road_env.py     procedurally generated random roads
- two_way_env.py         two-way traffic scenario
- u_turn_env.py          u-turn scenario
- road_generation_engine/  the PCG engine + generator.py used only by
                           random_road_env (gen_utils.py stayed in the core
                           package because observation.py needs it)

NOTE: ``engine/__init__.py`` inside the core package was rewritten when the
engine was moved here; if you ever move these files back, restore the
original ``env_core/road/generation/engine/__init__.py`` from the backup
in ``outofuse/me5418_env_vendor_backup_2026-09-10/``.
"""
