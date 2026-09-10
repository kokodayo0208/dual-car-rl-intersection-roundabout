"""Environment registry shim.

The scenario env files were split into two top-level folders inside
``me5418_env_vendor/``:

- ``env_using/``   — scenarios used by the project's main code (src/):
  ``intersection_env`` and ``roundabout_env``.
- ``env_standby/`` — spare scenarios not used by the project (exit, highway,
  lane-keeping, merge, parking, racetrack, random-road, two-way, u-turn).

Import them directly, e.g.::

    from env_using.intersection_env import IntersectionEnv
    from env_using.roundabout_env import RoundaboutEnv
"""
