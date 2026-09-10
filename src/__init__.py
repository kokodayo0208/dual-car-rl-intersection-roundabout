"""ME5418 final-project source package.

Run training/evaluation as a module from the project root, e.g.:
    python -m src.train --scene roundabout --algo ppo --use-ttc
"""

# Auto-locate the vendored simulation core so the project runs out of the box
# after a plain `git clone` / ZIP download — setting PYTHONPATH manually is no
# longer required (SETUP.md still documents it for shells that want it).
import os as _os
import sys as _sys

_VENDOR_DIR = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "me5418_env_vendor",
)
if _os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in _sys.path:
    _sys.path.append(_VENDOR_DIR)
