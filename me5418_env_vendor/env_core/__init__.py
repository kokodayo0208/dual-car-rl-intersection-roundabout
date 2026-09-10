import os
import sys

__version__ = "1.12.2.dev0"

try:
    from farama_notifications import notifications

    if "env_core" in notifications and __version__ in notifications["env_core"]:
        print(notifications["env_core"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_me5418_envs():
    """Backwards-compatible no-op.

    The gymnasium string-ID registrations were removed when the scenario
    env files were split into ``env_using/`` / ``env_standby/`` (top-level
    folders in ``me5418_env_vendor/``). The project imports env classes
    directly, so the gymnasium registry is no longer used.
    """
    return
