"""Geometry helpers kept in the core package.

The full PCG engine (agents / boundaries / optimize / rectify / validation)
has been moved to ``env_standby/road_generation_engine/`` together with
``generator.py`` — they are only needed by the (standby) RandomRoadEnv.
``gen_utils`` stays here because ``envs/common/observation.py`` depends on it.
"""

from .gen_utils import (
    Endpoint,
    Lane,
    do_line_segments_intersect,
    find_line_intersection,
    get_junction_pos,
    get_nodeset,
    get_radially_sorted_endpoints,
    line_intersection_t,
    wrap_with_tqdm,
)

__all__ = [
    "Lane",
    "Endpoint",
    "get_radially_sorted_endpoints",
    "get_junction_pos",
    "get_nodeset",
    "line_intersection_t",
    "do_line_segments_intersect",
    "find_line_intersection",
    "wrap_with_tqdm",
]
