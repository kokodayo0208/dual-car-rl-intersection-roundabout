"""Build the compound map: a cross intersection + a roundabout in ONE network.

The two topologies are the *proven* base-env geometries (``IntersectionEnv`` and
``RoundaboutEnv`` from the vendored highway-env), reused whole so we never
re-derive the tricky intersection/ring mathematics by hand. We instantiate each
base env to grab its already-built ``road.network``, then:

    * deep-copy every lane of the intersection (unchanged, sits at the origin),
    * deep-copy every lane of the roundabout and shift it to (RB_X, RB_Y),
    * TRIM the outer access arms of both features so the whole map fits in a
      single panoramic camera shot (the stock arms are 111 m / 170 m long, which
      would force the camera so far out that cars become 5-pixel specks), and
    * bridge the pair with a two-lane-per-direction link road so the map is one
      closed loop:

          intersection east exit (il3 -> o3)   -> bridge -> roundabout
          roundabout west exit   (wxr)         -> bridge -> o3 (east entry)

A car can therefore travel: intersection -> east bridge -> roundabout ring (one
or more laps, on either of the two ring lanes) -> west bridge -> back into the
intersection -> on around again forever.

Multi-lane
----------
The roundabout ring keeps BOTH of its original lanes (inner r=20, outer r=24)
and the two bridge carriageways have ``BRIDGE_LANES`` lanes each, so vehicles
genuinely drive side by side, overtake and change lane. This is what makes
"multi-lane avoidance" visible. ``env_core.road.road`` has been patched
(``position_heading_along_route``) so route following no longer crashes when a
2-lane road feeds a 1-lane one.

Right of way
------------
Lane ``priority`` (used by ``RegulatedRoad``) is re-tuned so the roundabout
behaves like a real one: circulating traffic (PRIORITY_RING) never yields and
entering traffic (PRIORITY_RB_ENTRY) must give way.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import configs as _cfg
from env_using.intersection_env import IntersectionEnv
from env_using.roundabout_env import RoundaboutEnv
from env_core.road.lane import (
    AbstractLane,
    CircularLane,
    LineType,
    SineLane,
    StraightLane,
)
from env_core.road.regulation import RegulatedRoad
from env_core.road.road import Road, RoadNetwork

LANE_WIDTH = AbstractLane.DEFAULT_WIDTH  # 4.0 [m]

# --- 1. Geometry knobs (all in metres) ------------------------------------
# Distance from each feature's centre out to the tip of its arms. Short arms
# keep the whole map inside one camera frame, which is what makes the panoramic
# (top-down, non-following) video view usable.
INTER_ARM_EXTENT = 72.0   # stock: 111 (access_length 100 + outer_distance 11)
RB_ARM_EXTENT = 85.0      # stock: 170
BRIDGE_LENGTH = 65.0      # straight link between the two features

RB_Y = 0.0
RB_X = INTER_ARM_EXTENT + BRIDGE_LENGTH + RB_ARM_EXTENT  # 222.0

# Radius (from the feature centre) of the INNER endpoint of the trimmable
# straight access lanes -- everything beyond that is what we shorten.
_INTER_INNER_RADIUS = 11.0   # IntersectionEnv: outer_distance
_RB_INNER_RADIUS = 42.5      # RoundaboutEnv: dev / 2 (dev = 85)

# Lanes per direction on the bridge (1 = single lane, 2 = overtake possible).
BRIDGE_LANES = 2

# --- 2. Right-of-way priorities (RegulatedRoad: higher = has right of way) --
PRIORITY_RING = 5        # traffic already circulating on the ring
PRIORITY_RB_EXIT = 4     # ring -> access ramp (leaving the roundabout)
PRIORITY_BRIDGE = 2      # the link road between the two features
PRIORITY_RB_ENTRY = 0    # access ramp -> ring (MUST give way to the ring)

# --- 3. Camera / screen ---------------------------------------------------
CAMERA_SCALING = 4.0      # pixels per metre of the panoramic top-down view
CAMERA_MARGIN = 22.0      # [m] of empty space kept around the map
CAMERA_MAX_WIDTH = 2400   # px; height follows from the map aspect ratio

# --- 4. Topology tables ---------------------------------------------------
# Lanes where obstacle traffic ENTERS the map (all at the map periphery, so no
# car ever materialises in the middle of a junction).
ENTRY_LANES = [
    ("o0", "ir0"), ("o1", "ir1"), ("o2", "ir2"),          # intersection S/W/N
    ("ser", "ses"), ("eer", "ees"), ("ner", "nes"),       # roundabout S/E/N
]
# Lanes whose far end leaves the map -- a car reaching the end of one of these
# is despawned. The two lanes feeding the bridge are deliberately excluded.
EXIT_LANES = [
    ("il0", "o0"), ("il1", "o1"), ("il2", "o2"),          # intersection exits
    ("sxs", "sxr"), ("exs", "exr"), ("nxs", "nxr"),       # roundabout exits
]
# Route destinations that make a car drive off the map (see EXIT_LANES).
EXIT_NODES = ["o0", "o1", "o2", "sxr", "exr", "nxr"]
# Same, split by feature, so a spawned car can be routed ACROSS the map
# (intersection -> roundabout or back) and therefore uses both junctions.
EXIT_NODES_INTER = ["o0", "o1", "o2"]
EXIT_NODES_RB = ["sxr", "exr", "nxr"]
# Nodes of the two bridge carriageways, used to tag which feature an entry
# lane belongs to.
_INTER_ENTRY_NODES = {"o0", "o1", "o2", "o3"}
# Destinations used to keep the two RL agents circulating forever.
LOOP_DESTINATIONS = ["o0", "o1", "o2", "o3", "sxr", "exr", "nxr", "wxr"]
# SAFE subset, used for the RL agents. Every node here still has OUTGOING lanes,
# so a car that reaches one can keep driving. sxr / exr / nxr are the
# roundabout's OFF-MAP exits: they are dead ends with no successor lane. Obstacle
# traffic reaching them is despawned, but the controlled agents are never
# removed -- so an agent routed to one parks at the end of the lane forever and
# blocks the junction behind it (the "orange car blocks the roundabout exit"
# deadlock). Never route a controlled agent to a dead end.
SAFE_LOOP_DESTINATIONS = ["o0", "o1", "o2", "o3", "wxr"]


# --- 5. Lane helpers ------------------------------------------------------
def _remake(lane, start=None, end=None, priority=None):
    """Rebuild a lane of the same concrete type with new endpoints/priority."""
    start = lane.start if start is None else start
    end = lane.end if end is None else end
    priority = lane.priority if priority is None else priority
    if isinstance(lane, SineLane):
        return SineLane(
            start, end, lane.amplitude, lane.pulsation, lane.phase, lane.width,
            lane.line_types, lane.forbidden, lane.speed_limit, priority,
        )
    if isinstance(lane, CircularLane):
        # Circular lanes are only translated, never reshaped.
        raise TypeError("CircularLane is handled by _shift_lane")
    return StraightLane(
        start, end, lane.width, lane.line_types, lane.forbidden,
        lane.speed_limit, priority,
    )


def _shift_lane(lane, shift: np.ndarray, priority=None):
    """Deep-copy a lane, translated by `shift` (and optionally re-prioritised)."""
    if isinstance(lane, SineLane):
        return SineLane(
            lane.start + shift, lane.end + shift, lane.amplitude, lane.pulsation,
            lane.phase, lane.width, lane.line_types, lane.forbidden,
            lane.speed_limit,
            lane.priority if priority is None else priority,
        )
    if isinstance(lane, CircularLane):
        return CircularLane(
            lane.center + shift, lane.radius, lane.start_phase, lane.end_phase,
            lane.clockwise, lane.width, lane.line_types, lane.forbidden,
            lane.speed_limit, lane.priority if priority is None else priority,
        )
    if isinstance(lane, StraightLane):
        return StraightLane(
            lane.start + shift, lane.end + shift, lane.width, lane.line_types,
            lane.forbidden, lane.speed_limit,
            lane.priority if priority is None else priority,
        )
    raise TypeError(f"unsupported lane type {type(lane).__name__}")


def _trim_lane(lane, center: np.ndarray, extent: float):
    """Shorten a straight access lane so its outer tip is `extent` [m] from `center`.

    Only StraightLanes are affected (the junction arcs and the sine transitions
    are already compact). The endpoint closest to `center` is kept as-is; the
    outer endpoint is pulled back along the lane direction. Lanes that are
    already shorter than the target are returned untouched.
    """
    if not isinstance(lane, StraightLane):
        return lane
    length = float(np.linalg.norm(lane.end - lane.start))
    if length <= 1e-6:
        return lane
    direction = (lane.end - lane.start) / length
    r_start = float(np.linalg.norm(lane.start - center))
    r_end = float(np.linalg.norm(lane.end - center))
    new_length = extent - min(r_start, r_end)
    if new_length <= 1.0 or new_length >= length:
        return lane  # nothing sensible to trim
    if r_start > r_end:                      # `start` is the outer tip
        return _remake(lane, start=lane.end - direction * new_length)
    return _remake(lane, end=lane.start + direction * new_length)  # `end` is outer


def _ring_lane_priority(lane_index: tuple[str, str]) -> int | None:
    """Right-of-way of a roundabout lane, or None to keep the stock priority."""
    _from, _to = lane_index[0], lane_index[1]
    ring_nodes = {"se", "ex", "ee", "nx", "ne", "wx", "we", "sx"}
    if _from in ring_nodes:                       # circulating on the ring
        return PRIORITY_RING
    if _to in ring_nodes:                         # ramp -> ring (must yield)
        return PRIORITY_RB_ENTRY
    if _from in {"sx", "ex", "nx", "wx"}:         # ring -> ramp (leaving)
        return PRIORITY_RB_EXIT
    return None


# Phase 4B: ring-mouth topology for the Dedicated Ring-Merge Controller.
_RING_NODES_4B = {"se", "ex", "ee", "nx", "ne", "wx", "we", "sx"}
_RING_MOUTH_NODES_4B = {"ne", "ee", "se", "we"}   # mouth where a ring arc meets a slip

# --- Phase 4C-B1.2: GROUND-TRUTH scope counters ----------------------------
# These are bumped on the REAL B1 execution path inside `enforce_road_rules`,
# NOT by a replica probe. B1.1 showed the replica probe disagrees with the real
# decision whenever the B1 rank is active (the probe's rank key is the frozen
# centroid key), so scope/eligibility must NEVER be inferred from it -- read
# these counters instead.
B1_SCOPE_STATS = {
    "groups_total": 0,                          # conflict components evaluated
    "pairwise_rank_applied": 0,                 # vehicles ranked with the B1 key
    "pairwise_rank_applied_intersection": 0,    # components where B1 rank applied
    "pairwise_rank_skipped_same_lane": 0,       # A lane == B lane (ring followers)
    "pairwise_rank_skipped_ring": 0,            # ring-mouth / ring-arc interaction
    "pairwise_rank_skipped_merge": 0,           # registered merge / same-node converge
    "pairwise_rank_skipped_other": 0,           # does not touch the intersection
}


def reset_b1_scope_stats() -> None:
    """Zero the B1 scope counters (call before a scenario / an A/B arm)."""
    for k in B1_SCOPE_STATS:
        B1_SCOPE_STATS[k] = 0


def get_b1_scope_stats() -> dict:
    """Snapshot of the B1 scope counters."""
    return dict(B1_SCOPE_STATS)


def _absorb(network: RoadNetwork, source: RoadNetwork, shift: np.ndarray,
            extent: float | None = None) -> None:
    """Copy every lane of `source` into `network`, shifted by `shift`.

    When `extent` is given, straight access lanes are trimmed to that radius
    (measured from the feature centre, which is the ORIGIN in the source frame,
    before `shift` is applied).
    """
    center = np.zeros(2)
    for _from, edges in source.graph.items():
        for _to, lanes in edges.items():
            for lane in lanes:
                if extent is not None:
                    lane = _trim_lane(lane, center, extent)
                network.add_lane(
                    _from, _to,
                    _shift_lane(lane, shift, _ring_lane_priority((_from, _to))),
                )


# --- 6. Network -----------------------------------------------------------
def build_compound_network() -> RoadNetwork:
    """Intersection at the origin + roundabout at (RB_X, RB_Y), bridged."""
    # Constructing an env also builds one vehicle, but we only keep the network.
    inter_net = IntersectionEnv().road.network
    rb_net = RoundaboutEnv().road.network

    net = RoadNetwork()
    _absorb(net, inter_net, np.zeros(2), extent=INTER_ARM_EXTENT)
    _absorb(net, rb_net, np.array([RB_X, RB_Y]), extent=RB_ARM_EXTENT)

    # Terminals of the two features, after the arms have been trimmed.
    o3_out = np.array([INTER_ARM_EXTENT, 2.0])      # il3 -> o3  (leaving the X)
    o3_in = np.array([INTER_ARM_EXTENT, -2.0])      # o3 -> ir3  (entering the X)
    wer = np.array([RB_X - RB_ARM_EXTENT, 2.0])     # roundabout west entry
    wxr = np.array([RB_X - RB_ARM_EXTENT, -2.0])    # roundabout west exit

    # Two-lane link road, one carriageway per direction. Lane 0 is the right
    # lane (the straight continuation of the junction lane), lane 1 the
    # overtaking lane; the dashed line between them lets vehicles change lane.
    n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
    for k in range(BRIDGE_LANES):
        y_out = 2.0 + k * LANE_WIDTH        # eastbound (intersection -> ring)
        y_in = -2.0 - k * LANE_WIDTH        # westbound (ring -> intersection)
        line_out = [s, c] if k == 0 else [c, s]
        line_in = [s, c] if k == 0 else [c, s]
        net.add_lane(
            "o3", "wer",
            StraightLane([o3_out[0], y_out], [wer[0], y_out],
                         line_types=line_out, priority=PRIORITY_BRIDGE),
        )
        net.add_lane(
            "wxr", "o3",
            StraightLane([wxr[0], y_in], [o3_in[0], y_in],
                         line_types=line_in, priority=PRIORITY_BRIDGE),
        )
    return net


def _travel_dir_v(v) -> np.ndarray:
    """Unit travel direction of a vehicle (fallback: heading)."""
    d = getattr(v, "direction", None)
    if d is not None and np.linalg.norm(d) > 1e-6:
        return np.asarray(d, dtype=float) / np.linalg.norm(d)
    h = getattr(v, "heading", None)
    if h is not None:
        return np.array([np.cos(h), np.sin(h)], dtype=float)
    return np.array([1.0, 0.0])


def _segment_intersection(p1, p2, p3, p4):
    """2-D segment/segment intersection point, or None."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p1 + t * d1
    return None


class CompoundRoad(RegulatedRoad):
    """RegulatedRoad with curve-aware conflict prediction.

    env_core's stock ``RegulatedRoad`` predicts every vehicle along a STRAIGHT
    line at constant speed. On the roundabout ring (and on the intersection's
    turn arcs) that prediction leaves the lane almost immediately, so any two
    cars inside the junction look like they are about to collide. The
    lower-priority one is then ordered to stop, the phantom conflict never
    clears, and the whole map deadlocks within a few seconds.

    Predicting along each vehicle's CURRENT LANE instead keeps the ring
    free-flowing while genuine conflicts (a car pulling onto the ring, two
    paths crossing inside the intersection) are still resolved by priority.
    Conflicts inside a single lane are skipped too -- IDM car-following already
    handles those far better than a hard "stop".
    """

    @staticmethod
    def _predict_along_lane(vehicle, times):
        lane = vehicle.lane
        s0, lateral = lane.local_coordinates(vehicle.position)
        heading_error = (vehicle.heading - lane.heading_at(s0) + np.pi) % (2 * np.pi) - np.pi
        positions, headings = [], []
        for t in times:
            s = s0 + vehicle.speed * t
            alignment = np.exp(-abs(vehicle.speed * t) / vehicle.LENGTH)
            positions.append(lane.position(s, lateral * alignment))
            headings.append(lane.heading_at(s) + heading_error * alignment)
        return positions, headings

    @staticmethod
    def _route_lane_chain(vehicle, max_lanes: int = 4):
        """Aligned route chain for `vehicle`, starting at its CURRENT lane.

        Never trust route[0]/route[1] blindly: routes go stale the moment the
        car leaves the lane they were planned for. Instead, locate the car's
        live lane INSIDE the route (exact (from, to, k) match first, then a
        UNIQUE (from, to) prefix match -- k can be stale after a lane change)
        and return [current, next, next-next, ...] from that point.

        Returns None when the route is missing or cannot be aligned -- the
        caller must then fall back to the current-lane prediction rather than
        guessing a next lane (a guessed wrong turn manufactures phantom
        conflicts, which deadlocks the map as surely as misses do).
        """
        li = tuple(vehicle.lane_index)
        route = getattr(vehicle, "route", None) or []
        idx = None
        for i, r in enumerate(route):
            if tuple(r) == li:
                idx = i
                break
        if idx is None:
            matches = [i for i, r in enumerate(route)
                       if tuple(r)[:2] == li[:2]]
            if len(matches) == 1:
                idx = matches[0]
        if idx is None:
            return None
        chain = [li] + [tuple(r) for r in route[idx + 1:idx + max_lanes]]
        return chain

    def _predict_along_route(self, vehicle, times):
        """Constant-speed trajectory that follows the vehicle's ROUTE.

        Starts on the current lane at the car's arc position and consumes the
        prediction distance s = s0 + v*t across lane boundaries: when s runs
        past the end of the current lane the remainder is spent on the next
        route lane (aligned by `_route_lane_chain`), then the next-next, up to
        `max_lanes`. Beyond the end of route knowledge the last lane's
        geometry is extrapolated -- identical to the old current-lane
        behaviour, so an exhausted route degrades gracefully instead of
        inventing a turn.

        Per-vehicle fallback: a stale/unresolvable route yields the old
        `_predict_along_lane` trajectory for THAT vehicle only; the other
        car keeps its route-aware trajectory if it has one.
        """
        try:
            chain = self._route_lane_chain(vehicle)
            if chain is None:
                return CompoundRoad._predict_along_lane(vehicle, times)
            lane = self.network.get_lane(chain[0])
            s0, lateral = lane.local_coordinates(vehicle.position)
            heading_error = (vehicle.heading - lane.heading_at(s0) + np.pi) % (2 * np.pi) - np.pi
        except Exception:
            return CompoundRoad._predict_along_lane(vehicle, times)
        positions, headings = [], []
        speed = float(vehicle.speed)
        # Resolve the chain lazily but ONCE: walk the FULL distance ahead of
        # the car (speed*t from its CURRENT position) from scratch for every
        # prediction time -- recomputing s against an already-advanced lane
        # silently forgets the arc length consumed by earlier transitions
        # (measured as 20-50 m position jumps). Resolved lane ids are cached
        # in `chain` so next_lane_given_next_road runs once per boundary.
        for t in times:
            dist = speed * float(t)          # travel from the car, not lane 0
            lane = self.network.get_lane(chain[0])
            s = s0
            ci = 0
            while True:
                avail = lane.length - s      # remaining arc on this lane
                if dist <= avail or ci + 1 >= len(chain):
                    s += dist                # beyond the chain: extrapolate
                    break
                dist -= avail
                cur, nxt = chain[ci], chain[ci + 1]
                if chain[ci + 1][2] is None:  # resolve lane id once
                    try:
                        proj = lane.position(float(lane.length), 0.0)
                        nid, _ = self.network.next_lane_given_next_road(
                            cur[0], cur[1], cur[2], nxt[1], nxt[2], proj)
                    except Exception:
                        nid = nxt[2]
                    if nid is None:
                        nid = 0
                    chain[ci + 1] = (nxt[0], nxt[1], int(nid))
                ci += 1
                lane = self.network.get_lane(chain[ci])
                s = 0.0
            # Forecast starts at the physical pose, including a partially
            # completed turn/lane change. Alignment requires travelled distance:
            # a stationary car must never snap onto the lane's centreline.
            alignment = np.exp(-abs(speed * float(t)) / vehicle.LENGTH)
            positions.append(lane.position(s, lateral * alignment))
            headings.append(lane.heading_at(s) + heading_error * alignment)
        return positions, headings

    def is_conflict_possible(
        self,
        v1,
        v2,
        horizon: int = 3,
        step: float = 0.25,
    ) -> bool:
        from env_core import utils

        if v1.lane_index == v2.lane_index:
            return False  # same lane -> IDM car-following handles it
        times = np.arange(step, horizon, step)
        if getattr(_cfg, "CROSS_ROUTE_AWARE_PREDICTION", False):
            # Phase 2C.1: follow each car's actual route across lane
            # boundaries. _predict_along_route falls back to the current-lane
            # prediction per vehicle, so a route-less/stale car degrades to
            # exactly the old behaviour instead of guessing a turn.
            positions_1, headings_1 = self._predict_along_route(v1, times)
            positions_2, headings_2 = self._predict_along_route(v2, times)
        else:
            positions_1, headings_1 = CompoundRoad._predict_along_lane(v1, times)
            positions_2, headings_2 = CompoundRoad._predict_along_lane(v2, times)
        for position_1, heading_1, position_2, heading_2 in zip(
            positions_1, headings_1, positions_2, headings_2, strict=False
        ):
            # Fast spherical pre-check
            if np.linalg.norm(position_2 - position_1) > v1.LENGTH:
                continue
            # Accurate rectangular check
            if utils.rotated_rectangles_intersect(
                (position_1, 1.5 * v1.LENGTH, 0.9 * v1.WIDTH, heading_1),
                (position_2, 1.5 * v2.LENGTH, 0.9 * v2.WIDTH, heading_2),
            ):
                return True
        return False

    def step(self, dt: float) -> None:
        if getattr(_cfg, 'CONFLICT_RESERVATIONS', False) and hasattr(self, '_reservations'):
            self._reservations.constrain_acceleration(dt)
        # Background IDM cars must never reverse: with a leader at ~1 m the
        # IDM commands sustained negative acceleration. Clamping the SPEED
        # alone still allows a negative speed between steps. Bound braking
        # by the zero-speed endpoint of this integration step instead.
        from env_core.vehicle.controller import (ControlledVehicle,
                                                    MDPVehicle)
        for v in self.vehicles:
            if not (isinstance(v, ControlledVehicle)
                    and not isinstance(v, MDPVehicle)):
                continue
            if v.speed < 0.0:
                v.speed = 0.0
            if isinstance(v.action, dict) \
                    and "acceleration" in v.action:
                # Do not cancel braking at a positive speed: the old <0.3 m/s
                # clamp held that speed forever and crept into stopped leaders.
                # Bound deceleration by exactly the amount needed to reach zero
                # this physics step, preventing reverse motion without creep.
                v.action["acceleration"] = max(
                    float(v.action["acceleration"]), -float(v.speed) / dt)
        return super().step(dt)

    # -- Phase 2: crossing group arbitration --------------------------------

    def _lane_prio(self, li) -> float:
        # NOTE: graph[from][to] is a LIST of lane objects (not dicts), so the
        # priority is an ATTRIBUTE of the lane, not a dict key. Reading it as a
        # dict key raised and silently fell back to 1.0 -- which meant lane
        # priority was NEVER actually used by the ranking (every car ranked with
        # -1.0 for priority). Read the lane object's .priority attribute instead.
        try:
            lane = self.network.get_lane(li)
            return float(getattr(lane, "priority", 1.0))
        except Exception:
            return 1.0

    def _lane_crossing_table(self) -> dict:
        """Where do lane centrelines geometrically CROSS, keyed by link pair.

        The constant-speed predictor is blind to STOPPED face-offs: both
        predicted trajectories stand still, so nothing ever intersects and the
        stock pairwise regulation never fires (the seed-1 gridlock). This
        table supplies the missing geometry: which links cross and where.
        Built lazily, once per road.
        """
        table = getattr(self, "_cross_table", None)
        if table is not None:
            return table
        table = {}
        lanes = []
        graph = self.network.graph
        for i_from, tos in graph.items():
            for i_to, lanelist in tos.items():
                for idx, ldata in enumerate(lanelist):
                    # The vendored env_core stores graph[from][to] as a LIST of
                    # lane objects directly. A previous edit assumed a
                    # {"lane": <Lane>} dict wrapper and so kept `lane` None for
                    # every entry -> the table was ALWAYS empty, which silently
                    # disabled the static-mouth and ETA arbitration branches (they
                    # both key off this table). Use the lane object directly when
                    # it is not a dict; this makes the intended mechanism run.
                    lane = ldata if not isinstance(ldata, dict) else ldata.get("lane")
                    if lane is None:
                        continue
                    lanes.append(((str(i_from), str(i_to), int(idx)), lane))
        step = 2.0
        for x in range(len(lanes)):
            li_a, lane_a = lanes[x]
            pa = [np.asarray(lane_a.position(float(s), 0.0))
                  for s in np.arange(0.0, lane_a.length, step)]
            pa.append(np.asarray(lane_a.position(float(lane_a.length), 0.0)))
            ba = (min(p[0] for p in pa), min(p[1] for p in pa),
                  max(p[0] for p in pa), max(p[1] for p in pa))
            for y in range(x + 1, len(lanes)):
                li_b, lane_b = lanes[y]
                if li_a[:2] == li_b[:2]:
                    continue        # same link: parallel lanes never cross
                pb = [np.asarray(lane_b.position(float(s), 0.0))
                      for s in np.arange(0.0, lane_b.length, step)]
                pb.append(np.asarray(lane_b.position(float(lane_b.length), 0.0)))
                bb = (min(p[0] for p in pb), min(p[1] for p in pb),
                      max(p[0] for p in pb), max(p[1] for p in pb))
                if (ba[2] < bb[0] or bb[2] < ba[0]
                        or ba[3] < bb[1] or bb[3] < ba[1]):
                    continue        # bounding boxes apart
                for m in range(len(pa) - 1):
                    hit = None
                    for k in range(len(pb) - 1):
                        pt = _segment_intersection(
                            pa[m], pa[m + 1], pb[k], pb[k + 1])
                        if pt is not None:
                            hit = pt
                            break
                    if hit is not None:
                        key = (li_a[:2], li_b[:2])
                        key = key if key[0] <= key[1] else (key[1], key[0])
                        table.setdefault(key, []).append(
                            [float(hit[0]), float(hit[1])])
                        break
        self._cross_table = table
        return table

    def _static_mouth_conflict(self, a, b) -> bool:
        """Topology+geometry conflict for SLOW cars the predictor cannot see.

        Both cars crawling/stopped, their lanes' centrelines cross, and the
        crossing point is within the mouth window of BOTH (ahead of them, or
        just past -- still occupying the zone). Deterministic, cheap, and it
        never fires for fast traffic (the dynamic predictor covers that).
        """
        la = tuple(getattr(a, "lane_index", ()))[:2]
        lb = tuple(getattr(b, "lane_index", ()))[:2]
        if not la or not lb or la == lb:
            return False
        key = (la, lb) if la <= lb else (lb, la)
        pts = self._lane_crossing_table().get(key)
        if not pts:
            return False
        try:
            lane_a = self.network.get_lane(a.lane_index)
            lane_b = self.network.get_lane(b.lane_index)
            sa = lane_a.local_coordinates(a.position)[0]
            sb = lane_b.local_coordinates(b.position)[0]
        except Exception:
            return False
        ahead, behind = _cfg.CROSS_STATIC_AHEAD, _cfg.CROSS_STATIC_BEHIND
        for p in pts:
            p = np.asarray(p)
            da = float(lane_a.local_coordinates(p)[0]) - sa
            db = float(lane_b.local_coordinates(p)[0]) - sb
            if (-behind <= da <= ahead) and (-behind <= db <= ahead):
                return True
        return False

    def _arbitration_conflict(self, a, b) -> bool:
        """Conflict test for the ARBITRATION (predictor identity untouched).

        Dynamic part: the stock curve-aware constant-speed predictor at 5 s.
        Its blind spot for FAST crossing pairs is geometry: two boxes passing
        a 37-degree crossing only overlap ~0.7 s before impact, which leaves
        no braking distance. ETA part: for lane pairs whose centrelines cross
        (the static table), flag a conflict when both cars are heading for
        that crossing and their arrival times are within CROSS_ETA_WINDOW --
        this fires seconds earlier and is what makes high-speed crossing
        pairs yieldable at all.
        """
        try:
            if self.is_conflict_possible(a, b, horizon=5.0):
                return True
        except Exception:
            pass
        if max(a.speed, b.speed) < _cfg.CROSS_STATIC_SPEED:
            return self._static_mouth_conflict(a, b)   # stopped face-off
        la = tuple(getattr(a, "lane_index", ()))[:2]
        lb = tuple(getattr(b, "lane_index", ()))[:2]
        if not la or not lb or la == lb:
            return False
        key = (la, lb) if la <= lb else (lb, la)
        pts = self._lane_crossing_table().get(key)
        if not pts:
            return False
        try:
            lane_a = self.network.get_lane(a.lane_index)
            lane_b = self.network.get_lane(b.lane_index)
            sa = lane_a.local_coordinates(a.position)[0]
            sb = lane_b.local_coordinates(b.position)[0]
        except Exception:
            return False
        look = _cfg.MERGE_LOOKAHEAD
        for p in pts:
            p = np.asarray(p)
            da = float(lane_a.local_coordinates(p)[0]) - sa
            db = float(lane_b.local_coordinates(p)[0]) - sb
            if not (0.0 < da <= look and 0.0 < db <= look):
                continue                    # crossing not ahead of both
            eta_a = da / max(a.speed, 0.5)
            eta_b = db / max(b.speed, 0.5)
            if abs(eta_a - eta_b) <= _cfg.CROSS_ETA_WINDOW:
                return True
        return False

    # --- Phase 2C.3: registered ring-mouth MERGE pairs ----------------------
    # Only THREE pairs are registered (2C.1b-proven blind spots). They are NOT
    # crossings: both lanes feed the SAME downstream ring node, so the correct
    # handling is "form a unique passage order" (one yields, one GOes) -- exactly
    # what the conflict-group arbitration already does. The point of registering
    # them is to GUARANTEE the conflict edge exists (the generic predictor/ETA
    # window detects them late at the approach->internal-lane transition, which
    # is why they crash ~10x in baseline). Kept tightly scoped: no "all shared
    # endpoints" or "all converging lanes" generalisation.
    @staticmethod
    def _ring_merge_pairs():
        """The three registered ring-mouth merge pairs as unordered lane-2-tuple
        sets. (circ_from, circ_to) and (entry_from, entry_to) both END at the
        same downstream ring node -- that shared 'to' node is the convergence."""
        return {
            frozenset((("sx", "se"), ("ses", "se"))),   # south ring mouth
            frozenset((("nx", "ne"), ("nes", "ne"))),   # north ring mouth
            frozenset((("ex", "ee"), ("ees", "ee"))),   # east ring mouth
        }

    def _is_registered_merge_pair(self, a, b) -> bool:
        """True iff (a,b) is one of the three registered ring-mouth merge pairs."""
        if not getattr(_cfg, "CROSS_RING_MERGE_PAIRS", False):
            return False
        la = tuple(getattr(a, "lane_index", ()))[:2]
        lb = tuple(getattr(b, "lane_index", ()))[:2]
        if not la or not lb or la == lb:
            return False
        return frozenset((la, lb)) in self._ring_merge_pairs()

    def _ring_merge_conflict(self, a, b) -> bool:
        """Merge conflict for the three registered pairs: both cars are still
        approaching the SAME downstream ring node and within
        CROSS_RING_MERGE_LOOKAHEAD of it. Proximity-gated so a far / already
        cleared car never triggers a spurious limitation. Uses the shared 'to'
        node (== lane end) as the convergence point, which is robust and does not
        depend on the geometric crossing table.
        """
        if not getattr(_cfg, "CROSS_RING_MERGE_PAIRS", False):
            return False
        la = tuple(getattr(a, "lane_index", ()))[:2]
        lb = tuple(getattr(b, "lane_index", ()))[:2]
        if not la or not lb or la == lb or la[1] != lb[1]:
            return False
        if frozenset((la, lb)) not in self._ring_merge_pairs():
            return False
        try:
            lane_a = self.network.get_lane(a.lane_index)
            lane_b = self.network.get_lane(b.lane_index)
            sa = lane_a.local_coordinates(a.position)[0]
            sb = lane_b.local_coordinates(b.position)[0]
        except Exception:
            return False
        # distance remaining to the shared downstream node (lane end)
        da = lane_a.length - sa
        db = lane_b.length - sb
        look = _cfg.CROSS_RING_MERGE_LOOKAHEAD
        return (0.0 < da <= look) and (0.0 < db <= look)

    # --- Phase 4B: Dedicated Ring-Merge topology rule ----------------------
    def _is_ring_mouth_pair(self, a, b):
        """True iff (a,b) is a dedicated ring-mouth conflict.

        Both cars converge on the SAME mouth node M; exactly ONE of them is on a
        ring arc (from-node in _RING_NODES_4B => circulating) and the other on an
        external slip (feeder). Returns (circulating, feeder, mouth_node) or None.

        This is a TOPOLOGY rule -- it does NOT read lane priority (which is 1.0 on
        both mouth lanes, the root cause of the Phase-4A wrong right-of-way), and
        it is order-independent. The four mouths are exactly:
            N: nx->ne (circ) vs nes->ne (feed)
            E: ex->ee (circ) vs ees->ee (feed)
            S: sx->se (circ) vs ses->se (feed)
            W: wx->we (circ) vs wes->we (feed)
        """
        la = tuple(getattr(a, "lane_index", ()))[:2]
        lb = tuple(getattr(b, "lane_index", ()))[:2]
        if not la or not lb or la == lb or la[1] != lb[1]:
            return None
        M = la[1]
        if M not in _RING_MOUTH_NODES_4B:
            return None
        ra = "circ" if la[0] in _RING_NODES_4B else "feed"
        rb = "circ" if lb[0] in _RING_NODES_4B else "feed"
        if ra == rb:
            return None                     # both ring or both non-ring: not this pair
        C = a if ra == "circ" else b
        F = b if ra == "circ" else a
        return (C, F, M)

    # --- Phase 3B: narrow route-aware LEADER fallback ----------------------
    def _lstats(self):
        """Per-road leader-lookup counters (lazily created)."""
        st = getattr(self, "_leader_stats", None)
        if st is None:
            from collections import Counter
            st = Counter()
            self._leader_stats = st
        return st

    def _route_aware_leader(self, v):
        """Narrow fallback leader for `v`, used ONLY when the stock exact-lane
        search found nobody (see `neighbour_vehicles` below).

        Admits exactly two kinds of candidate and nothing else:

          (1) direct successor -- a car on the ego's ACTUAL route successor lane.
              The route is aligned to the live lane first via
              `_route_lane_chain`; a stale / unalignable route returns None
              rather than guessing (a guessed successor is a phantom leader).
          (2) strict converging -- a car on ANOTHER lane feeding the SAME
              downstream node, that additionally satisfies ALL of:
                  * heading within ROUTE_LEADER_HEADING_MAX of the ego
                    (otherwise it is crossing traffic, not converging);
                  * closer to that node than the ego (it is AHEAD in the merge);
                  * both cars within ROUTE_LEADER_MERGE_DIST of the node.

        Deliberately rejected (counted, never adopted):
            - parallel lanes of the ego's own link   -> rejected_parallel
            - crossing / heading-mismatched traffic  -> rejected_crossing
            - unalignable (stale) route              -> rejected_not_on_route
            - topologically valid but beyond a gate  -> rejected_too_far

        The gap is measured ALONG THE ROUTE (remaining distance to the boundary
        plus the candidate's progress past it), never as a Euclidean distance.
        """
        if not getattr(_cfg, "ROUTE_AWARE_LEADER", False):
            return None
        stats = self._lstats()
        try:
            li = tuple(v.lane_index)
            lane = self.network.get_lane(li)
            s = float(lane.local_coordinates(v.position)[0])
        except Exception:
            return None
        ego_remaining = float(lane.length) - s
        chain = self._route_lane_chain(v)
        if chain is None or len(chain) < 2:
            # stale / unalignable route: do NOT guess a successor
            stats["rejected_not_on_route"] += 1
            return None
        succ = tuple(chain[1])
        try:
            ego_heading = float(v.heading)
        except Exception:
            ego_heading = None
        best = None                       # (gap, candidate, kind)
        gap_max = _cfg.ROUTE_LEADER_GAP_MAX
        merge_dist = _cfg.ROUTE_LEADER_MERGE_DIST
        head_max = _cfg.ROUTE_LEADER_HEADING_MAX

        for u in self.vehicles:
            if u is v or getattr(u, "crashed", False):
                continue
            uli = tuple(getattr(u, "lane_index", ()) or ())
            if len(uli) < 2:
                continue
            # ---- (1) the ego's actual route successor --------------------
            if uli[:2] == succ[:2] and (len(succ) < 3 or succ[2] is None
                                        or len(uli) < 3
                                        or uli[2] == succ[2]):
                try:
                    ul = self.network.get_lane(uli)
                    us = float(ul.local_coordinates(u.position)[0])
                except Exception:
                    continue
                gap = ego_remaining + us
                if gap <= gap_max:
                    if best is None or gap < best[0]:
                        best = (gap, u, "successor")
                else:
                    stats["rejected_too_far"] += 1
                continue
            # ---- parallel lane of the ego's OWN link: never a leader -----
            if uli[:2] == li[:2]:
                stats["rejected_parallel"] += 1
                continue
            # ---- (2) strict converging into the same downstream node -----
            if len(uli) > 1 and uli[1] == li[1]:
                try:
                    ul = self.network.get_lane(uli)
                    us = float(ul.local_coordinates(u.position)[0])
                except Exception:
                    continue
                u_remaining = float(ul.length) - us
                if ego_remaining > merge_dist or u_remaining > merge_dist:
                    stats["rejected_too_far"] += 1
                    continue
                if u_remaining >= ego_remaining:
                    # not ahead of the ego in the merge -> not a leader
                    stats["rejected_too_far"] += 1
                    continue
                if ego_heading is not None:
                    try:
                        dh = (float(u.heading) - ego_heading + np.pi) \
                            % (2 * np.pi) - np.pi
                        if abs(np.degrees(dh)) > head_max:
                            stats["rejected_crossing"] += 1
                            continue
                    except Exception:
                        pass
                gap = ego_remaining + us
                if gap <= gap_max:
                    if best is None or gap < best[0]:
                        best = (gap, u, "converging")
                else:
                    stats["rejected_too_far"] += 1
                continue
            # ---- anything else: crossing / unrelated connected lane -----
            stats["rejected_crossing"] += 1

        if best is None:
            return None
        stats["fallback_" + best[2]] += 1
        return best[1]

    def neighbour_vehicles(self, vehicle, lane_index=None):
        """Stock search first; fill ONLY the exact blind spot Phase 3A measured.

        The native (exact-lane) leader always wins -- when one exists the result
        is bit-for-bit the stock behaviour, which is why this fills a blind spot
        instead of rewriting IDM. The fallback is additionally confined to:
          * vehicles with NO native leader;
          * background traffic (never the RL-controlled MDPVehicle agents);
          * the ego's OWN lane probe -- the lane-change policy asks about a
            different (target) lane and must keep the stock answer.
        """
        front, back = super().neighbour_vehicles(vehicle, lane_index)
        if front is not None:
            self._lstats()["native_exact"] += 1
            return front, back
        if not getattr(_cfg, "ROUTE_AWARE_LEADER", False):
            return front, back
        if lane_index is not None and \
                tuple(lane_index) != tuple(getattr(vehicle, "lane_index", ()) or ()):
            return front, back          # lane-change probe: keep stock
        try:
            from env_core.vehicle.controller import MDPVehicle
            if isinstance(vehicle, MDPVehicle):
                return front, back      # RL agents keep the stock semantics
        except Exception:
            pass
        fb = self._route_aware_leader(vehicle)
        if fb is None:
            return front, back
        self._lstats()["native_absent_filled"] += 1
        return fb, back

    # --- Phase 4C-B1: pairwise actual-conflict-point geometry (rank helpers) --
    # These mirror src/_diag_4c_b1_rankcheck.py so the READ-ONLY regression and
    # the wired-in behaviour use identical math. They are only consulted when
    # CROSS_PAIRWISE_CONFLICT_RANK=True and the conflict group is a pure
    # intersection crossing group (no ring-mouth / registered-merge pair).
    def _lane_polyline_cr(self, li, step: float = 2.0):
        lane = self.network.get_lane(li)
        L = float(lane.length)
        pts = []
        s = 0.0
        while s < L:
            pts.append(np.asarray(lane.position(s, 0.0)))
            s += step
        pts.append(np.asarray(lane.position(L, 0.0)))
        return pts

    @staticmethod
    def _seg_intersect_cr(p1, p2, p3, p4):
        r = p2 - p1
        s = p4 - p3
        rxs = float(r[0] * s[1] - r[1] * s[0])
        if abs(rxs) < 1e-9:
            return None
        qp = p3 - p1
        t = float(qp[0] * s[1] - qp[1] * s[0]) / rxs
        u = float(qp[0] * r[1] - qp[1] * r[0]) / rxs
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return p1 + t * r
        return None

    def _path_of_cr(self, v):
        """Full trajectory polyline = current lane + up to 2 ROUTE lanes.

        vehicle.route layout (env_core controller): first entry == self.lane_index
        (a 3-tuple lane index), then continuation entries are 3-tuples
        (origin_node, destination_node, None). Build each continuation lane index
        as (origin_node, destination_node, 0).
        """
        pts = self._lane_polyline_cr(v.lane_index)
        route = getattr(v, "route", None) or []
        added = 0
        for entry in route:
            if added >= 2:
                break
            if entry == v.lane_index:
                continue
            try:
                li = (entry[0], entry[1], 0)
                pts += self._lane_polyline_cr(li)[1:]
                added += 1
            except Exception:
                break
        return pts

    def _conflict_point_v(self, a, b):
        """Actual pairwise conflict point = first geometric crossing of the two
        vehicles' route-aware trajectories. Falls back to the midpoint of the two
        trajectory ends (should not happen for a real conflict pair)."""
        pa = self._path_of_cr(a)
        pb = self._path_of_cr(b)
        for i in range(len(pa) - 1):
            for j in range(len(pb) - 1):
                p = self._seg_intersect_cr(pa[i], pa[i + 1], pb[j], pb[j + 1])
                if p is not None:
                    return p
        return 0.5 * (pa[-1] + pb[-1])

    def _movement_class_cr(self, v):
        """0 = through, 1 = left, 2 = right (consistent frame; only a tie-break)."""
        lane = self.network.get_lane(v.lane_index)
        try:
            inb = np.asarray(lane.position(lane.length * 0.5, 0.0)) - \
                np.asarray(lane.position(0.0, 0.0))
        except Exception:
            return 0
        nxt = None
        route = getattr(v, "route", None) or []
        if len(route) >= 2:
            try:
                nli = (route[0][1], route[1][1], 0)
                nl = self.network.get_lane(nli)
                nxt = np.asarray(nl.position(nl.length * 0.5, 0.0)) - \
                    np.asarray(nl.position(0.0, 0.0))
            except Exception:
                nxt = None
        outb = nxt if nxt is not None else inb
        n1 = np.linalg.norm(inb)
        n2 = np.linalg.norm(outb)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0
        ua = inb / n1
        ub = outb / n2
        cross = float(ua[0] * ub[1] - ua[1] * ub[0])   # 2D cross
        ang = float(np.arctan2(cross, float(np.dot(ua, ub))))
        if abs(ang) < 0.35:
            return 0
        return 1 if ang > 0 else 2

    # --- Phase 4C-B1.2: POSITIVE intersection-crossing whitelist -----------
    # B1.1 proved the old gate was a NEGATIVE exclusion ("anything that is not a
    # ring-mouth / registered-merge pair may be re-ranked"), which let B1 onto
    # same-lane ring followers (nx->ne|nx->ne, ex->ee|ex->ee, sx->se|sx->se) and
    # produced 4 DIRECT_SCOPE_LEAK ring crashes. B1.2 replaces it with a
    # whitelist over the conflict edge itself. NOTE: this changes WHERE the
    # pairwise rank is allowed -- never HOW it ranks (key/ETA/movement/serial,
    # occupancy threshold and yield caps are untouched).
    def _b1_touches_intersection(self, la) -> bool:
        """True iff either endpoint of the lane 2-tuple belongs to the signalised
        intersection (approach arms o*/c and the internal ir*/il* lanes)."""
        return any(str(n).startswith(("ir", "il", "o")) or n == "c"
                   for n in (la or ())[:2])

    def _b1_edge_class(self, a, b) -> str:
        """Classify one pairwise conflict edge for B1 eligibility.

        Only "crossing" may use the pairwise actual-conflict-point rank; every
        other class is hard-excluded and returns its reason so the ground-truth
        counters can attribute each skip.

            same_lane  - A lane == B lane          (ring followers / car-following)
            ring       - canonical ring-mouth pair, or either lane is a ring arc
            merge      - registered merge pair, or both lanes feed the SAME
                         downstream node (converging / merge)
            other      - the pair does not touch the intersection at all
            crossing   - genuine intersection crossing  -> B1 ELIGIBLE
        """
        la = tuple(getattr(a, "lane_index", ()))[:2]
        lb = tuple(getattr(b, "lane_index", ()))[:2]
        if not la or not lb:
            return "other"
        if la == lb:
            return "same_lane"
        # canonical ring-mouth (all four mouths, incl. the unregistered W mouth)
        if self._is_ring_mouth_pair(a, b) is not None:
            return "ring"
        # the three registered ring-mouth merge pairs
        if self._is_registered_merge_pair(a, b):
            return "merge"
        # either lane is a ring arc (both endpoints are ring nodes)
        if (la[0] in _RING_NODES_4B and la[1] in _RING_NODES_4B) or \
                (lb[0] in _RING_NODES_4B and lb[1] in _RING_NODES_4B):
            return "ring"
        # converging / merge: both lanes feed the SAME downstream node
        if la[1] == lb[1]:
            return "merge"
        # must actually touch the intersection to be an intersection crossing
        if not (self._b1_touches_intersection(la)
                or self._b1_touches_intersection(lb)):
            return "other"
        return "crossing"

    def _b1_group_eligible(self, bg, members, conflict_m):
        """A conflict component is B1-eligible iff EVERY conflicting edge inside
        it is a genuine intersection crossing.

        Deliberately NOT "all members are internal lanes": that would drop the
        real approach->internal crossings (e.g. o3->ir3 vs ir2->il3) that B1 was
        built to fix. One excluded edge (same-lane follower, ring, merge) still
        disqualifies the whole component, so it falls back to the frozen rank.
        """
        m = len(members)
        n_cross = 0
        for a in range(m):
            for b in range(a + 1, m):
                if not conflict_m[a][b]:
                    continue
                cls = self._b1_edge_class(bg[members[a]], bg[members[b]])
                if cls != "crossing":
                    return False, cls
                n_cross += 1
        if n_cross == 0:
            return False, "other"
        return True, None

    def act(self) -> None:
        from env_core.vehicle.controller import ControlledVehicle, MDPVehicle
        if getattr(_cfg, 'CONFLICT_RESERVATIONS', False):
            if not hasattr(self, '_reservations'):
                from .conflict_reservations import ConflictReservations
                self._reservations = ConflictReservations(self)
            self._reservations.update()
        requested = [(v, v.target_speed) for v in self.vehicles
                     if isinstance(v, ControlledVehicle) and not isinstance(v, MDPVehicle)]
        try:
            for v, target in requested:
                v.target_speed = min(target, getattr(v, '_regulation_speed_cap', float('inf')))
                if getattr(_cfg, 'CONFLICT_RESERVATIONS', False):
                    v.target_speed = min(v.target_speed, getattr(v, '_reservation_speed_cap', float('inf')))
            super().act()
        finally:
            for v, target in requested:
                v.target_speed = target

    def enforce_road_rules(self) -> None:
        from env_core.vehicle.controller import ControlledVehicle
        if getattr(_cfg, 'CONFLICT_RESERVATIONS', False):
            for v in self.vehicles:
                v._regulation_speed_cap = float('inf')
            return
        # Arbitration is an independent constraint. The environment refreshes
        # IDM cruise/merge targets every policy step; that must not erase an
        # outstanding STOP between the slower arbitration ticks.
        requested = [(v, v.target_speed) for v in self.vehicles
                     if isinstance(v, ControlledVehicle)]
        for v, _ in requested:
            v.target_speed = float('inf')
        try:
            self._enforce_compound_rules()
            for v, _ in requested:
                v._regulation_speed_cap = v.target_speed
        finally:
            for v, target in requested:
                v.target_speed = target

    def _enforce_compound_rules(self) -> None:
        """Phase-2 crossing arbitration: conflict GROUPS + hierarchical yielding.

        Replaces RegulatedRoad's pairwise stop-and-go ENTIRELY (no super()):
          2A. conflict edges (curve-aware predictor OR static mouth) are
              union-find'd into groups;
          2B. stopped face-offs are arbitrated via the static mouth table;
          2C. yielders get a PROGRESSIVE speed cap (6 / 3 / 0 by distance to the
              crossing point) instead of target_speed=0; the base target is
              restored by the env every step (_apply_merge_yield), so caps never
              accumulate and release resumes IDM, not lane.speed_limit.

        Hierarchical yielding (the ① fix, CROSS_HIERARCHICAL_YIELD=True):
          every conflict group is sorted into a DETERMINISTIC TOTAL RANKING
          (occupancy -> lane priority -> distance-to-centroid -> stable serial).
          A vehicle of rank k yields to EVERY higher-ranked group member it
          ACTUALLY conflicts with (the same `_arbitration_conflict` used to form
          the group), and its final cap is the MINIMUM of those per-pair caps.
          This closes the GROUP_INTERNAL gap: two yielders no longer only watch
          the single winner, they watch EACH OTHER.

          A vehicle physically BLOCKED by another member (that member projects
          onto its own lane ahead) is forced to yield to that member regardless
          of rank -- this preserves the old "a blocked car cannot be the GO
          winner" safety net and stops a blocked yielder being driven into its
          blocker. Cars with no higher-ranked conflicting member get GO and no
          cap.

          Flipping CROSS_HIERARCHICAL_YIELD=False restores the OLD single-winner
          behaviour (one winner per group, every other member caps against only
          that winner) for a clean single-variable A/B.
        """
        from collections import defaultdict
        from env_core.vehicle.controller import (ControlledVehicle,
                                                    MDPVehicle)

        bg = [v for v in self.vehicles
              if isinstance(v, ControlledVehicle)
              and (getattr(_cfg, 'CONTROLLED_ARBITRATION', False)
                   or not isinstance(v, MDPVehicle))
              and not getattr(v, "crashed", False)]
        for v in bg:
            v._group_go = False
            v._group_yield = False
            v._group_wcap = False
        if not getattr(_cfg, "CROSS_GROUP_ARBITRATION", True) or len(bg) < 2:
            return
        hierarchical = getattr(_cfg, "CROSS_HIERARCHICAL_YIELD", True)
        ownership = getattr(_cfg, "CROSS_RING_MOUTH_OWNERSHIP", True)
        n = len(bg)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                a, b = bg[i], bg[j]
                try:
                    # horizon 5 s (up from the stock 3 s): the diagnosis shows
                    # fast 8 m/s crossing pairs only enter each other's 3 s
                    # constant-speed window when it is already too late to
                    # brake -- trace evidence, per the Phase-2 protocol.
                    # Phase 2C.3: the three registered ring-mouth MERGE pairs are
                    # OR'd in so the conflict edge is GUARANTEED to exist even
                    # when the generic predictor/ETA window detects them late.
                    conflict = (self._arbitration_conflict(a, b)
                                or self._ring_merge_conflict(a, b))
                except Exception:
                    conflict = False
                # Phase 4B: a dedicated ring-mouth pair is OWNED by the Dedicated
                # Ring-Merge Controller and must be ISOLATED from the generic
                # crossing/group arbitration (prevents mode-D: the group yielding
                # the circulating car, and double-controller: the group also
                # yielding the feeder). The pair's specific (feeder, circulating)
                # edge is skipped; the circulating car keeps its right of way
                # against OTHER (non-feeder) hazards via the normal group path.
                if conflict and not (getattr(_cfg, "DEDICATED_RING_MERGE", False)
                                    and self._is_ring_mouth_pair(a, b)):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        for members in groups.values():
            if len(members) < 2:
                continue
            centroid = np.mean([bg[i].position for i in members], axis=0)

            def blocked(i: int) -> bool:
                # A car with another group member projecting ONTO ITS LANE
                # ahead (curve-aware: straight-line cones miss blockers round
                # a bend, measured crash) is PHYSICALLY blocked: making it
                # the GO winner would drive it into the blocker. The radius
                # scales with speed -- a fast winner covers half the radius
                # within one 0.5 s regulation tick.
                v = bg[i]
                try:
                    lane_v = self.network.get_lane(v.lane_index)
                    s_v = lane_v.local_coordinates(v.position)[0]
                except Exception:
                    return False
                radius = max(7.0, 1.5 * float(v.speed))
                for j in members:
                    if j == i:
                        continue
                    # Phase 4B: the Dedicated Ring-Merge Controller owns this pair.
                    # Neither member may be treated as a physical block of the other
                    # by the generic arbitration (the circulating car must never be
                    # force-yielded by the feeder, and the feeder must not be
                    # double-capped here). The dedicated controller meters the feeder.
                    if getattr(_cfg, "DEDICATED_RING_MERGE", False) and \
                            self._is_ring_mouth_pair(v, bg[j]):
                        continue
                    # A circulating ring vehicle is NEVER "blocked" by an entering
                    # vehicle at the mouth (absolute right of way). Without this,
                    # a ring+entry pair would both project onto each other's lane
                    # and read as mutually blocked -> the all-blocked gridlock
                    # breaker would despawn a legitimate right-of-way relationship.
                    if ownership and \
                            int(self._lane_prio(v.lane_index)) == PRIORITY_RING and \
                            int(self._lane_prio(bg[j].lane_index)) == PRIORITY_RB_ENTRY:
                        continue
                    # Phase 2C.3: a registered ring-mouth MERGE pair must not read
                    # as a physical block either (its members converge ON the shared
                    # downstream node, so they legitimately project onto each
                    # other's lanes there). Skipping the block check prevents a
                    # false all-blocked -> gridlock-despawn at the mouth; the merge
                    # arbitration (unique passage order) handles the pair instead.
                    if self._is_registered_merge_pair(v, bg[j]):
                        continue
                    try:
                        s_u = lane_v.local_coordinates(bg[j].position)[0]
                    except Exception:
                        continue
                    if 0.0 < s_u - s_v < radius:
                        return True
                return False

            def blocks(i: int, j: int) -> bool:
                # True if member j projects onto member i's lane AHEAD of i,
                # i.e. i physically cannot advance without hitting j. Used for
                # the forced-yield safety net (same-lane blockers are not
                # "arbitration conflicts" but ARE physical blocks).
                # A circulating ring vehicle is NEVER blocked by an entering
                # vehicle at the mouth -- it has absolute right of way, so the
                # pair must not read as a block (which would let the entry car
                # wrongly freeze or force the circulating car to yield).
                # Phase 4B: a dedicated ring-mouth pair is owned by the Dedicated
                # Ring-Merge Controller -- never read as a block by the group.
                if getattr(_cfg, "DEDICATED_RING_MERGE", False) and \
                        self._is_ring_mouth_pair(bg[i], bg[j]):
                    return False
                if ownership and \
                        int(self._lane_prio(bg[i].lane_index)) == PRIORITY_RING and \
                        int(self._lane_prio(bg[j].lane_index)) == PRIORITY_RB_ENTRY:
                    return False
                # Phase 2C.3: a registered ring-mouth MERGE pair is never a
                # physical block (see blocked() above).
                if self._is_registered_merge_pair(bg[i], bg[j]):
                    return False
                v = bg[i]
                try:
                    lane_v = self.network.get_lane(v.lane_index)
                    s_v = lane_v.local_coordinates(v.position)[0]
                    s_u = lane_v.local_coordinates(bg[j].position)[0]
                except Exception:
                    return False
                radius = max(7.0, 1.5 * float(v.speed))
                return 0.0 < s_u - s_v < radius

            # Phase 4C-B1.2: the B1 pairwise rank is now gated by a POSITIVE
            # intersection-crossing whitelist (see _b1_group_eligible), evaluated
            # below once conflict_m is known.

            # Cache pairwise arbitration conflicts inside this group so the
            # hierarchical loop (and the candidate rank) do not recompute
            # _arbitration_conflict.
            m = len(members)
            conflict_m = [[False] * m for _ in range(m)]
            for a in range(m):
                for b in range(a + 1, m):
                    ci, cj = members[a], members[b]
                    try:
                        # Phase 2C.3: registered merge pairs must appear as a real
                        # conflict here too, otherwise the hierarchical ranking
                        # would never ask the feeder to yield and the group would
                        # resolve to "everyone GO" (no control).
                        c = (self._arbitration_conflict(bg[ci], bg[cj])
                             or self._ring_merge_conflict(bg[ci], bg[cj]))
                    except Exception:
                        c = False
                    conflict_m[a][b] = conflict_m[b][a] = c

            # Phase 4C-B1.2: positive whitelist + ground-truth counters, bumped on
            # the REAL execution path (never inferred from a replica probe).
            b1_eligible = False
            if getattr(_cfg, "CROSS_PAIRWISE_CONFLICT_RANK", False):
                b1_eligible, b1_skip = self._b1_group_eligible(
                    bg, members, conflict_m)
                B1_SCOPE_STATS["groups_total"] += 1
                if b1_eligible:
                    B1_SCOPE_STATS["pairwise_rank_applied_intersection"] += 1
                else:
                    key = "pairwise_rank_skipped_" + b1_skip
                    B1_SCOPE_STATS[key] = B1_SCOPE_STATS.get(key, 0) + 1
            # Stash the REAL decision on every member so diagnostics read ground
            # truth instead of re-deriving eligibility from lane snapshots.
            for _mi in members:
                bg[_mi]._b1_eligible = b1_eligible

            def rank(a: int):
                v = bg[members[a]]
                # Phase 4C-B1.2: pairwise actual-conflict-point rank, applied ONLY
                # to genuine intersection-crossing components.
                if b1_eligible:
                    # Occupancy / clearance of the NEAREST real pairwise conflict
                    # point -> ETA to it -> movement tie-break -> stable serial.
                    min_d = float("inf")
                    for b in range(m):
                        if b == a or not conflict_m[a][b]:
                            continue
                        P = self._conflict_point_v(v, bg[members[b]])
                        d = float(np.linalg.norm(np.asarray(v.position, dtype=float) - P))
                        if d < min_d:
                            min_d = d
                    occ = min_d <= _cfg.CROSS_PAIRWISE_OCC_DIST
                    eta = min_d / max(float(v.speed), 0.5)
                    mv = self._movement_class_cr(v)
                    return (0 if occ else 1, eta, mv, a)
                # OLD rank key (frozen baseline behaviour): occupancy-on-centroid
                # -> dormant lane-priority -> distance-to-centroid -> serial.
                d = float(np.linalg.norm(v.position - centroid))
                if getattr(_cfg, "CROSS_LANE_PRIORITY_RANKING", False):
                    prio = -self._lane_prio(v.lane_index)
                else:
                    prio = -1.0
                return (0 if d <= _cfg.CROSS_OCCUPANCY_DIST else 1,
                        prio, round(d, 2), a)

            rkey = [rank(a) for a in range(m)]
            if b1_eligible:
                B1_SCOPE_STATS["pairwise_rank_applied"] += m
            is_blocked = [blocked(i) for i in members]

            if all(is_blocked):
                # EVERY member is physically blocked (a true cluster lock).
                # No right-of-way rule can unlock it (measured: partial GO
                # orders produce low-speed crunches during the shuffle), so
                # the WHOLE cluster is handed to the gridlock hook -- the env
                # recycles every member back to fresh entry lanes with honest
                # per-car counting. Fallback (no hook): freeze solid.
                if getattr(self, "gridlock_hook", None) and \
                        self.gridlock_hook([bg[i] for i in members]):
                    continue
                for i in members:
                    bg[i]._group_yield = True
                    bg[i].target_speed = min(bg[i].target_speed, 0.0)
                continue

            if not hierarchical:
                # OLD single-winner behaviour (A/B baseline). One winner per
                # group; every other member caps against ONLY that winner.
                pool = [a for a in range(m) if not is_blocked[a]]
                if not pool:
                    # (already handled by `all(is_blocked)` above)
                    for a in range(m):
                        bg[members[a]]._group_yield = True
                        bg[members[a]].target_speed = min(
                            bg[members[a]].target_speed, 0.0)
                    continue
                winner_a = min(pool, key=lambda a: rkey[a])
                w = bg[members[winner_a]]
                for a in range(m):
                    v = bg[members[a]]
                    if a == winner_a:
                        v._group_go = True
                        continue
                    v._group_yield = True
                    v.target_speed = min(v.target_speed,
                                         self._yield_cap(v, w, centroid))
                continue

            # NEW hierarchical yielding: rank k yields to EVERY higher-ranked
            # member it actually conflicts with; final cap = min of per-pair caps.
            #
            # Phase 2B ring-mouth ownership: a circulating ring vehicle (priority
            # == PRIORITY_RING) has ABSOLUTE right of way over an entering vehicle
            # (priority == PRIORITY_RB_ENTRY) at the mouth. For that specific
            # pair the generic ranking is OVERRIDDEN so a circulating car can
            # never yield to an entering car (the old occupancy-first rank let a
            # close entry car outrank a far circulating car -> both yielded ->
            # mouth freeze), and an entering car ALWAYS yields to a circulating
            # one. Scope is strictly ring(prio 5) vs entry(prio 0) pairs that
            # actually conflict; everything else keeps the normal hierarchy.
            for a in range(m):
                i = members[a]
                v = bg[i]
                va_prio = int(self._lane_prio(v.lane_index))
                v_is_ring = (va_prio == PRIORITY_RING)
                v_is_entry = (va_prio == PRIORITY_RB_ENTRY)
                caps = []
                for b in range(m):
                    if b == a:
                        continue
                    j = members[b]
                    vb_prio = int(self._lane_prio(bg[j].lane_index))
                    # Phase 4B: the Dedicated Ring-Merge Controller owns this pair.
                    # Skip the (feeder, circulating) edge entirely so the group
                    # never yields the circulating car (mode D) nor double-caps the
                    # feeder. Isolation is per-pair; the circulating car still obeys
                    # the normal hierarchy against any OTHER conflicting member.
                    if getattr(_cfg, "DEDICATED_RING_MERGE", False) and \
                            self._is_ring_mouth_pair(bg[i], bg[j]):
                        continue
                    ring_entry_pair = (ownership and conflict_m[a][b]
                                       and ((v_is_ring and vb_prio == PRIORITY_RB_ENTRY)
                                            or (v_is_entry and vb_prio == PRIORITY_RING)))
                    if ring_entry_pair:
                        if v_is_ring:
                            # circulating never yields to the entering car
                            continue
                        # entering always yields to the circulating car
                        caps.append(self._yield_cap(v, bg[j], centroid))
                        continue
                    if is_blocked[a] and blocks(a, b):
                        # Physically blocked by j: forced yield regardless of
                        # rank (preserves the blocked-cannot-be-winner safety).
                        caps.append(self._yield_cap(v, bg[j], centroid))
                        continue
                    if rkey[b] < rkey[a] and conflict_m[a][b]:
                        # Higher-ranked AND actually conflicting -> yield to it.
                        caps.append(self._yield_cap(v, bg[j], centroid))
                if caps:
                    v._group_yield = True
                    v.target_speed = min(v.target_speed, min(caps))
                else:
                    v._group_go = True

            # Phase 2C.2: winner collision-aware cap (single-variable, OFF by
            # default). A group winner keeps its GO / priority (never yields)
            # but receives a PROGRESSIVE cap -- strictly above the yielder's --
            # when a path-intersecting member currently occupies / is about to
            # occupy their shared conflict point. The cap releases the instant
            # the point clears. Only path-intersecting members (conflict_m[a]
            # True) are considered, so a non-conflicting 3rd/4th car in the
            # same group never slows the winner.
            if getattr(_cfg, "CROSS_WINNER_COLLISION_CAP", False):
                for a in range(m):
                    i = members[a]
                    w = bg[i]
                    if not getattr(w, "_group_go", False):
                        continue
                    wcap = self._winner_cap(w, bg, members,
                                            conflict_m[a], centroid)
                    if wcap is not None and wcap < w.target_speed:
                        w.target_speed = min(w.target_speed, wcap)
                        w._group_wcap = True

        # Phase 4D-B: latch + freeze the EXISTING ring-mouth GO/YIELD assignment
        # as a clearance-based reservation. Does NOT re-decide the winner; only
        # extends the lifetime of the first valid baseline assignment.
        self._ring_reservation_step(bg)

    # --- Phase 4D-B: Ring conflict reservation / clearance persistence ------
    # Minimal, pair-specific, clearance-based. NOT a new arbitration controller.
    # The existing baseline arbitration produces the FIRST valid GO/YIELD for a
    # genuine ring-mouth feeder<->circulator conflict (see _is_ring_mouth_pair);
    # 4D-B only LATCHES that assignment as a reservation and freezes its identity
    # until the owner has PHYSICALLY cleared the mouth. It never re-selects the
    # winner. Scope is strictly genuine feeder<->circulator ring-mouth pairs;
    # same-lane ring followers, ring<->ring, feeder<->feeder, ordinary merge,
    # intersection crossing and any other conflict group are excluded.
    # State machine: FREE -> RESERVED(owner, yielder, conflict_point) -> FREE.
    # Release = explicit physical clearance only (owner whole body past the
    # conflict point + margin); NOT predictor-edge / ETA / time / cycle based.
    # A stale fail-safe (owner crash/despawn/reroute) cleans up without freezing
    # the yielder. All counters live on self._ring_res_stats.
    RING_RESERVATION_CLEARANCE_MARGIN = 1.0  # [m] body-past-conflict-point margin

    def _ring_res_init(self):
        if not hasattr(self, "_ring_reservations"):
            self._ring_reservations = {}
        if not hasattr(self, "_ring_res_stats"):
            self._ring_res_stats = {
                "reservation_created": 0, "reservation_held": 0,
                "reservation_released_clearance": 0,
                "owner_flip_prevented": 0, "edge_loss_bridged": 0,
                "none_owner_prevented": 0, "reservation_stale_cleanup": 0,
                "max_hold_steps": 0,
            }
        return self._ring_reservations, self._ring_res_stats

    def _ring_res_clearance(self, owner, cp, mouth):
        """True iff owner has PHYSICALLY cleared the conflict point.

        Geometry-based ONLY: owner's whole body has passed cp + margin along its
        lane. Never time / ETA / cycle based. If owner is no longer converging on
        the mouth (lane changed past it, or rerouted), it is treated as cleared so
        a reservation can never freeze the yielder forever.
        """
        try:
            li = tuple(getattr(owner, "lane_index", ()))[:2]
            if li and li[1] != mouth:
                return True  # owner has left the mouth link -> physically past it
        except Exception:
            pass
        try:
            lane = self.network.get_lane(owner.lane_index)
            s_owner = float(lane.local_coordinates(owner.position)[0])
            s_cp = float(lane.local_coordinates(np.asarray(cp))[0])
            half = getattr(owner, "length", 5.0) / 2.0
            return (s_owner - s_cp) > (half + self.RING_RESERVATION_CLEARANCE_MARGIN)
        except Exception:
            return True  # unreadable geometry -> release, never hold forever

    def _ring_reservation_step(self, bg):
        """Latch + freeze ring-mouth GO/YIELD assignments as reservations.

        Runs at the END of enforce_road_rules, AFTER the group arbitration has set
        each vehicle's _group_go/_group_yield this cycle, so it can override a
        flip/none that the instantaneous predictor produced while the physical
        conflict is still live.
        """
        if not getattr(_cfg, "RING_RESERVATION_PERSISTENCE", False):
            return
        reservations, stats = self._ring_res_init()
        bgset = set(id(v) for v in bg)

        # --- (1) detect + create/refresh reservations from THIS cycle's assignment
        # A pair already under an active reservation must NOT be re-latched from a
        # flipped instantaneous assignment -- the enforce loop below freezes the
        # ORIGINAL owner identity. Re-latching a flip would defeat 4D-B.
        existing_pairs = {(r["mouth"], frozenset((id(r["owner"]), id(r["yielder"]))))
                          for r in reservations.values()}
        for i in range(len(bg)):
            for j in range(i + 1, len(bg)):
                a, b = bg[i], bg[j]
                rm = self._is_ring_mouth_pair(a, b)
                if rm is None:
                    continue            # not a genuine feeder<->circulator pair
                C, F, M = rm
                if (M, frozenset((id(a), id(b)))) in existing_pairs:
                    continue            # already reserved -> freeze loop handles it
                if not (self._arbitration_conflict(a, b)
                        or self._ring_merge_conflict(a, b)):
                    continue            # no live conflict -> nothing to latch yet
                # latch whatever the baseline produced (do NOT re-decide winner)
                if getattr(a, "_group_go", False) and getattr(b, "_group_yield", False):
                    owner, yielder = a, b
                elif getattr(b, "_group_go", False) and getattr(a, "_group_yield", False):
                    owner, yielder = b, a
                else:
                    continue            # no valid first assignment this cycle
                key = (id(owner), id(yielder), M)
                if key not in reservations:
                    reservations[key] = {
                        "owner": owner, "yielder": yielder, "mouth": M,
                        "cp": self._conflict_point_v(owner, yielder),
                        "hold_steps": 0,
                    }
                    stats["reservation_created"] += 1

        # --- (2) enforce / release / stale-clean every active reservation
        for key in list(reservations.keys()):
            r = reservations[key]
            owner, yielder, M = r["owner"], r["yielder"], r["mouth"]
            # stale fail-safe: owner/yielder gone, crashed, despawned or rerouted
            if (id(owner) not in bgset or getattr(owner, "crashed", False)
                    or id(yielder) not in bgset or getattr(yielder, "crashed", False)):
                stats["reservation_stale_cleanup"] += 1
                del reservations[key]
                continue
            # normal release: explicit physical clearance of the conflict point
            if self._ring_res_clearance(owner, r["cp"], M):
                stats["reservation_released_clearance"] += 1
                del reservations[key]
                continue
            # not cleared -> FREEZE the latched assignment (forbid flip / none)
            r["hold_steps"] = r.get("hold_steps", 0) + 1
            stats["reservation_held"] += 1
            stats["max_hold_steps"] = max(stats["max_hold_steps"], r["hold_steps"])
            owner_go = getattr(owner, "_group_go", False)
            yielder_yield = getattr(yielder, "_group_yield", False)
            flipped = (not owner_go) or yielder_yield
            none_assigned = (not owner_go) and (not yielder_yield)
            if flipped:
                stats["owner_flip_prevented"] += 1
            if none_assigned:
                stats["none_owner_prevented"] += 1
                stats["edge_loss_bridged"] += 1
            # re-assert the latched identity (baseline's first winner stays GO)
            owner._group_go = True
            owner._group_yield = False
            yielder._group_yield = True
            try:
                cap = self._yield_cap(yielder, owner, owner.position)
                yielder.target_speed = min(yielder.target_speed, cap)
            except Exception:
                pass

    def _winner_tier(self, d_w: float):
        """Progressive cap for a WINNER by its distance to the conflict point.

        Strictly ABOVE the yielder tiers (6/3/0) so the invariant
        winner_cap > yielder_cap holds -- a winner is never told to slow more
        than the car it outranks. Returns None when out of cap range (no cap).
        """
        d = d_w - _cfg.CROSS_WINNER_MARGIN
        if d > _cfg.CROSS_WINNER_CAP_DIST:
            return None
        if d > 8.0:
            return _cfg.CROSS_WINNER_CAP_FAR
        if d > 4.0:
            return _cfg.CROSS_WINNER_CAP_MID
        return _cfg.CROSS_WINNER_CAP_NEAR

    def _point_occupied(self, y, p, arrival: float) -> bool:
        """Occupancy check for the conflict point p (geometric, target-agnostic).

        True if y is within CROSS_WINNER_OCC_RADIUS of p now, OR will be near p
        around the winner's arrival time (predicted with its CURRENT speed along
        its route). This is what makes the winner slow for a yielder that has
        ALREADY stopped ON the point (target 0 but body still there) -- the cap
        keys off occupancy, not off the yielder's commanded speed.
        """
        p = np.asarray(p, dtype=float)
        try:
            if np.linalg.norm(y.position - p) <= _cfg.CROSS_WINNER_OCC_RADIUS:
                return True
        except Exception:
            pass
        try:
            times = np.arange(0.0, _cfg.CROSS_WINNER_OCC_TIME + 1e-6, 0.25)
            if getattr(_cfg, "CROSS_ROUTE_AWARE_PREDICTION", False):
                pos, _ = self._predict_along_route(y, times)
            else:
                pos, _ = CompoundRoad._predict_along_lane(y, times)
            for t, pp in zip(times, pos):
                if abs(t - arrival) <= 1.0 and \
                        np.linalg.norm(np.asarray(pp) - p) <= _cfg.CROSS_WINNER_OCC_RADIUS:
                    return True
        except Exception:
            pass
        return False

    def _is_merge_conflict(self, la_idx, lb_idx, p) -> bool:
        """Scope guard for the Winner Cap: is the (la,lb) conflict at point p a
        MERGE / CONVERGENCE rather than a genuine crossing?

        The cap must NOT slow a GO winner at a merge -- the yielder is already
        capped by _yield_cap there, and slowing the winner too creates mutual
        hesitation (the measured 2C.2 crash migration: ring 10->18, merge 10->22).
        Genuine crossings send the two lanes to DIFFERENT downstream nodes; merges
        CONVERGE on the same node. No reliable runtime ctype exists, so this is
        the geometric fallback per the Phase-2C.2a spec:

            merge == (both lanes feed the SAME downstream node)
                     AND (the intersection point sits within
                          CROSS_WINNER_MERGE_MARGIN of EITHER lane's end)

        The endpoint margin is a COMPONENT of the detector, not the sole
        condition: a real crossing that merely happens to lie near a lane end is
        never mis-excluded because its downstream nodes differ (same_dest False).
        If the geometry cannot be read, a structural same_dest merge is
        conservatively excluded anyway.
        """
        la = tuple(la_idx)[:2]
        lb = tuple(lb_idx)[:2]
        if not la or not lb or la == lb:
            return False
        if la[1] != lb[1]:
            return False            # downstream diverges -> genuine crossing
        try:
            lane_a = self.network.get_lane(la_idx)
            lane_b = self.network.get_lane(lb_idx)
            sa = float(lane_a.local_coordinates(np.asarray(p))[0]) \
                / max(lane_a.length, 1e-6)
            sb = float(lane_b.local_coordinates(np.asarray(p))[0]) \
                / max(lane_b.length, 1e-6)
        except Exception:
            return True             # structural merge, geometry unreadable -> exclude
        m = _cfg.CROSS_WINNER_MERGE_MARGIN
        return (sa > 1.0 - m) or (sb > 1.0 - m)

    def _winner_cap(self, winner, bg, members, conflict_row, centroid):
        """Collision-aware cap for one group winner.

        Iterates only members whose path ACTUALLY intersects the winner's
        (conflict_row True -- filters non-conflicting group members), and only
        when that member occupies / is about to occupy the shared conflict
        point. Returns the tightest applicable cap, or None if nothing warrants
        one.
        """
        best = None
        lw = tuple(getattr(winner, "lane_index", ()))[:2]
        for b in range(len(members)):
            if not conflict_row[b]:
                continue
            y = bg[members[b]]
            ly = tuple(getattr(y, "lane_index", ()))[:2]
            if not lw or not ly or lw == ly:
                p = centroid
            else:
                key = (lw, ly) if lw <= ly else (ly, lw)
                pts = self._lane_crossing_table().get(key)
                if not pts:
                    p = centroid
                else:
                    try:
                        lane_w = self.network.get_lane(winner.lane_index)
                        sw = lane_w.local_coordinates(winner.position)[0]
                    except Exception:
                        continue
                    bestd = None
                    p = centroid
                    for pp in pts:
                        d = float(lane_w.local_coordinates(np.asarray(pp))[0]) - sw
                        if -3.0 <= d <= 25.0 and (bestd is None or d < bestd):
                            bestd = d
                            p = pp
            # Phase 2C.2a scope guard: never cap a winner at a MERGE / convergence.
            # Genuine crossings remain capped; merges are excluded (see
            # _is_merge_conflict for the rationale and the crash-migration evidence).
            if lw and ly and lw != ly and self._is_merge_conflict(
                    winner.lane_index, y.lane_index, p):
                continue
            # distance from winner to the conflict point along its lane
            try:
                lane_w = self.network.get_lane(winner.lane_index)
                sw = lane_w.local_coordinates(winner.position)[0]
                d_w = float(lane_w.local_coordinates(np.asarray(p))[0]) - sw
            except Exception:
                continue
            if d_w < -3.0 or d_w > _cfg.CROSS_WINNER_CAP_DIST + 5.0:
                continue
            arrival = d_w / max(float(winner.speed), 0.5)
            if not self._point_occupied(y, p, arrival):
                continue
            cap = self._winner_tier(d_w)
            if cap is not None and (best is None or cap < best):
                best = cap
        return best

    def _yield_cap(self, v, winner, centroid) -> float:
        """Progressive cap for one yielder.

        Measured to the point where the yielder's OWN lane crosses the
        WINNER's lane, so the yielder comes to a halt BEFORE the crossing
        instead of creeping through the winner's path (the diagnosis: four
        crashes were all a fast winner hitting a yielder that a centroid
        distance cap had let creep into the conflict point). Falls back to
        the group-centroid distance for tangent merges with no crossing.
        """
        def tiers(d: float) -> float:
            # Stop-line margin: the cap-0 radius is measured from the car
            # CENTRE, so without a margin the stopped nose still pokes past
            # the crossing point (measured touch-crash with a creeping
            # winner). Hold 3 m earlier than the raw geometry suggests.
            d -= 3.0
            if d > 12.0:
                return _cfg.CROSS_CAP_FAR
            if d > _cfg.CROSS_ZONE_DIST:
                return _cfg.CROSS_CAP_NEAR
            return 0.0

        fallback = tiers(float(np.linalg.norm(v.position - centroid)))
        la = tuple(getattr(v, "lane_index", ()))[:2]
        lb = tuple(getattr(winner, "lane_index", ()))[:2]
        if not la or not lb or la == lb:
            return fallback
        key = (la, lb) if la <= lb else (lb, la)
        pts = self._lane_crossing_table().get(key)
        if not pts:
            return fallback
        try:
            lane_v = self.network.get_lane(v.lane_index)
            s_v = lane_v.local_coordinates(v.position)[0]
        except Exception:
            return fallback
        best = None
        for p in pts:
            d = float(lane_v.local_coordinates(np.asarray(p))[0]) - s_v
            if -2.0 <= d <= 25.0 and (best is None or d < best):
                best = d
        if best is None:
            return fallback
        return tiers(best)

    # NOTE on neighbour_vehicles: the earlier "winner ignores a yielding leader"
    # hook was removed (with the breaker in place it is redundant, and measured
    # crashes showed a fast winner using it to close on a stationary yielder
    # faster than its own braking could stop).
    # Phase 3B adds a DIFFERENT, purely additive override: it only ever fills in
    # a leader when the native exact-lane search found NONE, so it can never
    # hide a leader the car would otherwise have seen. See `neighbour_vehicles`.


def build_compound_road(
    np_random: np.random.RandomState | None = None,
    record_history: bool = False,
    neighbour_vehicles_connected_lanes: bool = False,
) -> Road:
    """A CompoundRoad wrapping the compound network.

    RegulatedRoad mirrors the base intersection's convention: it lets vehicles
    observe each other's right-of-way (`priority`) so approaching traffic
    actually yields at both junctions instead of driving through each other.
    CompoundRoad additionally predicts conflicts ALONG each vehicle's current
    lane (curve-aware) so the roundabout ring no longer deadlocks.

    `np_random` (and the regulation kwargs) are forwarded from the env so the
    road shares the env's RNG stream and config.
    """
    net = build_compound_network()
    return CompoundRoad(
        network=net,
        np_random=np_random if np_random is not None else np.random.default_rng(0),
        record_history=record_history,
        neighbour_vehicles_connected_lanes=neighbour_vehicles_connected_lanes,
    )


# --- 7. Panoramic camera --------------------------------------------------
def network_bounds(net: RoadNetwork, samples: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (min, max) of every lane in `net` [m].

    Lanes are sampled instead of using only their endpoints, so circular arcs
    and sine transitions contribute their real extent.
    """
    pts: list[np.ndarray] = []
    for _from, edges in net.graph.items():
        for _to, lanes in edges.items():
            for lane in lanes:
                for s in np.linspace(0.0, lane.length, samples):
                    pts.append(lane.position(s, 0.0))
    pts = np.asarray(pts)
    return pts.min(axis=0), pts.max(axis=0)


@lru_cache(maxsize=1)
def camera_config() -> dict:
    """Screen size / scaling / centre for a fixed, map-wide top-down camera."""
    lo, hi = network_bounds(build_compound_network())
    center = (lo + hi) / 2.0
    span_x = float(hi[0] - lo[0]) + 2 * CAMERA_MARGIN
    span_y = float(hi[1] - lo[1]) + 2 * CAMERA_MARGIN
    width = int(min(CAMERA_MAX_WIDTH, span_x * CAMERA_SCALING))
    height = int(span_y * CAMERA_SCALING)
    # Re-derive the scaling from the (possibly clamped) width so the map always
    # fits in BOTH directions.
    scaling = min(width / span_x, height / span_y)
    return {
        "center": center,
        "scaling": float(scaling),
        "screen_width": width,
        "screen_height": height,
        "bounds": (lo, hi),
    }


if __name__ == "__main__":  # pragma: no cover - manual geometry check
    cfg = camera_config()
    lo, hi = cfg["bounds"]
    print(f"map bounds   : x [{lo[0]:.1f}, {hi[0]:.1f}]  y [{lo[1]:.1f}, {hi[1]:.1f}]")
    print(f"map size     : {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} m")
    print(f"camera centre: {cfg['center']}")
    print(f"screen       : {cfg['screen_width']} x {cfg['screen_height']} px"
          f"  @ {cfg['scaling']:.2f} px/m")
    net = build_compound_network()
    n_lanes = sum(len(l) for e in net.graph.values() for l in e.values())
    multi = [k for k, v in
             ((f"{a}->{b}", lanes) for a, e in net.graph.items()
              for b, lanes in e.items()) if len(v) > 1]
    print(f"roads        : {len(multi)} multi-lane -> {multi}")
    print(f"lanes total  : {n_lanes}")
