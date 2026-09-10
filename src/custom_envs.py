"""Custom highway-env environments implementing the three differentiation angles.

1. TTC-based safety reward  -> added inside the env reward (MyRoundabout / MyIntersection)
2. Perception-noise robustness -> NoiseWrapper, applied OUTSIDE the env
3. Multi-agent game equilibrium -> MultiAgentRoundabout (experimental extension)

Items (1) and (2) are enabled by default and run out-of-the-box. Item (3) is
provided as a documented, optional extension (see README).
"""

from __future__ import annotations

import functools
import numpy as np
import gymnasium as gym
from gymnasium import register

from env_using.roundabout_env import RoundaboutEnv
from env_using.intersection_env import IntersectionEnv
from env_core.envs.common.abstract import AbstractEnv
from env_core.vehicle.controller import MDPVehicle
from env_core.vehicle.kinematics import Vehicle
# Import the configs MODULE (not just names) so the USE_TTC / CRASH_TERMINATE
# toggles set by train.py/evaluate.py are read LIVE. A plain `from .configs
# import USE_TTC` binds the value at import time, so later rebinding
# configs.USE_TTC would never reach default_config (--no-ttc silently did
# nothing -- a real bug found while wiring crash-terminate).
from . import configs as _configs
from .configs import (
    SHARED_OBS,
    SHARED_ACTION,
    REWARD,
    TTC_REWARD,
    TTC_THRESHOLD,
    STALL_SPEED_THRESHOLD,
    STALL_REBOOT_STEPS,
    NOISE_STD,
    N_AGENTS,
    JOINT_OBS_DIM,
    JOINT_ACTION_DIM,
    COMPOUND_DURATION,
    COMPOUND_SPAWN_PROB,
    COMPOUND_INITIAL_VEHICLES,
    COMPOUND_TARGET_VEHICLES,
    COMPOUND_MAX_VEHICLES,
    COMPOUND_SPAWN_MIN_GAP,
    COMPOUND_SPAWN_LEADER_GAP,
    COMPOUND_SPAWN_CONFLICT_CHECK,
    COMPOUND_CROSS_TRAFFIC_PROB,
    MERGE_YIELD_ENABLED,
    MERGE_LOOKAHEAD,
    MERGE_ETA_WINDOW,
    MERGE_YIELD_MIN_SPEED,
    MERGE_MAX_ANGLE,
    MERGE_MAX_ANGLE_RING,
    MERGE_CATCHUP_WINDOW,
    MERGE_MOUTH_DIST,
    MERGE_MOUTH_SPEED,
    DEDICATED_RING_MERGE,
    RING_MERGE_FAR,
    RING_MERGE_NEAR,
    RING_MERGE_SAFETY,
    RING_MERGE_TRANSIT,
    RING_MERGE_MIN_SPEED,
    GRIDLOCK_BREAK_STEPS,
    PANORAMIC_VIEW,
    ENV_IDS,
)

# --- Collision taxonomy ---------------------------------------------------
# Every evaluation must speak the SAME language about what "a collision" was.
# The old habit was to count every `_reset_controlled()` call as a crash, which
# lumped real impacts together with stall-watchdog reboots and dead-end resets.
# From now on a collision is tagged by WHO hit whom:
#   BB = background <-> background  (the ENVIRONMENT is broken -- must be 0)
#   CB = controlled <-> background  (the agent hit traffic, or vice versa)
#   CC = controlled <-> controlled  (the two RL agents hit each other)
COLLISION_TYPES = ("BB", "CB", "CC")


def compute_min_ttc(road, vehicle) -> float:
    """Smallest time-to-collision from `vehicle` to any other vehicle."""
    min_ttc = float("inf")
    for other in road.vehicles:
        if other is vehicle:
            continue
        rel_pos = other.position - vehicle.position
        rel_vel = other.velocity - vehicle.velocity
        dist = float(np.linalg.norm(rel_pos))
        if dist < 1e-6:
            continue
        approach = -float(np.dot(rel_vel, rel_pos) / dist)
        if approach > 0:
            min_ttc = min(min_ttc, dist / approach)
    return min_ttc


# --- Background-collision diagnostics --------------------------------------
# Step 2 of the environment-repair protocol: every background-vs-background
# (BB) crash must be classified as SAME-LANE REAR-END / MERGE / CROSSING /
# OTHER, and tagged with the pre-crash state that explains WHY it happened
# (leader-gap seen by IDM, fraction along the lane -> lane-boundary transition,
# closing speed). These are READ-ONLY diagnostics -- they never change how the
# env simulates, they only record what already happened. That matters: the user
# explicitly forbade any change to conflict logic / IDM params / priority /
# horizon / PPO / reward at this stage, so this is pure instrumentation.

def _travel_dir(v) -> np.ndarray:
    """Unit vector the vehicle is actually travelling along."""
    d = getattr(v, "direction", None)
    if d is not None and np.linalg.norm(d) > 1e-6:
        return np.asarray(d, dtype=float) / np.linalg.norm(d)
    h = getattr(v, "heading", None)
    if h is not None:
        return np.array([np.cos(h), np.sin(h)], dtype=float)
    sp = getattr(v, "velocity", None)
    if sp is not None and np.linalg.norm(sp) > 1e-6:
        return sp / np.linalg.norm(sp)
    return np.array([1.0, 0.0])


def _heading_deg(v) -> float:
    d = _travel_dir(v)
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _heading_angle_deg(v1, v2) -> float:
    """Angle between two vehicles' travel directions, folded to [0, 90].

    0 = parallel (same direction), 90 = perpendicular (crossing paths)."""
    a = _travel_dir(v1)
    b = _travel_dir(v2)
    cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cos)))
    return min(ang, 180.0 - ang)


def _leader_gap(road, v, exclude=frozenset()):
    """Gap [m] and TTC [s] to the nearest vehicle directly AHEAD on the SAME
    lane -- exactly what IDM's car-following (and the safety shield) looks at.

    Returns (None, None) when there is no leader in the lane ahead. A None
    leader gap is the smoking gun for the rear-end hypothesis: IDM "forgot" the
    car in front (e.g. the leader dropped out of view across a lane-segment
    boundary, so the follower accelerated into it).
    """
    d = _travel_dir(v)
    lateral = np.array([-d[1], d[0]])
    best_gap, best_ttc = None, None
    for o in road.vehicles:
        if o is v or id(o) in exclude:
            continue
        rel = o.position - v.position
        along = float(np.dot(rel, d))
        if along <= 0:                                     # behind us
            continue
        if abs(float(np.dot(rel, lateral))) > (v.WIDTH + o.WIDTH) / 2.0 + 0.5:
            continue                                       # not in our path
        gap = along - (v.LENGTH + o.LENGTH) / 2.0
        closing = v.speed - float(np.dot(o.velocity, d))
        ttc = gap / closing if closing > 1e-6 else float("inf")
        if best_gap is None or gap < best_gap:
            best_gap, best_ttc = gap, ttc
    return best_gap, best_ttc


def _next_lane(v):
    """The lane a vehicle is about to ENTER next (route[1]), normalised to a
    (from, to, idx) tuple, or None.

    Only trusted when route[0] is still the vehicle's CURRENT lane: a route
    that has fallen behind the vehicle (it left its planned path) makes
    route[1] stale garbage -- e.g. a car on a ring arc "whose next lane" is a
    ramp on the other side of the map. Stale routes must not drive merge
    decisions, so they read as None.
    """
    r = getattr(v, "route", None)
    if not isinstance(r, (list, tuple)) or len(r) < 2:
        return None
    cur = tuple(getattr(v, "lane_index", ()) or ())
    head = r[0]
    if isinstance(head, (list, tuple)) and tuple(head[:2]) != cur[:2]:
        return None                      # stale route -- do not trust it
    nxt = r[1]
    if isinstance(nxt, (list, tuple)) and len(nxt) >= 2:
        to = nxt[2] if len(nxt) > 2 and nxt[2] is not None else 0
        return (str(nxt[0]), str(nxt[1]), int(to))
    return None


def _classify_pair(v1, v2) -> str:
    """Classify a two-vehicle crash as rear-end / merge / crossing / other."""
    li1 = tuple(getattr(v1, "lane_index", ()) or ())
    li2 = tuple(getattr(v2, "lane_index", ()) or ())
    same = bool(li1) and li1 == li2
    ang = _heading_angle_deg(v1, v2)
    nl1, nl2 = _next_lane(v1), _next_lane(v2)
    # Merge: one car is entering the other's lane, OR both feed the same node.
    merge = (not same) and (
        (nl1 is not None and nl1 == li2)
        or (nl2 is not None and nl2 == li1)
        or (li1 and li2 and li1[1] == li2[1])
    )
    if same:
        return "rear-end"
    if merge or ang < 40.0:
        return "merge"
    if ang > 50.0:
        return "crossing"
    return "other"


def _s_frac(road, v):
    """Fraction [0,1] of the way along the vehicle's current lane.

    ~1.0 means the car is at the very end of its lane -- i.e. sitting on a
    lane-segment boundary / junction mouth, the spot where leader detection is
    most likely to drop the car ahead."""
    li = getattr(v, "lane_index", None)
    if not li:
        return None
    try:
        lane = road.network.get_lane(li)
        s, _ = lane.local_coordinates(v.position)
        return s / lane.length if lane.length else None
    except Exception:
        return None


# -- Phase 4E-B collision accounting helpers (instrumentation ONLY) ---------- #
def _serial_of(v) -> int:
    """Persistent per-object serial for canonical event identity.

    Unlike ``id()``, the serial is never reused for a different vehicle, so
    despawn/respawn recycling cannot pollute (step, vehicle-pair) event keys.
    Pure bookkeeping: it never influences simulation."""
    from src.collision_accounting import ensure_serial
    return ensure_serial(v)


def _pre_pos_of(v):
    """Pre-physics position of the vehicle (captured by
    ``_snapshot_pre_crash``), falling back to the current position."""
    p = getattr(v, "_pre_position", None)
    if p is None:
        p = v.position
    try:
        return [round(float(p[0]), 2), round(float(p[1]), 2)]
    except Exception:
        return None


class MyRoundabout(RoundaboutEnv):
    """Roundabout with aligned obs/action spaces + optional TTC safety reward."""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({"observation": dict(SHARED_OBS), "action": dict(SHARED_ACTION)})
        cfg.update(dict(REWARD))
        cfg["ttc_reward"] = TTC_REWARD if _configs.USE_TTC else 0.0
        cfg["ttc_threshold"] = TTC_THRESHOLD
        cfg["normalize_reward"] = False
        return cfg

    def _rewards(self, action: int) -> dict:
        min_ttc = compute_min_ttc(self.road, self.vehicle)
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road,
            "ttc_reward": float(min_ttc < self.config["ttc_threshold"]),
        }


class MyIntersection(IntersectionEnv):
    """Intersection with aligned obs/action spaces + optional TTC safety reward.

    Note: IntersectionEnv computes reward per-agent (it supports multiple
    controlled vehicles), so the TTC term is added inside `_agent_rewards`.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({"observation": dict(SHARED_OBS), "action": dict(SHARED_ACTION)})
        cfg.update(dict(REWARD))
        cfg["ttc_reward"] = TTC_REWARD if _configs.USE_TTC else 0.0
        cfg["ttc_threshold"] = TTC_THRESHOLD
        cfg["reward_speed_range"] = [7.0, 9.0]
        cfg["normalize_reward"] = False
        return cfg

    def _agent_rewards(self, action: int, vehicle) -> dict:
        rewards = super()._agent_rewards(action, vehicle)
        min_ttc = compute_min_ttc(self.road, vehicle)
        rewards["ttc_reward"] = float(min_ttc < self.config["ttc_threshold"])
        return rewards


# --- Multi-agent compound scene (intersection + roundabout, ONE shared policy) ---
from .compound_road import (  # noqa: E402
    ENTRY_LANES,
    EXIT_LANES,
    EXIT_NODES,
    EXIT_NODES_INTER,
    EXIT_NODES_RB,
    LOOP_DESTINATIONS,
    SAFE_LOOP_DESTINATIONS,
    CompoundRoad,
    build_compound_network,
    build_compound_road,
    camera_config,
)


class _FixedCamera:
    """Pin the renderer to a fixed world point instead of the ego vehicle.

    ``EnvViewer.window_position()`` returns ``observer_vehicle.position`` when
    that attribute is set, so feeding it this tiny stub is all that is needed to
    get a stable, map-wide panoramic view.
    """

    __slots__ = ("position",)

    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)


class SafeMDPVehicle(MDPVehicle):
    """MDPVehicle plus an emergency-braking safety layer.

    Why this exists
    ---------------
    A stock MDPVehicle has NO collision avoidance whatsoever -- it simply tracks
    `target_speed`, whatever happens to be in front of it. The obstacle traffic
    uses IDMVehicle, which ships with full IDM car-following (and MOBIL lane
    changing). That asymmetry is exactly why the two RL cars look worse than the
    background traffic: the background cars brake for each other automatically,
    while the agents have to rediscover braking from reward alone.

    This is the standard "safety shield / constrained RL" construction: the
    policy keeps full control of the high-level decision (which of the five
    meta-actions to take), but the execution layer may brake HARDER than the
    policy asked for when a crash is imminent. It can never accelerate into
    danger, so it cannot be exploited to game the speed reward.
    """

    BRAKE_TTC = 2.0        # s -- closing too fast on the leader: drop one gear
    BRAKE_GAP = 12.0       # m -- leader simply too close: drop one gear
    EMERG_TTC = 1.0        # s -- crash imminent: brake to a stop
    EMERG_GAP = 6.0        # m -- ditto

    # Cross-traffic conflict (the gap the plain leader check leaves open):
    # forward-simulate both vehicles and brake if their bounding circles will
    # overlap within this horizon. These are deliberately eager (tuned down from
    # 1.2/0.6s and 2.0s/0.5s to cut the intersection crash rate): the map moves
    # ~8-16 m/s, so reacting a second earlier is the difference between a
    # controlled stop and a T-bone. Trade-off: the agent may brake a little
    # earlier/softer at junctions, which the stall watchdog already tolerates.
    CROSS_BRAKE_TTC = 1.8   # s -- a crossing car will hit us soon: drop a gear
    CROSS_EMERG_TTC = 1.0   # s -- collision essentially unavoidable: stop
    CROSS_HORIZON = 3.0     # s -- how far ahead to forward-simulate
    CROSS_DT = 0.25         # s -- simulation step for the forward prediction
    CROSS_RANGE = 30.0      # m -- ignore vehicles farther than this

    def _leader_situation(self):
        """(gap, ttc) to the nearest vehicle directly ahead, else (None, None)."""
        best_gap, best_ttc = None, None
        chain = self.road._route_lane_chain(self, max_lanes=4) or [self.lane_index]
        offset = -self.lane.local_coordinates(self.position)[0]
        previous = self.lane_index
        for li in chain:
            if li[2] is None:
                prev_lane = self.road.network.get_lane(previous)
                k, _ = self.road.network.next_lane_given_next_road(
                    *previous, li[1], None, prev_lane.position(prev_lane.length, 0))
                li = (*li[:2], k)
            lane = self.road.network.get_lane(li)
            for other in self.road.vehicles:
                if other is self:
                    continue
                other_s, other_lateral = lane.local_coordinates(other.position)
                # A turning arc must not be extrapolated beyond its end: doing
                # so makes a car in an opposite access lane look like a leader.
                if not (0 <= other_s <= lane.length):
                    continue
                along = float(offset + other_s)
                if not 0 < along <= 60:
                    continue
                if abs(other_lateral) > (self.WIDTH + other.WIDTH) / 2 + 0.5:
                    continue
                gap = along - (self.LENGTH + other.LENGTH) / 2.0
                heading = lane.heading_at(other_s)
                d = np.array([np.cos(heading), np.sin(heading)])
                closing = self.speed - float(np.dot(other.velocity, d))
                ttc = gap / closing if closing > 1e-6 else float("inf")
                if best_gap is None or gap < best_gap:
                    best_gap, best_ttc = gap, ttc
            offset += lane.length
            previous = li
            if offset > 60:
                break
        return best_gap, best_ttc

    def _cross_situation(self):
        """Predicted time-to-collision with a CROSSING vehicle (intersection /
        roundabout merge conflict).

        Unlike `_leader_situation`, which only watches the car dead ahead in our
        own lane, this forward-simulates BOTH vehicles and flags the first
        instant their bounding circles would overlap. Returns that time
        (seconds), or None if no conflict within the horizon. This is what
        catches the lateral conflicts the leader check is blind to.

        The prediction follows each vehicle along ITS OWN lane rather than a
        straight line: a car on the roundabout ring (or on a curved junction
        arc) is turning, so constant-heading extrapolation mis-places it and the
        shield either brakes too late at a merge or brakes for a "ghost" that is
        actually sweeping away. Projecting along the lane fixes exactly that
        (same insight that makes CompoundRoad's conflict prediction curve-aware).
        """
        from env_core import utils
        times = np.arange(self.CROSS_DT, self.CROSS_HORIZON + 1e-6, self.CROSS_DT)
        predictor = getattr(self.road, '_predict_along_route', None)
        def trajectory(vehicle):
            if predictor is not None:
                return predictor(vehicle, times)
            return ([self._projected_future(vehicle, t) for t in times],
                    [vehicle.heading] * len(times))
        ego_path, ego_headings = trajectory(self)
        best = None
        for other in self.road.vehicles:
            if other is self or getattr(other, "crashed", False):
                continue
            if float(np.linalg.norm(other.position - self.position)) > self.CROSS_RANGE:
                continue
            other_path, other_headings = trajectory(other)
            # Width-only circles miss front/rear and oblique contact between
            # five-metre vehicles. Check the full oriented footprint instead.
            for k, t in enumerate(times):
                if best is not None and t >= best:
                    break
                if np.linalg.norm(ego_path[k] - other_path[k]) > self.LENGTH + other.LENGTH:
                    continue
                if utils.rotated_rectangles_intersect(
                        (ego_path[k], self.LENGTH + 1.0, self.WIDTH + 0.4, ego_headings[k]),
                        (other_path[k], other.LENGTH + 1.0, other.WIDTH + 0.4, other_headings[k])):
                    if best is None or t < best:
                        best = t
                    break
        return best

    def _projected_future(self, vehicle, t: float) -> np.ndarray:
        """Where `vehicle` will be `t` seconds ahead, projected along its lane.

        Follows the vehicle's current lane (so curves are honoured) using its
        current signed speed along that lane. Falls back to a straight line
        along its heading when it has no usable lane. The projection is clipped
        to the lane so a car near the end of a lane is not teleported past the
        map edge by the predictor.
        """
        predictor = getattr(self.road, "_predict_along_route", None)
        if predictor is not None:
            positions, _ = predictor(vehicle, [t])
            return np.asarray(positions[0], dtype=float)
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            d = np.asarray(vehicle.direction, dtype=float)
            return vehicle.position + d * float(vehicle.speed) * t
        try:
            s0, lat0 = lane.local_coordinates(vehicle.position)
            heading = lane.heading_at(s0)
        except Exception:
            d = np.asarray(vehicle.direction, dtype=float)
            return vehicle.position + d * float(vehicle.speed) * t
        along = np.array([np.cos(heading), np.sin(heading)])
        signed = float(np.dot(vehicle.velocity, along))
        s_future = s0 + signed * t
        length = getattr(lane, "length", None)
        if length is not None:
            s_future = float(np.clip(s_future, 0.0, length))
        return np.asarray(lane.position(s_future, lat0), dtype=float)

    def act(self, action=None):
        # 1. the policy's own speed request (same bookkeeping as MDPVehicle)
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        self.speed_index = int(np.clip(self.speed_index, 0, self.target_speeds.size - 1))
        # Keep the policy request separate from temporary execution constraints.
        # Otherwise 15 physics ticks repeatedly subtract gears and permanently
        # latch IDLE at zero after a hazard has gone away.
        safe_index = self.speed_index
        braking_index = max(0, self.speed_to_index(self.speed) - 1)
        # 2. safety layer -- may only cap the policy's requested speed
        gap, ttc = self._leader_situation()
        self._shield_gap = gap
        if gap is not None:
            if ttc < self.EMERG_TTC or gap < self.EMERG_GAP:
                safe_index = 0                                    # emergency stop
            elif ttc < self.BRAKE_TTC or gap < self.BRAKE_GAP:
                safe_index = min(safe_index, braking_index)
        # 3. cross-traffic safety: forward-simulate and brake for a crossing
        #    vehicle that is about to collide. Same rule -- never accelerates.
        cross = self._cross_situation()
        self._shield_cross = cross
        if cross is not None:
            if cross < self.CROSS_EMERG_TTC:
                safe_index = 0
            elif cross < self.CROSS_BRAKE_TTC:
                safe_index = min(safe_index, braking_index)
        self.target_speed = self.index_to_speed(safe_index)
        self.target_speed = min(self.target_speed,
                                getattr(self, '_regulation_speed_cap', float('inf')))
        if getattr(_configs, 'CONFLICT_RESERVATIONS', False):
            self.target_speed = min(self.target_speed, getattr(self, '_reservation_speed_cap', float('inf')))
        # 4. ControlledVehicle applies the lateral action + the speed control
        # Speed meta-actions have already been consumed above. Passing FASTER
        # to ControlledVehicle would add DELTA_SPEED after the safety stop,
        # overriding braking precisely when the policy requests acceleration.
        lateral_action = action if action in ("LANE_LEFT", "LANE_RIGHT") else None
        super(MDPVehicle, self).act(lateral_action)


class CompoundEnv(IntersectionEnv):
    """Two RL vehicles negotiating BOTH a cross intersection AND a roundabout.

    This is the multi-agent version of the "compound" map. It reuses
    IntersectionEnv (which natively supports `controlled_vehicles > 1` and
    per-agent rewards) and swaps in the compound road from compound_road.py.

    Behavioural tweaks over the base:
      * `_is_terminated` returns False -> the episode never ends early (cars keep
        looping / negotiating until the time limit, not on first crash/arrival).
      * Controlled vehicles are continuously re-routed so they keep circulating
        between the intersection and the roundabout instead of arriving once.
      * Obstacle traffic is maintained as a *population* of ~20 cars that are
        routed ACROSS the map, so both junctions stay busy and every car has to
        negotiate the intersection AND the roundabout.
      * Rendering uses a fixed, map-wide top-down camera.
    """

    # Entry lanes of each feature (split so a spawned car can be routed to the
    # far side of the map).
    INTER_ENTRY_LANES = [e for e in ENTRY_LANES if e[0].startswith("o")]
    RB_ENTRY_LANES = [e for e in ENTRY_LANES if not e[0].startswith("o")]
    # (from, to) pairs whose far end leaves the map -> despawn there.
    EXIT_LANE_SET = {tuple(x) for x in EXIT_LANES}

    # Where the two controlled agents START. Both begin on intersection arms
    # (south and north) at almost the same distance from the junction, so they
    # arrive together and must negotiate the crossing with each other.
    CONTROL_STARTS = [("o0", "ir0"), ("o2", "ir2")]
    CONTROL_STARTS_S = [26.0, 20.0]           # longitudinal offset along the arm
    # First destination: cross the intersection and get onto the roundabout ring.
    # These MUST be ring ENTRY nodes, which still have successor lanes. The old
    # values ("exr" / "sxr") are the roundabout's OFF-MAP EXITS -- lanes with no
    # successor at all -- so both agents were routed straight into a dead end
    # where they parked and blocked the junction for the rest of the episode.
    CONTROL_DESTS = ["ses", "nes"]
    # Roundabout-half re-entry lanes. When an agent fails in the roundabout half
    # (east of the bridge) it is rebooted onto a ring ENTRY ramp instead of being
    # dragged back to the intersection arms. With a fixed step budget the old
    # always-to-the-intersection reset starved the ring merge of training samples
    # (the agent re-crossed the whole map before it could try the merge again) --
    # the crash -> reset -> never-learns-the-roundabout loop. Re-entering on the
    # feature where it failed makes that region dense in the replay.
    EAST_X = 100.0               # position.x above this counts as "roundabout half"
    EAST_STARTS = [("ser", "ses"), ("ner", "nes")]
    EAST_STARTS_S = [25.0, 20.0]
    # Radius [m] kept free of obstacle traffic when an agent is (re)placed.
    CONTROL_CLEAR_RADIUS = 30.0
    # Colours so the two agents are easy to tell apart in the video (obstacle
    # traffic keeps highway-env's default blue).
    AGENT_COLORS = [(50, 220, 60), (255, 160, 0)]
    CRASH_COLOR = (255, 60, 60)
    # How long an obstacle car may stay on the map before it is re-routed to an
    # exit, so the population keeps turning over instead of looping forever.
    OBSTACLE_MAX_AGE = 90

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg["controlled_vehicles"] = 2
        cfg["observation"] = {
            "type": "MultiAgentObservation",
            "observation_config": dict(SHARED_OBS),
        }
        cfg["action"] = {
            "type": "MultiAgentAction",
            "action_config": dict(SHARED_ACTION),
        }
        cfg["duration"] = COMPOUND_DURATION
        cfg["spawn_probability"] = COMPOUND_SPAWN_PROB
        cfg["offroad_terminal"] = False
        cfg["normalize_reward"] = False
        cfg["reward_speed_range"] = [7.0, 9.0]
        # Traffic population (obstacle cars only; the two agents are extra).
        cfg["initial_vehicle_count"] = COMPOUND_INITIAL_VEHICLES + 2
        cfg["target_vehicle_count"] = COMPOUND_TARGET_VEHICLES
        cfg["max_vehicle_count"] = COMPOUND_MAX_VEHICLES + 2
        cfg["spawn_min_gap"] = COMPOUND_SPAWN_MIN_GAP
        cfg["spawn_leader_gap"] = COMPOUND_SPAWN_LEADER_GAP
        cfg["spawn_conflict_check"] = COMPOUND_SPAWN_CONFLICT_CHECK
        cfg["cross_traffic_prob"] = COMPOUND_CROSS_TRAFFIC_PROB
        # Panoramic top-down camera.
        cam = camera_config()
        cfg["panoramic_view"] = PANORAMIC_VIEW
        cfg["camera_center"] = cam["center"]
        cfg["screen_width"] = cam["screen_width"]
        cfg["screen_height"] = cam["screen_height"]
        cfg["scaling"] = cam["scaling"]
        cfg["centering_position"] = [0.5, 0.5]
        cfg.update(dict(REWARD))
        cfg["ttc_reward"] = TTC_REWARD if _configs.USE_TTC else 0.0
        cfg["ttc_threshold"] = TTC_THRESHOLD
        cfg["stall_speed_threshold"] = STALL_SPEED_THRESHOLD
        cfg["crash_terminate"] = _configs.CRASH_TERMINATE
        return cfg

    # -- road -------------------------------------------------------------
    def _make_road(self) -> None:
        # Build the merged intersection+roundabout network and wrap it in a
        # CompoundRoad (a RegulatedRoad subclass) so lane priorities are
        # respected AND conflicts are predicted ALONG each vehicle's lane.
        # The stock RegulatedRoad predicts in a straight line, which makes every
        # pair of cars on the curved ring look like a collision -> the whole map
        # deadlocks. CompoundRoad fixes that; see compound_road.py.
        self.road = build_compound_road(
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            neighbour_vehicles_connected_lanes=self.config[
                "neighbour_vehicles_connected_lanes"
            ],
        )
        # Hard-locked conflict clusters are handed back to the env, which
        # recycles their members with honest per-car accounting.
        self.road.gridlock_hook = self._gridlock_break_group

    # -- vehicles ---------------------------------------------------------
    def _make_vehicles(self, n_vehicles: int = 24) -> None:
        # Per-controlled-vehicle previous position, used to measure forward
        # progress for the reward (see _agent_rewards / _agent_reward).
        self._agent_prev = {}
        # Consecutive stalled steps per agent index, for the anti-deadlock
        # watchdog in step() (an agent parked in a junction is rebooted).
        self._stall_steps = {}
        # Crash EVENTS this episode. A controlled agent's `crashed` flag is
        # cleared inside the SAME step() by _reset_controlled, so checking it at
        # episode end (as evaluate.py used to) always reads False and reports a
        # bogus ~0 collision rate. This counter is captured BEFORE the reset so
        # the real number of collisions per episode is observable. Reset every
        # episode (this method runs on each reset).
        self._crash_count = 0
        # Collisions by taxonomy: BB / CB / CC (see COLLISION_TYPES). This is the
        # ONE definition every script should report from now on -- "how many
        # _reset_controlled() calls" is not a collision count.
        self._collision_classes = {k: 0 for k in COLLISION_TYPES}
        self._collision_log = []
        # Background-traffic health counters (spawn / despawn / ...). Used by
        # src/_test_background_traffic.py and cheap enough to keep always on.
        self._traffic_stats = dict(
            spawned=0, despawned_exit=0, despawned_crash=0,
            spawn_blocked_gate=0, spawn_blocked_cap=0,
            spawn_blocked_slot=0, spawn_blocked_conflict=0,
            merge_yields=0, despawned_gridlock=0,
            # Phase 4B: Dedicated Ring-Merge Controller counters.
            ring_merge_interventions=0, ring_merge_feeder_stops=0,
            ring_merge_circ_restricted=0, ring_merge_double_yield=0,
        )
        # Stock DiscreteMetaAction drives an MDPVehicle, which has no collision
        # avoidance whatsoever (unlike the IDM obstacle traffic). Re-bind the
        # SAME vehicle factory onto our safety-shielded subclass.
        #
        # NB: `action_type.vehicle_class` is a functools.partial that already
        # carries the configured `target_speeds`. Passing the bare SafeMDPVehicle
        # class instead silently drops them and falls back to MDPVehicle's
        # [20, 25, 30] defaults -- measured: the cars then cruise at 20-30 m/s
        # and crash far more than before.
        base_factory = self.action_type.vehicle_class
        speeds = getattr(base_factory, "keywords", {}).get("target_speeds")
        vehicle_type = (
            functools.partial(SafeMDPVehicle, target_speeds=speeds)
            if speeds is not None
            else SafeMDPVehicle
        )
        # These must be set on the CLASS: the old code assigned them onto the
        # functools.partial object, where they had no effect whatsoever.
        SafeMDPVehicle.DISTANCE_WANTED = 7
        SafeMDPVehicle.COMFORT_ACC_MAX = 6
        SafeMDPVehicle.COMFORT_ACC_MIN = -3

        self.controlled_vehicles = []
        for ego_id in range(self.config["controlled_vehicles"]):
            start = self.CONTROL_STARTS[ego_id % len(self.CONTROL_STARTS)]
            dst = self.CONTROL_DESTS[ego_id % len(self.CONTROL_DESTS)]
            lane = self.road.network.get_lane((*start, 0))
            s = min(self.CONTROL_STARTS_S[ego_id % len(self.CONTROL_STARTS_S)],
                    lane.length - 5.0)
            # Cap the spawn speed to the top gear. The road's lane speed_limit
            # (20 m/s) sits ABOVE the gearbox max, so a fresh agent would launch
            # at 20 m/s and slam the traffic ahead before the speed controller
            # could pull it back -- the #1 cause of the rear-end crashes. Start
            # at the top gear instead so the first frames are already in range.
            v0 = min(lane.speed_limit, float(max(speeds)) if speeds is not None else 16.0)
            ego = vehicle_type(
                self.road,
                lane.position(s, 0),
                speed=v0,
                heading=lane.heading_at(s),
            )
            try:
                ego.plan_route_to(dst)
                ego.speed_index = ego.speed_to_index(v0)
                ego.target_speed = ego.index_to_speed(ego.speed_index)
            except (AttributeError, KeyError):
                pass
            ego.color = self.AGENT_COLORS[ego_id % len(self.AGENT_COLORS)]
            ego._spawn_step = 0
            self.road.vehicles.append(ego)
            self.controlled_vehicles.append(ego)
            # Remove any non-controlled cars that would collide at spawn time.
            for v in self.road.vehicles.copy():
                if v not in self.controlled_vehicles and np.linalg.norm(
                    v.position - ego.position
                ) < 25:
                    self.road.vehicles.remove(v)

        # Initial obstacle traffic so BOTH junctions are busy from frame 0.
        for _ in range(max(0, n_vehicles - len(self.controlled_vehicles))):
            self._spawn_vehicle(spawn_probability=1.0, spread=True)
        # Let the traffic settle briefly so nothing starts out overlapping.
        for _ in range(3 * self.config["simulation_frequency"]):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])

    def _is_terminated(self) -> bool:
        # By default NEVER terminate early: cars keep looping / negotiating until
        # the time limit (a crash just gives a negative reward, then the agent is
        # rebooted in place). When CRASH_TERMINATE is switched on (an A/B ablation
        # for the report), a crash DOES end the episode so it costs the whole
        # remaining return -- the strongest signal that a collision is
        # unacceptable, at the cost of re-introducing the "park to avoid crashing"
        # freeze risk. Default False keeps the loop-forever design.
        if self.config.get("crash_terminate", False):
            return any(getattr(v, "crashed", False)
                       for v in self.controlled_vehicles)
        return False

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    # -- reward shaping (compound-specific) ----------------------------------
    # The inherited IntersectionEnv reward only fires a *binary* -5 when a crash
    # actually happens. With no graded signal, PPO rapidly learns the safest
    # "policy": never enter the junction -> stop at the entrance (reward 0 beats
    # -5). That is exactly the "cars stop at the entrance and won't move" symptom.
    #
    # Two fixes are wired in here -- the TTC safety reward that configs.py claims
    # is active but which was never actually applied to the compound env (only to
    # MyRoundabout / MyIntersection), plus a forward-progress reward so *driving*
    # (and completing the route) is explicitly better than freezing:
    #   * TTC reward     -- continuous penalty as time-to-collision to the
    #                       nearest vehicle drops below TTC_THRESHOLD, nudging the
    #                       agent to slow / yield / change lane *before* impact.
    #   * Progress reward -- positive reward for forward distance covered, which
    #                       removes the "stop = 0 reward" local optimum.
    def _agent_rewards(self, action: int, vehicle) -> dict:
        rewards = super()._agent_rewards(action, vehicle)
        # Continuous TTC penalty: 0 when safe, ramps up to 1 as TTC -> 0.
        min_ttc = compute_min_ttc(self.road, vehicle)
        if min_ttc >= self.config["ttc_threshold"]:
            ttc_val = 0.0
        else:
            ttc_val = (self.config["ttc_threshold"] - min_ttc) / self.config["ttc_threshold"]
        rewards["ttc_reward"] = float(ttc_val)
        # Forward progress: projection of this step's displacement onto the
        # vehicle's heading. Lateral moves / reversals contribute ~0.
        prev = self._agent_prev.get(id(vehicle))
        if prev is None:
            progress = 0.0
        else:
            disp = vehicle.position - prev
            fwd = float(np.dot(disp, vehicle.direction))
            progress = float(np.clip(fwd, 0.0, 10.0))   # clamp teleport spikes
        # Stall indicator: 1 while the agent is (essentially) not moving. The
        # safety terms alone cannot fix freezing -- with ~14 obstacle cars there
        # is nearly always something with a low TTC nearby, so "do not move"
        # looks safe to the policy. This makes waiting explicitly costly.
        rewards["stall_reward"] = float(
            vehicle.speed < self.config.get(
                "stall_speed_threshold", STALL_SPEED_THRESHOLD
            )
        )
        rewards["progress_reward"] = progress
        self._agent_prev[id(vehicle)] = vehicle.position.copy()
        return rewards

    def _agent_reward(self, action: int, vehicle) -> float:
        rewards = self._agent_rewards(action, vehicle)
        cfg = self.config
        reward = (
            cfg["collision_reward"] * rewards["collision_reward"]
            + cfg["high_speed_reward"] * rewards["high_speed_reward"]
            + cfg["arrived_reward"] * rewards["arrived_reward"]
            + cfg["ttc_reward"] * rewards["ttc_reward"]
            + cfg["progress_reward"] * rewards["progress_reward"]
            + cfg["stall_reward"] * rewards["stall_reward"]
        )
        # Off-road -> zero reward (discourage leaving the network).
        reward *= rewards["on_road_reward"]
        return float(reward)

    # -- traffic ----------------------------------------------------------
    @staticmethod
    def _lane_free(road, position, min_gap: float) -> bool:
        return all(
            np.linalg.norm(v.position - position) >= min_gap for v in road.vehicles
        )

    def _destination_for(self, entry) -> str:
        """Pick a destination on the FAR side of the map whenever possible.

        A car entering at the intersection is routed to a roundabout exit (and
        vice versa), so essentially every obstacle car drives through BOTH
        junctions. That is what makes intersection yielding, roundabout yielding
        and multi-lane manoeuvring show up in the video.
        """
        cross = self.np_random.uniform() < self.config["cross_traffic_prob"]
        if entry[0].startswith("o"):            # entered at the intersection
            options = EXIT_NODES_RB if cross else EXIT_NODES_INTER
        else:                                   # entered at the roundabout
            options = EXIT_NODES_INTER if cross else EXIT_NODES_RB
        return options[self.np_random.integers(0, len(options))]

    def _entry_lane_clear(self, lane_index, lane, s, leader_gap: float) -> bool:
        """Room AHEAD of `s` on the SAME entry lane?

        The plain Euclidean ``_lane_free`` radius says nothing about whether the
        car we are about to drop in can actually survive: 18 m from a vehicle
        BEHIND us is fine, 18 m from a queue directly ahead at 8 m/s is not.
        This rejects a slot whose leader on the same lane is closer than
        `leader_gap`, which is the check that stops "spawned into the back of a
        stopped queue" (the classic background-traffic crash).
        """
        for v in self.road.vehicles:
            if tuple(getattr(v, "lane_index", ())[:2]) != tuple(lane_index[:2]):
                continue
            try:
                s_v = lane.local_coordinates(v.position)[0]
            except Exception:
                continue
            if 0.0 <= s_v - s < leader_gap:
                return False
        return True

    def _spawn_conflicts(self, vehicle) -> bool:
        """Would this freshly built vehicle be in immediate conflict?

        Reuses CompoundRoad's curve-aware predictor (the same one RegulatedRoad
        uses to regulate the junctions) so a car is never dropped into a
        conflict it has no time to react to.
        """
        for other in self.road.vehicles:
            if other is vehicle:
                continue
            try:
                if self.road.is_conflict_possible(vehicle, other):
                    return True
            except Exception:
                continue
        return False

    def _spawn_vehicle(self, longitudinal: float = 0, spread: bool = False, **kwargs):
        """Spawn one obstacle car entering the map from a random periphery arm.

        `spread=True` drops the car anywhere along the entry arm instead of just
        at its tip; that is used for the initial fill so the whole map is busy
        from frame 0 rather than filling up over the first few seconds.

        Spawn safety (per the review): a car is only created when ALL of these
        hold -- (1) it is `spawn_min_gap` away from every existing car,
        (2) its leader on the same entry lane is at least `spawn_leader_gap`
        ahead, and (3) a forward simulation finds no immediate conflict.
        If no entry slot passes, NOTHING is spawned this attempt: we never
        squeeze a car in "somewhere", which is exactly how background cars used
        to appear on top of each other.

        Returns the vehicle, or None if this step's spawn gate was not passed or
        no entry slot was free.
        """
        if self.np_random.uniform() > kwargs.get("spawn_probability", 1.0):
            self._traffic_stats["spawn_blocked_gate"] += 1
            return None
        if len(self.road.vehicles) >= self.config["max_vehicle_count"]:
            self._traffic_stats["spawn_blocked_cap"] += 1
            return None
        from env_core import utils

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        min_gap = self.config["spawn_min_gap"]
        leader_gap = float(self.config.get("spawn_leader_gap",
                                           COMPOUND_SPAWN_LEADER_GAP))
        check_conflict = bool(self.config.get("spawn_conflict_check",
                                              COMPOUND_SPAWN_CONFLICT_CHECK))
        # Try several entry lanes so one blocked arm does not waste the step.
        for idx in self.np_random.permutation(len(ENTRY_LANES)):
            entry = ENTRY_LANES[int(idx)]
            lane_index = (*entry, 0)
            lane = self.road.network.get_lane(lane_index)
            far = max(3.0, lane.length - 8.0)
            if spread:
                s = float(self.np_random.uniform(2.0, far))
            else:
                s = float(np.clip(longitudinal + self.np_random.uniform(0.0, 16.0),
                                  2.0, far))
            position = lane.position(s, 0.0)
            if not self._lane_free(self.road, position, min_gap):
                continue
            if not self._entry_lane_clear(lane_index, lane, s, leader_gap):
                continue
            vehicle = vehicle_type.make_on_lane(
                self.road,
                lane_index,
                longitudinal=s,
                speed=8.0 + max(0.0, self.np_random.normal()),
            )
            if check_conflict and self._spawn_conflicts(vehicle):
                self._traffic_stats["spawn_blocked_conflict"] += 1
                continue        # never added to road.vehicles -> no leak
            try:
                vehicle.plan_route_to(self._destination_for(entry))
            except (AttributeError, KeyError):
                pass
            vehicle.randomize_behavior()
            vehicle._spawn_step = int(self.time)
            self.road.vehicles.append(vehicle)
            self._traffic_stats["spawned"] += 1
            return vehicle
        self._traffic_stats["spawn_blocked_slot"] += 1
        return None

    def _clear_vehicles(self) -> None:
        """Despawn obstacle cars once they have driven OFF the map -- OR crashed.

        Only the lanes listed in EXIT_LANES lead out of the network. The old
        implementation checked "near the end of ANY lane", which deleted cars
        right before they entered a junction -- that is why traffic used to be
        so sparse and why nothing ever happened at the intersections.

        Crashed OBSTACLE cars are also removed: a crashed vehicle is frozen on
        the spot and never drives again, so leaving it on the map just blocks a
        lane and (over a few hundred steps) freezes the whole junction. The two
        RL agents are handled separately in `step()` (they get rebooted, not
        deleted), so they are excluded here.

        NOTE (honest accounting): a crash-induced despawn looks exactly like the
        "car mysteriously disappears" artefact in the video. That is why it is
        counted separately (`despawned_crash`) instead of being silently merged
        into normal despawns -- if the number is not 0, the ENVIRONMENT is
        still unstable, no matter how the video looks.
        """
        def is_leaving(vehicle) -> bool:
            if vehicle in self.controlled_vehicles:
                return False
            if vehicle.crashed:
                self._traffic_stats["despawned_crash"] += 1
                return True  # a stuck obstacle -- remove it so the lane frees up
            if tuple(vehicle.lane_index[:2]) not in self.EXIT_LANE_SET:
                return False
            try:
                lane = self.road.network.get_lane(vehicle.lane_index)
            except (KeyError, IndexError):
                self._traffic_stats["despawned_exit"] += 1
                return True  # off-network -> it left the map
            s = lane.local_coordinates(vehicle.position)[0]
            left = s >= lane.length - 3 * vehicle.LENGTH
            if left:
                self._traffic_stats["despawned_exit"] += 1
            return left

        self.road.vehicles = [
            v for v in self.road.vehicles
            if v in self.controlled_vehicles or not is_leaving(v)
        ]

    def _snapshot_pre_crash(self) -> None:
        """Stash every car's state at the START of the crashing step.

        Called immediately BEFORE `_simulate`, so the values captured are the
        last ones before the impact that the step is about to integrate. The
        crash is detected afterwards in `_record_collisions`, which reads these
        stashed values to answer "what did IDM see one step before it hit?".

        Captured per vehicle:
          * pre_leader_gap  -- gap to the car ahead on the SAME lane (None = no
                               leader was visible; the rear-end smoking gun)
          * pre_leader_ttc  -- time-to-collision to that leader
          * pre_s_frac      -- fraction along the current lane (~1 = at a
                               lane-boundary / junction mouth)
          * pre_speed       -- speed before the step

        This is pure bookkeeping -- it does not touch how the simulation runs.
        """
        for v in self.road.vehicles:
            g, t = _leader_gap(self.road, v)
            v._pre_leader_gap = g
            v._pre_leader_ttc = t
            v._pre_s_frac = _s_frac(self.road, v)
            v._pre_speed = float(v.speed)
            v._pre_velocity = np.asarray(v.velocity, dtype=float).copy()
            v._pre_position = np.asarray(v.position, dtype=float).copy()

    # -- background merge-yield (environment fix #1) -----------------------
    # IDM brakes only for a leader on the SAME lane, so two background cars
    # converging into the same downstream lane never see each other (proven by
    # the Step-2/3 diagnosis: 18/28 crashes were merges, pre_leader_gap=None).
    # We do NOT fake a leader vehicle; we arbitrate explicitly and make exactly
    # ONE side of each conflicting pair yield by capping its IDM target_speed.

    def _lane_priority(self, v) -> float:
        """Road-graph priority of the vehicle's current lane (default 1.0)."""
        li = getattr(v, "lane_index", None)
        try:
            return float(self.road.network.graph[li[0]][li[1]][0]["priority"])
        except Exception:
            return 1.0

    def _merge_distance(self, v):
        """Remaining distance [m] along the CURRENT lane to its end (the merge
        point when the lane converges with another). None = unknown."""
        li = getattr(v, "lane_index", None)
        if not li:
            return None
        try:
            lane = self.road.network.get_lane(li)
            s, _ = lane.local_coordinates(v.position)
            return max(0.0, lane.length - s)
        except Exception:
            return None

    def _lane_progress(self, v):
        """Distance [m] already travelled along the CURRENT lane (its s)."""
        li = getattr(v, "lane_index", None)
        if not li:
            return None
        try:
            lane = self.road.network.get_lane(li)
            s, _ = lane.local_coordinates(v.position)
            return max(0.0, s)
        except Exception:
            return None

    @staticmethod
    def _is_intersection_lane(li) -> bool:
        """True for lanes touching the cross intersection (nodes ir*/il*/o*).

        The intersection's angular gate stays strict (crossing is a later
        phase); roundabout-area lanes get the relaxed gate because the entry
        ramps genuinely meet the ring at ~55-60 degrees."""
        return any(str(n).startswith(("ir", "il", "o")) for n in li[:2])

    def _merge_pairs(self, bg):
        """Background pairs converging on the same downstream capacity.

        Five geometric patterns are merges (everything else, notably the
        ~90-degree intersection arms, is a CROSSING and stays untouched this
        round). Returns [(a, b, d_a, d_b, mode), ...]:
          mode "feeder"  -- a is d_a BEFORE the merge point, b has already
                            passed it (a feeds into b's lane or reaches the
                            node where b's lane starts);
          mode "eta"     -- both still approaching a shared merge point (same
                            next lane, or lanes ending at the same node) at
                            distances d_a / d_b;
          mode "stagger" -- near-parallel lanes leaving the same node, a behind
                            b (progress s_rear = d_a, s_front = d_b).

        Yielding is resolved downstream by capping the IDM target_speed of
        exactly ONE car per pair.
        """
        pairs = []
        n = len(bg)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = bg[i], bg[j]
                la = tuple(getattr(a, "lane_index", ()) or ())
                lb = tuple(getattr(b, "lane_index", ()) or ())
                if not la or not lb or la == lb:
                    continue
                # Same-lane pairs are already handled by IDM car-following.
                na, nb = _next_lane(a), _next_lane(b)
                # ROUTE-based merges (any heading angle): both cars converge
                # into the same downstream lane, so the conflict is real even
                # at 60+ degrees (e.g. two intersection arms feeding one exit).
                # Case 1: A's next lane IS B's current lane -- B already passed
                # the merge point, A is the feeder.
                if na is not None and na == lb:
                    d_a = self._merge_distance(a)
                    if d_a is not None and d_a <= MERGE_LOOKAHEAD:
                        pairs.append((a, b, d_a, 0.0, "feeder"))
                    continue
                if nb is not None and nb == la:
                    d_b = self._merge_distance(b)
                    if d_b is not None and d_b <= MERGE_LOOKAHEAD:
                        pairs.append((b, a, d_b, 0.0, "feeder"))
                    continue
                # Case 2: shared next lane -- both feed the same downstream.
                if na is not None and na == nb:
                    d_a, d_b = self._merge_distance(a), self._merge_distance(b)
                    if d_a is None or d_b is None:
                        continue
                    if min(d_a, d_b) > MERGE_LOOKAHEAD:
                        continue
                    pairs.append((a, b, d_a, d_b, "eta"))
                    continue
                # GEOMETRIC merges. Angular gate: strict for intersection
                # lanes (crossing stays untouched this round), relaxed for
                # roundabout lanes (the ramps meet the ring at ~55-60 deg).
                ang = _heading_angle_deg(a, b)
                gate = (MERGE_MAX_ANGLE
                        if (self._is_intersection_lane(la)
                            or self._is_intersection_lane(lb))
                        else MERGE_MAX_ANGLE_RING)
                if ang > gate:
                    continue
                # Case 3: same destination node -- lanes converge at the node.
                # BUT if both routes are known and DIVERGE past the node, the
                # cars run in parallel through it (parallel ring lanes) and
                # must NOT be staggered -- forcing a 2 s gap there throttled
                # the whole ring into a crawl (stall 28.8% in the first fix).
                if la[1] == lb[1]:
                    if na is not None and nb is not None and na != nb:
                        continue
                    d_a, d_b = self._merge_distance(a), self._merge_distance(b)
                    if d_a is None or d_b is None:
                        continue
                    if min(d_a, d_b) > MERGE_LOOKAHEAD:
                        continue
                    pairs.append((a, b, d_a, d_b, "eta"))
                    continue
                # Case 4: node hand-off -- A's lane ends exactly where B's lane
                # starts (or vice versa). The car whose lane STARTS at the node
                # is already past the merge; the arriving car must not run into
                # it (IDM cannot see across the segment boundary).
                if la[1] == lb[0] or lb[1] == la[0]:
                    if la[1] == lb[0]:
                        d_a = self._merge_distance(a)   # a ends at the node
                        if d_a is not None and d_a <= MERGE_LOOKAHEAD:
                            pairs.append((a, b, d_a, 0.0, "feeder"))
                    else:
                        d_b = self._merge_distance(b)
                        if d_b is not None and d_b <= MERGE_LOOKAHEAD:
                            pairs.append((b, a, d_b, 0.0, "feeder"))
                    continue
                # Case 5: near-parallel lanes leaving the SAME node (a ring arc
                # beside an exit ramp) -- sideswipe risk while the lanes are
                # still within a car width of each other. Same-link parallel
                # lanes are skipped: Case 3 already arbitrates their arrival.
                if la[0] == lb[0] and la[1] != lb[1]:
                    s_a, s_b = self._lane_progress(a), self._lane_progress(b)
                    if s_a is None or s_b is None:
                        continue
                    if max(s_a, s_b) > MERGE_LOOKAHEAD:
                        continue
                    # The car with MORE progress is ahead along the path.
                    if s_a >= s_b:
                        pairs.append((b, a, s_b, s_a, "stagger"))
                    else:
                        pairs.append((a, b, s_a, s_b, "stagger"))
        return pairs

    def _merge_yielder(self, a, b, d_a, d_b, eta_a, eta_b):
        """Deterministically pick WHICH car yields. A pair must always resolve
        to exactly one yielder -- both-yield is the merge-deadlock bug.

        Tie-break chain: lane priority -> remaining distance -> ETA -> id.
        """
        pa, pb = self._lane_priority(a), self._lane_priority(b)
        if pa != pb:
            return b if pa > pb else a          # lower priority yields
        if d_a != d_b:
            return b if d_a < d_b else a        # the car closer to the merge goes
        if eta_a != eta_b:
            return b if eta_a < eta_b else a    # earlier ETA goes
        return b if id(a) < id(b) else a        # stable final tie-break

    def _apply_merge_yield(self) -> None:
        """One arbitration pass, run every step BEFORE the physics integrate.

        Each background car's IDM `target_speed` is first restored to its base
        value, then capped for yielders -- so the cap never accumulates and a
        car resumes cruising as soon as its conflict clears.
        """
        if not MERGE_YIELD_ENABLED:
            return
        bg = [v for v in self.road.vehicles
              if v not in self.controlled_vehicles
              and not getattr(v, "crashed", False)]
        if len(bg) < 2:
            return
        for v in bg:
            if getattr(v, "_merge_base_speed", None) is None:
                v._merge_base_speed = float(getattr(v, "target_speed", 0) or 0)
            else:
                v.target_speed = v._merge_base_speed
        if getattr(_configs, 'CONFLICT_RESERVATIONS', False):
            return  # the reservation controller owns conflicting movements
        for a, b, d_a, d_b, mode in self._merge_pairs(bg):
            # Phase 4B: a ring-mouth pair is OWNED by the Dedicated Ring-Merge
            # Controller. Skip it here so the generic resolver cannot (wrongly)
            # yield the circulating car -- the dedicated controller meters the
            # feeder instead. Both cars had their base target_speed restored
            # above, so skipping leaves the circulating car at base (GO).
            # NOTE: read the flag LIVE from the configs module (the imported
            # constant is a frozen copy and would not see runtime flips).
            if getattr(_configs, "DEDICATED_RING_MERGE", False) \
                    and self.road._is_ring_mouth_pair(a, b):
                continue
            if mode == "feeder":
                # B already occupies/passed the merge: A must not catch up
                # across the lane boundary (IDM cannot see B). Cap A to B's
                # pace only when A is actually closing in on B.
                s_b = self._lane_progress(b) or 0.0
                path_gap = d_a + s_b - (a.LENGTH + b.LENGTH) / 2.0
                closing = a.speed - b.speed
                if closing > 0.5 and path_gap / closing < MERGE_CATCHUP_WINDOW:
                    a.target_speed = min(
                        a.target_speed, max(b.speed, MERGE_YIELD_MIN_SPEED)
                    )
                    self._traffic_stats["merge_yields"] += 1
                continue
            if mode == "stagger":
                # Near-parallel lanes leaving the same node: B is ahead, A
                # behind -- A must fall in behind instead of rubbing alongside.
                path_gap = d_b - d_a - (a.LENGTH + b.LENGTH) / 2.0
                closing = a.speed - b.speed
                if closing > 0.5 and path_gap / closing < MERGE_CATCHUP_WINDOW:
                    a.target_speed = min(
                        a.target_speed, max(b.speed, MERGE_YIELD_MIN_SPEED)
                    )
                    self._traffic_stats["merge_yields"] += 1
                continue
            # mode == "eta": both approaching the shared merge point.
            eta_a = d_a / max(a.speed, 1.0)
            eta_b = d_b / max(b.speed, 1.0)
            # A slow car already inside the merge mouth OWNS the mouth: the
            # other car must follow its pace instead of racing past on an ETA
            # win. The crashers' IDM cannot see across lanes, so "my ETA is
            # earlier" does NOT mean the mouth is clear (measured: a ramp car
            # creeping at 0.5 m/s got T-boned by a ring car that had "won" the
            # ETA comparison).
            a_in = d_a <= MERGE_MOUTH_DIST and a.speed <= MERGE_MOUTH_SPEED
            b_in = d_b <= MERGE_MOUTH_DIST and b.speed <= MERGE_MOUTH_SPEED
            if a_in != b_in:
                blocker, passer = (a, b) if a_in else (b, a)
                passer.target_speed = min(
                    passer.target_speed, max(blocker.speed, MERGE_YIELD_MIN_SPEED)
                )
                self._traffic_stats["merge_yields"] += 1
                continue
            if abs(eta_a - eta_b) > MERGE_ETA_WINDOW:
                continue        # one car clearly arrives first -- no conflict
            yielder = self._merge_yielder(a, b, d_a, d_b, eta_a, eta_b)
            d_y = d_a if yielder is a else d_b
            eta_go = eta_b if yielder is a else eta_a
            # Arrive at least MERGE_ETA_WINDOW after the other car.
            v_cap = d_y / (eta_go + MERGE_ETA_WINDOW)
            v_cap = max(v_cap, MERGE_YIELD_MIN_SPEED)
            yielder.target_speed = min(yielder.target_speed, v_cap)
            self._traffic_stats["merge_yields"] += 1

    # -- Phase 4B: Dedicated Ring-Merge Controller --------------------------
    # OWNS the four ring mouths. For each topologically-confirmed ring-mouth pair
    # (circulating on a ring arc, feeder on an external slip, both converging on
    # the same mouth node) it enforces, EVERY step:
    #     circulating = GO    (never capped by this controller or the group)
    #     feeder      = YIELD (cap metered to the convergence node; never both yield)
    # Detection reuses the SAME proximity as _merge_pairs Case-3 (distance <=
    # MERGE_LOOKAHEAD) -- lookahead is NOT expanded. The cap is GRADUAL (a creep
    # floor between FAR and NEAR) and only allows a full hold (cap 0) within
    # RING_MERGE_NEAR of the mouth while the circulating car has not yet cleared,
    # so the feeder "enters late" instead of "stopping dead early". See configs.

    def _apply_dedicated_ring_merge(self) -> None:
        """One arbitration pass for the Dedicated Ring-Merge Controller.

        Runs once per step BEFORE the physics integrate, after the generic
        merge-yield (which has already restored every background car's base
        target_speed and skipped the ring-mouth pairs). Only the feeder of each
        ring-mouth pair is ever capped here; the circulating car keeps its base.
        """
        # Read the flag LIVE from the configs module (the imported constant is a
        # frozen copy and would not see runtime flips by the A/B driver / tests).
        if not getattr(_configs, "DEDICATED_RING_MERGE", False):
            return
        bg = [v for v in self.road.vehicles
              if v not in self.controlled_vehicles
              and not getattr(v, "crashed", False)]
        if len(bg) < 2:
            return
        n = len(bg)
        for i in range(n):
            for j in range(i + 1, n):
                rm = self.road._is_ring_mouth_pair(bg[i], bg[j])
                if not rm:
                    continue
                C, F, M = rm
                self._dedicated_ring_merge_cap(C, F, M)

    def _dedicated_ring_merge_cap(self, C, F, M) -> None:
        """Meter the FEEDER (F) to the convergence node M; never cap C."""
        d_C = self._merge_distance(C)
        d_F = self._merge_distance(F)
        if d_C is None or d_F is None:
            return
        # Do NOT expand lookahead: act only within MERGE_LOOKAHEAD of the mouth
        # (same radius as _merge_pairs Case-3 detection).
        if min(d_C, d_F) > MERGE_LOOKAHEAD:
            return
        self._traffic_stats["ring_merge_interventions"] += 1
        # The circulating car must never be capped by this controller. Isolation
        # in enforce_road_rules keeps the group off it too; this counter measures
        # any residual restriction from elsewhere (should stay 0).
        c_base = getattr(C, "_merge_base_speed", C.target_speed)
        if C.target_speed < c_base - 0.05:
            self._traffic_stats["ring_merge_circ_restricted"] += 1
        # Far out: the feeder keeps cruising -- do not brake early (anti-queue).
        if d_F > RING_MERGE_FAR:
            return
        eta_C = d_C / max(C.speed, 0.1)
        eta_F = d_F / max(F.speed, 0.1)
        # If the feeder would clear the mouth comfortably BEFORE the circulating
        # car arrives, it may proceed -- "late entry, not early stop".
        if eta_F < eta_C - RING_MERGE_TRANSIT:
            return
        # Feeder must arrive AFTER the circulating car clears the mouth.
        v_cap = d_F / max(eta_C + RING_MERGE_SAFETY, 0.1)
        base = getattr(F, "_merge_base_speed", None) or getattr(F, "target_speed", 0)
        v_cap = min(base, v_cap)
        if d_F > RING_MERGE_NEAR:
            v_cap = max(v_cap, RING_MERGE_MIN_SPEED)   # creep, never pin to 0 far out
        before = F.target_speed
        if v_cap < before:
            if v_cap <= 0.05:
                self._traffic_stats["ring_merge_feeder_stops"] += 1
            F.target_speed = v_cap
        # Double-yield guard: both cars in the pair yielded -> must never happen.
        f_base = getattr(F, "_merge_base_speed", F.target_speed)
        if (F.target_speed < f_base - 0.05
                and C.target_speed < c_base - 0.05):
            self._traffic_stats["ring_merge_double_yield"] += 1

    def _gridlock_break_group(self, members) -> bool:
        """Recycle a hard-locked conflict cluster (called by the road).

        Only fires once EVERY member has been stalled GRIDLOCK_BREAK_STEPS --
        until then the road simply holds the cluster frozen. Each recycled
        car is counted (`despawned_gridlock`) and immediately re-entered from
        a fresh periphery lane so the population does not dip.
        """
        if any(v in self.controlled_vehicles for v in members):
            return False
        ready = [v for v in members
                 if getattr(v, "_stall_count", 0) >= GRIDLOCK_BREAK_STEPS
                 and v in self.road.vehicles]
        if len(ready) < len(members):
            return False
        for v in ready:
            if v in self.road.vehicles:
                self.road.vehicles.remove(v)
                self._traffic_stats["despawned_gridlock"] += 1
        for _ in range(len(ready)):
            self._spawn_vehicle(spawn_probability=1.0)
        return True

    def _break_gridlocks(self) -> None:
        """Recycle stalled clusters (honest per-car accounting).

        Once ANY background car has been stalled GRIDLOCK_BREAK_STEPS in a
        row, the whole stalled cluster around it (slow cars within 20 m) is
        recycled back to fresh periphery lanes. Deliberately decisive: a
        packed 10-car junction lock cannot be choreographed out one car at a
        time -- the partial unlock itself produced low-speed crunches
        (measured) -- so the cluster dissolves in a single step instead.
        Every removal is counted (`despawned_gridlock`); the population is
        refilled immediately from entry lanes.
        """
        if getattr(_configs, 'CONFLICT_RESERVATIONS', False):
            return  # expose deadlocks; do not hide them by recycling queues
        bg = [v for v in self.road.vehicles
              if v not in self.controlled_vehicles
              and not getattr(v, "crashed", False)]
        for v in bg:
            if v.speed < 0.5:
                v._stall_count = getattr(v, "_stall_count", 0) + 1
            else:
                v._stall_count = 0
        stuck_long = [v for v in bg
                      if getattr(v, "_stall_count", 0) >= GRIDLOCK_BREAK_STEPS]
        if not stuck_long:
            return
        victims = set()
        for v in stuck_long:
            for u in bg:
                if u.speed < 2.0 and float(
                        np.linalg.norm(u.position - v.position)) <= 20.0:
                    victims.add(id(u))
        removed = 0
        for u in bg:
            if id(u) in victims and u in self.road.vehicles:
                self.road.vehicles.remove(u)
                self._traffic_stats["despawned_gridlock"] += 1
                removed += 1
        for _ in range(removed):
            self._spawn_vehicle(spawn_probability=1.0)

    @staticmethod
    def _veh_detail(env, v) -> dict:
        """Compact diagnostic record for one vehicle in a crash."""
        li = tuple(getattr(v, "lane_index", ()) or ())
        g = getattr(v, "_pre_leader_gap", None)
        t = getattr(v, "_pre_leader_ttc", None)
        sf = getattr(v, "_pre_s_frac", None)
        return {
            "lane": li,
            "speed": round(float(v.speed), 2),
            "heading": round(_heading_deg(v), 1),
            "next": _next_lane(v),
            "s_frac": None if sf is None else round(sf, 3),
            "pre_leader_gap": None if g is None else round(g, 2),
            "pre_leader_ttc": (None if t is None or t == float("inf")
                               else round(t, 2)),
            "pre_speed": round(float(getattr(v, "_pre_speed", v.speed)), 2),
        }

    def _record_collisions(self) -> None:
        """Tag every NEW collision this step as BB / CB / CC and count it.

        Runs right after the physics step (and BEFORE `_clear_vehicles` deletes
        crashed obstacle cars), so no impact is missed. Each vehicle is only
        counted once per crash -- the flag is cleared when it is rebooted or
        despawned, so a later, separate impact counts again.

        A crash is attributed to the pair of vehicles whose bounding boxes
        actually intersect, which is how highway-env itself decides `crashed`.

        DIAGNOSTICS (Step 2): every crash is also classified (rear-end / merge /
        crossing / other / single) and tagged with the pre-crash leader gap,
        lane-boundary fraction and closing speed. This is what lets the stress
        test report "which crash mechanism dominates" instead of a single BB=N
        number.
        """
        controlled = {id(v) for v in self.controlled_vehicles}
        newly = [v for v in self.road.vehicles
                 if getattr(v, "crashed", False)
                 and not getattr(v, "_crash_counted", False)]
        if not newly:
            return
        for v in newly:
            v._crash_counted = True
        # On impact highway-env pushes the two cars apart (impact vector), so
        # by the END of the step their boxes may no longer overlap even though
        # they just collided. Pair them by proximity as a fallback, otherwise
        # genuine two-car crashes get mis-filed as mysterious "singles".
        pair_dist = 7.0
        used = set()
        for v in newly:
            if id(v) in used:
                continue
            partner = None
            pairing_source = None
            for u in newly:
                if u is v or id(u) in used:
                    continue
                try:
                    if v._is_colliding(u, 0)[0]:
                        partner = u
                        pairing_source = "direct_overlap"
                        break
                except Exception:
                    continue
            if partner is None:
                best = None
                for u in newly:
                    if u is v or id(u) in used:
                        continue
                    d = float(np.linalg.norm(v.position - u.position))
                    if d <= pair_dist and (best is None or d < best[0]):
                        best = (d, u)
                if best is not None:
                    partner = best[1]
                    pairing_source = "fallback_7m"
            if partner is not None:
                used.add(id(partner))
            used.add(id(v))
            pair = [v] if partner is None else [v, partner]
            if partner is None:
                # No overlap with another NEWLY-crashed car: this is either a
                # secondary impact inside a same-step pile-up (the partner
                # crashed at an earlier sub-step and was already paired) or an
                # off-road excursion. Tag it so pile-ups don't masquerade as a
                # separate crash mechanism.
                wreck, on_road = None, True
                for u in self.road.vehicles:
                    if u is v or not getattr(u, "crashed", False):
                        continue
                    if float(np.linalg.norm(v.position - u.position)) <= pair_dist:
                        wreck = u
                        break
                    try:
                        if v._is_colliding(u, 0)[0]:
                            wreck = u
                            break
                    except Exception:
                        continue
                try:
                    on_road = self.road.network.is_on_road(v.position)
                except Exception:
                    pass
                ctype = ("pileup" if wreck is not None
                         else "offroad" if not on_road else "unknown-single")
                if wreck is not None:
                    # Keep the wreck as the "B" side so the report shows what
                    # was hit (and which lane the initiating crash was on).
                    pair = [v, wreck]
                    pairing_source = "wreck_search"
                else:
                    pairing_source = "unresolved_single"
            flags = [id(x) in controlled for x in pair]
            if len(pair) == 2:
                kind = "CC" if all(flags) else ("BB" if not any(flags) else "CB")
            else:
                kind = "CB" if flags[0] else "BB"
            self._collision_classes[kind] += 1
            # Classification + pre-crash geometry. For a primary 2-vehicle
            # impact the mechanism is geometric; for pileup/offroad/unknown
            # the tag was already set above.
            if partner is not None:
                ctype = _classify_pair(pair[0], pair[1])
            angle = (None if len(pair) == 1
                     else round(_heading_angle_deg(pair[0], pair[1]), 1))
            if len(pair) == 2:
                u_vec = pair[0].position - pair[1].position
                nu = float(np.linalg.norm(u_vec))
                rel_dist = nu
                closing = 0.0
                if nu > 1e-6:
                    rel_vel = pair[0].velocity - pair[1].velocity
                    closing = -float(np.dot(rel_vel, u_vec / nu))
                mid = (pair[0].position + pair[1].position) / 2.0
                # Pre-crash conflict TTC: extrapolate the PRE-STEP kinematics of
                # the two crash partners (positions/velocities captured by
                # _snapshot_pre_crash) to the moment their boxes would touch.
                u0 = pair[0]._pre_position - pair[1]._pre_position
                nu0 = float(np.linalg.norm(u0))
                pre_ttc_pair = None
                if nu0 > 1e-6:
                    vrel0 = pair[0]._pre_velocity - pair[1]._pre_velocity
                    closing0 = -float(np.dot(vrel0, u0 / nu0))
                    if closing0 > 1e-6:
                        gap0 = nu0 - (pair[0].LENGTH + pair[1].LENGTH) / 2.0
                        pre_ttc_pair = max(0.0, gap0) / closing0
            else:
                rel_dist = 0.0
                closing = 0.0
                mid = pair[0].position
                pre_ttc_pair = None
            self._collision_log.append(dict(
                step=int(self.time), kind=kind, cause=ctype, ctype=ctype,
                angle=angle,
                ids=[id(x) for x in pair],   # lets diagnostics match traces
                # Phase 4E-B (accounting-only, additive fields):
                #  * serials  -- persistent per-object identity, immune to
                #                Python id() reuse across despawn/respawn
                #  * pre_pos  -- positions captured BEFORE this step's physics
                #                (by _snapshot_pre_crash) so the accounting
                #                layer can re-pair impact-separated partners
                #  * lengths  -- body lengths for contact-distance checks
                #  * pairing_source -- how the recorder built this pair
                serials=[_serial_of(x) for x in pair],
                pre_pos=[_pre_pos_of(x) for x in pair],
                lengths=[float(getattr(x, "LENGTH", 5.0)) for x in pair],
                pairing_source=(pairing_source if partner is not None
                                else pairing_source or "unresolved_single"),
                pre_ttc_pair=(None if pre_ttc_pair is None
                              else round(pre_ttc_pair, 2)),
                lanes=[tuple(x.lane_index) for x in pair],
                pos=[round(float(c), 1) for c in mid.tolist()],
                pos_all=[np.round(x.position, 1).tolist() for x in pair],
                rel_dist=round(float(rel_dist), 2),
                closing=round(float(closing), 2),
                A=self._veh_detail(self, pair[0]),
                B=(self._veh_detail(self, pair[1]) if len(pair) == 2 else None),
            ))
            if kind != "BB":
                # _crash_count stays the "controlled vehicle crashed" number
                # evaluate.py already reports; keep it consistent.
                self._crash_count += sum(1 for f in flags if f)

    def _recycle_stale_traffic(self) -> None:
        """Re-route long-lived obstacle cars towards an exit so traffic turns over."""
        max_age = self.OBSTACLE_MAX_AGE
        for v in self.road.vehicles:
            if v in self.controlled_vehicles:
                continue
            if self.time - getattr(v, "_spawn_step", 0) < max_age:
                continue
            if getattr(v, "route", None):       # still heading somewhere -> leave it
                continue
            try:
                v.plan_route_to(
                    EXIT_NODES[self.np_random.integers(0, len(EXIT_NODES))]
                )
            except (AttributeError, KeyError):
                pass
            v._spawn_step = int(self.time) - max_age // 2

    def _maintain_traffic(self) -> None:
        """Top the obstacle population back up to `target_vehicle_count`."""
        target = self.config["target_vehicle_count"] + len(self.controlled_vehicles)
        attempts = 0
        while len(self.road.vehicles) < target and attempts < 8:
            attempts += 1
            self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])

    # -- step -------------------------------------------------------------
    def step(self, action):
        self._snapshot_pre_crash()
        self._apply_merge_yield()
        self._apply_dedicated_ring_merge()
        self._break_gridlocks()
        # IntersectionEnv.step removes wrecks and spawns traffic before returning.
        # Compound owns that lifecycle: record impacts before any removal and
        # maintain the population exactly once.
        obs, reward, terminated, truncated, info = AbstractEnv.step(self, action)
        # Tag + count every impact BEFORE crashed obstacles are despawned, and
        # BEFORE the agents' `crashed` flag is cleared by the reboot below.
        self._record_collisions()
        self._clear_vehicles()
        self._recycle_stale_traffic()
        self._maintain_traffic()
        # Keep the controlled agents circulating forever:
        #  * a crashed agent is rebooted onto a fresh start lane (in place, so the
        #    MultiAgentObservation bound to this object keeps observing it); and
        #  * an agent that has (almost) finished its route is re-directed so it
        #    never stops moving.
        for i, ego in enumerate(self.controlled_vehicles):
            ego.color = (
                self.CRASH_COLOR if ego.crashed
                else self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            )
            if ego.crashed or self._at_dead_end(ego):
                if ego.crashed:
                    # NOTE: the crash is already counted (and classified BB/CB/CC)
                    # by _record_collisions() above -- do not count it twice.
                    if self.config.get("crash_terminate", False):
                        # Display/eval mode (--end-on-crash): leave the car
                        # crashed in place so the video shows the real collision
                        # instead of a teleport. _is_terminated has already
                        # flagged the episode as done, so just skip the reboot.
                        continue
                self._reset_controlled(ego, i)
                self._stall_steps[i] = 0
                continue
            # Anti-deadlock watchdog. An agent parked inside a junction (or at
            # the end of a dead-end lane) blocks everything behind it for the
            # REST of the episode -- that is the visible "traffic jam" in the
            # video. Yielding for a few seconds is normal and is tolerated; only
            # a long freeze is treated as a deadlock and cleared by a reboot.
            # (The reward-side cure is the new stall_reward; this is the
            # environment-side safety net so one stuck car cannot jam the map.)
            if ego.speed < self.config.get(
                "stall_speed_threshold", STALL_SPEED_THRESHOLD
            ):
                self._stall_steps[i] = self._stall_steps.get(i, 0) + 1
            else:
                self._stall_steps[i] = 0
            if self._stall_steps[i] >= STALL_REBOOT_STEPS and not getattr(_configs, 'CONFLICT_RESERVATIONS', False):
                self._reset_controlled(ego, i)
                self._stall_steps[i] = 0
            elif self._nearly_at_destination(ego):
                self._reroute(ego, i)
        # The next policy action must observe the state it will act on, including
        # respawns and newly maintained traffic. Keep reward/termination from the
        # physical transition, so rebooting cannot erase a collision penalty.
        obs = self.observation_type.observe()
        return obs, reward, terminated, truncated, info

    def _reset_controlled(self, vehicle, index: int) -> None:
        """Reboot a crashed/stopped controlled agent onto a fresh re-entry lane.

        Which lane depends on which half of the map it failed in (see EAST_X):
        an agent that goes down in the roundabout half is re-entered on a ring
        ENTRY ramp so it can re-practise the merge immediately; only an agent
        that fails in the intersection half goes back to the intersection arms.
        The old always-to-the-intersection reboot made the agent cross the whole
        map for every attempt at the far junction -- which, with a fixed step
        budget, is exactly why the roundabout merge never got enough samples.
        """
        if float(vehicle.position[0]) > self.EAST_X:
            starts, offsets = self.EAST_STARTS, self.EAST_STARTS_S
        else:
            starts, offsets = self.CONTROL_STARTS, self.CONTROL_STARTS_S
        start = starts[index % len(starts)]
        lane = self.road.network.get_lane((*start, 0))
        s = min(offsets[index % len(offsets)], lane.length - 5.0)
        # Clear a bubble around the reboot point, otherwise the agent re-spawns
        # nose-to-tail with a queue of traffic and crashes again on the very
        # next step (it used to look "frozen" at its start position).
        spot = lane.position(s, 0)
        self.road.vehicles = [
            v for v in self.road.vehicles
            if v in self.controlled_vehicles
            or np.linalg.norm(v.position - spot) > self.CONTROL_CLEAR_RADIUS
        ]
        # Reuse the vehicle object so the observation hook stays valid.
        vehicle.road = self.road
        vehicle.position = spot
        vehicle.heading = lane.heading_at(s)
        # Cap the reboot speed to the top gear (see _make_vehicles: lane speed limit
        # exceeds the gearbox max, so resetting to it would relaunch at 20 m/s).
        v0 = min(lane.speed_limit, float(max(vehicle.target_speeds)))
        vehicle.speed = v0  # velocity is derived from speed*heading
        vehicle.crashed = False
        vehicle.impact = None
        vehicle._regulation_speed_cap = float('inf')
        # Allow the NEXT, separate impact of this car to be counted again.
        vehicle._crash_counted = False
        vehicle.target_speed = v0
        # MDPVehicle stores the desired speed as an INDEX into `target_speeds`.
        # If the agent had braked to a stop before crashing, that index is 0, so
        # on the very next act() `target_speed` snaps back to 0 and the car
        # brakes to a standstill and NEVER moves again -- measured: 282
        # consecutive stalled steps after a single crash. The index must be
        # reset together with the speed itself.
        try:
            vehicle.speed_index = vehicle.speed_to_index(v0)
        except Exception:  # pragma: no cover - non-MDP vehicle classes
            pass
        vehicle.target_lane_index = (start[0], start[1], 0)
        vehicle.lane_index = (start[0], start[1], 0)
        vehicle.lane = lane
        # No route yet: the next step() sees route=None and (re)routes the agent
        # to a safe loop destination via the same robust fallback used during
        # normal circulation.
        vehicle.route = None
        self._reroute(vehicle, index)
        # The vehicle was teleported; drop its progress tracker so the next step
        # does not register a huge (fake) forward jump.
        self._agent_prev.pop(id(vehicle), None)

    @staticmethod
    def _at_dead_end(vehicle) -> bool:
        """True when the vehicle is near the end of a lane that has NO successor.

        The compound map's off-map exits (sxr / exr / nxr and the outer arms)
        simply stop. Obstacle traffic reaching them is despawned, but the
        controlled agents never are -- so an agent that reaches one parks in the
        last few metres of the lane forever and blocks everything behind it.
        Re-routing cannot rescue it either: with no successor lane there is no
        route out of a dead end at all, so the only cure is a reboot.
        """
        try:
            lane = vehicle.road.network.get_lane(vehicle.lane_index)
            s = lane.local_coordinates(vehicle.position)[0]
        except (KeyError, IndexError, AttributeError):
            return False
        if s < lane.length - 3 * vehicle.LENGTH:
            return False      # still has road ahead -- not stuck yet
        # lane_index[1] is the node this lane leads TO; no outgoing edge means
        # the network ends here.
        return not vehicle.road.network.graph.get(vehicle.lane_index[1], {})

    @staticmethod
    def _nearly_at_destination(vehicle) -> bool:
        if vehicle.route is None or len(vehicle.route) <= 1:
            return True
        target = vehicle.route[-1]
        try:
            lane = vehicle.road.network.get_lane((target[0], target[1], 0))
        except (KeyError, IndexError):
            return True
        long, _ = lane.local_coordinates(vehicle.position)
        return vehicle.lane_index[0] == target[0] and long >= lane.length - 30

    def _reroute(self, vehicle, index: int) -> None:
        """Send an agent to a new destination that ALWAYS has a successor lane.

        Two failure modes are fixed here:

        * Dead-end destinations. sxr / exr / nxr are the roundabout's off-map
          exits with no successor lane. Obstacle cars reaching them are
          despawned, but the controlled agents never are -- so an agent routed
          there parks at the end of the lane forever and blocks the junction
          behind it (measured: orange car stalled at ('sxs','sxr',0)). Only
          SAFE_LOOP_DESTINATIONS is used now.
        * A failed route plan. The compound network's planner is a heuristic at
          our merged junctions and can fail. The old code swallowed the
          exception and left the PREVIOUS route in place, so the car, already at
          its destination, re-triggered the reroute on every single step
          (measured: 37 reroutes in one episode, car never moving). Now every
          candidate is tried in turn until one actually produces a route.
        """
        start = int(self.np_random.integers(0, len(SAFE_LOOP_DESTINATIONS)))
        for k in range(len(SAFE_LOOP_DESTINATIONS)):
            dst = SAFE_LOOP_DESTINATIONS[
                (start + k) % len(SAFE_LOOP_DESTINATIONS)
            ]
            try:
                vehicle.plan_route_to(dst)
            except (AttributeError, KeyError):
                continue
            if getattr(vehicle, "route", None):
                return

    # -- rendering --------------------------------------------------------
    def _setup_camera(self) -> None:
        if not self.config.get("panoramic_view", False):
            return
        if self.render_mode is None or self.viewer is not None:
            return
        from env_core.envs.common.graphics import EnvViewer

        # Create the viewer ourselves so the fixed camera is in place BEFORE the
        # first frame is drawn (AbstractEnv.render would otherwise draw frame 0
        # centred on the ego vehicle).
        self.viewer = EnvViewer(self)
        self.viewer.observer_vehicle = _FixedCamera(self.config["camera_center"])
        # EnvViewer disables itself (`self.enabled = False`) whenever
        # SDL_VIDEODRIVER == "dummy" so it can run headless. That flag ALSO makes
        # `display()` a no-op, which would render every frame black. With
        # offscreen rendering (auto-enabled for rgb_array mode) all drawing goes
        # to `sim_surface` and needs no real window, so we must keep drawing
        # enabled even in headless/dummy mode.
        self.viewer.enabled = True

    def render(self):
        self._setup_camera()
        return super().render()


class JointAgentWrapper(gym.Wrapper):
    """Expose the two-agent compound env to SB3 as single Box(150) + Discrete(25).

    The inner CompoundEnv uses gymnasium Tuple observation/action (one entry per
    controlled agent). stable-baselines3 cannot ingest Tuple spaces, so this
    wrapper concatenates the two per-agent observations into one flat vector and
    encodes the joint action as a single integer `a1 * 5 + a2`.

    Inner reward is already the mean over the two cooperative agents, so it is
    passed through unchanged.
    """

    def __init__(self, env, n_agents: int = N_AGENTS, action_dim: int = 5):
        super().__init__(env)
        self.n_agents = n_agents
        self.action_dim = action_dim
        # The inner env is built with render_mode already set by gym.make; the
        # wrapper only reshapes obs/action, so preserve its render_mode.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(JOINT_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(JOINT_ACTION_DIM)

    def _flat(self, obs) -> np.ndarray:
        obs = np.asarray([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        return obs.reshape(-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flat(obs), info

    def step(self, action: int):
        a1 = int(action) // self.action_dim
        a2 = int(action) % self.action_dim
        inner_action = (a1, a2) if self.n_agents == 2 else (a1,)
        obs, reward, terminated, truncated, info = self.env.step(inner_action)
        return self._flat(obs), reward, terminated, truncated, info


class NoiseWrapper(gym.Wrapper):
    """Inject zero-mean Gaussian noise into the kinematic observation.

    Models imperfect perception (radar/camera noise) so you can measure how
    robust the learned policy is to noisy sensors.
    """

    def __init__(self, env, noise_std: float = 0.0):
        super().__init__(env)
        self.noise_std = noise_std

    def _noisy(self, obs):
        if self.noise_std > 0:
            rng = self.env.unwrapped.np_random
            obs = obs + rng.normal(0.0, self.noise_std, obs.shape).astype(obs.dtype)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._noisy(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy(obs), reward, terminated, truncated, info


# --- 3. Multi-agent game equilibrium (EXPERIMENTAL extension) ----------------
# Guarded so a missing/renamed internal NEVER breaks the core single-agent
# pipeline. Uses the real env_core MultiAgentObservation / MultiAgentAction
# wrappers (verified present in the vendored 1.12.2.dev0 source). The factory
# reads `observation_config` / `action_config` keys, so the SAME per-vehicle
# SHARED_OBS / SHARED_ACTION are stacked across all controlled vehicles.
try:
    from env_core.envs.common.observation import MultiAgentObservation  # noqa: F401
    from env_core.envs.common.action import MultiAgentAction  # noqa: F401

    class MultiAgentRoundabout(MyRoundabout):
        """EXPERIMENTAL multi-agent variant (bonus experiment only).

        Sets `controlled_vehicles` > 1 and switches the observation/action types
        to env_core's MultiAgentObservation / MultiAgentAction, which stack
        the SAME per-vehicle SHARED_OBS / SHARED_ACTION across all controlled
        vehicles. Because the per-vehicle spaces are identical (from
        configs.py), a single MlpPolicy can be trained on the stacked space --
        the basis for the multi-agent game-equilibrium experiment. Not wired
        into train.py / evaluate.py; use as a starting point.
        """

        @classmethod
        def default_config(cls) -> dict:
            cfg = super().default_config()
            cfg["controlled_vehicles"] = 2
            cfg["observation"] = {
                "type": "MultiAgentObservation",
                "observation_config": dict(SHARED_OBS),
            }
            cfg["action"] = {
                "type": "MultiAgentAction",
                "action_config": dict(SHARED_ACTION),
            }
            return cfg

except Exception:  # pragma: no cover - guarded; core pipeline unaffected
    MultiAgentRoundabout = None  # type: ignore


def register_custom_envs() -> None:
    """Register the custom env ids (idempotent)."""
    specs = {
        "MyRoundabout-v0": MyRoundabout,
        "MyIntersection-v0": MyIntersection,
        "MyCompound-v0": CompoundEnv,
    }
    if MultiAgentRoundabout is not None:
        specs["MultiAgentRoundabout-v0"] = MultiAgentRoundabout
    for env_id, cls in specs.items():
        try:
            register(id=env_id, entry_point=cls)
        except gym.error.RegisteredEnvironmentError:
            pass


def make_env(scene: str = "roundabout", noise_std: float = NOISE_STD,
             seed: int = 0, render_mode: str | None = None):
    """Build a registered, noise-wrapped environment for `scene`.

    `render_mode` must be passed through `gym.make` (not set afterwards) because
    gymnasium wraps every env in `OrderEnforcing`, and `Wrapper.render_mode` is a
    read-only property.

    For the multi-agent ``compound`` scene the raw env exposes gymnasium Tuple
    observation/action spaces; we wrap it in a :class:`JointAgentWrapper` so
    stable-baselines3 sees a single Box(150,) + Discrete(25). ``NoiseWrapper`` is
    skipped for compound because it would broadcast noise onto a Tuple obs.
    """
    register_custom_envs()
    if scene == "compound":
        # The compound map makes env_core's route-planner fall back to a
        # closest-lane heuristic at our (merged) junctions; that path is handled,
        # but it logs a warning per intersection. Silence those benign messages.
        import logging

        logging.getLogger("env_core").setLevel(logging.ERROR)
        raw = gym.make(ENV_IDS[scene], render_mode=render_mode)
        env = JointAgentWrapper(raw)
    else:
        env = gym.make(ENV_IDS[scene], render_mode=render_mode)
        if noise_std > 0:
            env = NoiseWrapper(env, noise_std=noise_std)
    env.reset(seed=seed)
    return env
