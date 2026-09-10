"""Unified configuration for the dual-scene roundabout / intersection RL project.

Every knob you might want to tune for the NUS ME5418 final project lives here.
The whole point of this file is to make the two scenes (roundabout and
intersection) share ONE observation/action space, so a single PPO/DQN network
can be trained and compared across the two intersection topologies.
"""

# --- 1. TTC-based safety reward --------------------------------------------
# When True, the custom envs add a penalty whenever the time-to-collision (TTC)
# to the nearest vehicle drops below TTC_THRESHOLD. Set to False (or pass
# --no-ttc to train.py) to reproduce the baseline reward for ablation studies.
USE_TTC = True

# When True, a crash TERMINATES the episode (training = "survive a clean run").
# When False (default) the compound env reboots the crashed car in place and
# keeps going ("loop forever"). Terminating makes a collision expensive (it costs
# the whole remaining return), but risks re-introducing the "park to avoid
# crashing" freeze we just fixed -- so it is exposed as a flag (--crash-terminate)
# for an A/B ablation and has NO effect unless main() sets it before the env is
# built. Read live by CompoundEnv._is_terminated.
CRASH_TERMINATE = False
# Weight on the TTC term, raised from -0.15. The safety layer can only prevent
# REAR-END collisions (it looks solely at the leader straight ahead), so the
# crossing conflicts at the intersection and the roundabout merges have to be
# LEARNED. A stronger TTC gradient is what teaches the agent to yield early in
# those situations instead of only reacting once it has already been hit.
TTC_REWARD = -0.25       # weight applied to the ttc term in the reward sum
TTC_THRESHOLD = 3.0      # seconds; below this TTC the penalty is triggered

# --- 2. Perception-noise robustness ---------------------------------------
# Default std of zero-mean Gaussian noise injected into the kinematic
# observation. Overridden per-run by the --noise argument of train.py / evaluate.py.
NOISE_STD = 0.0

# --- 3. Observation / action alignment -------------------------------------
# Both scenes are forced to the SAME spaces so one network fits both.
SHARED_OBS = {
    "type": "Kinematics",
    "vehicles_count": 15,
    "features": ["presence", "x", "y", "vx", "vy"],
    # Relative (ego-centric) coordinates: x = lateral offset, y = longitudinal
    # offset, vx/vy = relative velocity, all w.r.t. the observer vehicle.
    # This is FAR easier for the policy to learn from than absolute world
    # coordinates (the old value), because the network no longer has to subtract
    # the ego position to know "where is that other car relative to me". It is
    # also what makes "is there room to change lane / overtake" learnable: a
    # car in the adjacent lane shows up at x ~ +/- lane_width, a car ahead at
    # y > 0 -- directly readable from the observation.
    "absolute": False,
    "flatten": True,        # flattened 1-D vector (needed for SB3 MlpPolicy)
}
SHARED_ACTION = {
    "type": "DiscreteMetaAction",
    "longitudinal": True,
    "lateral": True,        # open lane-change so BOTH scenes expose 5 actions
    # Five speed steps instead of three. With only [0, 8, 16] a single SLOWER
    # action drops the target from 16 to 8, and the next one to a FULL STOP --
    # so the only way to slow down is to slam on the brakes. Measured on the
    # trained policy: the green car issued SLOWER 155/300 times and ended up
    # stalled 73.7% of the time, because every brake was a brake to zero.
    # Finer steps let the agent slow down gently (16 -> 12 -> 8 -> 4) and keep
    # rolling, which is what the IDM background traffic does naturally.
    "target_speeds": [0, 4, 8, 12, 16],
}

# --- Unified reward weights (applied inside custom_envs.py) ----------------
REWARD = {
    # V3 balance. V1 (-12 / speed 1.0) was "drives but crashes too much"; V2
    # (-25 / speed 0.5) over-corrected into "parks ~90% of the time" (91%/82%
    # stall, 0.6-1.4 m/s, yet STILL 4 crashes -- parking did not even buy
    # safety). The scene never terminates (a crash is amortised over the whole
    # 300-step episode), so we walk back to the middle instead of pushing
    # collision further up. -18 keeps one crash clearly worse than ~60 steps of
    # stall without making "never move" the safest play.
    "collision_reward": -18.0,
    # V2 0.5 -> 0.75. Halving speed removed the "hold FASTER and let the shield
    # brake" degenerate policy but over-shot, leaving the agent no reason to
    # roll. 0.75 restores a real incentive to make progress while staying below
    # V1's 1.0 (which invited "accelerate, the shield saves me").
    "high_speed_reward": 0.75,
    # Lane changes are deliberately COST-NEUTRAL. Note this weight is not even
    # read by CompoundEnv._agent_reward -- it is pinned to 0 to state the intent
    # unambiguously. Charging for a lane change would penalise the exact
    # manoeuvre this project must demonstrate (overtaking / avoiding a slow
    # car); rewarding it would invite pointless lane weaving. Cost-neutral means
    # the agent may change lane whenever it buys progress or defuses a low-TTC
    # situation, and both of those are already paid for by progress / TTC.
    # (Measured: the trained policy already emits 43-47% lateral actions -- what
    # hid them from the video was the cars being frozen, not this weight.)
    "lane_change_reward": 0.0,
    "on_road_reward": 1.0,
    "arrived_reward": 1.0,
    # Reward for forward progress (distance advanced along the heading each
    # step). This is the key missing ingredient in the compound scene -- without
    # it the agent's only "safe" move is to sit still (reward 0) rather than
    # risk the -5 collision penalty by driving into the junction. V3 nudges it
    # 0.05 -> 0.06: V2 showed "parking" did NOT actually avoid crashes, so we
    # pay a little more for safe CONTINUED movement, not just for avoiding danger.
    "progress_reward": 0.06,
    # Explicit penalty for sitting still -- the reward-side cure for "the car
    # waits forever / freezes". V3 raises it -0.15 -> -0.30 because the V2 data
    # (91%/82% stall) proved stalling is NOW the dominant failure mode: 60 steps
    # of stall then costs ~-18, on par with one -18 collision, so "park = free
    # safety" no longer works. NOTE: this also fires when the safety shield
    # forces a stop (the shield writes speed_index=0), so a value much higher
    # than -0.30 would punish correct yielding -- this is the deliberate ceiling.
    "stall_reward": -0.30,
}

# Speed [m/s] below which an agent counts as "stalled / waiting".
STALL_SPEED_THRESHOLD = 0.5
# Watchdog: an agent stalled for this many CONSECUTIVE steps is rebooted onto a
# fresh start lane instead of blocking a junction for the rest of the episode.
# Raised 30 -> 60 because the (deliberately) more conservative safety shield
# brakes earlier and the agent now legitimately WAITS longer at a busy ring
# entry; 30 s of patient yielding used to be enough to trigger a reboot (the
# teleport-back "the car disappears" symptom). Long enough that real yielding is
# never punished, short enough to still clear a genuine deadlock.
STALL_REBOOT_STEPS = 60

# --- 4. Multi-agent compound scene ------------------------------------------
# The 'compound' map is both a cross intersection (at the origin) AND a
# roundabout (to the east), bridged into one closed loop. Two RL-controlled
# vehicles are trained by ONE shared policy. `joint` action space = 5x5.
N_AGENTS = 2
SHARED_OBS_DIM = 15 * 5              # per-agent Kinematics (vehicles_count=15, 5 features)
JOINT_OBS_DIM = N_AGENTS * SHARED_OBS_DIM  # 150 (two agents concatenated)
JOINT_ACTION_DIM = 5 * 5             # 25 (two 5-action agents combined)

# How long one episode runs before truncation ([s]; 1 simulated second per step).
COMPOUND_DURATION = 300.0

# --- Traffic density ------------------------------------------------------
# The compound map used to hold only 2-9 obstacle cars, which is why no
# lane-change / yielding behaviour was ever visible. Traffic is now maintained
# as a *population*: every step the env tops the number of obstacle cars up
# towards COMPOUND_TARGET_VEHICLES, hard-capped by COMPOUND_MAX_VEHICLES.
#
# V3-MD (medium density): the old 14/14/18 + spawn 0.8 + gap 12 m packed the
# map so tightly that the BACKGROUND traffic itself collided, jammed and had to
# be despawned -- i.e. the environment the agent was supposed to learn from was
# not even self-consistent (visible in the video as obstacle cars crashing into
# each other and vanishing). Per the review, the environment must be stable
# BEFORE the policy is judged, so the density is dialled back to
# 8 / 10 / 12 with a larger spawn gap and a lower cross-traffic fraction.
# The goal is NOT an empty map: it is "continuous traffic with real
# interaction, but no self-inflicted background crashes".
COMPOUND_INITIAL_VEHICLES = 8      # obstacle cars present right after reset
COMPOUND_TARGET_VEHICLES = 10      # keep at least this many on the map
COMPOUND_MAX_VEHICLES = 12         # never exceed this (keeps it drivable)
# Legacy per-step spawn gate (only limits how fast the population refills).
COMPOUND_SPAWN_PROB = 0.5
# Minimum gap [m] between a newly spawned car and any existing car. Lower it to
# pack traffic tighter; raise it if you see cars spawning on top of each other.
# 12 m was small enough that a car could appear right in front of a moving one.
COMPOUND_SPAWN_MIN_GAP = 18.0
# Extra spawn-safety gate: minimum longitudinal clearance [m] to the nearest
# vehicle AHEAD on the SAME entry lane. A plain Euclidean radius check says
# nothing about whether the car we are about to drop in has room to brake --
# this is the check that actually stops "spawned into the back of a queue".
COMPOUND_SPAWN_LEADER_GAP = 25.0
# Forward-simulate the candidate against the traffic already on the map
# (CompoundRoad's curve-aware conflict prediction) and skip the spawn when it
# would immediately be in conflict. "No safe slot -> do not spawn" is the rule;
# we never squeeze a car in anyway.
COMPOUND_SPAWN_CONFLICT_CHECK = True
# Every obstacle car is routed across the map (intersection <-> roundabout) so
# that BOTH junctions are permanently busy.
COMPOUND_CROSS_TRAFFIC_PROB = 0.6

# --- Background merge-yield (environment fix #1: the 18/28 merge crashes) ---
# IDM only brakes for a leader straight ahead on the SAME lane. Two background
# cars converging into the same downstream lane (parallel entry lanes, ramp
# into the ring, two arcs feeding one segment) do not see each other at all --
# that is what the Step-2/3 diagnosis proved: 18 of 28 crashes were merges and
# almost every crasher had pre_leader_gap=None. The fix is NOT to fake a
# leader; it is an explicit merge-yield arbitration:
#   1. find background pairs converging on the same downstream capacity;
#   2. compare their ETAs to the merge point;
#   3. if the ETAs are within MERGE_ETA_WINDOW, the LOWER-priority car yields
#      (its IDM target_speed is capped so it arrives clearly later);
#   4. the yielding side is chosen DETERMINISTICALLY (lane priority, then
#      remaining distance, then ETA, then vehicle id) so a pair can never end
#      up with both sides yielding -> no merge deadlock.
# These are INITIAL experimental values, not tuned truth.
MERGE_YIELD_ENABLED = True
MERGE_LOOKAHEAD = 30.0        # [m] physical distance to the merge point to care
MERGE_ETA_WINDOW = 2.0        # [s] |ETA_A - ETA_B| below this = conflict
MERGE_YIELD_MIN_SPEED = 1.5   # [m/s] floor for the yielded car (never fully stop)
MERGE_MAX_ANGLE = 40.0        # [deg] > this is a CROSSING conflict, not a merge
MERGE_MAX_ANGLE_RING = 75.0   # [deg] same gate for roundabout-area lanes: the
                              # entry ramps meet the ring at ~55-60 deg, which
                              # the diagnosis showed was still a merge (a 40 deg
                              # gate left 8/12 post-fix merges unfixed). Lanes
                              # touching the intersection (ir*/il*/o*) keep the
                              # strict 40 deg gate so crossing stays untouched.
MERGE_CATCHUP_WINDOW = 4.0    # [s] feeder-vs-downstream closing-time gate
MERGE_MOUTH_DIST = 8.0        # [m] "merge mouth": a car this close to the node
MERGE_MOUTH_SPEED = 2.5       # [m/s] ...and this slow OWNS the mouth -- the
                              # other car must follow its pace instead of
                              # racing past on an ETA win (its IDM cannot see
                              # across lanes, so an earlier ETA does not mean
                              # the mouth is clear).

# --- Crossing arbitration (Phase 2: replaces pairwise stop-and-go) ----------
# The stock RegulatedRoad regulation is PAIRWISE ("each conflicting pair gets
# one yielder, forced to target_speed=0, released later to FULL speed limit").
# Diagnosed failure modes (see _diagnose_crossing.py):
#   * 3+ car conflict groups resolve into yield CYCLES (A yields B yields C
#     yields A) -> everyone frozen;
#   * the constant-speed predictor is blind to STOPPED face-offs -> no
#     arbitration at all -> IDM leader-cycle gridlock;
#   * stop-and-go with full-speed release re-injects cars into conflicts.
# Phase 2 fix, all deterministic:
#   2A. conflict GROUPS (union-find over conflict edges) -> ONE winner per
#       group (lane priority -> zone occupancy -> distance -> serial);
#   2B. static mouth conflicts: a geometric lane-crossing table so stopped
#       face-offs are arbitrated even though the predictor sees nothing;
#   2C. yielders get a progressive speed cap (never target_speed=0 forever),
#       and caps never accumulate -- base is restored every env step.
CROSS_GROUP_ARBITRATION = True
# ① Hierarchical yielding (the GROUP_INTERNAL fix). When True, every conflict
# group is sorted into a DETERMINISTIC TOTAL RANKING (occupancy -> lane
# priority -> distance-to-centroid -> stable serial) and a vehicle of rank k
# yields to EVERY higher-ranked group member it ACTUALLY conflicts with, with
# its final cap = the MINIMUM of those per-pair caps. This closes the
# GROUP_INTERNAL gap (two yielders no longer only watch the single winner).
# When False the OLD single-winner behaviour is restored (one winner per group,
# every other member caps against only that winner) so the change can be A/B'd
# as a single variable. Default True = the fix is live.
CROSS_HIERARCHICAL_YIELD = True
# Phase 2B: roundabout-mouth ownership (the remaining 84% of GROUP_INTERNAL
# lives at the ring entries). A circulating ring vehicle (lane priority ==
# PRIORITY_RING = 5) has ABSOLUTE right of way over an entering vehicle
# (priority == PRIORITY_RB_ENTRY = 0) at the mouth. The generic conflict-group
# ranking is OVERRIDDEN for that specific pair so a circulating car can never be
# made to yield to an entering car -- the old occupancy-first rank let a close
# entry car outrank a far circulating car, both ended up yielding, and the mouth
# froze (the DEADLOCK+GROUP_INTERNAL hotspot ses->se|sx->se / nes->ne|nx->ne).
# Scope is strictly ring(prio 5) vs entry(prio 0) pairs that actually conflict;
# same-lane, ring-ring and entry-entry keep the normal hierarchy.
# PHASE 2B VERDICT (5x1000 A/B, 2026-09-07): FAILED. Ownership ON doubled the
# ring GROUP_INTERNAL count (18 -> 37) and raised total BB 65 -> 73: the
# override only fires AFTER a conflict is detected, but ring-mouth merge pairs
# are detected LATE (approach->internal lane transition blind spot), so forced
# GO + forced yield meet too fast in live traffic. Deterministic 2/3-car
# regressions pass but the stress verdict is negative. Default False; the
# mechanism stays available for Phase 2C (fix detection first, then re-trial).
CROSS_RING_MOUTH_OWNERSHIP = False
# Lane priorities take part in the hierarchical ranking. PHASE 2B ABLATION
# (5x1000, ownership OFF): enabling this cost +15 BB (52 -> 67) because the
# intersection internal lanes (ir*->il*) carry MEANINGLESS priority values
# (e.g. ir2->il3=0.0 vs the crossing ir3->il1=3.0) that had never been read by
# any arbitration before. Keep False until those priorities are re-authored;
# ring/entry priorities (5/0) are unaffected -- CROSS_RING_MOUTH_OWNERSHIP and
# the blocked() guards read the real lane priority regardless of this flag.
CROSS_LANE_PRIORITY_RANKING = False
# Phase 2C.1: route-aware trajectory prediction. The old predictor extrapolated
# along the vehicle's CURRENT lane only, so a fast approach car (o*->ir*, 8 m/s)
# turning onto an internal lane 1 s later was invisible -- 5 of the 7 residual
# PREDICTOR_MISS crashes (and the ring-mouth late detections that killed the
# ownership trial) are exactly this current->next lane blind spot. When True,
# is_conflict_possible() walks the vehicle's ALIGNED route chain (current lane
# -> route next -> next-next) and consumes the prediction distance across lane
# boundaries. Stale/unresolvable routes fall back to the old current-lane
# prediction per vehicle (never guess a next lane). Single-variable A/B only:
# everything else (table, ETA window, lookahead, yield caps, priorities,
# ownership, GRIDLOCK_BREAK_STEPS) stays unchanged. The validated route-aware
# profile is now the default for ordinary training and evaluation as well.
# Match the validated MD27_ROUTE_ON_V1 profile for ordinary train/evaluate runs.
CROSS_ROUTE_AWARE_PREDICTION = True
# Experimental joint traffic arbitration; enable only in explicit A/B runs.
CONTROLLED_ARBITRATION = False
# Validated production safety layer: results/validation/long_test_controlled_reservation_5x1000.json.
CONFLICT_RESERVATIONS = True
CROSS_CAP_FAR = 6.0           # [m/s] yielder cap when > 12 m from the group
CROSS_CAP_NEAR = 3.0          # [m/s] yielder cap when 5-12 m away
CROSS_ZONE_DIST = 7.0         # [m] inside this radius of the yield point: hold
                              #     (cap 0). Must exceed half a car length or
                              #     the stopped nose still pokes past the
                              #     crossing point (measured touch-crash)
CROSS_OCCUPANCY_DIST = 6.0    # [m] closer than this to the group centroid =
                              #     "occupies the zone" (beats ETA in ranking)
CROSS_STATIC_SPEED = 4.0      # [m/s] static mouth conflicts apply below this.
                              #     3.0 was too tight: a 3.1 m/s turning car
                              #     and a 0.2 m/s creeper met exactly in the
                              #     gap between the ETA test (creeper's ETA is
                              #     huge) and the static gate (measured)
CROSS_STATIC_AHEAD = 18.0     # [m] crossing point must be within this ahead
CROSS_STATIC_BEHIND = 6.0     # [m] ...or this far past it (still in the zone)
CROSS_ETA_WINDOW = 1.5        # [s] crossing-pair conflict if |ETA_A - ETA_B|
                              #     to the shared crossing point is below this
                              #     -- the constant-speed box test only sees a
                              #     fast crossing pair ~0.7 s before impact
                              #     (measured), which leaves no braking room
CROSS_GO_SUPPRESS_DIST = 8.0  # [m] the group winner may ignore a YIELDING car
                              #     as IDM leader only beyond this distance --
                              #     closer than that it is a physical block
                              #     (proven: pushing into <8 m hits it, BB>0).

# Phase 2C.2: winner collision-aware speed cap. After route-aware prediction the
# dominant residual failure is CONTROL_FAIL: the yielder has correctly slowed /
# stopped, but the higher-ranked WINNER still barges through at full speed
# because it is left completely unrestricted (the pre-2C.2 code gave the winner
# GO and NO cap). This flag keeps the winner's priority (it still GOes -- never
# yields) but applies a PROGRESSIVE speed cap -- STRICTLY ABOVE the yielder's
# cap, preserving the invariant winner_cap > yielder_cap -- when a
# path-intersecting group member currently occupies / is about to occupy their
# shared conflict point. The cap releases the instant the point clears, so an
# empty or already-cleared crossing leaves the winner at base speed (no
# throughput loss). It keys off OCCUPANCY of the conflict point (using the
# member's predicted position, not its commanded target_speed), so a yielder
# that has already stopped ON the point still reads as occupied. Single-variable
# A/B only: everything else (table, route prediction, hierarchy, yield caps,
# priorities, ownership, breaker) stays frozen. Default False = baseline.
CROSS_WINNER_COLLISION_CAP = False
CROSS_WINNER_CAP_DIST = 15.0   # [m] winner beyond this from occupied point: no cap
CROSS_WINNER_MARGIN = 2.0      # [m] start capping this much before the point
CROSS_WINNER_CAP_FAR = 7.0     # [m/s] winner cap 8-15 m out (> yielder CROSS_CAP_FAR 6)
CROSS_WINNER_CAP_MID = 4.0     # [m/s] winner cap 4-8 m out (> yielder CROSS_CAP_NEAR 3)
CROSS_WINNER_CAP_NEAR = 2.0    # [m/s] winner cap <4 m & still occupied (> yielder 0)
CROSS_WINNER_OCC_RADIUS = 5.0  # [m] a conflicting car within this of the point = occupied
CROSS_WINNER_OCC_TIME = 5.0     # [s] prediction horizon for occupancy

# Phase 2C.3: register exactly THREE ring-mouth MERGE pairs (the blind spots
# proven by 2C.1b to be un-registered / weakly-covered). These enter the SAME
# conflict-group arbitration as crossings but are treated as MERGE / CONVERGENCE
# (two lanes feeding the same downstream ring node -- a "form a unique passage
# order" problem), NOT as a genuine X-crossing. The active mechanism is identical
# to the crossing machinery (one winner GO, the other yields with a progressive
# cap); what changes is only that the edge is GUARANTEED to exist for these three
# pairs (the generic predictor/ETA window detects them unreliably at the
# approach->internal-lane transition, which is why they crash ~10x in baseline).
# NO generalisation: only these three pairs are registered, not "all shared
# endpoints" or "all converging lanes". Winner cap (2C.2) stays OFF; this is a
# single-variable addition on top of the fixed route-aware baseline.
CROSS_RING_MERGE_PAIRS = False
CROSS_RING_MERGE_LOOKAHEAD = 12.0   # [m] both cars must be within this of the
                                     #     shared downstream node for the merge
                                     #     edge to fire (proximity-gated so a far
                                     #     / cleared car is never limited early)

# --- Phase 4C-B1: pairwise actual-conflict-point intersection ranking --------
# Single-variable replacement for the OLD intersection rank key
# (occupancy-on-centroid -> dormant lane-priority -> distance-to-centroid ->
# serial), which handed GROUP_GO to a car that was NOT the one actually in / about
# to enter the conflict box -- it merely won the distance-to-centroid or
# stable-serial tiebreak. Phase 4C-A proved 5 WRONG_ROW crashes are exactly this,
# and 0/5 were decided by lane priority. The new key ranks each conflict-group
# member by its ACTUAL pairwise conflict geometry, in the user-specified order:
#   1. occupancy / clearance of the real conflict point (who is in the box),
#   2. ETA / distance to that conflict point (who arrives first),
#   3. movement semantics (through > turn) -- ONLY a tie-break,
#   4. stable serial -- final deterministic tie-break.
# "through > turn" is deliberately NOT placed ahead of occupancy: the 4C-A data
# showed 0/5 cases were lane-priority-decided, so the geometry-first ordering is
# what the regression requires. Scope is strictly INTERSECTION crossing groups
# that contain NO ring-mouth / registered-merge pair (those keep their own
# mechanisms and the old rank, so ring/merge behaviour is never touched).
# Route-aware crossing detection is used because the frozen predictor misses the
# `o3->ir3` continuation crossing that causes 2 of the 5 WRONG_ROW cases.
# Default False = frozen baseline (old centroid/serial rank).
CROSS_PAIRWISE_CONFLICT_RANK = False
CROSS_PAIRWISE_OCC_DIST = 4.0   # [m] within this of the actual conflict point =
                                #     "occupies / clearing the box" (rank class 0)

# --- Phase 4D-B: Ring conflict reservation / clearance persistence (CLOSED 2026-09-09, NO-GO) ----------
# The SINGLE new gated variable for Phase 4D-B (approved 2026-09-09). It is NOT
# a new arbitration controller. The existing baseline arbitration produces the
# FIRST valid GO/YIELD for a genuine ring-mouth feeder<->circulator conflict
# (see compound_road._is_ring_mouth_pair); when RING_RESERVATION_PERSISTENCE is
# True, that assignment is LATCHED as a pair-specific reservation and its
# ownership identity is FROZEN until the owner has PHYSICALLY cleared the mouth
# (whole body past the conflict point + a small margin). Release is geometry /
# clearance-based ONLY -- never predictor-edge-disappearance, ETA threshold,
# fixed time, or regulation-cycle count. A stale fail-safe (owner crash /
# despawn / reroute) cleans up the reservation without freezing the yielder.
# Default False = frozen baseline (no reservation; every regulation cycle the
# GO/YIELD is recomputed from scratch, which is the 4D-A lifecycle defect).
# *** CLOSED AS NO-GO 2026-09-09: ring-mouth BB 12->12 unchanged, gridlock 0->11,
#     stale_cleanup 0->14. PERMANENT frozen default. 4D-B series closed: do NOT
#     implement 4D-B.1, do NOT tune clearance margin / hold time / stale budget /
#     REGULATION_FREQUENCY. Keep impl + 29/29 regression + instrumentation as
#     dormant ablation only. ***
RING_RESERVATION_PERSISTENCE = False

# --- Phase 3B: narrow route-aware LEADER (longitudinal, not crossing) --------
# Phase 3A proved the rear car is COMPLETELY BLIND (leader=None for the whole
# 20-step window) in 5/5 of the `wx->we|wx->we` rear-end crashes and in 12/27
# crashes overall, because the stock search matches from+to+LANE ID and so never
# sees a car on a successor / converging / differently-numbered lane.
# Phase 3C then proved the fix must be NARROW: enabling the stock
# `neighbour_vehicles_connected_lanes` search fixed the blindness
# (blind rear 12/27 -> 1/46, rear-end 4 -> 1) but ~51% of the new leaders came
# from non-route connected lanes -> phantom braking -> BB 27 -> 46, speed
# 7.84 -> 6.70, stall 1.2% -> 6.6%, merge 10 -> 38. That is NOT shipped.
# So this fallback fires ONLY when the native exact-lane search found nothing,
# and admits exactly two kinds of candidate:
#   1. the ego's ACTUAL route successor (route aligned to the live lane first;
#      a stale/unalignable route is never guessed);
#   2. a STRICT converging car: another lane feeding the SAME downstream node,
#      heading within ROUTE_LEADER_HEADING_MAX, closer to that node than the ego,
#      and inside a short merge-distance gate.
# Explicitly excluded: arbitrary parallel lanes of the same link, crossing
# traffic, connected-but-not-planned lanes, and anything beyond the gates.
# The native leader ALWAYS wins -- this never overrides an existing leader, so it
# fills a blind spot rather than rewriting IDM.
ROUTE_AWARE_LEADER = True
ROUTE_LEADER_MERGE_DIST = 20.0    # [m] BOTH cars must be within this of the
                                  #     shared convergence node (candidate 2)
ROUTE_LEADER_HEADING_MAX = 30.0   # [deg] converging candidate must head within
                                  #      this of the ego (else it is crossing)
ROUTE_LEADER_GAP_MAX = 25.0       # [m] never adopt a candidate farther than this
                                  #     along the route (anti-queue/phantom gate)
# NOTE: these three numbers are REGRESSION STARTING POINTS chosen so the
# deterministic positives pass -- not tuned truth. They are judged by the
# 5x1000 A/B (BB, speed, stall, leader-acquisition rate), not by intuition.

# Phase 2C.2a: scope the winner cap to GENUINE CROSSINGS only. The lane-crossing
# table (_lane_crossing_table) is purely geometric and ALSO records MERGE /
# convergence intersections (ring-mouth mouths), so the raw 2C.2 winner cap
# wrongly slowed the GO winner at a merge (the yielder is already capped by
# _yield_cap there) -> the measured crash migration (ring 10->18, merge 10->22).
# This is the single last scope correction before a Go/No-Go A/B. No reliable
# runtime ctype exists, so a merge is classified by the geometric fallback per
# the 2C.2a spec: BOTH lanes feed the SAME downstream node (downstream
# converges) AND the intersection point sits within this fraction of EITHER
# lane's END (a mouth merge converges at the lane end; a real crossing is
# mid-block). The endpoint margin is a COMPONENT of the detector, not the sole
# condition: a genuine crossing that merely lies near a lane end is never
# mis-excluded because its downstream nodes differ (same_dest is False).
CROSS_WINNER_MERGE_MARGIN = 0.30  # [0..1] last-N% of a lane counts as "at the mouth"

# --- Phase 4B: Dedicated Ring-Merge Controller (topology right-of-way) -------
# Phase 4A proved 25 ring-mouth BB crashes = 22 mode-B (merge-yield yields the
# CIRCULATING car) + 3 mode-D (crossing/group controller yields the circulating
# car); A/C/E = 0. Root cause: both mouth lanes carry graph priority 1.0 (NOT 5),
# so the priority branch in _merge_yielder is DEAD and the distance tie-break
# yields the farther (circulating) car. The fix is NOT to edit priorities -- the
# 2C.3 / 2B failures already showed prio wiring is unreliable. Phase 4A proved
# circulating/feeder are 100% determinable from TOPOLOGY (lane from-node in
# ring_nodes => circulating ring arc; external slip => feeder). This controller
# OWNS the four ring mouths and enforces ONE invariant:
#     CIRCULATING = GO        (never merge-yielded, never group-yielded)
#     FEEDER      = YIELD      (cap metered to the convergence node, not centroid)
#     same mouth pair NEVER both yield (no double-controller)
# It reuses the SAME proximity as _merge_pairs Case-3 (distance <=
# MERGE_LOOKAHEAD) -- lookahead is NOT expanded -- but classifies right-of-way by
# topology. The mouth pair is ISOLATED from generic crossing/group arbitration so
# the group can neither yield the circulating car (mode D) nor double-cap the
# feeder. Isolation is PER-PAIR (this specific feeder+circulating at this mouth),
# never a global immunity for the circulating car against other real hazards.
# Single gated variable, default OFF until the A/B passes. Final target BB=0.
DEDICATED_RING_MERGE = False
RING_MERGE_FAR = 12.0        # [m] feeder cap active only within this (no early stop)
RING_MERGE_NEAR = 6.0        # [m] within this of the mouth a feeder may hold (cap 0)
RING_MERGE_SAFETY = 2.0      # [s] feeder metered to arrive this long after C clears
RING_MERGE_TRANSIT = 2.0     # [s] feeder that clears M this before C needs no cap
RING_MERGE_MIN_SPEED = 1.5   # [m/s] creep floor between FAR and NEAR (never stuck)

# --- Gridlock breaker (last resort, honest accounting) ----------------------
# Hard geometric leader-cycles (cars each holding behind the next INSIDE the
# junction) cannot be resolved by any right-of-way rule -- proven by the
# regression: the winner pushes and collides (BB=1 at step 0). Once a leader
# cycle has been stalled this long, ONE cycle member is recycled back to a
# fresh entry lane and the removal is COUNTED (despawned_gridlock), never
# silently merged into normal despawns.
GRIDLOCK_BREAK_STEPS = 15     # consecutive stalled steps before breaking

# --- Video rendering ------------------------------------------------------
# One env step == one simulated second, but the env internally integrates
# `simulation_frequency` (15) sub-steps per step. The OLD recorder repeated a
# single rendered frame VIDEO_FRAME_REPEAT times, so 1 simulated second only
# produced ONE distinct picture -> the video looked like a slide show ("一卡一卡").
#
# The new recorder instead raises `policy_frequency` to VIDEO_SUBSTEPS so each
# simulated second is rendered as VIDEO_SUBSTEPS distinct frames (every internal
# sub-step is drawn). At VIDEO_FPS=60 that is 60/15 = 4x real-time playback with
# 15 distinct frames per simulated second -- smooth, not choppy.
VIDEO_FPS = 60
VIDEO_FRAME_REPEAT = 10          # legacy per-frame repeat (unused by new recorder)
VIDEO_SUBSTEPS = 15             # distinct frames rendered per simulated second
# Fixed, map-wide top-down camera instead of following the ego vehicle.
PANORAMIC_VIEW = True

# Map a short scene name to the registered gym id.
ENV_IDS = {
    "roundabout": "MyRoundabout-v0",
    "intersection": "MyIntersection-v0",
    "compound": "MyCompound-v0",
}
