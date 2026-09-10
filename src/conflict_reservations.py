"""Exclusive occupancy reservations for the compound road's conflict areas.

This controller constrains acceleration; it never relocates vehicles, changes
collision detection, or removes traffic. Route geometry determines stop lines.
"""
from __future__ import annotations

import numpy as np


class ConflictReservations:
    LOOKAHEAD = 55.0
    BRAKING = 3.0
    CLEARANCE = 3.0
    STOP_MARGIN = 2.0
    ZONE_SPEED = 8.0

    def __init__(self, road):
        self.road = road
        # Radius is expanded by half a vehicle diagonal before testing occupancy.
        self.zones = [
            ('intersection', np.array([0., 0.]), 14., None),
            ('south', np.array([222., 22.]), 10., None),
            ('east', np.array([244., 0.]), 10., None),
            ('north', np.array([222., -22.]), 10., None),
            ('west', np.array([200., 0.]), 10., None),
            ('bridge_w', np.array([72., -4.]), 8., {('wxr', 'o3'), ('o3', 'ir3')}),
            ('bridge_e', np.array([137., 4.]), 8., {('o3', 'wer'), ('wer', 'wes')}),
        ]
        self.owners = {}
        self.entered = set()
        self.waiting_since = {}
        self.intervals = {}
        self.ticks = 0
        self.stats = dict(grants=0, releases=0, conflicting_occupancy_ticks=0)

    def _lane_interval(self, lane_index, zone_index):
        key = (tuple(lane_index), zone_index)
        if key not in self.intervals:
            _, center, radius, links = self.zones[zone_index]
            if links is not None and tuple(lane_index[:2]) not in links:
                self.intervals[key] = None
            else:
                lane = self.road.network.get_lane(lane_index)
                ss = np.linspace(0, lane.length, max(2, int(np.ceil(lane.length * 2)) + 1))
                inside = [s for s in ss if np.linalg.norm(lane.position(s, 0) - center) <= radius + self.CLEARANCE]
                self.intervals[key] = (max(0., min(inside) - .5), max(inside) + .5) if inside else None
        return self.intervals[key]

    def _distances(self, vehicle):
        chain = self.road._route_lane_chain(vehicle, max_lanes=8) or [tuple(vehicle.lane_index)]
        result = {}
        offset = -vehicle.lane.local_coordinates(vehicle.position)[0]
        previous = None
        for li in chain:
            if li[2] is None:
                if previous is None:
                    li = (*li[:2], vehicle.lane_index[2])
                else:
                    prev_lane = self.road.network.get_lane(previous)
                    k, _ = self.road.network.next_lane_given_next_road(
                        *previous, li[1], None, prev_lane.position(prev_lane.length, 0))
                    li = (*li[:2], k)
            lane = self.road.network.get_lane(li)
            for z in range(len(self.zones)):
                interval = self._lane_interval(li, z)
                if interval and offset + interval[1] >= 0 and z not in result:
                    distance = offset + interval[0]
                    if distance <= self.LOOKAHEAD:
                        result[z] = max(0., distance)
            offset += lane.length
            previous = li
            if offset > self.LOOKAHEAD:
                break
        return result

    def update(self):
        self.ticks += 1
        vehicles = [v for v in self.road.vehicles if not v.crashed]
        distances = {v: self._distances(v) for v in vehicles}
        for v in vehicles:
            v._reservation_stop_distance = float('inf')
            v._reservation_speed_cap = float('inf')
        for z, (_, center, radius, links) in enumerate(self.zones):
            inside = [v for v in vehicles
                      if (links is None or tuple(v.lane_index[:2]) in links)
                      and np.linalg.norm(v.position - center) <= radius + self.CLEARANCE]
            for v in inside:
                distances[v].setdefault(z, 0.0)
            owner = self.owners.get(z)
            if owner in inside:
                self.entered.add(z)
            if owner is not None and (owner not in vehicles or
                    (z in self.entered and owner not in inside) or
                    (owner not in inside and z not in distances.get(owner, {}))):
                self.owners.pop(z, None)
                self.entered.discard(z)
                self.stats['releases'] += 1
                owner = None
            candidates = [v for v in vehicles if z in distances[v]]
            def blocked(v):
                chain = self.road._route_lane_chain(v, max_lanes=3) or [v.lane_index]
                roads_ahead = {tuple(li[:2]) for li in chain}
                return any(u is not v and tuple(u.lane_index[:2]) in roads_ahead
                           and distances[u][z] + 1.0 < distances[v][z]
                           for u in candidates)
            # A queue head may be stopped by this very reservation. Never give
            # its follower permission and ask the head to wait for that follower.
            if owner is not None and z not in self.entered and blocked(owner):
                self.owners.pop(z, None)
                owner = None
            for v in candidates:
                self.waiting_since.setdefault((z, v), self.ticks)
            if owner is None:
                # Occupied space wins. Otherwise use lane right-of-way, then
                # waiting time and ETA; never reshuffle an issued reservation.
                ready = inside or [v for v in candidates if distances[v][z] < 25 and not blocked(v)]
                if ready:
                    def order(v):
                        since = self.waiting_since[(z, v)]
                        aged = self.ticks - since >= 150
                        return (0 if aged else 1,
                                since if aged else -float(v.lane.priority),
                                -float(v.lane.priority) if aged else since,
                                distances[v].get(z, 0) / max(v.speed, .5))
                    owner = min(ready, key=order)
                    self.owners[z] = owner
                    self.stats['grants'] += 1
                    if owner in inside:
                        self.entered.add(z)
            if len(inside) > 1:
                self.stats['conflicting_occupancy_ticks'] += 1
            for v in candidates:
                distance = distances[v][z]
                # Slow before curved/conflicting geometry, including owners.
                v._reservation_speed_cap = min(v._reservation_speed_cap,
                    np.sqrt(self.ZONE_SPEED ** 2 + 2 * self.BRAKING * distance))
                if v is not owner:
                    # Euclidean separation is a conservative lower bound on
                    # travel to the boundary, even during an off-centre turn.
                    geometric = max(0., np.linalg.norm(v.position - center) - radius - self.CLEARANCE)
                    remaining = max(0., min(distance, geometric) - self.STOP_MARGIN)
                    v._reservation_stop_distance = min(v._reservation_stop_distance, remaining)
                    v._reservation_speed_cap = min(v._reservation_speed_cap,
                        np.sqrt(2 * self.BRAKING * remaining))
        self.waiting_since = {key: value for key, value in self.waiting_since.items()
                              if key[1] in distances and key[0] in distances[key[1]]}

    def constrain_acceleration(self, dt):
        for v in self.road.vehicles:
            distance = getattr(v, '_reservation_stop_distance', float('inf'))
            if distance <= .05:
                # At a red reservation boundary IDM must not restart between
                # braking ticks; otherwise alternating 0/0.2 m/s creeps across it.
                v.action['acceleration'] = min(v.action['acceleration'], -min(max(v.speed, 0) / dt, 6.0))
            elif np.isfinite(distance) and v.speed > 0:
                # Compensate explicit-Euler travel before speed integration.
                remaining = max(distance - v.speed * dt, .05)
                required = v.speed ** 2 / (2 * remaining)
                if required >= self.BRAKING or distance < 1.0:
                    v.action['acceleration'] = min(v.action['acceleration'], -min(required, 6.0))
