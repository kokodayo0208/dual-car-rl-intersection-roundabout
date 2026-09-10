"""Phase 4E-B -- Canonical Collision Accounting (instrumentation only).

WHY THIS EXISTS
---------------
The frozen baseline reports ``BB = 27``, but Phase 4E-A proved 4 of the 27 raw
collision records are instrumentation artifacts, and two steps (seed 1 t65,
seed 2 t328) produced TWO ``unknown-single`` BB records each -- the signature
of ONE physical crash whose partners were pushed >7 m apart by the impact
inside the same policy step (15 physics sub-steps), so the 7 m fallback could
not re-pair them.

This module defines the canonical ground truth:

    raw_collision_records      what ``_record_collisions`` appended (kept 1:1,
                               never deleted -- debug only)
    unique_physical_bb_events  the canonical BB count used by ALL formal
                               metrics from now on

EVENT IDENTITY (per the 4E-B directive)
---------------------------------------
    event key = (simulation_step, unordered pair of persistent serials)

* ``_diag_serial`` is assigned once per vehicle OBJECT and never reused, so
  Python ``id()`` recycling cannot pollute identity.
* A-B and B-A are the same event (frozenset).
* Same pair at the same step is one event, no matter how many raw records.
* Different pairs at the same step stay DIFFERENT events (a pile-up is N pair
  events) -- there is deliberately NO step-level deduplication.

PRE-IMPACT PAIR RECONSTRUCTION
------------------------------
When an impact happens early inside a policy step, the impact impulse can
separate the two (now crashed) partners by more than the recorder's 7 m
fallback radius by the time the log is written. Each partner is then filed as
an ``unresolved_single``. The accounting layer re-pairs two same-step single
records ONLY when their PRE-PHYSICS positions (``pre_pos``, captured by
``_snapshot_pre_crash`` before any integration) are within contact distance

    |preA - preB| <= (lenA + lenB) / 2 + PREIMPACT_MARGIN

with ``PREIMPACT_MARGIN = 1.0 m``. This can never pair two cars that were not
overlapping-or-contacting before the step, so nearby-but-innocent vehicles can
never be mispaired. A single record that cannot be re-paired keeps
``pairing_source = "unresolved_single"``.

Every event records ``merged_raw`` (how many raw records it absorbed) so the
raw log remains fully auditable.

This module touches ACCOUNTING ONLY. It never reads or writes simulation
state and cannot change behaviour.
"""

from __future__ import annotations

import itertools

PREIMPACT_MARGIN = 1.0   # [m] slack beyond half-length sum for pre-impact pairing

_SERIAL_COUNTER = itertools.count(1)


def ensure_serial(v) -> int:
    """Return the vehicle's persistent serial, assigning one on first use.

    The serial lives on the object (``_diag_serial``), so it survives as long
    as the object does and is NEVER reused for a different vehicle -- unlike
    ``id()``.
    """
    s = getattr(v, "_diag_serial", None)
    if s is None:
        s = next(_SERIAL_COUNTER)
        v._diag_serial = s
    return s


def _contact_dist(rec) -> float:
    lens = rec.get("lengths") or (5.0, 5.0)
    a = float(lens[0]) if len(lens) > 0 else 5.0
    b = float(lens[1]) if len(lens) > 1 else 5.0
    return (a + b) / 2.0 + PREIMPACT_MARGIN


def _pre_dist(r1, r2) -> float | None:
    """Min distance between ANY pre-physics point of r1 and ANY of r2.

    A single record carries one pre-position (its own); a pair record carries
    two (both members). Comparing all members is what lets a tertiary single
    be absorbed into a pair event."""
    pts1 = [p for p in (r1.get("pre_pos") or []) if p and len(p) >= 2]
    pts2 = [p for p in (r2.get("pre_pos") or []) if p and len(p) >= 2]
    if not pts1 or not pts2:
        return None
    best = None
    for p1 in pts1:
        for p2 in pts2:
            d = float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
            best = d if best is None else min(best, d)
    return best


def canonical_events(raw_records):
    """Fold raw collision records into unique physical BB events.

    Rules (deterministic, order-independent for the merge result):
      1. Two-vehicle records: key = (step, frozenset(serials)). Duplicates of
         the same pair at the same step merge into one event.
      2. Single-vehicle records ("singles"): try to merge with another single
         of the SAME step whose pre-physics positions were within contact
         distance -> one two-vehicle event, pairing_source="preimpact_pair".
         A single may also be absorbed into a same-step pair event when its
         pre-position contacts one of the pair members (tertiary pile-up).
      3. Everything else stays its own event.

    Returns (events, stats). Each event dict carries: step, serials, kind,
    ctypes, lanes, pos, merged_raw, pairing_source, record_indices, and the
    member raw records' ids for debugging.
    """
    pairs = {}          # (step, frozenset) -> event
    singles = []        # unresolved single records (index, rec)
    stats = {"raw_records": len(raw_records),
             "pair_records": 0, "single_records": 0}

    for idx, rec in enumerate(raw_records):
        if rec.get("kind") != "BB":
            # only BB is accounted into the canonical BB ground truth
            continue
        serials = rec.get("serials") or []
        if len(serials) >= 2:
            stats["pair_records"] += 1
            key = (int(rec.get("step", 0)), frozenset(int(s) for s in serials[:2]))
            ev = pairs.get(key)
            if ev is None:
                pairs[key] = _event_from(rec, idx, serials[:2])
            else:
                ev["merged_raw"] += 1
                ev["record_indices"].append(idx)
                ev["ctypes"].append(rec.get("ctype"))
                if rec.get("pairing_source"):
                    ev["pairing_sources"].append(rec["pairing_source"])
        else:
            stats["single_records"] += 1
            singles.append((idx, rec))

    events = list(pairs.values())

    # ---- pre-impact reconstruction for singles --------------------------- #
    used_single = set()
    for i, (idx_i, ri) in enumerate(singles):
        if i in used_single:
            continue
        merged = False
        for j in range(i + 1, len(singles)):
            if j in used_single:
                continue
            idx_j, rj = singles[j]
            if int(rj.get("step", 0)) != int(ri.get("step", 0)):
                continue
            d = _pre_dist(ri, rj)
            if d is not None and d <= max(_contact_dist(ri), _contact_dist(rj)):
                key = (int(ri.get("step", 0)),
                       frozenset((int((ri.get("serials") or [0])[0]),
                                  int((rj.get("serials") or [0])[0]))))
                ev = _event_from(ri, idx_i,
                                 [(ri.get("serials") or [None])[0],
                                  (rj.get("serials") or [None])[0]])
                ev["merged_raw"] += 1
                ev["record_indices"].append(idx_j)
                ev["ctypes"].append(rj.get("ctype"))
                ev["pairing_sources"] = [ri.get("pairing_source"),
                                         rj.get("pairing_source"),
                                         "preimpact_pair"]
                ev["pairing_source"] = "preimpact_pair"
                events.append(ev)
                used_single.update({i, j})
                merged = True
                break
        if merged:
            continue
        # absorb into an existing same-step pair event (tertiary contact)
        for ev in events:
            if ev["step"] != int(ri.get("step", 0)) or len(ev["serials"]) < 2:
                continue
            # pre-position contact with either member?
            member_rec = None
            for ridx in ev["record_indices"]:
                cand = raw_records[ridx]
                if cand is ri:
                    continue
                if len(cand.get("serials") or []) >= 2:
                    member_rec = cand
                    break
            if member_rec is None:
                continue
            d = _pre_dist(ri, member_rec)
            if d is not None and d <= max(_contact_dist(ri), _contact_dist(member_rec)):
                ev["merged_raw"] += 1
                ev["record_indices"].append(idx_i)
                ev["ctypes"].append(ri.get("ctype"))
                ev["pairing_sources"].append(ri.get("pairing_source"))
                ev["tertiary_singles"] = ev.get("tertiary_singles", 0) + 1
                used_single.add(i)
                merged = True
                break
        if not merged:
            ev = _event_from(ri, idx_i, [(ri.get("serials") or [None])[0]])
            ev["pairing_source"] = "unresolved_single"
            ev["pairing_sources"] = [ri.get("pairing_source")]
            events.append(ev)

    events.sort(key=lambda e: (e["step"], sorted(e["serials"])))
    stats["unique_bb_events"] = sum(1 for e in events)
    return events, stats


def _event_from(rec, idx, serials):
    return {
        "step": int(rec.get("step", 0)),
        "serials": [None if s is None else int(s) for s in serials],
        "kind": rec.get("kind"),
        "ctypes": [rec.get("ctype")],
        "lanes": [rec.get("lanes")],
        "pos": rec.get("pos"),
        "pre_pos": [rec.get("pre_pos")],
        "merged_raw": 1,
        "pairing_source": rec.get("pairing_source") or "unknown",
        "pairing_sources": [rec.get("pairing_source") or "unknown"],
        "record_indices": [idx],
        "raw_ids": [rec.get("ids")],
        "tertiary_singles": 0,
    }
