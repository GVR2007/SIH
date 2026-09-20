"""Flood hazard rating and depth-damage loss estimation.

Implements the "Depth/Velocity -> Derive Hazard Variables -> Hazard rating ->
Low/Moderate/Significant/Extreme -> Hazard raster + area/class ->
Depth-damage curves -> damage class -> loss estimate" branch.

Hazard rating follows the UK Defra / Environment Agency FD2320-FD2321 flood
risk-to-people method, which is the formulation most widely reused in
dam-break consequence studies:

    HR = d * (v + 0.5) + DF

  d  = depth (m), v = velocity (m/s), DF = debris factor (0, 0.5 or 1)

    HR < 0.75         Low          "caution"
    0.75 <= HR < 1.25 Moderate     "dangerous for some (children/elderly)"
    1.25 <= HR < 2.5  Significant  "dangerous for most people"
    HR >= 2.5         Extreme      "dangerous for all"

Depth-damage curve *shapes* are the JRC global flood depth-damage functions for
Asia (Huizinga, de Moel & Szewczyk, 2017, JRC Technical Report EUR 28552 EN).

IMPORTANT ON MONETARY LOSS
--------------------------
The curves give a damage *fraction*.  Converting that to currency needs asset
unit values, which are a policy/economic input, not something derivable from
the DEM or satellite data.  They are therefore explicit, overridable scenario
parameters and every monetary figure this module returns is tagged with the
unit rates that produced it.  Physical exposure counts (buildings, people,
road length, cropland area) come from real OSM/WorldPop data and carry no such
assumption -- they are the primary output; currency is derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- hazard classes --------------------------------------------------------
HAZARD_CLASSES = [
    (0.00, 0.75, "Low", "#2c7fb8", "Caution - shallow or slow-moving water"),
    (0.75, 1.25, "Moderate", "#7fcdbb", "Dangerous for some (children, elderly)"),
    (1.25, 2.50, "Significant", "#fdae61", "Dangerous for most people"),
    (2.50, 1e9, "Extreme", "#d7191c", "Dangerous for all, including emergency services"),
]

# Debris factor by land cover (Defra FD2321 Table 3.2).
# Higher where floating debris is likely to be generated.
DEBRIS_FACTOR = {
    "default": 0.5,
    "built_up": 1.0,
    "forest": 1.0,
    "open": 0.0,
}

# --- depth-damage functions -----------------------------------------------
# These are loaded from the published JRC database rather than written out
# here.  An earlier version of this file carried a hand-entered table under a
# "JRC Huizinga et al. (2017) Asia" heading whose values did not match the
# published tables (residential read 0.58 at 1 m against the published 0.49,
# and so on for every class except commerce), which over-predicted damage at
# every depth while citing a source that said otherwise.  `damage.py` reads
# `data/reference/jrc_flood_damage.json`, extracted from the source PDF with
# its citation, so the numbers can be checked against the report.
#
# NOTE for anyone reading the branch history: the audit branch independently
# flagged these ordinates as untraceable literals and "fixed" them by adding a
# confident citation to the very numbers that turn out to be wrong, which made
# them look verified. Loading them from the extracted source is the correct
# fix and supersedes that one entirely.
from . import damage as _damage                                   # noqa: E402

DD_DEPTHS = np.array(_damage.curve("residential", "IND").depth_m)
DD_CURVES = {c: _damage.curve(c, "IND").damage_factor
             for c in _damage.ASSET_CLASSES}

#: Retained so callers that record provenance keep working. It now reports the
#: extracted database's own citation rather than asserting a hand transcription.
DD_SOURCE = _damage.citation()

# Building occupancy classification lives in `damage.py`, which maps the full
# OSM tag set onto the JRC asset classes and reports the basis of each match.
# The audit branch grew a second, cruder copy of the same table here; keeping
# two implementations of one mapping is exactly the duplication that branch
# complained about elsewhere, so this one is gone. Use:
#
#     from .damage import classify_building


@dataclass
class AssetValues:
    """Unit replacement values. EXPLICIT ASSUMPTIONS - override per study area.

    Defaults are order-of-magnitude placeholders in Indian rupees and are
    reported alongside every monetary result so a reviewer can substitute
    audited CPWD / state PWD schedule-of-rates figures.

    EVERY factor that multiplies its way into a currency figure lives here and
    is echoed by `to_dict()`.  Nothing that scales a loss may be a literal
    buried in the accounting code -- if it changes the rupee total, it is a
    declared assumption.
    """
    currency: str = "INR"
    residential_per_building: float = 1_200_000.0
    commercial_per_building: float = 3_500_000.0
    road_per_km: float = 25_000_000.0
    cropland_per_hectare: float = 150_000.0
    # NOTE. There used to be a `road_partial_damage_factor` here, promoted out
    # of the loss code by the audit branch because a flat 0.35 buried in
    # `estimate_losses` was silently producing ~96% of the reported total.
    # `exposure.estimate_losses` now evaluates the JRC INFRASTRUCTURE
    # depth-damage curve at the sampled depth instead, which is both sourced
    # and depth dependent, so the flat factor has no caller. Declaring a knob
    # that no longer does anything would be the same failure the audit
    # complained about, so it is removed rather than left dangling.
    source: str = ("USER-SUPPLIED ASSUMPTION - not measured data. "
                   "Replace with audited schedule-of-rates for the study area.")

    def per_building(self, rate_key: str) -> float:
        return (self.commercial_per_building if rate_key == "commercial"
                else self.residential_per_building)

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "residential_per_building": self.residential_per_building,
            "commercial_per_building": self.commercial_per_building,
            "road_per_km": self.road_per_km,
            "cropland_per_hectare": self.cropland_per_hectare,
            "_provenance": self.source,
        }


# ---------------------------------------------------------------------------
# Hazard rating
# ---------------------------------------------------------------------------

def debris_factor_map(landcover: Optional[np.ndarray],
                      depth: np.ndarray) -> np.ndarray:
    """Defra debris factor: depends on land cover AND depth."""
    df = np.full(depth.shape, 0.5)
    if landcover is not None:
        built = landcover == 50
        forest = landcover == 10
        open_land = np.isin(landcover, (30, 40, 60, 70, 100))
        df[open_land] = 0.0
        df[built] = 1.0
        df[forest] = 1.0
        # below 0.25 m nothing significant floats
        df[depth < 0.25] = 0.0
    return df


# The Defra risk-to-people curves were fitted for depths of order a few metres
# and velocities of a few m/s -- the range in which a person can still stand.
# A dam-break wave in a Himalayan gorge reaches tens of metres and tens of m/s,
# where HR runs into the thousands and the number stops carrying meaning: every
# cell is already "dangerous for all". The rating is therefore reported clipped,
# with the raw maximum kept separately so nothing is silently hidden.
HR_REPORTING_CAP = 20.0


def hazard_rating(depth: np.ndarray, velocity: np.ndarray,
                  landcover: Optional[np.ndarray] = None,
                  debris: Optional[np.ndarray] = None,
                  clip: Optional[float] = HR_REPORTING_CAP) -> np.ndarray:
    """HR = d(v + 0.5) + DF  (Defra FD2321), clipped for reporting."""
    df = debris if debris is not None else debris_factor_map(landcover, depth)
    hr = depth * (velocity + 0.5) + df
    hr = np.where(depth > 0.05, hr, 0.0)
    if clip is not None:
        hr = np.minimum(hr, clip)
    return hr


def classify(hr: np.ndarray) -> np.ndarray:
    """Integer hazard class: 0 none, 1 Low, 2 Moderate, 3 Significant, 4 Extreme."""
    cls = np.zeros(hr.shape, dtype=np.int8)
    for idx, (lo, hi, *_rest) in enumerate(HAZARD_CLASSES, start=1):
        cls[(hr >= lo) & (hr < hi)] = idx
    cls[hr <= 0] = 0
    return cls


def class_areas(cls: np.ndarray, cell_area: float) -> List[dict]:
    out = []
    for idx, (lo, hi, name, colour, desc) in enumerate(HAZARD_CLASSES, start=1):
        n = int((cls == idx).sum())
        out.append({
            "class": idx, "name": name, "colour": colour, "description": desc,
            "hr_range": [lo, None if hi > 1e8 else hi],
            "cells": n,
            "area_km2": round(n * cell_area / 1e6, 4),
        })
    return out


# ---------------------------------------------------------------------------
# Depth-damage
# ---------------------------------------------------------------------------

def damage_fraction(depth: np.ndarray, curve: str = "residential") -> np.ndarray:
    """Interpolate a JRC Asia depth-damage curve onto a depth field/array."""
    if curve not in DD_CURVES:
        raise ValueError(f"unknown damage curve {curve!r}; "
                         f"available: {sorted(DD_CURVES)}")
    return np.interp(depth, DD_DEPTHS, DD_CURVES[curve], left=0.0, right=1.0)


def damage_class(frac: np.ndarray) -> np.ndarray:
    """0 none, 1 minor (<25%), 2 moderate (<50%), 3 major (<75%), 4 destroyed."""
    cls = np.zeros(frac.shape, dtype=np.int8)
    cls[frac > 0.001] = 1
    cls[frac >= 0.25] = 2
    cls[frac >= 0.50] = 3
    cls[frac >= 0.75] = 4
    return cls


DAMAGE_CLASS_NAMES = {0: "None", 1: "Minor", 2: "Moderate",
                      3: "Major", 4: "Destroyed"}


# ---------------------------------------------------------------------------
# Combined product
# ---------------------------------------------------------------------------

def _percentiles(field: np.ndarray, wet: np.ndarray,
                 qs: Tuple[float, ...] = (50.0, 90.0, 99.0, 99.9)) -> Dict[str, float]:
    """Distribution of a field over wet cells only.

    `max` on a dam-break field is dominated by a handful of cells next to the
    source patch and on steep dry fronts, where a shock-capturing scheme is at
    its least reliable.  The percentiles are what an operational reader should
    actually quote.
    """
    if not wet.any():
        return {f"p{q:g}": 0.0 for q in qs}
    vals = field[wet]
    return {f"p{q:g}": round(float(np.percentile(vals, q)), 3) for q in qs}


def structural_vulnerability(dv: np.ndarray) -> Dict[str, int]:
    """Depth-velocity product thresholds for building stability.

    Thresholds after Clausen & Clark (1990) / FEMA; d*v in m2/s:
        < 3     inundation damage only
        3 - 7   partial damage to masonry
        > 7     total destruction of unreinforced masonry
    """
    return {
        "dv_lt_3_m2s": int(((dv > 0) & (dv < 3)).sum()),
        "dv_3_to_7_m2s": int(((dv >= 3) & (dv < 7)).sum()),
        "dv_gt_7_m2s": int((dv >= 7).sum()),
        "reference": "Clausen & Clark (1990) masonry stability thresholds",
    }


@dataclass
class HazardProduct:
    hr: np.ndarray
    hazard_class: np.ndarray
    depth: np.ndarray
    velocity: np.ndarray
    dv: np.ndarray                     # depth x velocity, m2/s (structural proxy)
    arrival_s: Optional[np.ndarray] = None
    duration_s: Optional[np.ndarray] = None
    summary: Dict[str, object] = field(default_factory=dict)


def build_hazard(depth: np.ndarray, velocity: np.ndarray, cell_area: float,
                 landcover: Optional[np.ndarray] = None,
                 arrival_s: Optional[np.ndarray] = None,
                 duration_s: Optional[np.ndarray] = None,
                 wet_threshold: float = 0.05) -> HazardProduct:
    hr = hazard_rating(depth, velocity, landcover)
    hr_raw = hazard_rating(depth, velocity, landcover, clip=None)
    cls = classify(hr)
    dv = depth * velocity
    wet = depth > wet_threshold

    summary = {
        "wet_threshold_m": wet_threshold,
        "inundated_area_km2": round(float(wet.sum()) * cell_area / 1e6, 4),
        "max_depth_m": round(float(depth.max()), 2),
        "max_velocity_ms": round(float(velocity.max()), 2),
        "max_dv_m2s": round(float(dv.max()), 2),
        # A single extreme cell -- usually adjacent to the breach source patch,
        # where the scheme is least trustworthy -- sets `max_velocity_ms` and is
        # not representative of the flood.  Report the distribution alongside it
        # so nobody quotes the outlier as "the" velocity.
        "velocity_percentiles_ms": _percentiles(velocity, wet),
        "depth_percentiles_m": _percentiles(depth, wet),
        "max_hazard_rating_clipped": round(float(hr.max()), 2),
        "max_hazard_rating_raw": round(float(hr_raw.max()), 1),
        "hazard_rating_cap": HR_REPORTING_CAP,
        "hazard_rating_note": (
            "Defra FD2321 was calibrated for depths of a few metres and "
            "velocities of a few m/s. Beyond class 4 ('dangerous for all') the "
            "number carries no extra meaning, so it is clipped for reporting; "
            "the raw maximum is given alongside."),
        "mean_depth_where_wet_m": round(float(depth[wet].mean()), 3) if wet.any() else 0.0,
        "classes": class_areas(cls, cell_area),
        "structural_dv": structural_vulnerability(np.where(wet, dv, 0.0)),
        "method": "Defra FD2321 HR = d(v+0.5) + DF",
        "damage_curve_source": DD_SOURCE,
    }
    if arrival_s is not None:
        arrived = arrival_s >= 0
        if arrived.any():
            summary["arrival_min"] = {
                "earliest_min": round(float(arrival_s[arrived].min()) / 60, 2),
                "median_min": round(float(np.median(arrival_s[arrived])) / 60, 2),
                "latest_min": round(float(arrival_s[arrived].max()) / 60, 2),
            }
    return HazardProduct(hr=hr, hazard_class=cls, depth=depth, velocity=velocity,
                         dv=dv, arrival_s=arrival_s, duration_s=duration_s,
                         summary=summary)
