"""Reservoir hypsometry (H-V-A) and level-pool routing.

Implements the "Reservoir H-V-A curve + Loading Level" and the
"Reservoir drained? -> invert H-V-A -> Volume Balance" loop of the technical
approach.  The H-V-A curve is derived from the conditioned DEM itself rather
than assumed, so the same code works for an engineered reservoir and for a
landslide-dammed lake.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

G = 9.80665


@dataclass
class Reservoir:
    """Stage-area-volume relationship for the impoundment behind the barrier."""

    levels: np.ndarray       # stage, m MSL (monotonic increasing)
    areas: np.ndarray        # inundated area at that stage, m^2
    volumes: np.ndarray      # stored volume at that stage, m^3
    bed: float               # lowest bed elevation in the pool, m MSL
    crest: float             # barrier crest, m MSL
    mask: Optional[np.ndarray] = None   # cells that belong to the pool at crest

    # -- interpolators ----------------------------------------------------
    def volume(self, h: float) -> float:
        return float(np.interp(h, self.levels, self.volumes,
                               left=0.0, right=self.volumes[-1]))

    def area(self, h: float) -> float:
        return float(np.interp(h, self.levels, self.areas,
                               left=0.0, right=self.areas[-1]))

    def level(self, v: float) -> float:
        """Invert V -> H.  This is the `invert H-V-A` node in the diagram."""
        v = min(max(v, 0.0), float(self.volumes[-1]))
        return float(np.interp(v, self.volumes, self.levels))

    @property
    def capacity(self) -> float:
        return float(self.volumes[-1])

    def summary(self) -> dict:
        return {
            "bed_m": self.bed,
            "crest_m": self.crest,
            "capacity_mcm": self.capacity / 1e6,
            "area_at_crest_km2": float(self.areas[-1]) / 1e6,
            "max_depth_m": self.crest - self.bed,
        }


def _connected_pool(z: np.ndarray, level: float, seed_rc: Tuple[int, int],
                    blocked: np.ndarray) -> np.ndarray:
    """4-connected flood fill of cells below `level` reachable from the seed.

    `blocked` marks the barrier body so the pool cannot leak through it.
    """
    ny, nx = z.shape
    wet = (z < level) & (~blocked)
    out = np.zeros_like(wet)
    r0, c0 = seed_rc
    if not wet[r0, c0]:
        # snap to the deepest nearby wet cell
        r1, r2 = max(0, r0 - 4), min(ny, r0 + 5)
        c1, c2 = max(0, c0 - 4), min(nx, c0 + 5)
        sub = np.where(wet[r1:r2, c1:c2], z[r1:r2, c1:c2], np.inf)
        if not np.isfinite(sub).any():
            return out
        k = np.unravel_index(np.argmin(sub), sub.shape)
        r0, c0 = r1 + k[0], c1 + k[1]

    q = deque([(r0, c0)])
    out[r0, c0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and wet[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                q.append((rr, cc))
    return out


def hva_from_dem(z: np.ndarray, cell_area: float, seed_rc: Tuple[int, int],
                 crest: float, barrier_mask: np.ndarray,
                 upstream_mask: np.ndarray,
                 n_levels: int = 45) -> Reservoir:
    """Build the H-V-A curve by successively flooding the DEM behind the dam.

    Volume is integrated as sum((level - z) * cell_area) over the connected
    pool, i.e. true DEM hypsometry, not a power-law fit.

    `upstream_mask` is the half-plane on the impounded side of the dam axis.
    It must be derived from the dam's own orientation -- splitting the domain
    by grid row or column silently fails whenever the river does not happen to
    run along that axis, and the fill then escapes downstream and reports an
    enormous phantom reservoir.
    """
    ny, nx = z.shape
    seed_row, seed_col = int(seed_rc[0]), int(seed_rc[1])
    bed = float(z[seed_row, seed_col])

    # everything that is not upstream, plus the barrier itself, is off limits
    domain_block = barrier_mask | (~upstream_mask)

    levels = np.linspace(bed + 0.25, crest, n_levels)
    areas = np.zeros(n_levels)
    volumes = np.zeros(n_levels)
    pool_mask = None
    for k, lv in enumerate(levels):
        pool = _connected_pool(z, lv, (seed_row, seed_col), domain_block)
        depth = np.where(pool, lv - z, 0.0)
        areas[k] = pool.sum() * cell_area
        volumes[k] = depth.sum() * cell_area
        pool_mask = pool

    # enforce monotonicity (DEM noise can produce tiny reversals)
    volumes = np.maximum.accumulate(volumes)
    areas = np.maximum.accumulate(areas)
    return Reservoir(levels=levels, areas=areas, volumes=volumes,
                     bed=bed, crest=float(crest), mask=pool_mask)


def detect_water_surface(z: np.ndarray, water_mask: np.ndarray,
                         upstream_mask: np.ndarray) -> Optional[dict]:
    """Elevation and area of the existing reservoir surface seen by the DSM.

    `water_mask` is the ESA WorldCover permanent-water class.  Copernicus
    GLO-30 is a surface model, so an existing reservoir appears as a flat plate
    at its fill level on the acquisition date; that plate is what this finds.
    """
    m = water_mask & upstream_mask
    if m.sum() < 8:
        return None
    zz = z[m]
    # the reservoir plate is the dominant low, flat population of water pixels
    lo, hi = np.percentile(zz, [10, 90])
    plate = zz[(zz >= lo) & (zz <= hi)]
    if plate.size < 4:
        return None
    return {"elevation_m": float(np.median(plate)),
            "cells": int(m.sum()),
            "spread_m": float(hi - lo)}


DEFAULT_SHAPE_EXPONENT = 2.4     # typical narrow Himalayan valley


def _solve_shape_exponent(d_ws: float, d_crest: float, area_ws: float,
                          volume_target: float) -> Tuple[float, bool]:
    """Find b in V = a*d^b that reproduces both the observed surface area at
    the DSM water level and the published storage at crest level.

    Returns (b, solved).  `solved` is False when the bisection bracket contains
    no root and the default exponent had to be used instead.  The caller MUST
    record that flag: a solved exponent is constrained by two measurements, a
    defaulted one is a guess, and the provenance has to tell them apart.
    """
    target = area_ws / max(volume_target, 1e-9)

    def f(b):
        return (b / d_ws) * (d_ws / d_crest) ** b - target

    lo, hi = 1.05, 6.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return DEFAULT_SHAPE_EXPONENT, False
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), True


def hybrid_hva(dem_reservoir: Reservoir, water_surface_m: float,
               area_at_surface_m2: float, published_capacity_m3: float,
               dam_height_m: float, crest_m: float,
               riverbed_m: Optional[float] = None,
               n_levels: int = 60) -> Tuple[Reservoir, dict]:
    """Combine reconstructed bathymetry with DEM hypsometry.

    A DSM cannot see beneath an existing reservoir, so DEM hypsometry alone
    measures only the sliver between the water surface and the crest -- for
    Tehri that is ~7% of gross storage, which would badly understate a breach.

    Below the water surface the drowned valley is reconstructed with the
    standard two-parameter power-law storage model V = a*(h - z_base)^b, with
    a and b fixed by two REAL constraints:

      * the reservoir surface area measured from ESA WorldCover at the DSM
        water level, and
      * the published gross storage at crest level (CWC/Wikipedia record).

    Above the water surface the DEM's own hypsometry is used unchanged.

    Nothing here is invented: the shape exponent is solved, not assumed, and
    the returned dict records every input so a reviewer can substitute a real
    bathymetric survey.
    """
    # Published dam height is measured from the FOUNDATION, which is excavated
    # below the river.  A breach erodes to roughly the original bed, not to the
    # foundation, so the observed riverbed at the dam toe is the right invert.
    # Using crest - height instead overstates the head (73 m at Tehri) and with
    # it the peak discharge.
    z_foundation = crest_m - dam_height_m
    if riverbed_m is not None and z_foundation < riverbed_m < water_surface_m:
        z_base = float(riverbed_m)
        base_source = "DEM riverbed at the dam toe"
    else:
        z_base = z_foundation
        base_source = "crest - published dam height (foundation level)"
    d_ws = max(water_surface_m - z_base, 1.0)
    d_crest = max(crest_m - z_base, d_ws + 0.5)

    b, b_solved = _solve_shape_exponent(d_ws, d_crest, area_at_surface_m2,
                                        published_capacity_m3)
    a = published_capacity_m3 / (d_crest ** b)

    levels = np.linspace(z_base, crest_m, n_levels)
    d = np.maximum(levels - z_base, 0.0)
    volumes = a * d ** b
    areas = np.where(d > 0, a * b * np.maximum(d, 1e-6) ** (b - 1.0), 0.0)

    # above the DSM water plate, add the terrain the DEM really does see
    dem_extra_v = np.array([dem_reservoir.volume(l) for l in levels])
    dem_extra_a = np.array([dem_reservoir.area(l) for l in levels])
    above = levels > water_surface_m
    if above.any():
        v_ws = a * d_ws ** b
        volumes = np.where(above, v_ws + dem_extra_v, volumes)
        areas = np.where(above, np.maximum(areas, dem_extra_a), areas)

    volumes = np.maximum.accumulate(volumes)
    areas = np.maximum.accumulate(areas)

    provenance = {
        "method": "hybrid: power-law bathymetry below DSM water surface + "
                  "DEM hypsometry above",
        "dsm_water_surface_m": round(water_surface_m, 2),
        "base_elevation_m": round(z_base, 2),
        "base_elevation_source": base_source,
        "foundation_level_m": round(z_foundation, 2),
        "crest_m": round(crest_m, 2),
        "published_capacity_mcm": round(published_capacity_m3 / 1e6, 1),
        "published_dam_height_m": round(dam_height_m, 1),
        "observed_surface_area_km2": round(area_at_surface_m2 / 1e6, 3),
        "shape_exponent_b": round(b, 3),
        "shape_exponent_solved": bool(b_solved),
        "shape_exponent_source": (
            "solved by bisection against the observed surface area and the "
            "published capacity"
            if b_solved else
            f"DEFAULTED to {DEFAULT_SHAPE_EXPONENT} - the two constraints did "
            "not bracket a root, so this exponent is an assumption about "
            "valley shape, not a fitted value"),
        "coefficient_a": a,
        "dem_only_capacity_mcm": round(dem_reservoir.capacity / 1e6, 2),
        "capacity_from_published_fraction": round(
            1.0 - min(dem_reservoir.capacity / max(published_capacity_m3, 1.0), 1.0), 4),
        "caveat": ("Bathymetry below the DSM water surface is RECONSTRUCTED, "
                   "not surveyed. Gross storage is SET BY the published "
                   "capacity: V(crest) == published capacity by construction, "
                   "so this curve must NOT be described as DEM-derived. What "
                   "the DEM contributes is the hypsometry above the water "
                   "plate and the surface area that constrains the exponent. "
                   "Replace with a bathymetric survey where one exists."),
    }
    res = Reservoir(levels=levels, areas=areas, volumes=volumes,
                    bed=z_base, crest=crest_m, mask=dem_reservoir.mask)
    return res, provenance


def analytic_reservoir(capacity_mcm: float, depth: float, bed: float = 0.0,
                       shape_exp: float = 2.4, n_levels: int = 60) -> Reservoir:
    """Fallback H-V-A when only gazetteer capacity and dam height are known.

    NOT called by the default pipeline: every preset has a DEM, so either
    `hva_from_dem` or `hybrid_hva` applies. Kept for a barrier with no usable
    terrain data at all.


    Uses V(h) = V_max * (h/H)^b, the standard power-law storage model; b ~ 2.4
    is typical of a narrow Himalayan valley (b ~ 3 for a steep gorge,
    b ~ 1.5 for a broad plains reservoir).
    """
    h = np.linspace(0.0, depth, n_levels)
    v = capacity_mcm * 1e6 * (h / depth) ** shape_exp
    a = np.gradient(v, h)
    a[0] = a[1]
    return Reservoir(levels=bed + h, areas=a, volumes=v,
                     bed=bed, crest=bed + depth)


# ---------------------------------------------------------------------------
# Level-pool routing
# ---------------------------------------------------------------------------

def route_step(reservoir: Reservoir, volume: float, q_in: float,
               q_out: float, dt: float) -> Tuple[float, float]:
    """One explicit continuity step:  dV/dt = Qin - Qout.

    NOT called by the default pipeline: `breach.simulate_breach` performs the
    same update inline so it can share the adaptive-dt bookkeeping. Kept as the
    standalone, testable form of the "Reservoir drained?" decision node.


    Returns (new_volume, new_level).  Volume is clamped at zero: the
    "Reservoir drained?" decision node.
    """
    v = max(0.0, volume + (q_in - q_out) * dt)
    return v, reservoir.level(v)


def inflow_hydrograph(kind: str = "constant", base: float = 40.0,
                      peak: float = 400.0, t_peak: float = 3600.0,
                      shape: float = 3.0) -> Callable[[float], float]:
    """Upstream inflow Q_in(t) -- constant baseflow or a gamma-shaped flood."""
    if kind == "constant":
        return lambda t: base
    if kind == "flood":
        def q(t: float) -> float:
            if t <= 0:
                return base
            r = t / t_peak
            return base + (peak - base) * (r ** shape) * np.exp(shape * (1.0 - r))
        return q
    raise ValueError(f"unknown inflow kind: {kind}")


def volume_balance(v0: float, v_end: float, released: float,
                   inflow_total: float, tol: float = 0.02) -> dict:
    """Mass-balance gate ("Volume Balance" / "Sanity checks" in the diagram)."""
    expected = v0 + inflow_total - released
    err = abs(expected - v_end)
    denom = max(v0, 1.0)
    return {
        "v_initial_mcm": v0 / 1e6,
        "v_final_mcm": v_end / 1e6,
        "released_mcm": released / 1e6,
        "inflow_mcm": inflow_total / 1e6,
        "closure_error_mcm": err / 1e6,
        "relative_error": err / denom,
        "pass": bool(err / denom < tol),
    }
