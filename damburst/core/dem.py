"""DEM container, QC gate and conditioning.

Covers the "Input Data QC & Harmonization -> DEM conditioning -> Channel
geometry & bathymetry -> Computational domain & mesh -> Roughness map" branch
of the technical approach.

There is no synthetic terrain path in this module.  Elevation always comes from
a real mission raster (Copernicus GLO-30 / SRTM / ASTER / CartoDEM) supplied by
`damburst.core.datasources`.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin

NODATA = -9999.0


@dataclass
class DEM:
    """A projected, metre-referenced elevation grid plus its derived layers."""

    z: np.ndarray                      # elevation, m above EGM2008 geoid
    transform: Affine
    crs: str                           # projected CRS, e.g. "EPSG:32644"
    dx: float                          # cell size, metres (east)
    dy: float                          # cell size, metres (north)
    name: str = "dem"
    source: str = ""                   # provenance string, written to manifest
    manning: Optional[np.ndarray] = None
    landcover: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.z.shape

    @property
    def ny(self) -> int:
        return self.z.shape[0]

    @property
    def nx(self) -> int:
        return self.z.shape[1]

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def origin(self) -> Tuple[float, float]:
        return (self.transform.c, self.transform.f)

    def bounds(self) -> Tuple[float, float, float, float]:
        w, n = self.origin
        return (w, n - self.ny * self.dy, w + self.nx * self.dx, n)

    def xy(self, row, col):
        w, n = self.origin
        return (w + (np.asarray(col) + 0.5) * self.dx,
                n - (np.asarray(row) + 0.5) * self.dy)

    def rowcol(self, x, y):
        w, n = self.origin
        return (np.floor((n - np.asarray(y)) / self.dy).astype(int),
                np.floor((np.asarray(x) - w) / self.dx).astype(int))

    def coord_grids(self):
        w, n = self.origin
        xs = w + (np.arange(self.nx) + 0.5) * self.dx
        ys = n - (np.arange(self.ny) + 0.5) * self.dy
        return np.meshgrid(xs, ys)

    def copy(self) -> "DEM":
        return DEM(z=self.z.copy(), transform=self.transform, crs=self.crs,
                   dx=self.dx, dy=self.dy, name=self.name, source=self.source,
                   manning=None if self.manning is None else self.manning.copy(),
                   landcover=None if self.landcover is None else self.landcover.copy(),
                   meta=dict(self.meta))


# ---------------------------------------------------------------------------
# QC gate
# ---------------------------------------------------------------------------

def slope(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Terrain slope, degrees from horizontal.

    Standalone and reusable: this used to be computed ad hoc, once, inline
    inside `qc_report` with no way for another caller (the adaptive
    model-selection criterion, a terrain-products export) to get the same
    field without recomputing the gradient itself.
    """
    gy, gx = np.gradient(z, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def aspect(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compass bearing of the downslope direction, degrees 0-360 (0 = north,
    90 = east), the standard GIS aspect convention.

    Flat cells (both gradient components ~0, i.e. no defined downslope
    direction) are returned as NaN rather than an arbitrary bearing -- a flat
    cell genuinely has no aspect, and reporting one (typically 0/north from a
    naive `arctan2(0, 0)` -> 0) would silently misrepresent it as a
    north-facing slope.

    `z` rows are assumed to increase southward (the raster/array convention
    used throughout this codebase, e.g. `DEM.z`, `dem.xy`), so the north
    component of the gradient is the NEGATIVE row-gradient.
    """
    gy, gx = np.gradient(z, dy, dx)          # gy: d(z)/d(row) i.e. southward
    dz_dnorth = -gy                          # d(z)/d(north)
    dz_deast = gx
    flat = (np.abs(dz_dnorth) < 1e-9) & (np.abs(dz_deast) < 1e-9)
    # downslope direction vector = -gradient = (-dz_dnorth, -dz_deast);
    # bearing clockwise from north = atan2(east_component, north_component)
    bearing = np.degrees(np.arctan2(-dz_deast, -dz_dnorth)) % 360.0
    return np.where(flat, np.nan, bearing)


def qc_report(dem: DEM) -> dict:
    """Checks that must pass before a solver is built on this grid."""
    z = dem.z
    finite = np.isfinite(z) & (z != NODATA)
    voids = int((~finite).sum())
    zf = z[finite]
    slope_deg = slope(np.where(finite, z, np.nan), dem.dx, dem.dy)

    checks = {
        "source": dem.source,
        "cells": int(z.size),
        "grid": [int(dem.ny), int(dem.nx)],
        "void_cells": voids,
        "void_fraction": round(float(voids / z.size), 6),
        "z_min_m": round(float(zf.min()), 2) if zf.size else None,
        "z_max_m": round(float(zf.max()), 2) if zf.size else None,
        "relief_m": round(float(zf.max() - zf.min()), 2) if zf.size else None,
        "cell_size_m": [round(dem.dx, 2), round(dem.dy, 2)],
        "crs": dem.crs,
        "projected_metres": not dem.crs.upper().endswith("4326"),
        "max_slope_deg": round(float(np.nanmax(slope_deg)), 2),
        "mean_slope_deg": round(float(np.nanmean(slope_deg)), 2),
    }
    msgs = []
    if checks["void_fraction"] >= 0.02:
        msgs.append("More than 2% void cells - fill or re-source the DEM.")
    if not checks["projected_metres"]:
        msgs.append("DEM is geographic; reproject to UTM before routing.")
    if checks["relief_m"] is not None and checks["relief_m"] < 20:
        msgs.append("Relief under 20 m - domain may not contain the valley.")
    checks["messages"] = msgs
    checks["pass"] = len(msgs) == 0
    return checks


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

def fill_voids(z: np.ndarray, nodata: float = NODATA) -> np.ndarray:
    """Iterative 8-neighbour mean infill for mission voids (SRTM/ASTER gaps)."""
    out = z.astype(np.float64).copy()
    bad = ~np.isfinite(out) | (out == nodata)
    if not bad.any():
        return out
    out[bad] = np.nan
    for _ in range(300):
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            break
        padded = np.pad(out, 1, mode="edge")
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                if di == 1 and dj == 1:
                    continue
                nb = padded[di:di + out.shape[0], dj:dj + out.shape[1]]
                ok = np.isfinite(nb)
                acc += np.where(ok, nb, 0.0)
                cnt += ok
        out = np.where(nan_mask, np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), out)
    return np.nan_to_num(out, nan=float(np.nanmin(out[np.isfinite(out)])))


def fill_depressions(z: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """Priority-flood depression filling (Barnes, Lehman & Mulla 2014).

    Removes spurious sinks that would otherwise trap the flood wave and produce
    phantom ponding in the inundation map.
    """
    ny, nx = z.shape
    out = np.empty_like(z, dtype=np.float64)
    closed = np.zeros((ny, nx), dtype=bool)
    pq: list = []

    for i in range(ny):
        for j in (0, nx - 1):
            heapq.heappush(pq, (float(z[i, j]), i, j))
            closed[i, j] = True
            out[i, j] = z[i, j]
    for j in range(nx):
        for i in (0, ny - 1):
            if not closed[i, j]:
                heapq.heappush(pq, (float(z[i, j]), i, j))
                closed[i, j] = True
                out[i, j] = z[i, j]

    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while pq:
        zc, i, j = heapq.heappop(pq)
        for di, dj in nbrs:
            ii, jj = i + di, j + dj
            if 0 <= ii < ny and 0 <= jj < nx and not closed[ii, jj]:
                zn = max(float(z[ii, jj]), zc + epsilon)
                out[ii, jj] = zn
                closed[ii, jj] = True
                heapq.heappush(pq, (zn, ii, jj))
    return out


def burn_channel(z: np.ndarray, mask: np.ndarray, depth: float = 3.0) -> np.ndarray:
    """Stream burning: recover sub-pixel channel conveyance lost at 30 m posting.

    A 30 m DSM cannot resolve a 20 m wide incised channel, so the low-flow
    thalweg is enforced from the OSM waterway centreline.
    """
    out = z.copy()
    out[mask] -= depth
    return out


def condition(dem: DEM, channel_mask: Optional[np.ndarray] = None,
              burn: float = 0.0, fill_sinks: bool = True) -> DEM:
    """Full DEM-conditioning node of the technical approach."""
    out = dem.copy()
    out.z = fill_voids(out.z)
    steps = ["void_fill"]
    if burn > 0 and channel_mask is not None and channel_mask.any():
        out.z = burn_channel(out.z, channel_mask, burn)
        steps.append(f"channel_burn_{burn:g}m")
    if fill_sinks:
        out.z = fill_depressions(out.z)
        steps.append("priority_flood_sink_fill")
    out.meta = dict(out.meta)
    out.meta["conditioning"] = steps
    return out


# ---------------------------------------------------------------------------
# Roughness  (ESA WorldCover -> Manning n)
# ---------------------------------------------------------------------------
# Manning values follow Chow (1959) open-channel tables and the Arcement &
# Schneider (USGS WSP 2339) floodplain guidance.

WORLDCOVER_CLASSES = {
    10: ("Tree cover",              0.120),
    20: ("Shrubland",               0.060),
    30: ("Grassland",               0.040),
    40: ("Cropland",                0.045),
    50: ("Built-up",                0.090),
    60: ("Bare / sparse vegetation", 0.030),
    70: ("Snow and ice",            0.025),
    80: ("Permanent water bodies",  0.030),
    90: ("Herbaceous wetland",      0.055),
    95: ("Mangroves",               0.100),
    100: ("Moss and lichen",        0.035),
}


def roughness_from_worldcover(landcover: np.ndarray,
                              default: float = 0.045) -> np.ndarray:
    """Map ESA WorldCover v200 class codes to a Manning roughness field."""
    n = np.full(landcover.shape, default, dtype=np.float64)
    for code, (_, manning) in WORLDCOVER_CLASSES.items():
        n[landcover == code] = manning
    return n


def landcover_summary(landcover: np.ndarray) -> dict:
    total = landcover.size
    out = {}
    for code, (label, manning) in WORLDCOVER_CLASSES.items():
        c = int((landcover == code).sum())
        if c:
            out[label] = {"cells": c, "percent": round(100 * c / total, 2),
                          "manning_n": manning}
    return out


# ---------------------------------------------------------------------------
# Hydro-geometry helpers
# ---------------------------------------------------------------------------

def d8_flow_direction(z: np.ndarray, dx: float) -> np.ndarray:
    """Single-flow-direction (D8) target cell for every cell.

    Returns a flat int64 array of length `z.size`: `target[k]` is the raveled
    index of the cell that cell `k` drains into, steepest-descent among its 8
    neighbours. A cell with no downslope neighbour (a pit, or an edge cell
    whose lowest neighbour is still higher) drains to ITSELF -- `target[k]==k`
    -- which both `d8_flow_accumulation` and `delineate_watershed` use as the
    sink/boundary marker.

    Split out of what used to be a single `d8_flow_accumulation` function so
    the flow-direction grid -- needed for watershed delineation, which has to
    walk the network in the other direction -- is not a private detail nobody
    else can reuse.
    """
    ny, nx = z.shape
    zf = z.ravel()
    n = zf.size
    target = np.arange(n, dtype=np.int64)     # default: drains to self (sink)

    di = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dj = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    dist = np.hypot(di * dx, dj * dx)

    for idx in range(n):
        i, j = divmod(idx, nx)
        ii, jj = i + di, j + dj
        ok = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        if not ok.any():
            continue
        nidx = ii[ok] * nx + jj[ok]
        drop = (zf[idx] - zf[nidx]) / dist[ok]
        k = int(np.argmax(drop))
        if drop[k] > 0:
            target[idx] = nidx[k]
    return target


def d8_flow_accumulation(z: np.ndarray, dx: float,
                         flow_dir: Optional[np.ndarray] = None) -> np.ndarray:
    """D8 flow accumulation (cell counts) on a sink-filled DEM.

    Used to locate the real channel network for the downstream centreline and
    for the cross-section extraction. Pass a precomputed `flow_dir` (from
    `d8_flow_direction`) to avoid recomputing it when both are needed, e.g. by
    `delineate_watershed`.
    """
    order = np.argsort(z, axis=None)[::-1]          # high -> low
    acc = np.ones(z.size, dtype=np.float64)
    target = flow_dir if flow_dir is not None else d8_flow_direction(z, dx)

    for idx in order:
        t = target[idx]
        if t != idx:                                 # not a sink
            acc[t] += acc[idx]
    return acc.reshape(z.shape)


def delineate_watershed(z: np.ndarray, dx: float, pour_point_rc: Tuple[int, int],
                        flow_dir: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean catchment mask draining through `pour_point_rc`.

    This was previously MISSING entirely: the codebase had D8 flow
    accumulation (a cell-count field) and a single traced thalweg path, but
    nothing that answers "what area drains to this point" -- the "watershed
    boundary" node the technical-approach spec asks for.

    Method: build the REVERSE of the D8 flow-direction graph (which cells
    drain INTO each cell) and breadth-first search outward from the pour
    point. Every cell reached is, by construction, part of its catchment --
    this is the standard D8 watershed-delineation algorithm, just without a
    name-brand GIS library behind it.
    """
    ny, nx = z.shape
    target = flow_dir if flow_dir is not None else d8_flow_direction(z, dx)
    n = target.size

    # reverse adjacency: upstream[t] = list of cells draining into t
    upstream: dict = {}
    for idx in range(n):
        t = int(target[idx])
        if t == idx:
            continue                                 # sink, no edge to record
        upstream.setdefault(t, []).append(idx)

    pr, pc = pour_point_rc
    start = pr * nx + pc
    if not (0 <= pr < ny and 0 <= pc < nx):
        raise ValueError(f"pour point {pour_point_rc} is outside the {ny}x{nx} grid")

    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    stack = [start]
    while stack:
        cur = stack.pop()
        for up in upstream.get(cur, ()):
            if not seen[up]:
                seen[up] = True
                stack.append(up)
    return seen.reshape(z.shape)


def extract_contours(z: np.ndarray, dx: float, dy: float,
                     interval: float = 25.0,
                     origin_xy: Tuple[float, float] = (0.0, 0.0)
                     ) -> List[Dict[str, object]]:
    """Elevation contour lines at a fixed interval, as polylines in the DEM's
    own projected coordinates (metres) -- another terrain product the
    technical-approach spec names that had no implementation at all.

    Uses matplotlib's contouring engine (marching squares) purely as a
    geometry tracer -- no figure is drawn or rendered, `Agg` never touches a
    display, and matplotlib is only imported here, not at module load, so
    nothing that only needs elevation/slope/aspect/flow pays its import cost.

    `origin_xy` should be `(dem.transform.c, dem.transform.f)` -- the grid's
    own origin -- so the returned line coordinates line up with `dem.xy()`.
    Each returned dict is `{"elevation_m": level, "coords": [[x, y], ...]}`
    for one contour segment (a single level can produce several disjoint
    polylines, each returned separately rather than concatenated).
    """
    import matplotlib
    matplotlib.use("Agg")                    # headless: no display backend
    import matplotlib.pyplot as plt

    finite = np.isfinite(z)
    if not finite.any():
        return []
    zlo, zhi = float(z[finite].min()), float(z[finite].max())
    if zhi - zlo < interval:
        return []
    levels = np.arange(math.ceil(zlo / interval) * interval, zhi, interval)
    if levels.size == 0:
        return []

    ny, nx = z.shape
    xs = origin_xy[0] + np.arange(nx) * dx
    ys = origin_xy[1] - np.arange(ny) * dy    # rows increase southward

    fig = plt.figure()
    try:
        ax = fig.add_subplot(111)
        cs = ax.contour(xs, ys, np.where(finite, z, np.nan), levels=levels)
        out: List[Dict[str, object]] = []
        # matplotlib >=3.8 exposes contour geometry via `cs.allsegs`
        # (list per level of arrays of (x, y) vertices); this is the stable
        # public shape for the versions pinned in requirements.txt.
        for level, segs in zip(levels, cs.allsegs):
            for seg in segs:
                if len(seg) < 2:
                    continue
                out.append({"elevation_m": round(float(level), 2),
                           "coords": [[round(float(x), 2), round(float(y), 2)]
                                      for x, y in seg]})
        return out
    finally:
        plt.close(fig)


def steepest_descent_path(z: np.ndarray, start: Tuple[int, int],
                          max_steps: int = 20000) -> np.ndarray:
    """Trace the downstream thalweg from a seed cell (used for the long profile)."""
    ny, nx = z.shape
    i, j = start
    path = [(i, j)]
    seen = {(i, j)}
    di = (-1, -1, -1, 0, 0, 1, 1, 1)
    dj = (-1, 0, 1, -1, 1, -1, 0, 1)
    for _ in range(max_steps):
        best, bi, bj = 0.0, None, None
        for a, b in zip(di, dj):
            ii, jj = i + a, j + b
            if 0 <= ii < ny and 0 <= jj < nx and (ii, jj) not in seen:
                d = (z[i, j] - z[ii, jj]) / math.hypot(a, b)
                if d > best:
                    best, bi, bj = d, ii, jj
        if bi is None:
            break
        i, j = bi, bj
        seen.add((i, j))
        path.append((i, j))
    return np.array(path)
