"""SPH <-> 2D shallow-water coupling and the three-way model comparison.

Implements the centre column of the technical approach:

    SPH configuration -> Run SPH near field
        -> SPH-Delft3D Transfer Interface  (near field / far field / overlap zone)
        -> Balance  <->  Delft3D coupled
        -> Delft3D standalone <-> Stability watch
        -> Model comparison

Three model configurations are produced and compared:

  A. `grid_standalone`  far-field SWE driven directly by the empirical breach
                        weir hydrograph Q(t).  The conventional approach.
  B. `sph_nearfield`    particle model of the breach jet.  Resolves vertical
                        accelerations and the non-hydrostatic plunging jet that
                        a depth-averaged model cannot represent.
  C. `coupled`          SWE far field driven by the hydrograph that SPH
                        actually delivers across the transfer section.

The overlap zone is the reach between the breach and the transfer section: SPH
is authoritative upstream of it, the grid model downstream, and the transfer
section is where the handover happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .breach import G, BreachResult
from .dem import DEM, steepest_descent_path
from .sph import SPHConfig, SPHResult, build_slice, run_sph


@dataclass
class TransferInterface:
    """Time series handed from the particle model to the grid model."""
    t: np.ndarray
    depth: np.ndarray            # m at the transfer section
    u_mean: np.ndarray           # m/s depth-averaged
    q_unit: np.ndarray           # m2/s per unit width
    width_m: float               # effective conveying width at the section
    q_total: np.ndarray          # m3/s = q_unit * width
    transfer_x_m: float
    overlap_start_m: float
    overlap_end_m: float

    def hydrograph(self) -> Callable[[float], float]:
        t, q = self.t, self.q_total
        return lambda tt: float(np.interp(tt, t, q, left=0.0, right=q[-1]))

    def to_dict(self) -> dict:
        return {
            "transfer_x_m": round(self.transfer_x_m, 1),
            "overlap_zone_m": [round(self.overlap_start_m, 1),
                               round(self.overlap_end_m, 1)],
            "effective_width_m": round(self.width_m, 1),
            "peak_q_total_m3s": round(float(self.q_total.max()), 1),
            "peak_depth_m": round(float(self.depth.max()), 2),
            "peak_u_ms": round(float(self.u_mean.max()), 2),
            "samples": int(self.t.size),
        }


# ---------------------------------------------------------------------------
# Long profile extraction from the real DEM
# ---------------------------------------------------------------------------

def thalweg_profile(dem: DEM, start_rc: Tuple[int, int],
                    length_m: float = 1500.0,
                    smooth: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace the steepest-descent path downstream and return its long profile.

    Returns (s, z, path_rc): chainage in metres, bed elevation, and the cell
    indices of the path.  This is the real valley geometry the SPH slice uses.
    """
    path = steepest_descent_path(dem.z, start_rc,
                                 max_steps=int(length_m / dem.dx) + 50)
    if path.shape[0] < 3:
        raise ValueError("Could not trace a downstream path from the dam cell")

    d = np.hypot(np.diff(path[:, 0]) * dem.dy, np.diff(path[:, 1]) * dem.dx)
    s = np.concatenate([[0.0], np.cumsum(d)])
    keep = s <= length_m
    path = path[keep]
    s = s[keep]
    z = dem.z[path[:, 0], path[:, 1]].astype(float)

    if smooth > 1 and z.size > smooth:
        k = np.ones(smooth) / smooth
        z = np.convolve(z, k, mode="same")
        z[:smooth] = dem.z[path[:smooth, 0], path[:smooth, 1]]
        z[-smooth:] = dem.z[path[-smooth:, 0], path[-smooth:, 1]]

    # enforce monotonic descent so the SPH bed has no artificial sinks
    z = np.minimum.accumulate(z)
    return s, z, path


def upstream_profile(dem: DEM, dam_rc: Tuple[int, int], pool_mask: np.ndarray,
                     upstream_dir: np.ndarray, along_dir: np.ndarray,
                     length_m: float = 800.0,
                     half_width_cells: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Bed profile inside the reservoir, walking upstream along the dam normal.

    At each chainage the deepest pool cell across the valley is taken, so the
    profile follows the thalweg rather than an arbitrary grid row or column.
    """
    n = max(int(length_m / dem.dx), 2)
    r0, c0 = dam_rc
    ny, nx = dem.shape
    xs, zs = [], []
    for k in range(1, n + 1):
        best = None
        for m in range(-half_width_cells, half_width_cells + 1):
            r = int(round(r0 + upstream_dir[0] * k + along_dir[0] * m))
            c = int(round(c0 + upstream_dir[1] * k + along_dir[1] * m))
            if not (0 <= r < ny and 0 <= c < nx):
                continue
            zz = float(dem.z[r, c])
            if pool_mask is not None and pool_mask.any() and not pool_mask[r, c]:
                continue
            if best is None or zz < best:
                best = zz
        if best is None:
            for m in range(-half_width_cells, half_width_cells + 1):
                r = int(round(r0 + upstream_dir[0] * k + along_dir[0] * m))
                c = int(round(c0 + upstream_dir[1] * k + along_dir[1] * m))
                if 0 <= r < ny and 0 <= c < nx:
                    zz = float(dem.z[r, c])
                    best = zz if best is None else min(best, zz)
        if best is None:
            continue
        xs.append(-k * dem.dx)
        zs.append(best)
    return np.array(xs[::-1]), np.array(zs[::-1])


# ---------------------------------------------------------------------------
# Near-field SPH run
# ---------------------------------------------------------------------------

def run_near_field(dem: DEM, dam_rc: Tuple[int, int], pool_mask: np.ndarray,
                   water_level: float, breach: BreachResult,
                   upstream_dir: np.ndarray, along_dir: np.ndarray,
                   start_rc: Optional[Tuple[int, int]] = None,
                   cfg: Optional[SPHConfig] = None,
                   upstream_m: float = 600.0,
                   downstream_m: float = 900.0,
                   reservoir_base_m: Optional[float] = None,
                   water_surface_m: Optional[float] = None,
                   progress=None) -> Tuple[SPHResult, TransferInterface]:
    """Build the vertical slice from the real DEM and integrate the SPH model."""
    cfg = cfg or SPHConfig()

    s_dn, z_dn, _path = thalweg_profile(dem, start_rc or dam_rc,
                                        length_m=downstream_m)
    x_up, z_up = upstream_profile(dem, dam_rc, pool_mask, upstream_dir,
                                  along_dir, length_m=upstream_m)

    # The DSM shows the reservoir as a flat plate at its fill level, so the
    # upstream "bed" it reports is the water surface.  Where the H-V-A has been
    # reconstructed, carry the same drowned-valley geometry into the SPH slice;
    # otherwise the particle model would see an 11 m puddle instead of a
    # 260 m head and the near-field jet would be meaningless.
    if reservoir_base_m is not None and water_surface_m is not None and z_up.size:
        drowned = z_up <= water_surface_m + 1.0
        if drowned.any():
            xd = x_up[drowned]
            x_near, x_far = float(xd.max()), float(xd.min())
            f = ((xd - x_near) / (x_far - x_near)) if x_far != x_near else np.zeros_like(xd)
            z_up = z_up.copy()
            z_up[drowned] = reservoir_base_m + f * (water_surface_m - reservoir_base_m)

    # The downstream trace begins at the dam toe, which the DEM may still place
    # part-way up the structure.  Water cannot run uphill out of the breach, so
    # tie the downstream profile to the breach invert and force it to descend:
    # any residual step up at x = 0 would dam the particle model and it would
    # report zero discharge across the transfer section.
    if z_dn.size:
        invert = float(breach.geometry.invert)
        start = min(float(z_dn[0]), invert)
        z_dn = np.minimum(z_dn, start)
        z_dn = np.minimum.accumulate(z_dn)
        if z_up.size:
            z_up = np.maximum(z_up, start)

    bed_x = np.concatenate([x_up, s_dn])
    bed_z = np.concatenate([z_up, z_dn])
    o = np.argsort(bed_x)
    bed_x, bed_z = bed_x[o], bed_z[o]
    uniq = np.concatenate([[True], np.diff(bed_x) > 1e-6])
    bed_x, bed_z = bed_x[uniq], bed_z[uniq]

    setup = build_slice(bed_x, bed_z, water_level=water_level, dam_x=0.0,
                        cfg=cfg, breach_invert=breach.geometry.invert,
                        breach_open=True)

    transfer_x = min(12.0 * cfg.dp, downstream_m * 0.35)
    res = run_sph(setup, cfg, transfer_x=transfer_x, progress=progress)

    width = float(breach.b_top[-1]) if breach.b_top.size else breach.geometry.b_top()
    width = max(width, 1.0)

    iface = TransferInterface(
        t=res.t, depth=res.depth, u_mean=res.u_mean, q_unit=res.q_unit,
        width_m=width, q_total=res.q_unit * width,
        transfer_x_m=float(transfer_x),
        overlap_start_m=0.0, overlap_end_m=float(transfer_x),
    )
    return res, iface


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _peak(t: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
    if q.size == 0:
        return 0.0, 0.0
    k = int(np.argmax(q))
    return float(q[k]), float(t[k])


def nash_sutcliffe(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def compare_hydrographs(breach: BreachResult, iface: TransferInterface,
                        head_m: Optional[float] = None) -> dict:
    """Balance / stability-watch node.

    The two models must be compared like for like.  Sampling the weir
    hydrograph over the SPH clock window is meaningless: SPH is initialised
    with the breach already cut to its invert, whereas the weir closure grows
    the breach over its formation time -- often 60-90 minutes.  Ten seconds in,
    the weir discharge is still essentially zero, so a ratio against it is a
    division by zero, not a diagnostic.

    Instead:
      * SPH peak (an instantaneous, fully-open breach) is compared with the
        weir model's peak over the WHOLE breach, which is the closest
        equivalent quantity; SPH is expected to be the higher of the two.
      * SPH's peak unit discharge is checked against the critical-flow value
        for the same head, q_c = (2/3)^1.5 * sqrt(g) * H^1.5.  That is a hard
        physical ceiling for free discharge and is the real stability test on
        the particle model.
    """
    t = iface.t
    q_sph = iface.q_total
    ps, t_sph_peak = _peak(t, q_sph)
    pw = float(breach.q.max()) if breach.q.size else 0.0

    out: Dict[str, object] = {
        "sph_window_s": [round(float(t[0]), 3) if t.size else 0.0,
                         round(float(t[-1]), 3) if t.size else 0.0],
        "sph_peak_m3s": round(ps, 1),
        "sph_peak_at_s": round(t_sph_peak, 2),
        "weir_peak_m3s": round(pw, 1),
        "weir_t_peak_min": round(breach.t_peak / 60.0, 2),
        "peak_ratio_sph_over_weir": round(ps / pw, 3) if pw > 0 else None,
        "breach_formation_time_min": round(breach.geometry.t_form / 60.0, 2),
    }

    if head_m and head_m > 0:
        q_crit = (2.0 / 3.0) ** 1.5 * math.sqrt(G) * head_m ** 1.5
        q_sph_unit = float(iface.q_unit.max()) if iface.q_unit.size else 0.0
        ratio = q_sph_unit / q_crit if q_crit > 0 else float("nan")
        out["head_m"] = round(head_m, 2)
        out["critical_unit_discharge_m2s"] = round(q_crit, 1)
        out["sph_peak_unit_discharge_m2s"] = round(q_sph_unit, 1)
        out["ratio_to_critical_flow"] = round(ratio, 3) if np.isfinite(ratio) else None
        out["balance_pass"] = bool(np.isfinite(ratio) and 0.4 <= ratio <= 2.5)
        out["interpretation"] = (
            "SPH near-field discharge is within the expected band around the "
            "critical-flow ceiling for this head."
            if out["balance_pass"] else
            "SPH near-field discharge departs from the critical-flow ceiling; "
            "treat the particle result as indicative only and refine dp, "
            "artificial viscosity or the sound-speed coefficient.")
    else:
        out["balance_pass"] = False
        out["interpretation"] = "No head supplied; critical-flow check skipped."

    out["note"] = (
        "SPH models an instantaneous, fully-open breach and therefore peaks "
        "higher and far earlier than the weir closure, which grows the breach "
        "over its formation time. The two peaks are not expected to match; the "
        "critical-flow ratio is the meaningful gate.")
    return out


def build_comparison_table(runs: Dict[str, dict]) -> List[dict]:
    """Assemble the model-comparison table shown on the dashboard."""
    rows = []
    for key, r in runs.items():
        rows.append({
            "model": key,
            "description": r.get("description", ""),
            "peak_inflow_m3s": r.get("peak_inflow_m3s"),
            "inundated_km2": r.get("inundated_km2"),
            "max_depth_m": r.get("max_depth_m"),
            "max_velocity_ms": r.get("max_velocity_ms"),
            "volume_routed_mcm": r.get("volume_routed_mcm"),
            "wallclock_s": r.get("wallclock_s"),
            "cells_or_particles": r.get("resolution"),
        })
    return rows
