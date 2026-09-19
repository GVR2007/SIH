"""Weakly-compressible Smoothed Particle Hydrodynamics -- near-field breach jet.

This is the "SPH configuration -> Particle initialisation -> Run SPH near field"
branch of the technical approach.  It models a vertical (x-z) slice along the
breach centreline, taken from the *real* DEM long profile, and resolves what a
depth-averaged model structurally cannot:

  * the free-falling / plunging jet through the breach opening
  * vertical accelerations and the non-hydrostatic pressure field
  * the violent, fragmenting surge front in the first few hundred metres

Formulation (Monaghan 1992/1994; DualSPHysics conventions):

  kernel        cubic spline, 2D normalisation 10/(7*pi*h^2)
  continuity    drho_i/dt = sum_j m_j (v_i - v_j) . grad_i W_ij
  momentum      dv_i/dt   = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij) grad_i W_ij + g
  state         Tait,  p = B[(rho/rho0)^7 - 1],  B = rho0 c0^2 / 7
  viscosity     Monaghan artificial viscosity, alpha ~ 0.01-0.1
  boundary      dynamic boundary particles (DBC)
  integration   Verlet with periodic re-synchronisation
  density       Shepard filter re-initialisation every N steps

The near-field result is handed to the 2D far-field model through
`damburst.core.coupling`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                                      # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or not callable(a[0]) else a[0]

    prange = range

G = 9.80665
FLUID = 0
BOUND = 1


@dataclass
class SPHConfig:
    dp: float = 2.0                # initial particle spacing, m
    h_fac: float = 1.3             # smoothing length = h_fac * dp
    rho0: float = 1000.0
    gamma: float = 7.0
    coef_sound: float = 12.0       # c0 = coef_sound * sqrt(g * H_max)
    alpha_visc: float = 0.05
    eps_xsph: float = 0.5
    cfl: float = 0.20
    shepard_every: int = 30
    verlet_resync: int = 40
    t_end: float = 40.0
    out_every: float = 0.5         # snapshot interval, s
    # Hard budgets.  SPH timestep is set by sound speed and peak acceleration;
    # a 180 m reservoir column released against a steep bed can drive dt down by
    # orders of magnitude, so an unbounded loop can run effectively forever.
    # The near field is a diagnostic, not the deliverable -- bound it and report
    # how far it actually got.
    max_steps: int = 60_000
    max_wallclock_s: float = 180.0
    dt_floor: float = 1e-5         # below this the run is declared stalled


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, inline="always")
def _kernel_w(q, ad):
    if q >= 2.0:
        return 0.0
    if q < 1.0:
        return ad * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    t = 2.0 - q
    return ad * 0.25 * t * t * t


@njit(cache=True, fastmath=True, inline="always")
def _kernel_dwdq(q, ad):
    if q >= 2.0 or q <= 0.0:
        return 0.0
    if q < 1.0:
        return ad * (-3.0 * q + 2.25 * q * q)
    t = 2.0 - q
    return ad * (-0.75 * t * t)


# ---------------------------------------------------------------------------
# Uniform-grid neighbour search (counting sort)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _build_cells(x, z, cs, xmin, zmin, ncx, ncy, cell_start, cell_count, order):
    n = x.shape[0]
    cell_count[:] = 0
    cell_of = np.empty(n, dtype=np.int32)
    for i in range(n):
        cx = int((x[i] - xmin) / cs)
        cy = int((z[i] - zmin) / cs)
        if cx < 0:
            cx = 0
        elif cx >= ncx:
            cx = ncx - 1
        if cy < 0:
            cy = 0
        elif cy >= ncy:
            cy = ncy - 1
        c = cy * ncx + cx
        cell_of[i] = c
        cell_count[c] += 1
    s = 0
    for c in range(ncx * ncy):
        cell_start[c] = s
        s += cell_count[c]
    cell_start[ncx * ncy] = s
    fill = cell_start[:ncx * ncy].copy()
    for i in range(n):
        c = cell_of[i]
        order[fill[c]] = i
        fill[c] += 1


@njit(cache=True, fastmath=True, parallel=True)
def _interactions(x, z, vx, vz, rho, p, mass, ptype,
                  ax, az, drho, xsx, xsz,
                  hsm, ad, alpha, c0, rho0, cs, xmin, zmin, ncx, ncy,
                  cell_start, order):
    """Density rate, pressure acceleration, artificial viscosity, XSPH."""
    n = x.shape[0]
    h2 = hsm * hsm
    for i in prange(n):
        axi = 0.0
        azi = -G if ptype[i] == FLUID else 0.0
        dri = 0.0
        xsxi = 0.0
        xszi = 0.0

        xi = x[i]; zi = z[i]
        vxi = vx[i]; vzi = vz[i]
        rhoi = rho[i]; pi_ = p[i]
        pri = pi_ / (rhoi * rhoi)

        cx = int((xi - xmin) / cs)
        cy = int((zi - zmin) / cs)
        if cx < 0:
            cx = 0
        elif cx >= ncx:
            cx = ncx - 1
        if cy < 0:
            cy = 0
        elif cy >= ncy:
            cy = ncy - 1

        for gy in range(max(cy - 1, 0), min(cy + 2, ncy)):
            for gx in range(max(cx - 1, 0), min(cx + 2, ncx)):
                c = gy * ncx + gx
                for k in range(cell_start[c], cell_start[c + 1]):
                    j = order[k]
                    if j == i:
                        continue
                    dx = xi - x[j]
                    dz = zi - z[j]
                    r2 = dx * dx + dz * dz
                    if r2 >= 4.0 * h2 or r2 < 1e-12:
                        continue
                    r = math.sqrt(r2)
                    q = r / hsm
                    dwdr = _kernel_dwdq(q, ad) / hsm
                    gwx = dwdr * dx / r
                    gwz = dwdr * dz / r

                    dvx = vxi - vx[j]
                    dvz = vzi - vz[j]
                    mj = mass[j]

                    # continuity
                    dri += mj * (dvx * gwx + dvz * gwz)

                    # pressure + artificial viscosity
                    prj = p[j] / (rho[j] * rho[j])
                    visc = 0.0
                    vr = dvx * dx + dvz * dz
                    if vr < 0.0:
                        mu = hsm * vr / (r2 + 0.01 * h2)
                        rhobar = 0.5 * (rhoi + rho[j])
                        visc = -alpha * c0 * mu / rhobar
                    fac = -mj * (pri + prj + visc)
                    axi += fac * gwx
                    azi += fac * gwz

                    # XSPH velocity smoothing (fluid-fluid only)
                    if ptype[i] == FLUID and ptype[j] == FLUID:
                        w = _kernel_w(q, ad)
                        rhobar = 0.5 * (rhoi + rho[j])
                        xsxi -= mj * dvx * w / rhobar
                        xszi -= mj * dvz * w / rhobar

        ax[i] = axi
        az[i] = azi
        drho[i] = dri
        xsx[i] = xsxi
        xsz[i] = xszi


@njit(cache=True, fastmath=True, parallel=True)
def _shepard(x, z, rho, mass, ptype, rho_new,
             hsm, ad, cs, xmin, zmin, ncx, ncy, cell_start, order):
    """Shepard density re-initialisation -- removes pressure noise."""
    n = x.shape[0]
    h2 = hsm * hsm
    for i in prange(n):
        if ptype[i] != FLUID:
            rho_new[i] = rho[i]
            continue
        num = 0.0
        den = 0.0
        xi = x[i]; zi = z[i]
        cx = int((xi - xmin) / cs)
        cy = int((zi - zmin) / cs)
        if cx < 0:
            cx = 0
        elif cx >= ncx:
            cx = ncx - 1
        if cy < 0:
            cy = 0
        elif cy >= ncy:
            cy = ncy - 1
        for gy in range(max(cy - 1, 0), min(cy + 2, ncy)):
            for gx in range(max(cx - 1, 0), min(cx + 2, ncx)):
                c = gy * ncx + gx
                for k in range(cell_start[c], cell_start[c + 1]):
                    j = order[k]
                    dx = xi - x[j]
                    dz = zi - z[j]
                    r2 = dx * dx + dz * dz
                    if r2 >= 4.0 * h2:
                        continue
                    w = _kernel_w(math.sqrt(r2) / hsm, ad)
                    num += mass[j] * w
                    den += mass[j] * w / rho[j]
        rho_new[i] = num / den if den > 1e-12 else rho[i]


@njit(cache=True, fastmath=True, parallel=True)
def _eos(rho, p, ptype, b_tait, rho0, gamma):
    n = rho.shape[0]
    for i in prange(n):
        r = rho[i] / rho0
        val = b_tait * (r ** gamma - 1.0)
        # DBC boundary particles must never pull fluid in
        if val < 0.0 and ptype[i] == BOUND:
            val = 0.0
        p[i] = val


# ---------------------------------------------------------------------------
# Particle initialisation
# ---------------------------------------------------------------------------

@dataclass
class SPHResult:
    t: np.ndarray                     # transfer-section sample times
    depth: np.ndarray                 # flow depth at the transfer section, m
    u_mean: np.ndarray                # depth-averaged horizontal velocity, m/s
    q_unit: np.ndarray                # unit discharge, m2/s
    front_x: np.ndarray               # surge-front position, m
    snapshots: List[dict] = field(default_factory=list)
    config: Optional[SPHConfig] = None
    stats: Dict[str, object] = field(default_factory=dict)
    bed: Optional[np.ndarray] = None  # (x, z_bed) of the slice


def build_slice(bed_x: np.ndarray, bed_z: np.ndarray,
                water_level: float, dam_x: float,
                cfg: SPHConfig,
                breach_invert: Optional[float] = None,
                breach_open: bool = True,
                n_bound_layers: int = 3) -> dict:
    """Lay out fluid and boundary particles over a real bed profile.

    bed_x/bed_z come from the DEM thalweg long profile, so the near-field
    geometry is the actual valley, not an idealised flume.
    """
    dp = cfg.dp
    xmin, xmax = float(bed_x.min()), float(bed_x.max())

    def bed_at(xq):
        return np.interp(xq, bed_x, bed_z)

    # --- fluid: reservoir column behind the dam ---------------------------
    fx: List[float] = []
    fz: List[float] = []
    xs = np.arange(xmin + dp * 0.5, dam_x, dp)
    for xq in xs:
        zb = bed_at(xq)
        if water_level <= zb:
            continue
        zs = np.arange(zb + dp * 0.5, water_level, dp)
        for zq in zs:
            fx.append(xq)
            fz.append(zq)

    # --- boundary: bed + the residual dam body ----------------------------
    bx: List[float] = []
    bz: List[float] = []
    xs_all = np.arange(xmin, xmax + dp, dp)
    for xq in xs_all:
        zb = bed_at(xq)
        for L in range(n_bound_layers):
            bx.append(xq)
            bz.append(zb - L * dp)

    # residual barrier: the un-breached shoulders below the breach invert
    if not breach_open:
        crest = water_level + 5.0
        zs = np.arange(bed_at(dam_x), crest, dp)
        for zq in zs:
            for L in range(2):
                bx.append(dam_x + L * dp)
                bz.append(zq)
    elif breach_invert is not None:
        zb = bed_at(dam_x)
        if breach_invert > zb:
            zs = np.arange(zb, breach_invert, dp)
            for zq in zs:
                for L in range(2):
                    bx.append(dam_x + L * dp)
                    bz.append(zq)

    n_f = len(fx)
    n_b = len(bx)
    x = np.array(fx + bx, dtype=np.float64)
    z = np.array(fz + bz, dtype=np.float64)
    ptype = np.concatenate([np.full(n_f, FLUID, np.int32),
                            np.full(n_b, BOUND, np.int32)])
    return {"x": x, "z": z, "ptype": ptype, "n_fluid": n_f, "n_bound": n_b,
            "bed_x": bed_x, "bed_z": bed_z, "dam_x": dam_x,
            "water_level": water_level}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_sph(setup: dict, cfg: SPHConfig,
            transfer_x: Optional[float] = None,
            progress=None) -> SPHResult:
    """Integrate the near-field slice and sample the transfer section."""
    x = setup["x"].copy()
    z = setup["z"].copy()
    ptype = setup["ptype"]
    n = x.size
    n_fluid = setup["n_fluid"]

    dp = cfg.dp
    hsm = cfg.h_fac * dp
    ad = 10.0 / (7.0 * math.pi * hsm * hsm)
    mass = np.full(n, cfg.rho0 * dp * dp)
    rho = np.full(n, cfg.rho0)
    p = np.zeros(n)
    vx = np.zeros(n)
    vz = np.zeros(n)

    water_level = setup["water_level"]
    bed_z = setup["bed_z"]
    h_max = max(water_level - float(np.min(bed_z)), 1.0)
    c0 = cfg.coef_sound * math.sqrt(G * h_max)
    b_tait = cfg.rho0 * c0 * c0 / cfg.gamma

    if transfer_x is None:
        transfer_x = setup["dam_x"] + 12.0 * dp

    # grid
    cs = 2.0 * hsm
    xmin = float(x.min()) - 4 * dp
    zmin = float(z.min()) - 4 * dp
    xmax = float(x.max()) + 4 * dp
    zmax = float(z.max()) + 40.0
    ncx = max(int((xmax - xmin) / cs) + 1, 1)
    ncy = max(int((zmax - zmin) / cs) + 1, 1)
    cell_start = np.zeros(ncx * ncy + 1, dtype=np.int64)
    cell_count = np.zeros(ncx * ncy, dtype=np.int64)
    order = np.zeros(n, dtype=np.int64)

    ax = np.zeros(n); az = np.zeros(n); drho = np.zeros(n)
    xsx = np.zeros(n); xsz = np.zeros(n); rho_new = np.zeros(n)

    vx_prev = vx.copy(); vz_prev = vz.copy(); rho_prev = rho.copy()

    # hydrostatic initialisation of the reservoir column -> avoids a spurious
    # start-up transient while the column settles
    fl = ptype == FLUID
    depth0 = np.maximum(water_level - z, 0.0)
    rho[fl] = cfg.rho0 * (1.0 + cfg.rho0 * G * depth0[fl] / b_tait) ** (1.0 / cfg.gamma)
    rho_prev[:] = rho

    ts: List[float] = []
    dep: List[float] = []
    um: List[float] = []
    qs: List[float] = []
    fxs: List[float] = []
    snaps: List[dict] = []

    t = 0.0
    step = 0
    next_out = 0.0
    slab = 1.5 * dp
    bed_at_transfer = float(np.interp(transfer_x, setup["bed_x"], setup["bed_z"]))

    import time as _time
    t0 = _time.time()

    stop_reason = "reached t_end"
    while t < cfg.t_end:
        if step >= cfg.max_steps:
            stop_reason = f"step budget ({cfg.max_steps}) reached at t={t:.3f}s"
            break
        if _time.time() - t0 > cfg.max_wallclock_s:
            stop_reason = (f"wallclock budget ({cfg.max_wallclock_s:.0f}s) "
                           f"reached at t={t:.3f}s")
            break
        _build_cells(x, z, cs, xmin, zmin, ncx, ncy, cell_start, cell_count, order)
        _eos(rho, p, ptype, b_tait, cfg.rho0, cfg.gamma)
        _interactions(x, z, vx, vz, rho, p, mass, ptype,
                      ax, az, drho, xsx, xsz,
                      hsm, ad, cfg.alpha_visc, c0, cfg.rho0,
                      cs, xmin, zmin, ncx, ncy, cell_start, order)

        amax = float(np.sqrt(ax * ax + az * az).max()) if n else 0.0
        vmax = float(np.sqrt(vx * vx + vz * vz).max()) if n else 0.0
        dt_f = math.sqrt(hsm / max(amax, 1e-6))
        dt_cv = hsm / (c0 + vmax)
        dt = cfg.cfl * min(dt_f, dt_cv)
        if dt < cfg.dt_floor:
            stop_reason = (f"timestep collapsed to {dt:.2e}s at t={t:.3f}s "
                           "(pressure instability); near field truncated")
            break
        dt = max(min(dt, 0.05), cfg.dt_floor)

        # --- Verlet -------------------------------------------------------
        if step % cfg.verlet_resync == 0:
            vx_n = vx + dt * ax
            vz_n = vz + dt * az
            rho_n = rho + dt * drho
        else:
            vx_n = vx_prev + 2.0 * dt * ax
            vz_n = vz_prev + 2.0 * dt * az
            rho_n = rho_prev + 2.0 * dt * drho

        x_n = x + dt * (vx + cfg.eps_xsph * xsx) + 0.5 * dt * dt * ax
        z_n = z + dt * (vz + cfg.eps_xsph * xsz) + 0.5 * dt * dt * az

        # boundary particles are fixed in space with zero velocity (DBC)
        bnd = ptype == BOUND
        x_n[bnd] = x[bnd]
        z_n[bnd] = z[bnd]
        vx_n[bnd] = 0.0
        vz_n[bnd] = 0.0

        vx_prev, vz_prev, rho_prev = vx, vz, rho
        vx, vz, rho = vx_n, vz_n, rho_n
        x, z = x_n, z_n

        if cfg.shepard_every and step % cfg.shepard_every == 0 and step > 0:
            _build_cells(x, z, cs, xmin, zmin, ncx, ncy, cell_start, cell_count, order)
            _shepard(x, z, rho, mass, ptype, rho_new, hsm, ad, cs, xmin, zmin,
                     ncx, ncy, cell_start, order)
            rho = rho_new.copy()

        # discard particles that leave the domain
        out = (x < xmin + dp) | (x > xmax - dp) | (z < zmin)
        if out.any():
            keep = ~out
            x, z, vx, vz, rho, p = (a[keep] for a in (x, z, vx, vz, rho, p))
            mass = mass[keep]; ptype = ptype[keep]
            vx_prev = vx_prev[keep]; vz_prev = vz_prev[keep]; rho_prev = rho_prev[keep]
            n = x.size
            ax = np.zeros(n); az = np.zeros(n); drho = np.zeros(n)
            xsx = np.zeros(n); xsz = np.zeros(n); rho_new = np.zeros(n)
            order = np.zeros(n, dtype=np.int64)

        t += dt
        step += 1

        # --- transfer-section sampling ------------------------------------
        fl = ptype == FLUID
        sel = fl & (np.abs(x - transfer_x) < slab)
        if sel.any():
            zz = z[sel]
            surf = float(zz.max()) + 0.5 * dp
            d = max(surf - bed_at_transfer, 0.0)
            uu = float(np.average(vx[sel]))
        else:
            d = 0.0
            uu = 0.0
        ts.append(t); dep.append(d); um.append(uu); qs.append(d * uu)
        fxs.append(float(x[fl].max()) if fl.any() else transfer_x)

        if t >= next_out:
            snaps.append({
                "t": round(t, 3),
                "x": x[fl].astype(np.float32).tolist()[::3],
                "z": z[fl].astype(np.float32).tolist()[::3],
                "v": np.hypot(vx[fl], vz[fl]).astype(np.float32).tolist()[::3],
            })
            next_out += cfg.out_every

        if progress is not None and step % 100 == 0:
            progress(t, cfg.t_end, {"step": step, "particles": int(n), "dt": dt})

    elapsed = _time.time() - t0
    return SPHResult(
        t=np.array(ts), depth=np.array(dep), u_mean=np.array(um),
        q_unit=np.array(qs), front_x=np.array(fxs), snapshots=snaps, config=cfg,
        bed=np.column_stack([setup["bed_x"], setup["bed_z"]]),
        stats={
            "n_fluid_initial": int(n_fluid),
            "n_particles_final": int(n),
            "steps": step,
            "simulated_s": round(t, 3),
            "requested_s": cfg.t_end,
            "completed": bool(t >= cfg.t_end - 1e-6),
            "stop_reason": stop_reason,
            "wallclock_s": round(elapsed, 2),
            "dp_m": dp,
            "smoothing_length_m": round(hsm, 3),
            "sound_speed_ms": round(c0, 1),
            "transfer_x_m": round(float(transfer_x), 1),
            "bed_at_transfer_m": round(bed_at_transfer, 2),
            "peak_unit_discharge_m2s": round(float(np.max(qs)) if qs else 0.0, 2),
            "peak_depth_m": round(float(np.max(dep)) if dep else 0.0, 2),
            "peak_velocity_ms": round(float(np.max(um)) if um else 0.0, 2),
            "engine": "numba" if HAVE_NUMBA else "numpy-fallback",
        })


# ---------------------------------------------------------------------------
# Analytical check
# ---------------------------------------------------------------------------

def ritter_reference(h0: float, t: float, x: float) -> Tuple[float, float]:
    """Ritter solution at (x, t) for validating the SPH surge front."""
    c0 = math.sqrt(G * h0)
    if x <= -c0 * t:
        return h0, 0.0
    if x >= 2 * c0 * t:
        return 0.0, 0.0
    h = (2 * c0 - x / t) ** 2 / (9 * G)
    u = 2.0 / 3.0 * (x / t + c0)
    return h, u
