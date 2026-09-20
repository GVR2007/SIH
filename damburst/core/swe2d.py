"""2D depth-averaged shallow-water solver (the Delft3D-FM class far-field model).

Godunov finite-volume scheme on the structured DEM mesh:

  * HLL approximate Riemann solver at every cell face
  * optional second-order MUSCL reconstruction of the *water-surface elevation*
    (eta = z + h) with a minmod limiter, advanced by SSP-RK2.  Reconstructing
    eta rather than h is what keeps the scheme well balanced at second order:
    when the free surface is flat every slope is identically zero and the
    scheme collapses onto the first-order lake-at-rest solution.
  * Audusse et al. (2004) hydrostatic reconstruction at the face
  * Kurganov-Petrova depth desingularisation for wet/dry fronts
  * semi-implicit Manning friction
  * adaptive timestep from the CFL condition

This is the "Delft3D setup -> Delft3D mesh attribution -> Delft3D-FM" branch of
the technical approach, and the far-field half of the SPH coupling.

Governing equations (conservative form):

    d(h)/dt   + d(hu)/dx          + d(hv)/dy      = q_src
    d(hu)/dt  + d(hu^2+gh^2/2)/dx + d(huv)/dy     = -g h dz/dx - g n^2 u|U| h^(-1/3)
    d(hv)/dt  + d(huv)/dx + d(hv^2+gh^2/2)/dy     = -g h dz/dy - g n^2 v|U| h^(-1/3)

Verified against: lake-at-rest over irregular wet/dry bathymetry (machine
precision), the Ritter dry-bed dam-break analytical solution, and closed-basin
mass conservation.  See `tests/test_swe_benchmarks.py`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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

# Dry-cell threshold.  A newly wetted cell receives only dt*flux/dx of water on
# its first step -- typically ~1e-4 m.  If H_DRY sits above that, the update
# discards the momentum the face flux just injected and the front re-accelerates
# from rest every step, stalling the wave well behind the analytical position.
H_DRY = 1e-5

# Kurganov-Petrova desingularisation floor for q/h.  Must sit below H_DRY's
# physical scale for the same reason: too large a floor throttles the thin
# leading edge, where the true velocity is 2*sqrt(g*h0), not O(sqrt(g*h)).
H_VEL = 1e-4

H_THIN = 2e-2         # m, films thinner than this get the absolute speed cap
V_CAP = 50.0          # m/s, well above any physical dam-break velocity

# --- steep-terrain Froude limiter -------------------------------------------
#
# WHY THIS EXISTS.  The shallow-water equations assume a hydrostatic pressure
# distribution, which requires the bed slope to be small.  A Himalayan dam-break
# domain is not that: routing a 174 m release down a gorge with >1000 m of
# relief, the equations happily integrate water into free fall and report
# 120-130 m/s, because sqrt(2*g*400 m) really is ~88 m/s.  Those velocities are
# what the equations say; they are not what the physics says, because the
# equations stopped being valid several hundred metres upslope.  Left alone
# they propagate into the hazard rating, the d*v structural classes and the
# arrival times an evacuation plan would be built on.
#
# The limiter caps the Froude number ONLY on cells whose own bed slope already
# violates the hydrostatic assumption.  Flat-bed benchmarks (Ritter, lake at
# rest, closed-basin mass conservation) have zero slope everywhere, so they are
# untouched and the verification results are unchanged -- checked, not assumed.
#
# It is a modelling choice, not a silent correction: every run reports how many
# cells were limited and how often, in SWEResult.stats["froude_limiter"].
STEEP_SLOPE_DEG = 12.0   # beyond this the hydrostatic assumption is not tenable
FROUDE_MAX = 4.0         # supercritical, but not free-fall


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, inline="always")
def _vel(h, q):
    if h <= H_DRY:
        return 0.0
    return 2.0 * h * q / (h * h + max(h * h, H_VEL * H_VEL))


@njit(cache=True, fastmath=True, inline="always")
def _minmod(a, b):
    if a * b <= 0.0:
        return 0.0
    return a if abs(a) < abs(b) else b


@njit(cache=True, fastmath=True, parallel=True)
def _primitives(h, hu, hv, z, eta, u, v):
    ny, nx = h.shape
    for i in prange(ny):
        for j in range(nx):
            hh = h[i, j]
            eta[i, j] = z[i, j] + hh
            u[i, j] = _vel(hh, hu[i, j])
            v[i, j] = _vel(hh, hv[i, j])


@njit(cache=True, fastmath=True, parallel=True)
def _slopes_x(h, eta, u, v, se, su, sv, order):
    """Minmod slopes in x, with a positivity clamp on the eta slope."""
    ny, nx = h.shape
    for i in prange(ny):
        for j in range(nx):
            if order < 2 or j == 0 or j == nx - 1 or h[i, j] <= H_DRY:
                se[i, j] = 0.0
                su[i, j] = 0.0
                sv[i, j] = 0.0
                continue
            s = _minmod(eta[i, j] - eta[i, j - 1], eta[i, j + 1] - eta[i, j])
            # keep the reconstructed depth non-negative: |s|/2 <= h
            lim = 2.0 * h[i, j]
            if s > lim:
                s = lim
            elif s < -lim:
                s = -lim
            se[i, j] = s
            su[i, j] = _minmod(u[i, j] - u[i, j - 1], u[i, j + 1] - u[i, j])
            sv[i, j] = _minmod(v[i, j] - v[i, j - 1], v[i, j + 1] - v[i, j])


@njit(cache=True, fastmath=True, parallel=True)
def _slopes_y(h, eta, u, v, se, su, sv, order):
    ny, nx = h.shape
    for i in prange(ny):
        for j in range(nx):
            if order < 2 or i == 0 or i == ny - 1 or h[i, j] <= H_DRY:
                se[i, j] = 0.0
                su[i, j] = 0.0
                sv[i, j] = 0.0
                continue
            s = _minmod(eta[i, j] - eta[i - 1, j], eta[i + 1, j] - eta[i, j])
            lim = 2.0 * h[i, j]
            if s > lim:
                s = lim
            elif s < -lim:
                s = -lim
            se[i, j] = s
            su[i, j] = _minmod(u[i, j] - u[i - 1, j], u[i + 1, j] - u[i, j])
            sv[i, j] = _minmod(v[i, j] - v[i - 1, j], v[i + 1, j] - v[i, j])


# ---------------------------------------------------------------------------
# HLL fluxes with hydrostatic reconstruction
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def _flux_x(z, eta, u, v, se, su, sv, fh, fqx, fqy, wbL, wbR):
    ny, nx = z.shape
    for i in prange(ny):
        for j in range(nx - 1):
            etaL = eta[i, j] + 0.5 * se[i, j]
            etaR = eta[i, j + 1] - 0.5 * se[i, j + 1]
            uL = u[i, j] + 0.5 * su[i, j]
            uR = u[i, j + 1] - 0.5 * su[i, j + 1]
            vL = v[i, j] + 0.5 * sv[i, j]
            vR = v[i, j + 1] - 0.5 * sv[i, j + 1]

            zL = z[i, j]
            zR = z[i, j + 1]
            zf = zL if zL > zR else zR

            hLs = etaL - zf
            hRs = etaR - zf
            if hLs < 0.0:
                hLs = 0.0
            if hRs < 0.0:
                hRs = 0.0

            wbL[i, j] = hLs * hLs
            wbR[i, j] = hRs * hRs

            if hLs <= H_DRY and hRs <= H_DRY:
                fh[i, j] = 0.0
                fqx[i, j] = 0.0
                fqy[i, j] = 0.0
                continue

            cL = math.sqrt(G * hLs)
            cR = math.sqrt(G * hRs)
            if hLs <= H_DRY:
                sl = uR - 2.0 * cR
                sr = uR + cR
            elif hRs <= H_DRY:
                sl = uL - cL
                sr = uL + 2.0 * cL
            else:
                rl = math.sqrt(hLs)
                rr = math.sqrt(hRs)
                ubar = (uL * rl + uR * rr) / (rl + rr)
                cbar = math.sqrt(0.5 * G * (hLs + hRs))
                sl = min(uL - cL, ubar - cbar)
                sr = max(uR + cR, ubar + cbar)

            qL = hLs * uL
            qR = hRs * uR
            FhL = qL
            FhR = qR
            FqxL = qL * uL + 0.5 * G * hLs * hLs
            FqxR = qR * uR + 0.5 * G * hRs * hRs
            FqyL = qL * vL
            FqyR = qR * vR

            if sl >= 0.0:
                fh[i, j] = FhL
                fqx[i, j] = FqxL
                fqy[i, j] = FqyL
            elif sr <= 0.0:
                fh[i, j] = FhR
                fqx[i, j] = FqxR
                fqy[i, j] = FqyR
            else:
                inv = 1.0 / (sr - sl)
                fh[i, j] = (sr * FhL - sl * FhR + sl * sr * (hRs - hLs)) * inv
                fqx[i, j] = (sr * FqxL - sl * FqxR +
                             sl * sr * (hRs * uR - hLs * uL)) * inv
                fqy[i, j] = (sr * FqyL - sl * FqyR +
                             sl * sr * (hRs * vR - hLs * vL)) * inv


@njit(cache=True, fastmath=True, parallel=True)
def _flux_y(z, eta, u, v, se, su, sv, gh, gqx, gqy, wbB, wbT):
    ny, nx = z.shape
    for i in prange(ny - 1):
        for j in range(nx):
            etaB = eta[i, j] + 0.5 * se[i, j]
            etaT = eta[i + 1, j] - 0.5 * se[i + 1, j]
            uB = u[i, j] + 0.5 * su[i, j]
            uT = u[i + 1, j] - 0.5 * su[i + 1, j]
            vB = v[i, j] + 0.5 * sv[i, j]
            vT = v[i + 1, j] - 0.5 * sv[i + 1, j]

            zB = z[i, j]
            zT = z[i + 1, j]
            zf = zB if zB > zT else zT

            hBs = etaB - zf
            hTs = etaT - zf
            if hBs < 0.0:
                hBs = 0.0
            if hTs < 0.0:
                hTs = 0.0

            wbB[i, j] = hBs * hBs
            wbT[i, j] = hTs * hTs

            if hBs <= H_DRY and hTs <= H_DRY:
                gh[i, j] = 0.0
                gqx[i, j] = 0.0
                gqy[i, j] = 0.0
                continue

            cB = math.sqrt(G * hBs)
            cT = math.sqrt(G * hTs)
            if hBs <= H_DRY:
                sl = vT - 2.0 * cT
                sr = vT + cT
            elif hTs <= H_DRY:
                sl = vB - cB
                sr = vB + 2.0 * cB
            else:
                rb = math.sqrt(hBs)
                rt = math.sqrt(hTs)
                vbar = (vB * rb + vT * rt) / (rb + rt)
                cbar = math.sqrt(0.5 * G * (hBs + hTs))
                sl = min(vB - cB, vbar - cbar)
                sr = max(vT + cT, vbar + cbar)

            qB = hBs * vB
            qT = hTs * vT
            FhB = qB
            FhT = qT
            FqyB = qB * vB + 0.5 * G * hBs * hBs
            FqyT = qT * vT + 0.5 * G * hTs * hTs
            FqxB = qB * uB
            FqxT = qT * uT

            if sl >= 0.0:
                gh[i, j] = FhB
                gqx[i, j] = FqxB
                gqy[i, j] = FqyB
            elif sr <= 0.0:
                gh[i, j] = FhT
                gqx[i, j] = FqxT
                gqy[i, j] = FqyT
            else:
                inv = 1.0 / (sr - sl)
                gh[i, j] = (sr * FhB - sl * FhT + sl * sr * (hTs - hBs)) * inv
                gqx[i, j] = (sr * FqxB - sl * FqxT +
                             sl * sr * (hTs * uT - hBs * uB)) * inv
                gqy[i, j] = (sr * FqyB - sl * FqyT +
                             sl * sr * (hTs * vT - hBs * vB)) * inv


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def _residual(h, z, eta, sex, sey, fh, fqx, fqy, gh, gqx, gqy,
              wbXL, wbXR, wbYB, wbYT, dx, dy, rh, rhu, rhv):
    """dU/dt from face fluxes plus the well-balanced bed-slope source.

    The source is the Audusse & Bristeau (2005) second-order form

        S = g/2 [ (h*_{j+1/2,L})^2 - (h_j^+)^2 ]
          + g/2 [ (h_j^-)^2 - (h*_{j-1/2,R})^2 ]

    where h_j^+/h_j^- are the MUSCL-reconstructed depths at the right/left edge
    of cell j *before* the hydrostatic face correction.  The bed is kept
    piecewise constant within a cell, so the extra centred term
    -g*hbar*(z_j^+ - z_j^-) vanishes.

    Dropping the (h_j^+)^2 / (h_j^-)^2 pair is only legal at first order, where
    the reconstruction is constant and they cancel against h*_L / h*_R.  With a
    sloped reconstruction they do not cancel, and on a flat bed the omission
    leaves a spurious g*h*dh/dx body force that wrecks the solution.
    """
    ny, nx = h.shape
    for i in prange(ny):
        for j in range(nx):
            # Reflective ghost cell at the domain wall: zero mass flux but a
            # non-zero hydrostatic momentum flux g*h^2/2.  Omitting it leaves an
            # unbalanced pressure gradient and lake-at-rest decays from the edges.
            wall = 0.5 * G * h[i, j] * h[i, j]

            fw = fh[i, j - 1] if j > 0 else 0.0
            fe = fh[i, j] if j < nx - 1 else 0.0
            gs = gh[i - 1, j] if i > 0 else 0.0
            gn = gh[i, j] if i < ny - 1 else 0.0

            fqxw = fqx[i, j - 1] if j > 0 else wall
            fqxe = fqx[i, j] if j < nx - 1 else wall
            gqxs = gqx[i - 1, j] if i > 0 else 0.0
            gqxn = gqx[i, j] if i < ny - 1 else 0.0

            fqyw = fqy[i, j - 1] if j > 0 else 0.0
            fqye = fqy[i, j] if j < nx - 1 else 0.0
            gqys = gqy[i - 1, j] if i > 0 else wall
            gqyn = gqy[i, j] if i < ny - 1 else wall

            sxr = wbXL[i, j] if j < nx - 1 else h[i, j] * h[i, j]
            sxl = wbXR[i, j - 1] if j > 0 else h[i, j] * h[i, j]
            syt = wbYB[i, j] if i < ny - 1 else h[i, j] * h[i, j]
            syb = wbYT[i - 1, j] if i > 0 else h[i, j] * h[i, j]

            # reconstructed cell-edge depths (bed piecewise constant)
            zc = z[i, j]
            hxp = eta[i, j] + 0.5 * sex[i, j] - zc
            hxm = eta[i, j] - 0.5 * sex[i, j] - zc
            hyp = eta[i, j] + 0.5 * sey[i, j] - zc
            hym = eta[i, j] - 0.5 * sey[i, j] - zc
            if hxp < 0.0:
                hxp = 0.0
            if hxm < 0.0:
                hxm = 0.0
            if hyp < 0.0:
                hyp = 0.0
            if hym < 0.0:
                hym = 0.0

            rh[i, j] = -((fe - fw) / dx + (gn - gs) / dy)
            rhu[i, j] = (-((fqxe - fqxw) / dx + (gqxn - gqxs) / dy)
                         + 0.5 * G * (sxr - hxp * hxp + hxm * hxm - sxl) / dx)
            rhv[i, j] = (-((fqye - fqyw) / dx + (gqyn - gqys) / dy)
                         + 0.5 * G * (syt - hyp * hyp + hym * hym - syb) / dy)


@njit(cache=True, fastmath=True, parallel=True)
def _stage(h0, hu0, hv0, h1, hu1, hv1, rh, rhu, rhv, a0, a1, dt,
           ho, huo, hvo):
    """ho = a0*U0 + a1*(U1 + dt*R)   (SSP-RK2 convex combination)."""
    ny, nx = h0.shape
    for i in prange(ny):
        for j in range(nx):
            hh = a0 * h0[i, j] + a1 * (h1[i, j] + dt * rh[i, j])
            if hh < 0.0:
                hh = 0.0
            ho[i, j] = hh
            if hh <= H_DRY:
                huo[i, j] = 0.0
                hvo[i, j] = 0.0
            else:
                huo[i, j] = a0 * hu0[i, j] + a1 * (hu1[i, j] + dt * rhu[i, j])
                hvo[i, j] = a0 * hv0[i, j] + a1 * (hv1[i, j] + dt * rhv[i, j])


@njit(cache=True, fastmath=True, parallel=True)
def _finalise(h, hu, hv, n2, dt, open_edges, steep, fr_max, limited):
    """Semi-implicit Manning friction, thin-film cap, Froude limiter, edges.

    `steep` is a per-cell flag: 1 where the bed slope exceeds the angle at
    which the hydrostatic (shallow-water) assumption fails, 0 elsewhere.  On
    those cells only, the velocity is capped at `fr_max * sqrt(g*h)`.  Cells
    with a flat bed -- which is every cell in the analytical benchmarks -- never
    enter that branch.  `limited` accumulates the count of capped cell-steps so
    the run can report how much it had to intervene.
    """
    ny, nx = h.shape
    for i in prange(ny):
        for j in range(nx):
            hh = h[i, j]
            if hh <= H_DRY:
                h[i, j] = hh
                hu[i, j] = 0.0
                hv[i, j] = 0.0
                continue
            qx = hu[i, j]
            qy = hv[i, j]
            u = qx / hh
            v = qy / hh
            spd = math.sqrt(u * u + v * v)
            if spd > 1e-9:
                cf = G * n2[i, j] * spd / (hh ** (4.0 / 3.0))
                d = 1.0 + dt * cf
                qx /= d
                qy /= d
            if hh < H_THIN:
                u = qx / hh
                v = qy / hh
                spd = math.sqrt(u * u + v * v)
                if spd > V_CAP:
                    sc = V_CAP / spd
                    qx *= sc
                    qy *= sc
            if steep[i, j] and fr_max > 0.0:
                u = qx / hh
                v = qy / hh
                spd = math.sqrt(u * u + v * v)
                vmax = fr_max * math.sqrt(G * hh)
                if spd > vmax:
                    sc = vmax / spd
                    qx *= sc
                    qy *= sc
                    limited[i] += 1
            hu[i, j] = qx
            hv[i, j] = qy

    if open_edges:
        for i in prange(ny):
            h[i, 0] = h[i, 1]
            hu[i, 0] = hu[i, 1] if hu[i, 1] < 0.0 else 0.0
            hv[i, 0] = hv[i, 1]
            h[i, nx - 1] = h[i, nx - 2]
            hu[i, nx - 1] = hu[i, nx - 2] if hu[i, nx - 2] > 0.0 else 0.0
            hv[i, nx - 1] = hv[i, nx - 2]
        for j in prange(nx):
            h[0, j] = h[1, j]
            hv[0, j] = hv[1, j] if hv[1, j] > 0.0 else 0.0
            hu[0, j] = hu[1, j]
            h[ny - 1, j] = h[ny - 2, j]
            hv[ny - 1, j] = hv[ny - 2, j] if hv[ny - 2, j] < 0.0 else 0.0
            hu[ny - 1, j] = hu[ny - 2, j]


@njit(cache=True, fastmath=True, parallel=True)
def _rowmax_wavespeed(h, hu, hv, out):
    """Per-row max of |U|+sqrt(gh); reduced on the host (prange has no
    safe conditional-max reduction pattern)."""
    ny, nx = h.shape
    for i in prange(ny):
        loc = 1e-9
        for j in range(nx):
            hh = h[i, j]
            if hh <= H_DRY:
                continue
            u = _vel(hh, hu[i, j])
            v = _vel(hh, hv[i, j])
            s = math.sqrt(u * u + v * v) + math.sqrt(G * hh)
            if s > loc:
                loc = s
        out[i] = loc


@njit(cache=True, fastmath=True, parallel=True)
def _accumulate(h, hu, hv, hmax, vmax, arrival, duration, t, dt, h_thresh):
    ny, nx = h.shape
    for i in prange(ny):
        for j in range(nx):
            hh = h[i, j]
            if hh > hmax[i, j]:
                hmax[i, j] = hh
            if hh > H_DRY:
                u = _vel(hh, hu[i, j])
                v = _vel(hh, hv[i, j])
                s = math.sqrt(u * u + v * v)
                if s > vmax[i, j]:
                    vmax[i, j] = s
            if hh >= h_thresh:
                if arrival[i, j] < 0.0:
                    arrival[i, j] = t
                duration[i, j] += dt


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _peak_velocity_context(v_max: np.ndarray, h_max: np.ndarray,
                           slope_deg: np.ndarray, dx: float) -> dict:
    """Where the reported maximum velocity actually sits, and on what terrain.

    A single cell sets `max_velocity_ms`.  Without knowing its depth, its bed
    slope and its Froude number, that number cannot be judged -- and it is the
    number a reader will quote.  So report its context rather than the bare
    scalar.
    """
    if not np.isfinite(v_max).any() or v_max.max() <= 0:
        return {}
    k = int(np.argmax(v_max))
    i, j = divmod(k, v_max.shape[1])
    h = float(h_max[i, j])
    v = float(v_max[i, j])
    fr = v / math.sqrt(G * h) if h > 1e-6 else float("inf")
    wet = h_max > 0.05
    return {
        "row_col": [i, j],
        "depth_there_m": round(h, 3),
        "velocity_ms": round(v, 3),
        "bed_slope_deg": round(float(slope_deg[i, j]), 2),
        "froude_number": round(fr, 2) if np.isfinite(fr) else None,
        "p99_9_velocity_ms": round(float(np.percentile(v_max[wet], 99.9)), 3)
            if wet.any() else 0.0,
        "median_velocity_ms": round(float(np.median(v_max[wet])), 3)
            if wet.any() else 0.0,
        "note": ("The maximum is a single cell. Quote the percentiles for "
                 "anything operational; check bed_slope_deg before trusting "
                 "the maximum at all."),
    }


@dataclass
class PointSource:
    """Volumetric inflow injected over a set of cells (the breach opening)."""
    rows: np.ndarray
    cols: np.ndarray
    hydrograph: Callable[[float], float]     # Q(t), m3/s
    name: str = "breach"

    #: Velocity coefficient for water entering through the breach, applied to
    #: the free-jet value sqrt(2*g*h).  A frictionless free jet would enter at
    #: the full torricellian speed; a breach jet impinging on a plunge pool and
    #: spreading across the source patch arrives slower, and 0.6 is the same
    #: order as the standard orifice discharge coefficient.  It is a MODELLING
    #: CHOICE, exposed here so it can be varied, and it only sets the momentum
    #: of the incoming water -- the mass is fixed by the hydrograph.
    jet_velocity_coefficient: float = 0.6

    def apply(self, h, hu, hv, cell_area, dt, t, direction=None) -> float:
        q = self.hydrograph(t)
        if q <= 0.0:
            return 0.0
        n = len(self.rows)
        dh = q * dt / (cell_area * n)
        h[self.rows, self.cols] += dh
        if direction is not None:
            # Momentum MIXING, not accumulation: the incoming slab dh arrives
            # with speed `spd`, so the cell's new specific discharge is the
            # mass-weighted blend of what was there and what arrived.  This is
            # already bounded by max(u_old, spd), which is why the source patch
            # is not the origin of the extreme velocities.
            hh = h[self.rows, self.cols]
            spd = np.sqrt(2.0 * G * np.maximum(hh, 0.0)) \
                * self.jet_velocity_coefficient
            hu[self.rows, self.cols] += dh * spd * direction[0]
            hv[self.rows, self.cols] += dh * spd * direction[1]
        return q * dt


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SWEResult:
    h_max: np.ndarray
    v_max: np.ndarray
    arrival_s: np.ndarray            # -1 where never inundated
    duration_s: np.ndarray
    frame_times: List[float] = field(default_factory=list)
    frame_dir: Optional[Path] = None
    gauges: Dict[str, dict] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)

    def wet_mask(self, threshold: float = 0.05) -> np.ndarray:
        return self.h_max > threshold


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class SWE2D:
    """Structured-grid shallow-water model over a conditioned DEM."""

    def __init__(self, z: np.ndarray, dx: float, dy: float,
                 manning: np.ndarray, open_edges: bool = True,
                 order: int = 2,
                 froude_max: float = FROUDE_MAX,
                 steep_slope_deg: float = STEEP_SLOPE_DEG):
        self.z = np.ascontiguousarray(z, dtype=np.float64)
        self.dx = float(dx)
        self.dy = float(dy)
        self.n2 = np.ascontiguousarray(np.asarray(manning) ** 2, dtype=np.float64)
        self.open_edges = bool(open_edges)
        self.order = int(order)
        ny, nx = self.z.shape

        # Cells where the bed is too steep for the hydrostatic assumption.
        # Flat-bed benchmarks produce an all-zero mask, so they run exactly as
        # before the limiter existed.
        self.froude_max = float(froude_max)
        self.steep_slope_deg = float(steep_slope_deg)
        gy, gx = np.gradient(self.z, self.dy, self.dx)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        self.steep = np.ascontiguousarray(
            (slope_deg > self.steep_slope_deg).astype(np.uint8))
        self.slope_deg = slope_deg

        z2 = (ny, nx)
        self.h = np.zeros(z2)
        self.hu = np.zeros(z2)
        self.hv = np.zeros(z2)

        self._eta = np.zeros(z2)
        self._u = np.zeros(z2)
        self._v = np.zeros(z2)
        self._sex = np.zeros(z2); self._sux = np.zeros(z2); self._svx = np.zeros(z2)
        self._sey = np.zeros(z2); self._suy = np.zeros(z2); self._svy = np.zeros(z2)
        self._rh = np.zeros(z2); self._rhu = np.zeros(z2); self._rhv = np.zeros(z2)
        self._h1 = np.zeros(z2); self._hu1 = np.zeros(z2); self._hv1 = np.zeros(z2)

        self._fh = np.zeros((ny, nx - 1)); self._fqx = np.zeros((ny, nx - 1))
        self._fqy = np.zeros((ny, nx - 1))
        self._wbXL = np.zeros((ny, nx - 1)); self._wbXR = np.zeros((ny, nx - 1))
        self._gh = np.zeros((ny - 1, nx)); self._gqx = np.zeros((ny - 1, nx))
        self._gqy = np.zeros((ny - 1, nx))
        self._wbYB = np.zeros((ny - 1, nx)); self._wbYT = np.zeros((ny - 1, nx))
        self._rowmax = np.zeros(ny)

    # -- initial conditions ----------------------------------------------
    def set_still_water(self, level: float, mask: Optional[np.ndarray] = None):
        d = level - self.z
        if mask is not None:
            d = np.where(mask, d, 0.0)
        self.h[:] = np.maximum(d, 0.0)
        self.hu[:] = 0.0
        self.hv[:] = 0.0

    def add_baseflow(self, channel_mask: np.ndarray, depth: float = 1.0):
        """Pre-wet the channel to an antecedent low-flow depth.

        NOT called by the default pipeline: a dam-break run starts from a dry
        downstream bed by design, which is the conservative initial condition
        for arrival time. Needed for a river-blockage scenario where the reach
        is already carrying flow.
        """
        self.h[channel_mask] = np.maximum(self.h[channel_mask], depth)

    # -- one residual evaluation -----------------------------------------
    def _rhs(self, h, hu, hv):
        _primitives(h, hu, hv, self.z, self._eta, self._u, self._v)
        _slopes_x(h, self._eta, self._u, self._v,
                  self._sex, self._sux, self._svx, self.order)
        _slopes_y(h, self._eta, self._u, self._v,
                  self._sey, self._suy, self._svy, self.order)
        _flux_x(self.z, self._eta, self._u, self._v,
                self._sex, self._sux, self._svx,
                self._fh, self._fqx, self._fqy, self._wbXL, self._wbXR)
        _flux_y(self.z, self._eta, self._u, self._v,
                self._sey, self._suy, self._svy,
                self._gh, self._gqx, self._gqy, self._wbYB, self._wbYT)
        _residual(h, self.z, self._eta, self._sex, self._sey,
                  self._fh, self._fqx, self._fqy,
                  self._gh, self._gqx, self._gqy,
                  self._wbXL, self._wbXR, self._wbYB, self._wbYT,
                  self.dx, self.dy, self._rh, self._rhu, self._rhv)

    # -- main loop --------------------------------------------------------
    def run(self,
            t_end: float,
            sources: Sequence[PointSource] = (),
            source_direction: Optional[Tuple[float, float]] = None,
            cfl: float = 0.40,
            dt_max: float = 20.0,
            dt_min: float = 1e-4,
            n_frames: int = 60,
            frame_dir: Optional[Path] = None,
            gauge_cells: Optional[Dict[str, Tuple[int, int]]] = None,
            h_arrival: float = 0.30,
            progress: Optional[Callable[[float, float, dict], None]] = None,
            max_steps: int = 2_000_000) -> SWEResult:

        ny, nx = self.z.shape
        cell_area = self.dx * self.dy

        h_max = np.zeros((ny, nx))
        v_max = np.zeros((ny, nx))
        arrival = np.full((ny, nx), -1.0)
        duration = np.zeros((ny, nx))

        frame_times: List[float] = []
        if frame_dir is not None:
            frame_dir = Path(frame_dir)
            frame_dir.mkdir(parents=True, exist_ok=True)
        frame_every = t_end / max(n_frames, 1) if n_frames else float("inf")
        next_frame = 0.0

        gauges: Dict[str, dict] = {}
        if gauge_cells:
            for gname in gauge_cells:
                gauges[gname] = {"t": [], "h": [], "v": [], "eta": []}

        t = 0.0
        step = 0
        injected = 0.0
        t0 = time.time()
        vol0 = float(self.h.sum() * cell_area)
        limited = np.zeros(ny, dtype=np.int64)     # Froude-limiter counter

        while t < t_end and step < max_steps:
            _rowmax_wavespeed(self.h, self.hu, self.hv, self._rowmax)
            smax = float(self._rowmax.max())
            if not np.isfinite(smax):
                raise FloatingPointError(
                    f"SWE2D diverged at t={t:.3f}s step {step}: non-finite "
                    "wave speed. Lower cfl or dt_max.")
            dt = cfl * min(self.dx, self.dy) / max(smax, 1e-9)
            dt = min(dt, dt_max, t_end - t)
            dt = max(dt, dt_min)

            if frame_dir is not None and n_frames and t >= next_frame - 1e-9:
                self._write_frame(frame_dir, len(frame_times))
                frame_times.append(t)
                next_frame += frame_every

            # --- SSP-RK2 ---------------------------------------------------
            self._rhs(self.h, self.hu, self.hv)
            _stage(self.h, self.hu, self.hv, self.h, self.hu, self.hv,
                   self._rh, self._rhu, self._rhv, 0.0, 1.0, dt,
                   self._h1, self._hu1, self._hv1)

            if self.order >= 2:
                self._rhs(self._h1, self._hu1, self._hv1)
                _stage(self.h, self.hu, self.hv, self._h1, self._hu1, self._hv1,
                       self._rh, self._rhu, self._rhv, 0.5, 0.5, dt,
                       self.h, self.hu, self.hv)
            else:
                self.h[:] = self._h1
                self.hu[:] = self._hu1
                self.hv[:] = self._hv1

            _finalise(self.h, self.hu, self.hv, self.n2, dt, self.open_edges,
                      self.steep, self.froude_max, limited)

            for src in sources:
                injected += src.apply(self.h, self.hu, self.hv, cell_area, dt,
                                      t, source_direction)

            t += dt
            step += 1
            _accumulate(self.h, self.hu, self.hv, h_max, v_max, arrival,
                        duration, t, dt, h_arrival)

            if gauge_cells and step % 5 == 0:
                for gname, (gi, gj) in gauge_cells.items():
                    hh = float(self.h[gi, gj])
                    uu = float(self.hu[gi, gj] / hh) if hh > H_DRY else 0.0
                    vv = float(self.hv[gi, gj] / hh) if hh > H_DRY else 0.0
                    g = gauges[gname]
                    g["t"].append(t); g["h"].append(hh)
                    g["v"].append(math.hypot(uu, vv))
                    g["eta"].append(float(self.z[gi, gj]) + hh)

            if progress is not None and step % 50 == 0:
                progress(t, t_end, {"step": step, "dt": dt,
                                    "wet_cells": int((self.h > H_DRY).sum()),
                                    "max_depth": float(self.h.max())})

        vol_end = float(self.h.sum() * cell_area)
        elapsed = time.time() - t0
        stats = {
            "steps": step,
            "order": self.order,
            "sim_time_s": round(t, 2),
            "wallclock_s": round(elapsed, 2),
            "cells": int(self.z.size),
            "cell_updates_per_s": int(step * self.z.size / max(elapsed, 1e-6)),
            "volume_initial_mcm": round(vol0 / 1e6, 4),
            "volume_injected_mcm": round(injected / 1e6, 4),
            "volume_final_mcm": round(vol_end / 1e6, 4),
            "volume_left_domain_mcm": round((vol0 + injected - vol_end) / 1e6, 4),
            "max_depth_m": round(float(h_max.max()), 3),
            "max_velocity_ms": round(float(v_max.max()), 3),
            "inundated_km2": round(float((h_max > 0.05).sum() * cell_area) / 1e6, 4),
            "engine": "numba" if HAVE_NUMBA else "numpy-fallback",
            "froude_limiter": {
                "enabled": self.froude_max > 0.0,
                "froude_max": self.froude_max,
                "steep_slope_deg": self.steep_slope_deg,
                "steep_cells": int(self.steep.sum()),
                "steep_cell_fraction": round(float(self.steep.mean()), 4),
                "limited_cell_steps": int(limited.sum()),
                "limited_per_step": round(float(limited.sum()) / max(step, 1), 2),
                "note": ("Velocity capped at froude_max*sqrt(g*h) on cells "
                         "whose bed slope exceeds steep_slope_deg, where the "
                         "shallow-water hydrostatic assumption does not hold. "
                         "Flat-bed benchmarks have no steep cells and are "
                         "unaffected."),
            },
            "peak_velocity_context": _peak_velocity_context(
                v_max, h_max, self.slope_deg, self.dx),
        }
        return SWEResult(h_max=h_max, v_max=v_max, arrival_s=arrival,
                         duration_s=duration, frame_times=frame_times,
                         frame_dir=frame_dir, gauges=gauges, stats=stats)

    def _write_frame(self, frame_dir: Path, idx: int):
        hh = self.h.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(self.h > H_DRY, self.hu / np.maximum(self.h, H_DRY), 0.0)
            v = np.where(self.h > H_DRY, self.hv / np.maximum(self.h, H_DRY), 0.0)
        np.savez_compressed(frame_dir / f"frame_{idx:04d}.npz",
                            h=hh, speed=np.hypot(u, v).astype(np.float32))
