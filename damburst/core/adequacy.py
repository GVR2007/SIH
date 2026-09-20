"""Model adequacy and automated adaptive model selection.

WHAT THIS IS FOR
----------------
A dam-break framework that owns two solvers has to answer a question most of
them dodge: *which one is valid where?*  The usual answer is to run both and
put the numbers side by side, which tells a reader that the models disagree
but not which one to believe.  Worse, it makes the handover point a free
parameter -- in this codebase it was previously a multiple of the SPH particle
spacing, i.e. a numerical setting dressed up as a physical overlap zone.

This module computes the answer from the flow instead.  It evaluates, cell by
cell, how badly the assumptions behind the depth-averaged shallow-water
equations are violated, and it uses that field to:

  1. decide whether the non-hydrostatic near-field model is needed at all;
  2. place the SPH -> SWE transfer section where the flow actually becomes
     depth-averageable, rather than at an arbitrary offset;
  3. report, after the run, what fraction of the inundated area was simulated
     outside the validity of the equations used -- which is a statement about
     the result's trustworthiness that a single hazard raster cannot make.

THE CRITERION
-------------
The shallow-water equations follow from the Navier-Stokes equations under one
substantive assumption: that the pressure is HYDROSTATIC,

    p(x, z) = rho * g * (eta - z)

which requires the vertical acceleration Dw/Dt to be negligible against g.
Three distinct things break it, and the index tracks each separately rather
than collapsing them into one opaque number.

**1. Bed slope.**  The derivation assumes |dz/dx| << 1.  At 12 degrees the
neglected terms are already ~5% of gravity; in a Himalayan gorge the slope
reaches 40 degrees and the equations integrate water into free fall.

    N_bed = |grad z| / tan(theta_c)

**2. Streamline curvature -> vertical acceleration.**  Where the bed curves,
the flow must accelerate vertically to follow it.  With w ~ u * dz/dx,

    Dw/Dt ~ u^2 * d2z/dx2

so the ratio of that acceleration to gravity is a direct measure of the
non-hydrostatic pressure defect:

    N_curv = u^2 * |kappa_bed| / g

This is the term that fires at a breach jet, at a knickpoint and at the lip of
a plunge pool -- exactly the places a depth-averaged model cannot represent.

**3. Free-surface steepness.**  The long-wave assumption also requires the
surface to vary slowly compared with the depth.  A steep surface gradient means
the horizontal length scale has collapsed towards the depth scale and the
Boussinesq dispersion parameter (h/L)^2 is no longer small:

    N_surf = |grad eta| / s_c

The composite index is the worst offender,

    NHI = max(N_bed, N_curv, N_surf)

with NHI < 1 meaning "the depth-averaged equations are defensible here" and
NHI >= 1 meaning "they are not, and a non-hydrostatic model is required".

The Froude number is carried alongside but deliberately NOT folded in:
supercritical flow is perfectly well described by the shallow-water equations.
It matters for interpretation (hydraulic jumps are locally non-hydrostatic) but
it is not by itself an adequacy failure.

REFERENCES
----------
* Stoker, J.J. (1957) Water Waves, ch. 10 -- the long-wave derivation and the
  conditions under which the hydrostatic assumption holds.
* Peregrine, D.H. (1967) Long waves on a beach. J. Fluid Mech. 27(4), 815-827
  -- the (h/L)^2 dispersion parameter.
* Denlinger, R.P. & Iverson, R.M. (2004) Granular avalanches across irregular
  three-dimensional terrain: 1. Theory and computation. J. Geophys. Res. 109,
  F01014 -- bed-normal acceleration from curvature, u^2 * kappa, on steep and
  curved terrain.
* Castro-Orgaz, O. & Hager, W.H. (2017) Non-Hydrostatic Free Surface Flows.
  Springer -- the general treatment of when depth-averaging fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

G = 9.80665

# --- thresholds --------------------------------------------------------------
# Each is the value at which the corresponding neglected term reaches roughly
# 5-10% of the leading term -- i.e. the point at which the shallow-water
# equations stop being a small-error approximation and start being the wrong
# equations. They are exposed as scenario parameters, not buried constants.
BED_SLOPE_DEG_C = 12.0      # tan(12 deg) ~ 0.21; neglected terms ~ 5% of g
CURVATURE_RATIO_C = 0.10    # vertical acceleration reaching 10% of gravity
SURFACE_SLOPE_C = 0.10      # free surface at 10%: (h/L)^2 no longer small

MODEL_DEPTH_AVERAGED = "depth_averaged_swe"
MODEL_NON_HYDROSTATIC = "non_hydrostatic_sph"


# ---------------------------------------------------------------------------
# Field diagnostics on the 2D grid
# ---------------------------------------------------------------------------

def bed_slope(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """|grad z|, dimensionless."""
    gy, gx = np.gradient(z, dy, dx)
    return np.hypot(gx, gy)


def bed_curvature(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Magnitude of the mean bed curvature, 1/m.

    Uses the trace of the Hessian (the Laplacian) rather than a directional
    second derivative: the flow direction is not known a priori at every cell,
    and the trace bounds the curvature the flow can experience.
    """
    gy, gx = np.gradient(z, dy, dx)
    gxy, gxx = np.gradient(gx, dy, dx)
    gyy, gyx = np.gradient(gy, dy, dx)
    return np.abs(gxx + gyy)


def surface_slope(z: np.ndarray, h: np.ndarray,
                  dx: float, dy: float) -> np.ndarray:
    """|grad eta| , dimensionless.

    Dry cells are given eta = z, not zero.  A dry cell HAS a water surface
    elevation -- it is the bed, at zero depth -- and substituting 0 there puts
    a cliff of hundreds of metres at every wet/dry boundary, which then shows
    up as a spurious surface slope of O(10) along the entire flood edge.  With
    eta = z the field is continuous and the gradient at the margin is the real
    one: the slope of the water wedge running out onto dry land.
    """
    eta = np.where(h > 0.0, z + h, z)
    gy, gx = np.gradient(eta, dy, dx)
    return np.hypot(gx, gy)


@dataclass
class AdequacyField:
    """Per-cell verdict on whether the depth-averaged equations are valid."""
    nhi: np.ndarray                  # composite non-hydrostatic index
    n_bed: np.ndarray
    n_curv: np.ndarray
    n_surf: np.ndarray
    froude: np.ndarray
    wet: np.ndarray
    summary: Dict[str, object] = field(default_factory=dict)

    def inadequate_mask(self, threshold: float = 1.0) -> np.ndarray:
        return self.wet & (self.nhi >= threshold)


def adequacy_field(z: np.ndarray, h: np.ndarray, v: np.ndarray,
                   dx: float, dy: float,
                   bed_slope_deg_c: float = BED_SLOPE_DEG_C,
                   curvature_ratio_c: float = CURVATURE_RATIO_C,
                   surface_slope_c: float = SURFACE_SLOPE_C,
                   wet_threshold: float = 0.05) -> AdequacyField:
    """Evaluate the non-hydrostatic index over a 2D solution.

    `h` and `v` are the depth and speed fields to judge -- normally the maxima
    from a completed run, which gives the worst-case verdict over the event.
    """
    wet = h > wet_threshold

    slope = bed_slope(z, dx, dy)
    kappa = bed_curvature(z, dx, dy)
    s_eta = surface_slope(z, h, dx, dy)

    n_bed = slope / np.tan(np.radians(bed_slope_deg_c))
    n_curv = (v ** 2) * kappa / (G * curvature_ratio_c)
    n_surf = s_eta / surface_slope_c

    nhi = np.maximum(np.maximum(n_bed, n_curv), n_surf)
    nhi = np.where(wet, nhi, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(h > wet_threshold, v / np.sqrt(G * np.maximum(h, 1e-9)), 0.0)

    n_wet = int(wet.sum())
    cell_area = dx * dy

    def _frac(mask):
        return round(float(mask.sum()) / n_wet, 4) if n_wet else 0.0

    bad = wet & (nhi >= 1.0)

    # AREA fraction over-weights the flood margin.  In a gorge the water climbs
    # steep valley walls, so most of the WETTED CELLS sit on terrain that
    # violates the assumptions -- while carrying a few centimetres of water and
    # almost none of the volume.  Quoting the area fraction alone makes a
    # perfectly serviceable routing result look unusable.
    #
    # The volume-weighted fraction answers the question that actually matters:
    # of the water that is there, how much of it is being modelled with
    # equations that apply?
    vol = np.where(wet, h, 0.0)
    vol_total = float(vol.sum())
    vol_bad = float(vol[bad].sum()) if bad.any() else 0.0
    frac_vol = round(vol_bad / vol_total, 4) if vol_total > 0 else 0.0
    # Which term is responsible where the index fails -- this is what makes the
    # verdict actionable rather than just a red flag.
    dom = np.zeros(z.shape, dtype=np.int8)
    if bad.any():
        stack = np.stack([n_bed, n_curv, n_surf])
        dom = np.where(bad, np.argmax(stack, axis=0) + 1, 0).astype(np.int8)

    summary = {
        "wet_cells": n_wet,
        "inadequate_cells": int(bad.sum()),
        "inadequate_area_km2": round(float(bad.sum()) * cell_area / 1e6, 4),
        "inadequate_area_fraction": _frac(bad),
        "inadequate_volume_fraction": frac_vol,
        "mean_depth_where_inadequate_m": (
            round(vol_bad / max(float(bad.sum()), 1.0), 3) if bad.any() else 0.0),
        "mean_depth_where_adequate_m": (
            round((vol_total - vol_bad) / max(float((wet & ~bad).sum()), 1.0), 3)
            if n_wet else 0.0),
        "dominant_violation": {
            "bed_slope": int((dom == 1).sum()),
            "bed_curvature": int((dom == 2).sum()),
            "surface_slope": int((dom == 3).sum()),
        },
        "nhi_percentiles": {
            f"p{q:g}": round(float(np.percentile(nhi[wet], q)), 3)
            for q in (50, 90, 99, 99.9)
        } if n_wet else {},
        "supercritical_area_fraction": _frac(wet & (fr > 1.0)),
        "thresholds": {
            "bed_slope_deg": bed_slope_deg_c,
            "curvature_ratio": curvature_ratio_c,
            "surface_slope": surface_slope_c,
        },
        "verdict": _verdict(_frac(bad), frac_vol),
        "scope": (
            "FAR FIELD. This scores the domain the flood actually covered, "
            "which is a different question from the near-field decision made "
            "before the run. The two can legitimately disagree: at Tehri the "
            "thalweg below the dam is graded at 0.16 degrees (so the breach "
            "jet needs no particle model) while the flood then fills a gorge "
            "with 38-degree walls (so much of the routed volume sits on "
            "terrain the depth-averaged equations do not describe)."),
        "remedy": (
            "A near-field particle model would NOT fix this -- it resolves the "
            "breach jet, not the far field. Closing it needs either a "
            "non-hydrostatic far-field solver (Boussinesq or a 3D model) or a "
            "finer DEM on which the channel, rather than the valley side, "
            "carries the flow. Neither is in this prototype, which is why the "
            "number is reported rather than quietly absorbed."),
        "method": ("NHI = max(|grad z|/tan(theta_c), u^2*|kappa|/(g*c_curv), "
                   "|grad eta|/s_c); NHI >= 1 means the depth-averaged "
                   "(shallow-water) assumptions are violated at that cell"),
    }
    return AdequacyField(nhi=nhi, n_bed=n_bed, n_curv=n_curv, n_surf=n_surf,
                         froude=fr, wet=wet, summary=summary)


def _verdict(frac_area: float, frac_vol: float) -> str:
    """Verdict driven by the VOLUME fraction, with the area fraction for context.

    In steep terrain the two diverge sharply and it matters which one is
    quoted: a flood filling a gorge wets the valley walls, so most wet cells
    are steep, while nearly all the water sits in the channel where the
    equations are fine.  Judging on area would condemn a result that is
    substantially sound; judging on volume alone would hide a genuinely
    unreliable margin.  Report both, and decide on volume.
    """
    a, v = 100.0 * frac_area, 100.0 * frac_vol
    spread = (f" ({a:.0f}% of the wetted AREA is affected, but it holds only "
              f"{v:.0f}% of the water -- the difference is thin flow on steep "
              f"valley sides.)" if frac_area > 2 * frac_vol + 0.05 else
              f" ({a:.0f}% of the wetted area.)")

    if frac_vol < 0.02:
        return ("Depth-averaged modelling is adequate for essentially all of "
                "the flood volume." + spread)
    if frac_vol < 0.10:
        return (f"Depth-averaged modelling is adequate for {100-v:.0f}% of the "
                "flood volume; the remainder is in steep or strongly curved "
                "terrain where depths and velocities are indicative." + spread)
    if frac_vol < 0.30:
        return (f"{v:.0f}% of the flood volume sits where the shallow-water "
                "assumptions are violated. A non-hydrostatic near-field model "
                "is warranted and those depths should be read as "
                "order-of-magnitude." + spread)
    return (f"{v:.0f}% of the flood volume violates the shallow-water "
            "assumptions. This domain is not well suited to a purely "
            "depth-averaged treatment; treat the hazard map as indicative."
            + spread)


# ---------------------------------------------------------------------------
# 1D profile: where does the near field end?
# ---------------------------------------------------------------------------

def profile_nhi(s: np.ndarray, z: np.ndarray, u_ref: float,
                h_ref: float,
                bed_slope_deg_c: float = BED_SLOPE_DEG_C,
                curvature_ratio_c: float = CURVATURE_RATIO_C,
                smooth: int = 5) -> np.ndarray:
    """Non-hydrostatic index along a long profile (s, z) at a reference flow.

    Used before any 2D solution exists, to locate the handover point from the
    terrain plus the scale of the release.
    """
    if s.size < 3:
        return np.zeros_like(s)
    ds = float(np.median(np.diff(s))) or 1.0
    if smooth > 1 and z.size > smooth:
        # Edge-pad before smoothing.  `np.convolve(..., mode="same")` zero-pads,
        # which drags the first and last few samples of a profile sitting at
        # ~800 m elevation towards zero and manufactures an enormous artificial
        # slope and curvature at both ends -- enough to make a dead-flat reach
        # report a non-hydrostatic index of 9 and trigger the near-field model
        # for no physical reason.
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        z = np.convolve(np.pad(z, pad, mode="edge"), k, mode="same")[pad:pad + s.size]

    dz = np.gradient(z, ds)
    d2z = np.gradient(dz, ds)

    n_bed = np.abs(dz) / np.tan(np.radians(bed_slope_deg_c))
    n_curv = (u_ref ** 2) * np.abs(d2z) / (G * curvature_ratio_c)
    return np.maximum(n_bed, n_curv)


def locate_transfer_section(s: np.ndarray, z: np.ndarray, u_ref: float,
                            h_ref: float, min_x: float, max_x: float,
                            **kw) -> Tuple[float, dict]:
    """Place the SPH -> SWE handover where the flow becomes depth-averageable.

    Returns (x_transfer_m, rationale).  The section is the first point
    downstream of the breach at which the non-hydrostatic index falls below 1
    and STAYS below it -- i.e. the flow has recovered a hydrostatic profile and
    handing it to the depth-averaged solver is defensible.

    If the index never recovers inside the slice, the section is clamped to
    `max_x` and the rationale says so; that is a real physical statement about
    the reach (a continuously steep gorge has no clean near-field boundary),
    not a silent fallback.
    """
    nhi = profile_nhi(s, z, u_ref, h_ref, **kw)
    ok = nhi < 1.0

    x_rec = None
    for i in range(nhi.size):
        if s[i] < min_x:
            continue
        if ok[i:].all():
            x_rec = float(s[i])
            break

    if x_rec is None:
        x = float(min(max_x, s[-1]))
        reason = ("the non-hydrostatic index does not recover below 1 anywhere "
                  "in the modelled reach, so the section is clamped to the "
                  "downstream limit of the near-field slice; the receiving "
                  "reach is steep enough that no clean hydrostatic boundary "
                  "exists")
        recovered = False
    else:
        x = float(min(max(x_rec, min_x), max_x))
        reason = (f"the non-hydrostatic index falls below 1 at "
                  f"{x_rec:.0f} m downstream of the breach and stays below it, "
                  "so the flow has recovered a hydrostatic profile there")
        recovered = True

    return x, {
        "transfer_x_m": round(x, 1),
        "nhi_recovered": recovered,
        "nhi_recovery_x_m": round(x_rec, 1) if x_rec is not None else None,
        "reference_velocity_ms": round(u_ref, 2),
        "reference_depth_m": round(h_ref, 2),
        "search_bounds_m": [round(min_x, 1), round(max_x, 1)],
        "max_nhi_in_reach": round(float(nhi.max()), 2) if nhi.size else None,
        "rationale": reason,
        "method": ("first point downstream at which NHI < 1 for the remainder "
                   "of the profile; NHI from bed slope and u^2*curvature/g"),
    }


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------

@dataclass
class ModelPlan:
    """The framework's own decision about which physics to run, and why."""
    run_near_field: bool
    transfer_x_m: Optional[float]
    near_field_length_m: float
    reason: str
    decisions: List[dict] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_near_field": self.run_near_field,
            "selected_models": (
                [MODEL_NON_HYDROSTATIC, MODEL_DEPTH_AVERAGED]
                if self.run_near_field else [MODEL_DEPTH_AVERAGED]),
            "transfer_x_m": self.transfer_x_m,
            "near_field_length_m": round(self.near_field_length_m, 1),
            "reason": self.reason,
            "decisions": self.decisions,
            "diagnostics": self.diagnostics,
            "policy": ("The near-field model is run only where the flow is "
                       "predicted to be non-hydrostatic. This is a decision "
                       "the framework makes from the terrain and the release "
                       "scale, not a user setting."),
        }


def corridor_relief(dem_z: np.ndarray, path_rc: np.ndarray,
                    dx: float, dy: float, half_width_cells: int = 8
                    ) -> Optional[dict]:
    """Cross-valley relief statistics along the receiving reach.

    WHY THE THALWEG ALONE IS NOT ENOUGH.  `steepest_descent_path` follows the
    river channel, which in a Himalayan gorge is the one gently-graded line in
    the whole domain -- below Tehri it drops 4 m in 1.4 km, i.e. 0.16 degrees.
    Judging model adequacy on that profile alone reports a benign verdict for
    a valley whose walls stand at 30 degrees, because the flood does not stay
    in the channel: it fills the cross-section.

    So the pre-run test also samples a swath either side of the flowline and
    reports the transverse slope the flood will actually climb.
    """
    if path_rc is None or len(path_rc) < 3:
        return None
    ny, nx = dem_z.shape
    gy, gx = np.gradient(dem_z, dy, dx)
    slope = np.hypot(gx, gy)

    vals: List[float] = []
    for r0, c0 in path_rc:
        r1, r2 = max(0, r0 - half_width_cells), min(ny, r0 + half_width_cells + 1)
        c1, c2 = max(0, c0 - half_width_cells), min(nx, c0 + half_width_cells + 1)
        sub = slope[r1:r2, c1:c2]
        if sub.size:
            vals.append(float(np.percentile(sub, 90)))
    if not vals:
        return None
    arr = np.array(vals)
    return {
        "swath_half_width_m": round(half_width_cells * dx, 1),
        "p50_transverse_slope_deg": round(
            float(np.degrees(np.arctan(np.percentile(arr, 50)))), 2),
        "p90_transverse_slope_deg": round(
            float(np.degrees(np.arctan(np.percentile(arr, 90)))), 2),
        "max_transverse_slope_deg": round(
            float(np.degrees(np.arctan(arr.max()))), 2),
        "note": ("90th-percentile terrain slope in a swath around the "
                 "flowline: the valley sides the flood will climb, which the "
                 "thalweg profile does not see"),
    }


def select_models(s_dn: np.ndarray, z_dn: np.ndarray,
                  head_m: float, breach_width_m: float, peak_q_m3s: float,
                  dp_m: float, downstream_m: float,
                  force: Optional[bool] = None,
                  bed_slope_deg_c: float = BED_SLOPE_DEG_C,
                  curvature_ratio_c: float = CURVATURE_RATIO_C,
                  breach_invert_m: Optional[float] = None,
                  toe_bed_m: Optional[float] = None,
                  corridor: Optional[dict] = None) -> ModelPlan:
    """Decide whether the near field needs a non-hydrostatic model, and where
    it ends.

    The reference flow used to evaluate the criterion is the breach jet itself:
    critical depth and velocity at the opening, which is the fastest, deepest
    flow anywhere in the domain and therefore the strongest test of the
    depth-averaged assumption.

        h_c = (q^2 / g)^(1/3),      u_c = q / h_c = sqrt(g * h_c)

    with q = Q_peak / B the unit discharge through the breach.
    """
    decisions: List[dict] = []

    q_unit = peak_q_m3s / max(breach_width_m, 1.0)
    h_c = (q_unit ** 2 / G) ** (1.0 / 3.0) if q_unit > 0 else 0.0
    u_c = q_unit / h_c if h_c > 1e-6 else 0.0
    fr_jet = u_c / np.sqrt(G * h_c) if h_c > 1e-6 else 0.0

    decisions.append({
        "test": "breach jet scale",
        "unit_discharge_m2s": round(q_unit, 1),
        "critical_depth_m": round(h_c, 2),
        "critical_velocity_ms": round(u_c, 2),
        "froude": round(float(fr_jet), 3),
        "note": ("critical flow through the breach; by definition Fr = 1, "
                 "which is the reference state for the near field"),
    })

    nhi = profile_nhi(s_dn, z_dn, u_c, h_c,
                      bed_slope_deg_c=bed_slope_deg_c,
                      curvature_ratio_c=curvature_ratio_c)
    nhi_max = float(nhi.max()) if nhi.size else 0.0
    frac_bad = float((nhi >= 1.0).mean()) if nhi.size else 0.0

    decisions.append({
        "test": "non-hydrostatic index along the receiving reach",
        "max_nhi": round(nhi_max, 2),
        "fraction_of_reach_with_nhi_ge_1": round(frac_bad, 3),
        "threshold": 1.0,
        "note": ("computed from the DEM long profile at the breach-jet "
                 "velocity; >= 1 means depth-averaging is not valid there"),
    })

    # --- test 3: plunging jet at the structure ---------------------------
    # A breach that scours to the riverbed discharges at bed level and there is
    # no plunge.  A PARTIAL breach, a piping orifice, or a natural blockage
    # that does not scour to the original floor leaves the invert above the
    # toe, and the jet then free-falls that drop -- which is the textbook
    # non-hydrostatic near field and the case SPH exists to resolve.
    plunge = 0.0
    plunge_ratio = 0.0
    if breach_invert_m is not None and toe_bed_m is not None:
        plunge = max(float(breach_invert_m) - float(toe_bed_m), 0.0)
        plunge_ratio = plunge / max(h_c, 1e-6)
        decisions.append({
            "test": "plunging jet at the breach",
            "invert_above_toe_m": round(plunge, 2),
            "plunge_to_critical_depth_ratio": round(plunge_ratio, 3),
            "threshold": 1.0,
            "note": ("a jet that free-falls more than its own critical depth "
                     "is unambiguously non-hydrostatic; a breach scoured to "
                     "the riverbed has no plunge and does not trigger this"),
        })

    # --- test 4: cross-valley relief -------------------------------------
    # The flood does not stay in the channel. Confinement by steep valley
    # walls is the other way a depth-averaged treatment gets into trouble.
    corridor_trigger = False
    if corridor:
        corridor_trigger = corridor["p90_transverse_slope_deg"] > bed_slope_deg_c
        decisions.append({
            "test": "cross-valley relief in the flood corridor",
            **corridor,
            "threshold_deg": bed_slope_deg_c,
            "exceeds_threshold": bool(corridor_trigger),
        })

    needed = (nhi_max >= 1.0) or (plunge_ratio >= 1.0)
    if force is not None:
        decisions.append({
            "test": "user override",
            "forced_to": bool(force),
            "note": ("run_sph was set explicitly, so the automatic decision "
                     "is recorded but not applied"),
        })
        run_nf = bool(force)
    else:
        run_nf = needed

    if run_nf:
        min_x = max(8.0 * dp_m, 2.0 * h_c)
        max_x = downstream_m * 0.35
        x_t, rationale = locate_transfer_section(
            s_dn, z_dn, u_c, h_c, min_x, max_x,
            bed_slope_deg_c=bed_slope_deg_c,
            curvature_ratio_c=curvature_ratio_c)
        decisions.append({"test": "transfer-section placement", **rationale})
        trigger = ("a plunging jet of " f"{plunge:.0f} m "
                   f"({plunge_ratio:.1f}x the critical depth) at the breach"
                   if plunge_ratio >= 1.0 else
                   f"a non-hydrostatic index of {nhi_max:.1f} in the receiving "
                   f"reach at the breach-jet velocity ({u_c:.1f} m/s)")
        reason = (
            f"Near-field model SELECTED: {trigger}, so the depth-averaged "
            f"equations are not valid there. Handover to the 2D solver at "
            f"{x_t:.0f} m, where " + rationale["rationale"] + ".")
    else:
        x_t = None
        min_x = max_x = 0.0
        extra = ""
        if plunge_ratio > 0:
            extra = (f" The breach scours to within {plunge:.1f} m of the "
                     f"riverbed, so there is no plunging jet either "
                     f"({plunge_ratio:.2f}x critical depth).")
        if corridor_trigger:
            extra += (
                " NOTE: the flood corridor is confined by valley sides at "
                f"{corridor['p90_transverse_slope_deg']:.0f} degrees, so parts "
                "of the inundated area will still fall outside the "
                "depth-averaged assumptions once the water climbs them. That "
                "is a far-field accuracy caveat, reported after the run, not a "
                "reason to run a near-field particle model at the breach.")
        reason = (
            f"Near-field model NOT NEEDED: the non-hydrostatic index peaks at "
            f"{nhi_max:.2f} (< 1) along the receiving reach, so the "
            f"depth-averaged equations are valid from the breach onward and a "
            f"particle model would add cost without adding physics." + extra)

    return ModelPlan(
        run_near_field=run_nf,
        transfer_x_m=x_t,
        near_field_length_m=(x_t or 0.0),
        reason=reason,
        decisions=decisions,
        diagnostics={
            "jet_critical_depth_m": round(h_c, 2),
            "jet_critical_velocity_ms": round(u_c, 2),
            "max_nhi": round(nhi_max, 2),
            "plunge_m": round(plunge, 2),
            "plunge_to_critical_depth_ratio": round(plunge_ratio, 3),
            "corridor_confined": bool(corridor_trigger),
            "auto_decision": bool(needed),
            "applied_decision": bool(run_nf),
        },
    )
