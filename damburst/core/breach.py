"""Breach formation and the outflow hydrograph Q(t).

Implements the left-hand column of the technical approach:

    Barrier type -> Engineered / Natural -> Empirical Seed
      -> Scenario x Failure Mode
      -> Adaptive dt -> Geometry growth law -> head H -> Q_breach
      -> Volume balance -> invert H-V-A -> Reservoir drained?
      -> Sanity checks (breach geometry, Q(t), t_peak, volume, reservoir level)

Empirical closures are the standard dam-safety regressions:

  Froehlich (2008)              J. Hydraulic Eng. 134(12)  -- default
  Von Thun & Gillette (1990)    ASCE dam-breach workshop
  MacDonald & Langridge-Monopolis (1984) J. Hydraulic Eng. 110(5)
  Costa & Schuster (1988)       landslide-dam specific peak-discharge envelope

The weir relation, submergence correction and the erosion-limited growth law
follow the NWS BREACH / HEC-RAS formulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .reservoir import Reservoir, volume_balance

G = 9.80665

# --- failure-mode -> Froehlich K0 -------------------------------------------
# Overtopping breaches develop a wider final section than piping failures.
K0_OVERTOPPING = 1.3
K0_PIPING = 1.0

FAILURE_MODES = ("overtopping", "piping", "progressive_erosion", "instantaneous")
BARRIER_TYPES = ("engineered", "natural")

# --- natural-blockage (landslide dam) judgement factors ---------------------
# Applied to the Froehlich seed in `seed_breach(method="costa_schuster")`.
# These are CALIBRATION CONSTANTS, not regression outputs -- see the comment at
# the call site.  Every run records them in `BreachGeometry.notes`.
CS_WIDTH_FACTOR = 1.5           # landslide dams breach wider than engineered fills
CS_TIME_FACTOR = 2.0            # and more slowly (poorly sorted, cohesionless)
CS_SIDE_SLOPE = 1.5             # flatter breach side slopes, z H : 1 V
CS_RESIDUAL_HEIGHT_FRAC = 0.25  # fraction of barrier height left unscoured

# Growth fraction at which a piping roof collapses and the pressurised orifice
# becomes a free-surface weir.  Below it the opening is submerged on all sides
# and `orifice_discharge` governs; above it `weir_discharge` does.
PIPE_COLLAPSE_FRAC = 0.5
PIPE_ORIFICE_CD = 0.6           # discharge coefficient, submerged rectangular
                                # opening in an embankment (NWS BREACH)


@dataclass
class BreachGeometry:
    """Final trapezoidal breach section."""
    b_bottom: float          # bottom width, m
    side_slope: float        # z, horizontal:vertical
    invert: float            # final breach invert elevation, m MSL
    height: float            # breach height, m
    t_form: float            # formation time, s
    method: str = ""
    notes: Dict[str, float] = field(default_factory=dict)

    def b_top(self) -> float:
        return self.b_bottom + 2.0 * self.side_slope * self.height

    def b_avg(self) -> float:
        return self.b_bottom + self.side_slope * self.height


# ---------------------------------------------------------------------------
# Empirical seed
# ---------------------------------------------------------------------------

def froehlich_2008(volume_m3: float, h_breach: float,
                   mode: str = "overtopping") -> Tuple[float, float]:
    """Froehlich (2008) average breach width and formation time.

        B_avg = 0.27 * K0 * Vw^0.32 * hb^0.04      [m]
        tf    = 63.2 * sqrt(Vw / (g * hb^2))       [s]

    Vw = reservoir volume at failure (m3), hb = breach height (m).
    """
    k0 = K0_OVERTOPPING if mode == "overtopping" else K0_PIPING
    vw = max(volume_m3, 1.0)
    hb = max(h_breach, 1.0)
    b_avg = 0.27 * k0 * (vw ** 0.32) * (hb ** 0.04)
    t_f = 63.2 * math.sqrt(vw / (G * hb ** 2))
    return b_avg, t_f


def von_thun_gillette(volume_m3: float, h_water: float,
                      erodibility: str = "medium") -> Tuple[float, float]:
    """Von Thun & Gillette (1990).

        B_avg = 2.5 * hw + Cb                  [m]
        tf    = B / (4*hw)        highly erodible      [h]
        tf    = B / (4*hw + 61)   erosion resistant    [h]

    hw = depth of water above the breach invert at failure (m); Cb is the
    volume-dependent offset from their Table 1.

    NOTE. The width coefficient is 2.5, not 4.0.  The 4.0 belongs to the
    formation-time denominators below and had been copied into the width
    relation, which overstated breach width by ~60% at large heads.
    """
    vw = max(volume_m3, 1.0)
    hw = max(h_water, 1.0)
    if vw < 1.23e6:
        cb = 6.1
    elif vw < 6.17e6:
        cb = 18.3
    elif vw < 1.233e7:
        cb = 42.7
    else:
        cb = 54.9
    b_avg = 2.5 * hw + cb
    if erodibility == "high":
        t_f = 3600.0 * (b_avg / (4.0 * hw))
    else:
        t_f = 3600.0 * (b_avg / (4.0 * hw + 61.0))
    return b_avg, max(t_f, 900.0)


def macdonald_langridge(volume_m3: float, h_breach: float,
                        earthfill: bool = True) -> Tuple[float, float]:
    """MacDonald & Langridge-Monopolis (1984) breach-erosion volume regression."""
    vw = max(volume_m3, 1.0)
    hb = max(h_breach, 1.0)
    fe = vw * hb
    ver = 0.0261 * (fe ** 0.769) if earthfill else 0.00348 * (fe ** 0.852)
    # convert eroded volume to an equivalent average width for a 1V:0.5H section
    b_avg = ver / max(hb * hb, 1.0)
    t_f = 0.0179 * (ver ** 0.364) * 3600.0
    return max(b_avg, 0.5 * hb), max(t_f, 600.0)


def costa_schuster_peak(volume_m3: float, head_m: float) -> float:
    """Costa & Schuster (1988) peak-discharge envelope for landslide dams.

        Qp = 0.0158 * PE^0.41 ,  PE = rho*g*V*h   (potential energy, J)

    Used as an independent sanity envelope on the routed peak, not as input.
    """
    pe = 1000.0 * G * max(volume_m3, 1.0) * max(head_m, 1.0)
    return 0.0158 * (pe ** 0.41)


def froehlich_peak(volume_m3: float, h_water: float) -> float:
    """Froehlich (1995) peak-outflow regression -- second sanity envelope."""
    return 0.607 * (max(volume_m3, 1.0) ** 0.295) * (max(h_water, 1.0) ** 1.24)


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------

def seed_breach(barrier_type: str, failure_mode: str, reservoir: Reservoir,
                h_init: float, dam_crest: float,
                method: str = "auto",
                side_slope: Optional[float] = None,
                erodibility: str = "medium") -> BreachGeometry:
    """Empirical-seed node: pick the regression appropriate to the barrier."""
    vw = reservoir.volume(h_init)
    invert = reservoir.bed
    h_breach = max(dam_crest - invert, 1.0)
    h_water = max(h_init - invert, 1.0)

    if method == "auto":
        method = "costa_schuster" if barrier_type == "natural" else "froehlich_2008"

    if method == "froehlich_2008":
        b_avg, t_f = froehlich_2008(vw, h_breach, failure_mode)
        z = 1.0 if failure_mode == "overtopping" else 0.7
    elif method == "von_thun_gillette":
        b_avg, t_f = von_thun_gillette(vw, h_water, erodibility)
        z = 1.0
    elif method == "macdonald":
        b_avg, t_f = macdonald_langridge(vw, h_breach)
        z = 0.5
    elif method == "costa_schuster":
        # Landslide-dam branch.
        #
        # There is no published width/time regression for natural blockages
        # comparable to Froehlich's for engineered fills -- Costa & Schuster
        # (1988) give a PEAK-DISCHARGE envelope, not a geometry.  So the
        # geometry is the Froehlich seed adjusted by three JUDGEMENT FACTORS
        # that encode the qualitative differences reported in
        #   Costa & Schuster (1988), GSA Bulletin 100(7), 1054-1068, and
        #   Peng & Zhang (2012), Nat. Hazards 64, 903-933 (breach statistics
        #   for 114 landslide dams):
        # natural blockages are wider and more poorly sorted, they breach more
        # slowly, and they rarely scour down to the original valley floor.
        #
        # THESE THREE NUMBERS ARE CALIBRATION, NOT MEASUREMENT.  They are
        # exposed here as named constants so they can be varied, and they are
        # recorded in the returned `notes` so every run states what it used.
        b_avg, t_f = froehlich_2008(vw, h_breach, "overtopping")
        b_avg *= CS_WIDTH_FACTOR
        t_f *= CS_TIME_FACTOR
        z = CS_SIDE_SLOPE
        invert = reservoir.bed + CS_RESIDUAL_HEIGHT_FRAC * h_breach
        h_breach = max(dam_crest - invert, 1.0)
    else:
        raise ValueError(f"unknown breach method: {method}")

    if side_slope is not None:
        z = side_slope

    if failure_mode == "instantaneous":
        # full removal of the barrier over one computational step
        b_avg = max(b_avg, 0.9 * h_breach * 4.0)
        t_f = 1.0

    b_bottom = max(b_avg - z * h_breach, 0.1 * h_breach)
    return BreachGeometry(
        b_bottom=b_bottom, side_slope=z, invert=invert, height=h_breach,
        t_form=t_f, method=method,
        notes={"reservoir_volume_m3": vw, "h_water_m": h_water,
               "b_avg_m": b_avg, "K0": K0_OVERTOPPING if failure_mode == "overtopping"
               else K0_PIPING,
               **({"cs_width_factor": CS_WIDTH_FACTOR,
                   "cs_time_factor": CS_TIME_FACTOR,
                   "cs_side_slope": CS_SIDE_SLOPE,
                   "cs_residual_height_frac": CS_RESIDUAL_HEIGHT_FRAC,
                   "cs_factors_are": "judgement calibration, not regression"}
                  if method == "costa_schuster" else {})},
    )


# ---------------------------------------------------------------------------
# Growth law
# ---------------------------------------------------------------------------

def growth_fraction(t: float, t_form: float, law: str = "sine") -> float:
    """Fraction of the final breach realised at time t (0 -> 1)."""
    if t <= 0:
        return 0.0
    if t >= t_form:
        return 1.0
    r = t / t_form
    if law == "linear":
        return r
    if law == "sine":                      # slow start, fast middle (HEC-RAS)
        return 0.5 * (1.0 - math.cos(math.pi * r))
    if law == "erosion":                   # t^0.5, headcut-dominated
        return math.sqrt(r)
    raise ValueError(f"unknown growth law: {law}")


def breach_section(geom: BreachGeometry, frac: float,
                   mode: str = "overtopping") -> Tuple[float, float, float]:
    """Instantaneous (bottom width, side slope, invert) at growth fraction f.

    Overtopping cuts downward and outward together; piping opens a soffit that
    only becomes a free-surface weir once the roof collapses (f > 0.5).
    """
    if mode == "piping" and frac < PIPE_COLLAPSE_FRAC:
        # orifice phase: keep the invert high, widen slowly
        f2 = frac / PIPE_COLLAPSE_FRAC
        invert = geom.invert + geom.height * (1.0 - 0.35 * f2)
        b = geom.b_bottom * (0.25 + 0.35 * f2)
        return b, geom.side_slope, invert
    f = (frac if mode != "piping"
         else (frac - PIPE_COLLAPSE_FRAC) / (1.0 - PIPE_COLLAPSE_FRAC))
    f = min(max(f, 0.0), 1.0)
    invert = geom.invert + geom.height * (1.0 - f)
    b = geom.b_bottom * f
    return b, geom.side_slope * f, invert


# ---------------------------------------------------------------------------
# Outflow
# ---------------------------------------------------------------------------

def weir_discharge(h_res: float, b_bottom: float, side_slope: float,
                   invert: float, tailwater: Optional[float] = None,
                   c_rect: float = 1.70, c_tri: float = 1.35) -> float:
    """Trapezoidal broad-crested weir with Villemonte submergence correction.

        Q = c_rect * b * H^1.5  +  c_tri * z * H^2.5

    c_rect = 1.70 corresponds to Cd = 0.55 in Q = Cd*b*sqrt(2g)*H^1.5, the
    standard broad-crested value used by NWS DAMBRK / HEC-RAS.
    """
    head = h_res - invert
    if head <= 0.0 or (b_bottom <= 0.0 and side_slope <= 0.0):
        return 0.0
    q = c_rect * b_bottom * head ** 1.5 + c_tri * side_slope * head ** 2.5

    if tailwater is not None:
        ht = tailwater - invert
        if ht > 0.0:
            s = ht / head
            if s > 0.67:                     # modular limit
                s = min(s, 0.999)
                q *= (1.0 - s ** 1.5) ** 0.385
    return max(q, 0.0)


def orifice_discharge(h_res: float, area: float, centroid: float,
                      cd: float = 0.6) -> float:
    head = h_res - centroid
    if head <= 0 or area <= 0:
        return 0.0
    return cd * area * math.sqrt(2.0 * G * head)


# ---------------------------------------------------------------------------
# Coupled breach + reservoir routing
# ---------------------------------------------------------------------------

@dataclass
class BreachResult:
    t: np.ndarray                # s
    q: np.ndarray                # m3/s outflow
    h_res: np.ndarray            # m MSL reservoir level
    b_top: np.ndarray            # m instantaneous top width
    invert: np.ndarray           # m MSL instantaneous invert
    volume: np.ndarray           # m3 stored
    geometry: BreachGeometry
    peak_q: float
    t_peak: float
    released_m3: float
    regime: List[str] = field(default_factory=list)   # orifice|weir|weir_submerged
    checks: Dict[str, object] = field(default_factory=dict)

    def hydrograph(self) -> Callable[[float], float]:
        t, q = self.t, self.q
        return lambda tt: float(np.interp(tt, t, q, left=q[0], right=q[-1]))

    def to_dict(self) -> dict:
        return {
            "peak_q_m3s": round(self.peak_q, 1),
            "t_peak_s": round(self.t_peak, 1),
            "t_peak_min": round(self.t_peak / 60.0, 2),
            "released_mcm": round(self.released_m3 / 1e6, 3),
            "formation_time_s": round(self.geometry.t_form, 1),
            "final_top_width_m": round(float(self.b_top[-1]), 1),
            "final_invert_m": round(float(self.invert[-1]), 2),
            "breach_height_m": round(self.geometry.height, 2),
            "side_slope_zH_1V": self.geometry.side_slope,
            "seed_method": self.geometry.method,
            "checks": self.checks,
        }


def simulate_breach(
    reservoir: Reservoir,
    geom: BreachGeometry,
    h_init: float,
    failure_mode: str = "overtopping",
    growth_law: str = "sine",
    inflow: Optional[Callable[[float], float]] = None,
    tailwater: Optional[Callable[[float], float]] = None,
    t_end: float = 6 * 3600.0,
    dt_max: float = 10.0,
    dt_min: float = 0.05,
    stop_fraction: float = 0.02,
) -> BreachResult:
    """Adaptive-dt coupled breach growth and level-pool depletion.

    The timestep is limited so that no single step drains more than 0.5% of the
    live storage or advances the breach by more than 2% of its formation time --
    the "Adaptive dt" node of the technical approach.

    `tailwater(t)` returns the downstream water-surface elevation and drives the
    Villemonte submergence correction in `weir_discharge`.  Pass
    `normal_depth_tailwater(...)` for a physically-based rating; passing None
    leaves the outflow free-discharging, which overestimates Q once the
    downstream reach fills.

    In `piping` mode the opening is a PRESSURISED ORIFICE until the roof
    collapses at `PIPE_COLLAPSE_FRAC`, and a free-surface weir after.  The two
    branches are evaluated by `orifice_discharge` and `weir_discharge`
    respectively; `regime` records which governed at each step.
    """
    inflow = inflow or (lambda t: 0.0)
    v = reservoir.volume(h_init)
    v0 = v
    h = h_init

    ts: List[float] = [0.0]
    qs: List[float] = [0.0]
    hs: List[float] = [h]
    bs: List[float] = [0.0]
    ivs: List[float] = [geom.invert + geom.height]
    vs: List[float] = [v]
    regimes: List[str] = ["closed"]

    t = 0.0
    released = 0.0
    inflow_total = 0.0
    dt = dt_min
    n_submerged = 0
    n_orifice = 0
    n_steps = 0

    while t < t_end:
        frac = growth_fraction(t, geom.t_form, growth_law)
        b, z, inv = breach_section(geom, frac, failure_mode)
        tw = tailwater(t) if tailwater else None

        if failure_mode == "piping" and frac < PIPE_COLLAPSE_FRAC:
            # Pressurised orifice: the opening is submerged on all sides, so
            # discharge scales with sqrt(head to the CENTROID), not head^1.5.
            soffit = inv + max(geom.height * 0.35 * (frac / PIPE_COLLAPSE_FRAC),
                               0.05 * geom.height)
            area = max(b, 0.0) * max(soffit - inv, 0.0)
            centroid = 0.5 * (inv + soffit)
            q = orifice_discharge(h, area, centroid, cd=PIPE_ORIFICE_CD)
            regime = "orifice"
            n_orifice += 1
        else:
            q = weir_discharge(h, b, z, inv, tw)
            regime = "weir"
            if tw is not None and h > inv and (tw - inv) / max(h - inv, 1e-9) > 0.67:
                regime = "weir_submerged"
                n_submerged += 1

        q_in = inflow(t)

        # --- adaptive dt -------------------------------------------------
        dt = dt_max
        if q > 1.0:
            live = max(v, 1.0)
            dt = min(dt, 0.005 * live / q)           # <=0.5% storage per step
        dt = min(dt, max(0.02 * geom.t_form, dt_min))
        dt = max(min(dt, t_end - t), dt_min)

        v = max(0.0, v + (q_in - q) * dt)
        released += q * dt
        inflow_total += q_in * dt
        # "Reservoir drained?" decision node -- the explicit continuity step
        h = reservoir.level(v)
        t += dt
        n_steps += 1

        ts.append(t); qs.append(q); hs.append(h)
        bs.append(b + 2.0 * z * max(h - inv, 0.0)); ivs.append(inv); vs.append(v)
        regimes.append(regime)

        # Explicit one-step lag on the tailwater rating.
        if tailwater is not None and hasattr(tailwater, "update"):
            tailwater.update(q)

        drained = v <= stop_fraction * max(v0, 1.0)
        if drained and t > geom.t_form and q < 0.01 * max(qs):
            break

    t_arr = np.array(ts); q_arr = np.array(qs)
    k = int(np.argmax(q_arr))

    res = BreachResult(
        t=t_arr, q=q_arr, h_res=np.array(hs), b_top=np.array(bs),
        invert=np.array(ivs), volume=np.array(vs), geometry=geom,
        peak_q=float(q_arr[k]), t_peak=float(t_arr[k]), released_m3=released,
        regime=regimes,
    )
    res.checks = sanity_checks(res, reservoir, v0, inflow_total, h_init)
    res.checks["outflow_regime"] = {
        "steps": n_steps,
        "orifice_steps": n_orifice,
        "submerged_weir_steps": n_submerged,
        "tailwater_supplied": tailwater is not None,
        "villemonte_applied": n_submerged > 0,
        "note": ("Free-discharging weir throughout; no tailwater rating was "
                 "supplied, so the Villemonte submergence correction did not "
                 "engage and peak outflow is an upper bound."
                 if tailwater is None else
                 f"Submergence correction engaged on {n_submerged} of "
                 f"{n_steps} steps."),
    }
    return res


class TailwaterRating:
    """Downstream stage rating from Manning normal depth in a wide channel.

        h_n = ( q_unit * n / sqrt(S) ) ** 0.6          [m]

    `__call__(t)` returns the tailwater ELEVATION so `weir_discharge` can form
    the submergence ratio.  This is a rating curve, not a backwater solution:
    it assumes the receiving reach conveys the breach discharge at normal
    depth, which is conservative in a steep gorge (the true tailwater there is
    lower, so the true submergence is weaker).

    The coupling to the breach is EXPLICIT and lagged by one step:
    `simulate_breach` calls `update(q)` after each step, and the next step's
    rating uses that discharge.  An implicit solve inside the routing loop
    would be more consistent but is not worth the cost at these timesteps --
    the reservoir level changes by well under 1% per step.
    """

    def __init__(self, invert_m: float, width_m: float, slope: float,
                 manning_n: float):
        self.invert_m = float(invert_m)
        self.width_m = max(float(width_m), 1.0)
        self.slope = max(float(slope), 1e-4)
        self.n = max(float(manning_n), 0.01)
        self.q_prev = 0.0

    def update(self, q: float) -> None:
        self.q_prev = max(float(q), 0.0)

    def __call__(self, t: float) -> float:
        if self.q_prev <= 0.0:
            return self.invert_m
        q_unit = self.q_prev / self.width_m
        return self.invert_m + (q_unit * self.n / math.sqrt(self.slope)) ** 0.6


def normal_depth_tailwater(invert_m: float, width_m: float, slope: float,
                           manning_n: float) -> TailwaterRating:
    """Convenience constructor for `TailwaterRating`."""
    return TailwaterRating(invert_m, width_m, slope, manning_n)


def sanity_checks(res: BreachResult, reservoir: Reservoir, v0: float,
                  inflow_total: float, h_init: float) -> Dict[str, object]:
    """Solver verification gate.

    The routed peak is compared against two independent published regressions.
    Agreement inside the documented scatter of those envelopes (roughly a factor
    of three) is the pass condition; anything outside is flagged, not hidden.
    """
    head = h_init - res.geometry.invert
    q_fro = froehlich_peak(v0, head)
    q_cs = costa_schuster_peak(v0, head)

    v_end = float(res.volume[-1])
    closure = abs((v0 + inflow_total - res.released_m3) - v_end)
    rel = closure / max(v0, 1.0)

    ratios = [res.peak_q / q for q in (q_fro, q_cs) if q > 0]
    envelope_ok = all(0.33 <= r <= 3.0 for r in ratios)

    # Froehlich (1995) was fitted to 22 failures and Costa & Schuster (1988) to
    # landslide dams; the great majority were under ~100 m high and under
    # ~1 km3. A 260 m, 3.5 km3 structure sits well outside both fits, so the
    # envelopes become weak priors rather than acceptance criteria.
    in_calibration = (v0 <= 1.0e9) and (head <= 100.0)

    # A gate that returns `pass: true` on a 16x exceedance is not a gate.  When
    # the regressions cannot arbitrate, the gate returns None ("not
    # applicable") and the roll-up in pipeline.py must treat None as
    # inconclusive, NOT as a pass.
    envelope_pass: Optional[bool] = bool(envelope_ok) if in_calibration else None

    # --- scale-independent physical ceiling ------------------------------
    # Critical flow through the final breach section is a hard upper bound on
    # free discharge at ANY scale, which is exactly what the empirical
    # envelopes stop being for a 260 m structure.  This is the gate that
    # actually binds on the presets.
    b_top_final = float(res.b_top[-1]) if res.b_top.size else res.geometry.b_top()
    q_crit = (2.0 / 3.0) ** 1.5 * math.sqrt(G) * max(b_top_final, 1.0) * \
        max(head, 0.0) ** 1.5
    crit_ratio = res.peak_q / q_crit if q_crit > 0 else float("nan")
    crit_ok = bool(np.isfinite(crit_ratio) and crit_ratio <= 1.05)

    return {
        # Delegated to reservoir.volume_balance so there is exactly one
        # implementation of the continuity gate in the codebase.
        "mass_balance": {
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in volume_balance(v0, v_end, res.released_m3,
                                          inflow_total, tol=0.02).items()},
            "closure_error_mcm": round(closure / 1e6, 6),
            "relative_error": round(rel, 8),
        },
        "peak_discharge_critical_flow": {
            "routed_peak_m3s": round(res.peak_q, 1),
            "critical_flow_ceiling_m3s": round(q_crit, 1),
            "ratio_to_critical": round(crit_ratio, 3)
                if np.isfinite(crit_ratio) else None,
            "breach_top_width_m": round(b_top_final, 1),
            "head_m": round(head, 1),
            "pass": crit_ok,
            "method": "Q_crit = (2/3)^1.5 * sqrt(g) * B * H^1.5",
            "interpretation": (
                "Routed peak is at or below critical flow through the breach "
                "section, which is the hard physical ceiling for free "
                "discharge and is valid at any scale."
                if crit_ok else
                "Routed peak EXCEEDS critical flow through its own breach "
                "section. This is physically impossible for free discharge - "
                "check the weir coefficients, the breach width and the head."),
        },
        "peak_discharge_envelopes": {
            "routed_peak_m3s": round(res.peak_q, 1),
            "froehlich_1995_m3s": round(q_fro, 1),
            "costa_schuster_1988_m3s": round(q_cs, 1),
            "ratio_to_froehlich": round(res.peak_q / q_fro, 3) if q_fro else None,
            "ratio_to_costa_schuster": round(res.peak_q / q_cs, 3) if q_cs else None,
            "within_envelope": bool(envelope_ok),
            "within_regression_calibration_range": bool(in_calibration),
            "reservoir_volume_mcm": round(v0 / 1e6, 1),
            "head_m": round(head, 1),
            "pass": envelope_pass,
            "applicable": bool(in_calibration),
            "interpretation": (
                "Routed peak agrees with the published envelopes."
                if envelope_ok and in_calibration else
                ("NOT APPLICABLE: this reservoir is outside the range those "
                 "regressions were fitted to (<~1 km3, <~100 m head), so they "
                 "can neither confirm nor refute the routed peak. The gate "
                 "returns null, not a pass. Use the critical-flow ceiling "
                 "above, which is scale-independent."
                 if not in_calibration else
                 "Routed peak disagrees with the envelopes INSIDE their "
                 "calibration range - investigate the breach geometry.")),
        },
        "geometry": {
            "final_top_width_m": round(float(res.b_top[-1]), 1),
            "breach_height_m": round(res.geometry.height, 2),
            "formation_time_min": round(res.geometry.t_form / 60.0, 2),
            "width_to_height": round(float(res.b_top[-1]) / max(res.geometry.height, 1), 2),
            "pass": bool(0.5 <= float(res.b_top[-1]) / max(res.geometry.height, 1) <= 20.0),
        },
        "drawdown": {
            "h_initial_m": round(h_init, 2),
            "h_final_m": round(float(res.h_res[-1]), 2),
            "drawdown_m": round(h_init - float(res.h_res[-1]), 2),
            "fraction_drained": round(1.0 - v_end / max(v0, 1.0), 4),
        },
    }


def scenario_matrix(reservoir: Reservoir, h_init: float, dam_crest: float,
                    barrier_type: str,
                    modes: Tuple[str, ...] = ("overtopping", "piping",
                                              "instantaneous"),
                    **kw) -> Dict[str, BreachResult]:
    """Run a failure-mode ensemble in one call.

    NOT called by the default pipeline, and the dashboard does not use it -- the
    Models tab compares MODEL CONFIGURATIONS (grid / SPH / SPH-initialised) for
    a single failure mode, not failure modes against each other. To build the
    ensemble, invoke the CLI once per `--failure-mode`. Kept because a
    consequence study normally wants all three.
    """
    out = {}
    for mode in modes:
        geom = seed_breach(barrier_type, mode, reservoir, h_init, dam_crest)
        out[mode] = simulate_breach(reservoir, geom, h_init, failure_mode=mode, **kw)
    return out
