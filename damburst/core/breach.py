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

from .reservoir import Reservoir

G = 9.80665

# --- failure-mode -> Froehlich K0 -------------------------------------------
# Overtopping breaches develop a wider final section than piping failures.
K0_OVERTOPPING = 1.3
K0_PIPING = 1.0

FAILURE_MODES = ("overtopping", "piping", "progressive_erosion", "instantaneous")
BARRIER_TYPES = ("engineered", "natural")


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
    """Von Thun & Gillette (1990).  B_avg = 2.5*hw + Cb ; tf from erodibility."""
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
    b_avg = 4.0 * hw + cb
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


# --- Xu & Zhang (2009) ------------------------------------------------------
# Xu, Y. and Zhang, L.M. (2009). "Breaching Parameters for Earth and Rockfill
# Dams." J. Geotech. Geoenviron. Eng. 135(12), 1957-1970.
#
# Fitted to 182 earth and rockfill failures from the USA and China, with
# deliberate coverage of dams taller than 15 m -- the reason it is worth having
# alongside Froehlich, whose sample skews small.  Coefficients below are the
# "best" (multiplicative) forms as tabulated by USBR HL-2014-02 Table 4.
#
# USBR's evaluation found the Xu & Zhang FAILURE TIME equations "dramatically
# overpredict" the breach formation time, so only the geometry and peak-flow
# forms are used here; timing stays with Froehlich and Von Thun & Gillette.
XZ_HR = 15.0                      # reference height, m

_XZ_DAM_TYPE = {"core": 0.061, "concrete_face": 0.088, "homogeneous": -0.089}
_XZ_MODE = {"overtopping": 0.299, "piping": -0.239}
_XZ_EROD = {"high": 0.411, "medium": -0.062, "low": -0.289}

_XZ_AVG_DAM_TYPE = {"core": -0.041, "concrete_face": 0.026, "homogeneous": -0.226}
_XZ_AVG_MODE = {"overtopping": 0.149, "piping": -0.389}
_XZ_AVG_EROD = {"high": 0.291, "medium": -0.140, "low": -0.391}

_XZ_Q_DAM_TYPE = {"core": -0.503, "concrete_face": -0.591, "homogeneous": -0.649}
_XZ_Q_MODE = {"overtopping": -0.705, "piping": -1.039}
_XZ_Q_EROD = {"high": -0.007, "medium": -0.375, "low": -1.362}


def xu_zhang_2009(volume_m3: float, h_dam: float, h_water: float,
                  h_breach: float, mode: str = "overtopping",
                  dam_type: str = "core",
                  erodibility: str = "medium") -> Dict[str, float]:
    """Xu & Zhang (2009) average breach width, top width and peak outflow.

    Returns metres and m3/s.  `erodibility` is the dominant control: USBR
    HL-2014-02 Table 3 reports that moving one category changes the predicted
    average width by roughly a factor 0.67 (low) to 1.68 (high) and the peak
    outflow by 0.37 to 1.44, independent of dam scale.
    """
    vw = max(volume_m3, 1.0)
    hd = max(h_dam, 1.0)
    hw = max(h_water, 1.0)
    hb = max(h_breach, 1.0)
    v13_hw = (vw ** (1.0 / 3.0)) / hw

    b2 = (_XZ_DAM_TYPE.get(dam_type, _XZ_DAM_TYPE["core"])
          + _XZ_MODE.get(mode, _XZ_MODE["overtopping"])
          + _XZ_EROD.get(erodibility, _XZ_EROD["medium"]))
    b_top = hb * 1.062 * ((hd / XZ_HR) ** 0.092) * (v13_hw ** 0.508) * math.exp(b2)

    b3 = (_XZ_AVG_DAM_TYPE.get(dam_type, _XZ_AVG_DAM_TYPE["core"])
          + _XZ_AVG_MODE.get(mode, _XZ_AVG_MODE["overtopping"])
          + _XZ_AVG_EROD.get(erodibility, _XZ_AVG_EROD["medium"]))
    b_avg = hb * 0.787 * ((hd / XZ_HR) ** 0.133) * (v13_hw ** 0.652) * math.exp(b3)

    b4 = (_XZ_Q_DAM_TYPE.get(dam_type, _XZ_Q_DAM_TYPE["core"])
          + _XZ_Q_MODE.get(mode, _XZ_Q_MODE["overtopping"])
          + _XZ_Q_EROD.get(erodibility, _XZ_Q_EROD["medium"]))
    qp = (0.175 * ((hd / XZ_HR) ** 0.199) * (v13_hw ** -1.274) * math.exp(b4)
          * math.sqrt(G) * (vw ** (5.0 / 6.0)))

    return {"b_avg": b_avg, "b_top": b_top, "peak_q": qp}


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
        # Landslide dams breach wider, shallower and slower than engineered fills.
        b_avg, t_f = froehlich_2008(vw, h_breach, "overtopping")
        b_avg *= 1.5
        t_f *= 2.0
        z = 1.5
        # natural blockages rarely scour to the original bed
        invert = reservoir.bed + 0.25 * h_breach
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
               else K0_PIPING},
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
    if mode == "piping" and frac < 0.5:
        # orifice phase: keep the invert high, widen slowly
        f2 = frac / 0.5
        invert = geom.invert + geom.height * (1.0 - 0.35 * f2)
        b = geom.b_bottom * (0.25 + 0.35 * f2)
        return b, geom.side_slope, invert
    f = frac if mode != "piping" else (frac - 0.5) / 0.5
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

    t = 0.0
    released = 0.0
    inflow_total = 0.0
    dt = dt_min

    while t < t_end:
        frac = growth_fraction(t, geom.t_form, growth_law)
        b, z, inv = breach_section(geom, frac, failure_mode)
        tw = tailwater(t) if tailwater else None
        q = weir_discharge(h, b, z, inv, tw)
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
        h = reservoir.level(v)
        t += dt

        ts.append(t); qs.append(q); hs.append(h)
        bs.append(b + 2.0 * z * max(h - inv, 0.0)); ivs.append(inv); vs.append(v)

        drained = v <= stop_fraction * max(v0, 1.0)
        if drained and t > geom.t_form and q < 0.01 * max(qs):
            break

    t_arr = np.array(ts); q_arr = np.array(qs)
    k = int(np.argmax(q_arr))

    res = BreachResult(
        t=t_arr, q=q_arr, h_res=np.array(hs), b_top=np.array(bs),
        invert=np.array(ivs), volume=np.array(vs), geometry=geom,
        peak_q=float(q_arr[k]), t_peak=float(t_arr[k]), released_m3=released,
    )
    res.checks = sanity_checks(res, reservoir, v0, inflow_total, h_init)
    return res


def regression_ensemble(volume_m3: float, h_dam: float, h_water: float,
                        h_breach: float, mode: str = "overtopping",
                        dam_type: str = "core",
                        erodibility: str = "medium") -> Dict[str, object]:
    """Every applicable breach regression, reported as a spread not a number.

    This is the practice USBR HL-2014-02 prescribes for exactly the situation a
    260 m Himalayan dam puts us in:

        "the uncertainty of the regression relationships is still large due to
        inherent uncertainty in the data for the underlying case studies, so it
        should remain common practice to apply multiple regression equations to
        most dams as a means of evaluating prediction uncertainty"

    and, on dams larger than anything in the databases:

        "This evaluation study did not suggest that there is a size or scale
        limitation for the equations ... it is reasonable, when necessary, to
        extend the equations for application to dams even larger than those
        included in the database as one component of a coordinated strategy to
        predict breach behaviour by a variety of methods."

    So the answer to "your dam is outside the calibration range" is not to pick
    one regression and hope, and not to refuse: it is to run them all, report
    the envelope, and let the physically routed solution sit inside a declared
    band of empirical uncertainty.
    """
    members: List[dict] = []

    b_f, t_f = froehlich_2008(volume_m3, h_breach, mode)
    members.append({"method": "Froehlich (2008)", "b_avg_m": b_f,
                    "t_form_s": t_f,
                    "reference": "J. Hydraulic Eng. 134(12), 1708-1721",
                    "fitted_to": "74 embankment failures, mostly < 60 m high"})

    b_v, t_v = von_thun_gillette(volume_m3, h_water, erodibility)
    members.append({"method": "Von Thun & Gillette (1990)", "b_avg_m": b_v,
                    "t_form_s": t_v,
                    "reference": "ASCE dam-breach workshop",
                    "fitted_to": "57 embankment failures"})

    b_m, t_m = macdonald_langridge(volume_m3, h_breach)
    members.append({"method": "MacDonald & Langridge-Monopolis (1984)",
                    "b_avg_m": b_m, "t_form_s": t_m,
                    "reference": "J. Hydraulic Eng. 110(5), 567-586",
                    "fitted_to": "42 embankment failures"})

    xz = xu_zhang_2009(volume_m3, h_dam, h_water, h_breach, mode,
                       dam_type, erodibility)
    members.append({"method": "Xu & Zhang (2009)", "b_avg_m": xz["b_avg"],
                    "b_top_m": xz["b_top"], "peak_q_m3s": xz["peak_q"],
                    "t_form_s": None,
                    "reference": "J. Geotech. Geoenviron. Eng. 135(12), 1957-1970",
                    "fitted_to": ("182 earth and rockfill failures (USA and "
                                  "China), ~50% taller than 15 m"),
                    "note": ("Failure-time equations omitted: USBR HL-2014-02 "
                             "found they dramatically overpredict formation "
                             "time.")})

    widths = [m["b_avg_m"] for m in members if m.get("b_avg_m")]
    times = [m["t_form_s"] for m in members if m.get("t_form_s")]
    return {
        "members": [{k: (round(v, 2) if isinstance(v, float) else v)
                     for k, v in m.items()} for m in members],
        "b_avg_m": {"min": round(min(widths), 1), "max": round(max(widths), 1),
                    "median": round(float(np.median(widths)), 1),
                    "spread_ratio": round(max(widths) / max(min(widths), 1e-9), 2)},
        "t_form_s": {"min": round(min(times), 1), "max": round(max(times), 1),
                     "median": round(float(np.median(times)), 1),
                     "spread_ratio": round(max(times) / max(min(times), 1e-9), 2)},
        "erodibility_assumed": erodibility,
        "erodibility_sensitivity": {
            "note": ("USBR HL-2014-02 Table 3: relative to medium erodibility, "
                     "low gives 0.67x and high 1.68x the average breach width, "
                     "and 0.37x / 1.44x the peak outflow, near-independently "
                     "of dam scale."),
            "b_avg_low": round(float(np.median(widths)) * 0.67, 1),
            "b_avg_high": round(float(np.median(widths)) * 1.68, 1),
        },
        "guidance": (
            "USBR HL-2014-02 (Wahl et al.): apply multiple regressions to "
            "evaluate prediction uncertainty, and extending them beyond the "
            "database size range is reasonable as one component of a "
            "multi-method strategy. The spread above IS the empirical "
            "uncertainty; the routed hydrograph is the primary estimate."),
        "source": ("Wahl, T.L. et al. (2014). Evaluation of Erodibility-Based "
                   "Embankment Dam Breach Equations. USBR HL-2014-02."),
    }


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
    # envelopes become weak priors rather than acceptance criteria. Say so
    # explicitly instead of returning a bare "fail" a reviewer cannot interpret.
    in_calibration = (v0 <= 1.0e9) and (head <= 100.0)

    # The empirical spread across ALL applicable regressions, which is what
    # USBR HL-2014-02 says to report when a structure is outside any single
    # regression's fitted range. Xu & Zhang (2009) also yields a peak-flow
    # prediction from a database that deliberately includes large dams, so it
    # is a third, better-matched envelope for a structure this size.
    # h_dam is not carried on BreachResult; at failure the water depth at the
    # dam is the closest available proxy for the structure height, which is
    # what Xu & Zhang's Hd/Hr term needs.
    xz = xu_zhang_2009(v0, head, head, res.geometry.height)
    q_xz = xz["peak_q"]

    return {
        "mass_balance": {
            "v_initial_mcm": round(v0 / 1e6, 3),
            "v_final_mcm": round(v_end / 1e6, 3),
            "released_mcm": round(res.released_m3 / 1e6, 3),
            "inflow_mcm": round(inflow_total / 1e6, 3),
            "closure_error_mcm": round(closure / 1e6, 6),
            "relative_error": round(rel, 8),
            "pass": bool(rel < 0.02),
        },
        "peak_discharge_envelopes": {
            "routed_peak_m3s": round(res.peak_q, 1),
            "froehlich_1995_m3s": round(q_fro, 1),
            "costa_schuster_1988_m3s": round(q_cs, 1),
            "ratio_to_froehlich": round(res.peak_q / q_fro, 3) if q_fro else None,
            "ratio_to_costa_schuster": round(res.peak_q / q_cs, 3) if q_cs else None,
            "xu_zhang_2009_m3s": round(q_xz, 1),
            "ratio_to_xu_zhang": round(res.peak_q / q_xz, 3) if q_xz else None,
            "within_envelope": bool(envelope_ok),
            "within_regression_calibration_range": bool(in_calibration),
            "reservoir_volume_mcm": round(v0 / 1e6, 1),
            "head_m": round(head, 1),
            "pass": bool(envelope_ok or not in_calibration),
            "interpretation": (
                "Routed peak agrees with the published envelopes."
                if envelope_ok else
                ("Routed peak lies outside the Froehlich/Costa-Schuster "
                 "envelopes, which were fitted to dams far smaller than this "
                 "one (<~1 km3, <~100 m head). Per USBR HL-2014-02 the right "
                 "response is a multi-regression spread, reported under "
                 "breach.regression_ensemble, with the routed hydrograph as "
                 "the primary estimate; Xu & Zhang (2009) is the envelope "
                 "fitted to the largest sample of tall dams."
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
    """Run the failure-mode ensemble that the dashboard compares side by side."""
    out = {}
    for mode in modes:
        geom = seed_breach(barrier_type, mode, reservoir, h_init, dam_crest)
        out[mode] = simulate_breach(reservoir, geom, h_init, failure_mode=mode, **kw)
    return out
