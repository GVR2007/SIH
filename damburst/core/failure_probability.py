"""Possibility of breach: P(failure), separate from the consequence of failure.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Everything else in `core/` answers "what happens if the dam fails?".  The
hazard rasters, the exposure counts and the evacuation timeline are all
CONDITIONAL on failure:

    hazard map  =  P(depth > d | failure)

What an emergency planner needs is the unconditional form,

    risk  =  P(failure) x consequence | failure

and quoting the consequence with no probability attached overstates what the
framework knows.  A scenario is not a forecast.  This module supplies the left
factor for the one mechanism that can be computed from data the framework
already holds -- **overtopping** -- and gives documented base rates for the
mechanisms that cannot.

WHAT IS COMPUTED AND WHAT IS ASSUMED
------------------------------------
* **Computed.**  P(overtopping | flood of return period T), by routing a
  design-flood family through the SAME level-pool equation the breach model
  uses (`reservoir.route_step` / the loop in `breach.simulate_breach`) against
  a spillway rating.  Given a spillway rating and an inflow-frequency curve,
  this is a deterministic calculation, not a guess.
* **Assumed, and flagged as such.**  The spillway rating (not in OSM or
  Wikidata -- it belongs to the CWC National Register of Large Dams), the
  inflow frequency curve, and the fragility of every non-hydrological
  mechanism.  Each is an explicit input with a recorded provenance string, and
  every result carries the list of assumptions that produced it.

Read `docs/MATHEMATICAL_FORMULATION.md` Part B for the formulation.

REFERENCES
----------
* Foster, M., Fell, R., Spannagle, M. (2000) The statistics of embankment dam
  failures and accidents. Can. Geotech. J. 37(5), 1000-1024.
* ICOLD (1995) Dam Failures: Statistical Analysis. Bulletin 99.
* USBR (2011) Dam Safety Risk Analysis Best Practices Training Manual.
* Kalinina, A. et al. (2018) Application of a Bayesian hierarchical model to
  dam failure frequency. Reliab. Eng. Syst. Saf. 178, 89-102.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .reservoir import Reservoir, route_step

G = 9.80665


# ---------------------------------------------------------------------------
# Base rates -- the honest anchor where site data does not exist
# ---------------------------------------------------------------------------
#
# Annual probability of failure per dam-year, from the ICOLD / Foster et al.
# population statistics over ~11,000 embankment dams.  These are POPULATION
# AVERAGES: they are the right order of magnitude for a dam you know nothing
# about, and they are NOT a site-specific estimate.  Using them for a specific
# structure without adjusting for age, inspection record, spillway adequacy and
# seismicity is exactly the kind of unsupported number this codebase tries to
# avoid -- so every result that uses them says so.
BASE_RATES_PER_YEAR = {
    "overtopping":      {"p": 2.0e-5, "share": 0.48},
    "internal_erosion": {"p": 1.5e-5, "share": 0.33},
    "seismic":          {"p": 2.0e-6, "share": 0.04},
    "structural":       {"p": 5.0e-6, "share": 0.11},
    "other":            {"p": 2.0e-6, "share": 0.04},
}
BASE_RATE_SOURCE = (
    "ICOLD Bulletin 99 / Foster, Fell & Spannagle (2000), population "
    "statistics over ~11,000 embankment dams. POPULATION AVERAGE, not a "
    "site-specific estimate.")


# ---------------------------------------------------------------------------
# Spillway rating
# ---------------------------------------------------------------------------

@dataclass
class Spillway:
    """Ogee / broad-crested spillway rating.

        Q = C * L * (h - z_sill)^1.5

    `source` is mandatory in spirit: a rating that cannot say where it came
    from should not be used to make a probability statement.
    """
    crest_length_m: float
    sill_elevation_m: float
    discharge_coefficient: float = 2.2       # C in SI for a standard ogee
    gated: bool = False
    max_gate_discharge_m3s: Optional[float] = None
    source: str = "ASSUMED - substitute the CWC register value"

    def discharge(self, h: float) -> float:
        head = h - self.sill_elevation_m
        if head <= 0.0:
            return 0.0
        q = self.discharge_coefficient * self.crest_length_m * head ** 1.5
        if self.gated and self.max_gate_discharge_m3s:
            q = min(q, self.max_gate_discharge_m3s)
        return q

    def to_dict(self) -> dict:
        return {
            "crest_length_m": self.crest_length_m,
            "sill_elevation_m": round(self.sill_elevation_m, 2),
            "discharge_coefficient": self.discharge_coefficient,
            "gated": self.gated,
            "max_gate_discharge_m3s": self.max_gate_discharge_m3s,
            "_provenance": self.source,
        }


def default_spillway(crest_m: float, reservoir: Reservoir,
                     mean_annual_flood_m3s: Optional[float] = None,
                     flood_cv: float = 0.6,
                     design_return_period_y: float = 1000.0,
                     design_flood_m3s: Optional[float] = None,
                     capacity_factor: float = 1.0) -> Spillway:
    """A placeholder rating sized to pass a stated design flood.

    THIS IS AN ASSUMPTION, not a measurement.  Spillway geometry is not carried
    by OpenStreetMap or Wikidata; it lives in the CWC National Register of
    Large Dams.  The placeholder exists so the machinery can be exercised and
    so the SENSITIVITY to spillway capacity can be shown -- never so a number
    can be quoted as a property of a real dam.

    Sizing, in order of preference:

    1. `design_flood_m3s` if given;
    2. the `design_return_period_y` quantile of the Gumbel curve implied by
       `mean_annual_flood_m3s` and `flood_cv` -- i.e. a dam designed to Indian
       practice for a large structure;
    3. failing both, a crude volume-based scale.

    `capacity_factor` scales the result: 1.0 is a dam whose spillway exactly
    meets its design flood, 0.5 an under-provided one, 2.0 a conservative one.
    Sweeping it is the intended way to show how much P(overtopping) depends on
    a number the framework does not know.
    """
    depth = max(crest_m - reservoir.bed, 1.0)
    sill = crest_m - max(0.06 * depth, 3.0)
    head = crest_m - sill

    if design_flood_m3s is not None:
        q_design = design_flood_m3s
        basis = f"explicit design flood {q_design:,.0f} m3/s"
    elif mean_annual_flood_m3s:
        q_design = gumbel_quantile(mean_annual_flood_m3s, flood_cv,
                                   design_return_period_y)
        basis = (f"{design_return_period_y:,.0f}-year Gumbel quantile of the "
                 f"supplied flood statistics ({q_design:,.0f} m3/s)")
    else:
        q_design = max(reservoir.capacity / 86400.0, 500.0)
        basis = "crude volume/24h scale (no flood statistics supplied)"

    length = (q_design * capacity_factor) / max(2.2 * head ** 1.5, 1.0)
    return Spillway(
        crest_length_m=round(length, 1), sill_elevation_m=sill,
        source=(f"ASSUMED placeholder sized on {basis}, capacity_factor="
                f"{capacity_factor}; substitute the CWC National Register of "
                "Large Dams rating before quoting any probability"))


# ---------------------------------------------------------------------------
# Design flood family
# ---------------------------------------------------------------------------

def gumbel_quantile(mean: float, cv: float, return_period: float) -> float:
    """Gumbel (EV1) flood quantile for return period T.

        Q_T = mu + sigma * y_T ,     y_T = -ln(-ln(1 - 1/T))

    with mu, sigma from the mean and coefficient of variation.  EV1 is the
    standard first choice for annual maximum series in Indian regional flood
    frequency practice (CWC), and it is used here because it needs only two
    moments -- which is all a gauge-free estimate can honestly support.
    """
    sigma = cv * mean * math.sqrt(6.0) / math.pi
    mu = mean - 0.5772 * sigma
    y = -math.log(-math.log(max(1.0 - 1.0 / return_period, 1e-12)))
    return max(mu + sigma * y, 0.0)


def design_hydrograph(peak_m3s: float, t_peak_s: float = 18 * 3600.0,
                      base_m3s: float = 0.0,
                      shape: float = 3.0) -> Callable[[float], float]:
    """Gamma-shaped single-peak flood hydrograph normalised to `peak_m3s`."""
    def q(t: float) -> float:
        if t <= 0.0:
            return base_m3s
        r = t / t_peak_s
        return base_m3s + (peak_m3s - base_m3s) * (r ** shape) * math.exp(
            shape * (1.0 - r))
    return q


# ---------------------------------------------------------------------------
# Overtopping probability by routing
# ---------------------------------------------------------------------------

@dataclass
class OvertoppingResult:
    return_periods: List[float]
    rows: List[dict] = field(default_factory=list)
    annual_probability: Optional[float] = None
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mechanism": "overtopping",
            "method": ("design-flood family routed through the reservoir "
                       "against the spillway rating; overtopping when the "
                       "routed level exceeds the crest"),
            "return_periods_y": self.return_periods,
            "by_return_period": self.rows,
            "annual_probability_of_overtopping": self.annual_probability,
            "assumptions": self.assumptions,
            "caveat": ("This is P(reservoir level exceeds crest), NOT "
                       "P(dam fails). Overtopping initiates failure of an "
                       "embankment; a concrete gravity dam may pass a "
                       "substantial overtopping depth without breaching. "
                       "Multiply by a conditional failure probability "
                       "P(breach | overtopping) appropriate to the dam type."),
        }


def route_flood(reservoir: Reservoir, spillway: Spillway, crest_m: float,
                inflow: Callable[[float], float],
                h_start: float,
                t_end: float = 96 * 3600.0,
                dt: float = 300.0) -> dict:
    """Level-pool route one flood and report the peak level reached.

    Uses the same continuity step as the breach model
    (`reservoir.route_step`), so the two cannot drift apart:

        dV/dt = Q_in(t) - Q_spillway(h)
    """
    v = reservoir.volume(h_start)
    h = h_start
    h_max = h
    t = 0.0
    q_in_peak = 0.0
    q_out_peak = 0.0
    t_overtop = None

    while t < t_end:
        q_in = inflow(t)
        q_out = spillway.discharge(h)
        q_in_peak = max(q_in_peak, q_in)
        q_out_peak = max(q_out_peak, q_out)
        v, h = route_step(reservoir, v, q_in, q_out, dt)
        if h > h_max:
            h_max = h
        if h >= crest_m and t_overtop is None:
            t_overtop = t
        t += dt

    return {
        "peak_level_m": round(h_max, 2),
        "crest_m": round(crest_m, 2),
        "freeboard_remaining_m": round(crest_m - h_max, 2),
        "overtopped": bool(h_max >= crest_m),
        "time_to_overtop_h": round(t_overtop / 3600.0, 2) if t_overtop else None,
        "peak_inflow_m3s": round(q_in_peak, 1),
        "peak_spillway_outflow_m3s": round(q_out_peak, 1),
    }


def overtopping_probability(reservoir: Reservoir, crest_m: float,
                            spillway: Spillway,
                            mean_annual_flood_m3s: float,
                            flood_cv: float = 0.6,
                            return_periods: Sequence[float] = (
                                10, 25, 50, 100, 200, 500, 1000, 5000, 10000),
                            starting_level: Optional[float] = None,
                            t_peak_s: float = 18 * 3600.0) -> OvertoppingResult:
    """P(overtopping) by routing a design-flood family.

    The annual probability is obtained by integrating over the flood-frequency
    curve: if overtopping first occurs at return period T*, then

        P(overtopping in a year) = 1 / T*

    interpolated between the bracketing return periods actually routed.
    """
    h0 = starting_level if starting_level is not None else \
        spillway.sill_elevation_m
    rows: List[dict] = []
    for T in return_periods:
        peak = gumbel_quantile(mean_annual_flood_m3s, flood_cv, T)
        res = route_flood(reservoir, spillway, crest_m,
                          design_hydrograph(peak, t_peak_s), h0)
        rows.append({"return_period_y": T,
                     "annual_exceedance_probability": round(1.0 / T, 6),
                     "design_peak_inflow_m3s": round(peak, 1), **res})

    # first return period at which the crest is exceeded
    p_annual = None
    prev = None
    for r in rows:
        if r["overtopped"]:
            if prev is None:
                p_annual = r["annual_exceedance_probability"]
            else:
                # log-linear interpolation in return period on freeboard
                f0, f1 = prev["freeboard_remaining_m"], r["freeboard_remaining_m"]
                t0, t1 = prev["return_period_y"], r["return_period_y"]
                w = f0 / max(f0 - f1, 1e-9)
                t_star = math.exp(math.log(t0) + w * (math.log(t1) - math.log(t0)))
                p_annual = round(1.0 / max(t_star, 1.0), 8)
            break
        prev = r

    assumptions = [
        f"Spillway rating: {spillway.to_dict()['_provenance']}",
        (f"Flood frequency: Gumbel EV1 with mean annual flood "
         f"{mean_annual_flood_m3s:,.0f} m3/s and Cv {flood_cv}. Replace with a "
         "CWC gauge-based frequency curve for the site."),
        (f"Starting level {h0:.1f} m (spillway sill). A reservoir already at "
         "FRL when the flood arrives overtops at a much lower return period; "
         "this is the single most sensitive assumption here."),
        (f"Flood shape: gamma hydrograph peaking at {t_peak_s/3600:.0f} h. "
         "Volume, not just peak, governs routing through a large reservoir."),
    ]
    if p_annual is None:
        assumptions.append(
            f"No overtopping up to the {max(return_periods):,.0f}-year flood, "
            "so the annual probability is below "
            f"{1.0/max(return_periods):.1e} on these assumptions.")

    return OvertoppingResult(return_periods=list(return_periods), rows=rows,
                             annual_probability=p_annual,
                             assumptions=assumptions)


# ---------------------------------------------------------------------------
# Fragility (formulation only -- see docs/MATHEMATICAL_FORMULATION.md B.3)
# ---------------------------------------------------------------------------

def lognormal_fragility(im: np.ndarray, median: float,
                        beta: float) -> np.ndarray:
    """P(failure | IM) = Phi( ln(IM/theta) / zeta ), the standard form."""
    im = np.asarray(im, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(np.maximum(im, 1e-12) / median) / max(beta, 1e-9)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def reliability_index(mu_r: float, mu_s: float,
                      beta_r: float, beta_s: float) -> Tuple[float, float]:
    """Lognormal load-resistance: returns (beta, P_f)."""
    beta = math.log(mu_r / mu_s) / math.sqrt(beta_r ** 2 + beta_s ** 2)
    pf = 0.5 * (1.0 + math.erf(-beta / math.sqrt(2.0)))
    return beta, pf


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def failure_probability_report(
        reservoir: Reservoir, crest_m: float,
        spillway: Optional[Spillway] = None,
        mean_annual_flood_m3s: Optional[float] = None,
        flood_cv: float = 0.6,
        barrier_type: str = "engineered",
        starting_level: Optional[float] = None,
        conditional_breach_given_overtopping: Optional[float] = None) -> dict:
    """Event-tree roll-up of annual failure probability.

    Overtopping is routed; every other mechanism falls back to the ICOLD /
    Foster population base rate, which is stated as such.
    """
    out: Dict[str, object] = {
        "purpose": ("Likelihood of failure. The hazard and exposure products "
                    "elsewhere in this run are CONDITIONAL on failure; this is "
                    "the factor that converts them into risk."),
        "mechanisms": {},
        "base_rate_source": BASE_RATE_SOURCE,
    }

    # --- overtopping: computed where possible -----------------------------
    if spillway is not None and mean_annual_flood_m3s:
        ot = overtopping_probability(
            reservoir, crest_m, spillway, mean_annual_flood_m3s,
            flood_cv=flood_cv, starting_level=starting_level)
        d = ot.to_dict()
        p_ot_level = ot.annual_probability

        # P(breach | overtopping) -- an embankment erodes, a concrete gravity
        # dam may not. Defaults are USBR-style order-of-magnitude judgements.
        if conditional_breach_given_overtopping is None:
            conditional_breach_given_overtopping = (
                0.5 if barrier_type == "engineered" else 0.9)
        d["conditional_breach_given_overtopping"] = \
            conditional_breach_given_overtopping
        d["annual_probability_of_breach"] = (
            round(p_ot_level * conditional_breach_given_overtopping, 10)
            if p_ot_level else None)
        d["computed"] = True
        out["mechanisms"]["overtopping"] = d
    else:
        out["mechanisms"]["overtopping"] = {
            "computed": False,
            "annual_probability_of_breach": BASE_RATES_PER_YEAR["overtopping"]["p"],
            "basis": "ICOLD/Foster base rate",
            "missing_inputs": [
                "spillway rating (crest length, sill elevation, coefficient) "
                "-- CWC National Register of Large Dams",
                "mean annual flood and Cv at the site -- CWC gauge records",
            ],
        }

    # --- everything else: base rates, labelled --------------------------
    for mech in ("internal_erosion", "seismic", "structural", "other"):
        out["mechanisms"][mech] = {
            "computed": False,
            "annual_probability_of_breach": BASE_RATES_PER_YEAR[mech]["p"],
            "basis": "ICOLD/Foster population base rate",
            "missing_inputs": _missing_for(mech),
        }

    total = sum(float(m.get("annual_probability_of_breach") or 0.0)
                for m in out["mechanisms"].values())
    out["annual_probability_of_failure"] = round(total, 10)
    out["return_period_y"] = round(1.0 / total, 1) if total > 0 else None
    out["interpretation"] = (
        f"Annual probability of failure ~{total:.2e} per year "
        f"(~1 in {1.0/total:,.0f} years) on the stated assumptions. "
        "Dominated by the mechanisms that could NOT be computed from available "
        "data, so treat this as an order of magnitude, not an estimate."
        if total > 0 else "Not computable from the available inputs.")
    out["honest_summary"] = (
        "The framework computes P(overtopping) by routing, when a spillway "
        "rating and a flood-frequency curve are supplied. It does NOT compute "
        "site-specific fragility for internal erosion, seismic or structural "
        "failure; those use published population base rates and are labelled "
        "accordingly. No number here should be presented as a site-specific "
        "dam-safety assessment.")
    return out


def _missing_for(mech: str) -> List[str]:
    return {
        "internal_erosion": [
            "filter and core gradation, seepage monitoring record",
            "age and inspection history (CWC/DSO annual report)",
        ],
        "seismic": [
            "site seismic hazard curve (PGA vs return period)",
            "dam-specific seismic fragility (theta, zeta)",
        ],
        "structural": [
            "foundation and abutment condition assessment",
            "instrumentation record",
        ],
        "other": ["operational and human-factor review"],
    }.get(mech, [])
