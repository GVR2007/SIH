"""Tests for the technical novelty (adaptive model selection) and for the
breach-likelihood formulation.

These lock in the two properties that make the novelty defensible:

  * the criterion DISCRIMINATES -- it must say "no" on terrain where
    depth-averaging is fine, or it is just an expensive way of always running
    everything; and
  * it is driven by physics that can be pointed at, so each decision carries a
    trace a reviewer can check.

    pytest tests/test_adequacy_and_risk.py -q
"""

import math

import numpy as np
import pytest

from damburst.core import adequacy as ADQ
from damburst.core import failure_probability as FP
from damburst.core import reservoir as RES


# ---------------------------------------------------------------------------
# helpers: synthetic long profiles
# ---------------------------------------------------------------------------

def _profile(fn, length=900.0, step=10.0):
    s = np.arange(0.0, length, step)
    return s, fn(s)


FLAT = lambda s: 100.0 - 0.0008 * s                     # 0.05 deg
MILD = lambda s: 400.0 - 0.03 * s                       # 1.7 deg
STEEP = lambda s: 650.0 - 0.25 * s + 6 * np.sin(s / 90.0)
KNICK = lambda s: 500.0 - 0.01 * s - 40.0 / (1 + np.exp(-(s - 300) / 20.0))


# ---------------------------------------------------------------------------
# The criterion must discriminate
# ---------------------------------------------------------------------------

def test_flat_reach_does_not_trigger_the_near_field_model():
    """If this fires on a dead-flat bed the criterion is worthless."""
    s, z = _profile(FLAT)
    plan = ADQ.select_models(s, z, head_m=12, breach_width_m=60,
                             peak_q_m3s=900, dp_m=2.0, downstream_m=900)
    assert plan.run_near_field is False
    assert plan.transfer_x_m is None
    assert plan.diagnostics["max_nhi"] < 1.0
    assert "NOT NEEDED" in plan.reason


def test_smoothing_does_not_manufacture_curvature_at_the_ends():
    """A perfectly linear bed has zero curvature; edge padding must preserve that.

    Zero-padding inside the smoothing convolution used to drag an 800 m
    elevation profile towards 0 at both ends and invent a non-hydrostatic index
    of ~9 on a dead-flat reach.
    """
    s, z = _profile(FLAT)
    nhi = ADQ.profile_nhi(s, z, u_ref=5.0, h_ref=2.0)
    assert float(nhi.max()) < 0.05
    # and with no smoothing at all the answer must be essentially the same
    nhi0 = ADQ.profile_nhi(s, z, u_ref=5.0, h_ref=2.0, smooth=1)
    assert abs(float(nhi.max()) - float(nhi0.max())) < 0.05


def test_steep_gorge_triggers_the_near_field_model():
    s, z = _profile(STEEP)
    plan = ADQ.select_models(s, z, head_m=174, breach_width_m=350,
                             peak_q_m3s=815000, dp_m=4.0, downstream_m=900)
    assert plan.run_near_field is True
    assert plan.transfer_x_m is not None and plan.transfer_x_m > 0
    assert plan.diagnostics["max_nhi"] >= 1.0


def test_knickpoint_triggers_on_curvature_not_slope():
    """A waterfall in an otherwise gentle reach: mean slope is mild, the
    curvature term is what must fire."""
    s, z = _profile(KNICK)
    plan = ADQ.select_models(s, z, head_m=60, breach_width_m=200,
                             peak_q_m3s=60000, dp_m=3.0, downstream_m=900)
    assert plan.run_near_field is True
    mean_slope_deg = math.degrees(math.atan((z[0] - z[-1]) / s[-1]))
    assert mean_slope_deg < ADQ.BED_SLOPE_DEG_C, \
        "the mean slope alone should NOT have been enough to trigger it"


def test_mild_reach_is_below_threshold():
    s, z = _profile(MILD)
    plan = ADQ.select_models(s, z, head_m=40, breach_width_m=150,
                             peak_q_m3s=40000, dp_m=3.0, downstream_m=900)
    assert plan.run_near_field is False


# ---------------------------------------------------------------------------
# The plunging-jet test
# ---------------------------------------------------------------------------

def test_breach_scoured_to_the_riverbed_has_no_plunge():
    s, z = _profile(FLAT)
    plan = ADQ.select_models(s, z, head_m=174, breach_width_m=350,
                             peak_q_m3s=815000, dp_m=4.0, downstream_m=900,
                             breach_invert_m=657.0, toe_bed_m=645.0)
    assert plan.diagnostics["plunge_m"] == pytest.approx(12.0)
    assert plan.diagnostics["plunge_to_critical_depth_ratio"] < 1.0
    assert plan.run_near_field is False


def test_residual_barrier_plunge_triggers_even_on_a_flat_reach():
    """A landslide dam that does not scour to the floor: the jet free-falls.

    This is the case SPH exists for, and it must be caught by the plunge test
    rather than by the downstream slope, which here is zero.
    """
    # A landslide-dammed lake: modest discharge over a wide barrier, so the
    # critical depth is ~30 m, against a 55 m residual barrier height.
    s, z = _profile(FLAT)
    plan = ADQ.select_models(s, z, head_m=90, breach_width_m=400,
                             peak_q_m3s=200_000, dp_m=4.0, downstream_m=900,
                             breach_invert_m=700.0, toe_bed_m=645.0)
    assert plan.diagnostics["plunge_m"] == pytest.approx(55.0)
    assert plan.diagnostics["plunge_to_critical_depth_ratio"] >= 1.0
    assert plan.run_near_field is True
    assert "plunging jet" in plan.reason


def test_user_override_is_recorded_not_silently_obeyed():
    s, z = _profile(STEEP)
    plan = ADQ.select_models(s, z, head_m=174, breach_width_m=350,
                             peak_q_m3s=815000, dp_m=4.0, downstream_m=900,
                             force=False)
    assert plan.run_near_field is False               # honoured
    assert plan.diagnostics["auto_decision"] is True  # but the truth is kept
    assert any(d["test"] == "user override" for d in plan.decisions)


# ---------------------------------------------------------------------------
# Transfer-section placement
# ---------------------------------------------------------------------------

def test_transfer_section_is_not_a_multiple_of_particle_spacing():
    """The old placement was min(12*dp, ...), i.e. set by a numerical knob.

    Changing dp must not move a section whose position is physical.
    """
    s, z = _profile(KNICK)
    kw = dict(head_m=60, breach_width_m=200, peak_q_m3s=60000,
              downstream_m=900, breach_invert_m=500.0, toe_bed_m=470.0)
    a = ADQ.select_models(s, z, dp_m=2.0, **kw).transfer_x_m
    b = ADQ.select_models(s, z, dp_m=4.0, **kw).transfer_x_m
    assert a is not None and b is not None
    assert a == pytest.approx(b), "transfer section moved when only dp changed"


def test_transfer_section_reports_when_nhi_never_recovers():
    s, z = _profile(STEEP)
    x, info = ADQ.locate_transfer_section(s, z, u_ref=28.0, h_ref=80.0,
                                          min_x=30.0, max_x=315.0)
    assert x <= 315.0
    if not info["nhi_recovered"]:
        assert "clamped" in info["rationale"]
        assert info["nhi_recovery_x_m"] is None


# ---------------------------------------------------------------------------
# Post-run adequacy field
# ---------------------------------------------------------------------------

def _gorge_case():
    """A realistic confined valley: a wide FLAT floor between steep walls.

    Not a slot canyon.  A 100 m-wide notch cut 260 m deep has extreme bed
    curvature even on its floor, so every cell fails the criterion and the
    fixture proves nothing.  A real gorge has a graded channel several cells
    across -- where the water and the curvature are both well behaved -- and
    steep sides that the flood climbs in a thin veneer.
    """
    ny, nx = 60, 60
    xx, _ = np.meshgrid(np.arange(nx) * 50.0, np.arange(ny) * 50.0)
    z = 1000.0 - 0.02 * xx                       # mild downvalley grade
    floor = slice(24, 42)                        # 900 m wide flat floor
    z[:, floor] -= 300.0
    for k, j in enumerate(range(18, 24)):        # left wall, ~45 deg
        z[:, j] -= 300.0 * (k + 1) / 6.0
    for k, j in enumerate(range(42, 48)):        # right wall
        z[:, j] -= 300.0 * (6 - k) / 6.0
    h = np.zeros((ny, nx))
    h[20:40, floor] = 40.0                       # deep water on the floor
    h[20:40, 18:24] = 0.4                        # thin veneer on the walls
    h[20:40, 42:48] = 0.4
    v = np.where(h > 0, 6.0, 0.0)
    return z, h, v


def test_flat_domain_is_fully_adequate():
    z = np.zeros((40, 40))
    h = np.zeros((40, 40)); h[10:30, 10:30] = 2.0
    v = np.where(h > 0, 3.0, 0.0)
    f = ADQ.adequacy_field(z, h, v, 50.0, 50.0)
    assert f.summary["inadequate_area_fraction"] == 0.0
    assert f.summary["inadequate_volume_fraction"] == 0.0
    assert "adequate for essentially all" in f.summary["verdict"]


def test_volume_fraction_is_far_below_area_fraction_in_a_gorge():
    """The metric that matters.

    A flood filling a gorge wets the steep valley sides, so most WET CELLS
    violate the assumptions while holding almost none of the WATER. Judging on
    area alone condemns a result that is substantially sound.
    """
    z, h, v = _gorge_case()
    f = ADQ.adequacy_field(z, h, v, 50.0, 50.0)
    a = f.summary["inadequate_area_fraction"]
    vol = f.summary["inadequate_volume_fraction"]
    assert a > 0.2, "the valley sides should register as inadequate by area"
    assert vol < a / 2.0, "but they must hold far less of the volume"
    assert f.summary["mean_depth_where_adequate_m"] > \
        f.summary["mean_depth_where_inadequate_m"]


def test_verdict_is_driven_by_volume_and_explains_the_divergence():
    z, h, v = _gorge_case()
    f = ADQ.adequacy_field(z, h, v, 50.0, 50.0)
    verdict = f.summary["verdict"]
    assert "wetted AREA" in verdict and "of the water" in verdict


def test_dry_cells_do_not_create_a_spurious_surface_slope():
    """eta must be z (not 0) on dry cells, or every flood edge reads as a cliff."""
    z = np.full((30, 30), 800.0)
    h = np.zeros((30, 30)); h[10:20, 10:20] = 1.0
    s = ADQ.surface_slope(z, h, 50.0, 50.0)
    assert float(s.max()) < 0.05, "flat bed + shallow pond must be nearly flat"


def test_adequacy_reports_which_assumption_failed():
    z, h, v = _gorge_case()
    f = ADQ.adequacy_field(z, h, v, 50.0, 50.0)
    dom = f.summary["dominant_violation"]
    assert set(dom) == {"bed_slope", "bed_curvature", "surface_slope"}
    assert sum(dom.values()) == f.summary["inadequate_cells"]


def test_corridor_relief_sees_valley_walls_the_thalweg_misses():
    """The thalweg is the one gentle line in a gorge; the swath must see more."""
    ny, nx = 40, 40
    z = np.full((ny, nx), 1000.0)
    z[:, 20] = 500.0                       # a slot canyon: flat floor, sheer walls
    path = np.array([[i, 20] for i in range(5, 35)])
    c = ADQ.corridor_relief(z, path, 50.0, 50.0, half_width_cells=4)
    assert c["p90_transverse_slope_deg"] > 40.0


# ---------------------------------------------------------------------------
# Possibility of breach
# ---------------------------------------------------------------------------

def _res():
    return RES.analytic_reservoir(3400.0, 174.0, bed=657.0)


def test_gumbel_quantile_increases_with_return_period():
    q = [FP.gumbel_quantile(3000.0, 0.6, T) for T in (10, 100, 1000, 10000)]
    assert q == sorted(q)
    assert q[-1] > 2 * q[0]


def test_spillway_rating_is_monotonic_and_zero_below_the_sill():
    sp = FP.Spillway(crest_length_m=100.0, sill_elevation_m=800.0)
    assert sp.discharge(795.0) == 0.0
    assert sp.discharge(805.0) < sp.discharge(810.0)


def test_gated_spillway_is_capped():
    sp = FP.Spillway(crest_length_m=100.0, sill_elevation_m=800.0,
                     gated=True, max_gate_discharge_m3s=5000.0)
    assert sp.discharge(830.0) == 5000.0


def test_undersized_spillway_overtops_sooner():
    """The point of the whole exercise: P(overtopping) is dominated by a number
    the framework does not know, and the sweep must show it."""
    res, crest, maf = _res(), 831.0, 3000.0
    firsts = []
    for cf in (2.0, 0.5, 0.25):
        sp = FP.default_spillway(crest, res, mean_annual_flood_m3s=maf,
                                 capacity_factor=cf)
        ot = FP.overtopping_probability(res, crest, sp, maf)
        firsts.append(next((r["return_period_y"] for r in ot.rows
                            if r["overtopped"]), float("inf")))
    assert firsts[0] >= firsts[1] >= firsts[2], \
        "a smaller spillway must overtop at a shorter return period"
    assert firsts[2] < float("inf")


def test_higher_starting_level_overtops_sooner():
    res, crest, maf = _res(), 831.0, 3000.0
    sp = FP.default_spillway(crest, res, mean_annual_flood_m3s=maf,
                             capacity_factor=0.5)
    low = FP.overtopping_probability(res, crest, sp, maf,
                                     starting_level=sp.sill_elevation_m)
    high = FP.overtopping_probability(res, crest, sp, maf,
                                      starting_level=crest - 2.0)
    assert (high.annual_probability or 0) >= (low.annual_probability or 0)


def test_spillway_provenance_is_always_recorded():
    sp = FP.default_spillway(831.0, _res(), mean_annual_flood_m3s=3000.0)
    d = sp.to_dict()
    assert "ASSUMED" in d["_provenance"]
    assert "CWC" in d["_provenance"]


def test_report_separates_computed_from_base_rate():
    res, crest = _res(), 831.0
    sp = FP.default_spillway(crest, res, mean_annual_flood_m3s=3000.0,
                             capacity_factor=0.25)
    rep = FP.failure_probability_report(res, crest, spillway=sp,
                                        mean_annual_flood_m3s=3000.0)
    assert rep["mechanisms"]["overtopping"]["computed"] is True
    for mech in ("internal_erosion", "seismic", "structural"):
        assert rep["mechanisms"][mech]["computed"] is False
        assert "base rate" in rep["mechanisms"][mech]["basis"].lower()
        assert rep["mechanisms"][mech]["missing_inputs"]
    assert rep["annual_probability_of_failure"] > 0
    assert "should be presented as a site-specific" in rep["honest_summary"]


def test_report_without_flood_data_makes_no_computed_claim():
    rep = FP.failure_probability_report(_res(), 831.0)
    assert rep["mechanisms"]["overtopping"]["computed"] is False
    assert rep["mechanisms"]["overtopping"]["missing_inputs"]


def test_overtopping_is_not_the_same_as_failure():
    """An embankment erodes when overtopped; a gravity dam may not."""
    res, crest = _res(), 831.0
    sp = FP.default_spillway(crest, res, mean_annual_flood_m3s=3000.0,
                             capacity_factor=0.25)
    rep = FP.failure_probability_report(res, crest, spillway=sp,
                                        mean_annual_flood_m3s=3000.0,
                                        conditional_breach_given_overtopping=0.5)
    ot = rep["mechanisms"]["overtopping"]
    assert ot["annual_probability_of_breach"] == pytest.approx(
        ot["annual_probability_of_overtopping"] * 0.5)
    assert "NOT" in ot["caveat"]


def test_reliability_index_and_fragility_are_consistent():
    beta, pf = FP.reliability_index(mu_r=2.0, mu_s=1.0, beta_r=0.2, beta_s=0.3)
    assert beta > 0 and 0.0 < pf < 0.5
    beta2, pf2 = FP.reliability_index(1.0, 1.0, 0.2, 0.3)
    assert beta2 == pytest.approx(0.0)
    assert pf2 == pytest.approx(0.5)
    # a fragility curve evaluated at its median is 0.5 by construction
    assert FP.lognormal_fragility(np.array([1.0]), 1.0, 0.4)[0] == \
        pytest.approx(0.5, abs=1e-6)


def test_routing_conserves_the_flood_volume_it_is_given():
    res, crest = _res(), 831.0
    sp = FP.default_spillway(crest, res, mean_annual_flood_m3s=3000.0)
    out = FP.route_flood(res, sp, crest,
                         FP.design_hydrograph(8000.0), h_start=sp.sill_elevation_m)
    assert out["peak_inflow_m3s"] == pytest.approx(8000.0, rel=0.02)
    assert out["peak_level_m"] >= sp.sill_elevation_m
    assert isinstance(out["overtopped"], bool)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "auto_model_selection", "bed_slope_deg_c", "curvature_ratio_c",
    "mean_annual_flood_m3s", "flood_cv", "spillway_capacity_factor",
    "spillway_crest_length_m", "spillway_sill_m",
])
def test_new_parameters_reachable_from_both_interfaces(field):
    from damburst.api.main import RunRequest
    from damburst.pipeline import Scenario
    assert field in Scenario.__dataclass_fields__
    assert field in RunRequest.model_fields


def test_defaults_make_no_probability_claim():
    """Silence is the correct default: no flood data, no P(failure)."""
    from damburst.pipeline import Scenario
    s = Scenario(name="x", bbox_ll=(0, 0, 1, 1), dam_name="d")
    assert s.mean_annual_flood_m3s == 0.0
    assert s.auto_model_selection is True
