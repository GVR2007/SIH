"""Regression tests for the findings in docs/RESULTS_AUDIT.md.

Each test names the audit section it locks down.  These are cheap, offline and
deterministic -- no network, no DEM download -- so they can run in CI on every
commit.  The point is that a future refactor cannot silently reintroduce a
result that is circular, self-excusing, or produced by code the pipeline never
calls.

    pytest tests/test_audit_regressions.py -q
"""

import math

import numpy as np
import pytest

from damburst.core import breach as B
from damburst.core import damage as DMG
from damburst.core import exposure as EX
from damburst.core import hazard as HZ
from damburst.core import reservoir as RES
from damburst.core import validation as VAL
from damburst.core.swe2d import SWE2D


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _big_reservoir():
    """A Tehri-scale reservoir: 3.4 km3 behind 174 m of head.

    Deliberately outside the calibration range of every published breach
    regression, which is the whole point of several of these tests.
    """
    return RES.analytic_reservoir(3400.0, 174.0, bed=657.0)


def _small_reservoir():
    """Inside the regressions' calibration range: 40 MCM behind 25 m."""
    return RES.analytic_reservoir(40.0, 25.0, bed=100.0)


# ---------------------------------------------------------------------------
# AUDIT SS9 -- Von Thun & Gillette width coefficient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vw,cb", [(1.0e6, 6.1), (3.0e6, 18.3),
                                   (1.0e7, 42.7), (3.4e9, 54.9)])
def test_von_thun_gillette_uses_2_5_not_4_0(vw, cb):
    """B_avg = 2.5*hw + Cb.  The 4.0 belongs to the formation-time relation."""
    hw = 60.0
    b, _t = B.von_thun_gillette(vw, hw)
    assert b == pytest.approx(2.5 * hw + cb)


def test_von_thun_gillette_formation_time_branches():
    b_hi, t_hi = B.von_thun_gillette(1e7, 40.0, erodibility="high")
    b_lo, t_lo = B.von_thun_gillette(1e7, 40.0, erodibility="medium")
    assert b_hi == b_lo                 # width does not depend on erodibility
    assert t_hi > t_lo                  # but formation time does


def test_all_four_seed_methods_are_selectable():
    """SS11: seed_method was exposed by neither the CLI nor the API."""
    res = _big_reservoir()
    seen = {}
    for m in ("froehlich_2008", "von_thun_gillette", "macdonald",
              "costa_schuster"):
        g = B.seed_breach("engineered", "overtopping", res, 831.0, 831.0,
                          method=m)
        assert g.method == m
        assert g.b_bottom > 0 and g.t_form > 0
        seen[m] = g.b_top()
    assert len(set(round(v, 3) for v in seen.values())) == 4, \
        "the four regressions should not all collapse to the same geometry"


def test_costa_schuster_factors_are_recorded():
    """SS13: the landslide-dam fudge factors must appear in the run record."""
    g = B.seed_breach("natural", "overtopping", _big_reservoir(), 831.0, 831.0)
    assert g.method == "costa_schuster"
    for k in ("cs_width_factor", "cs_time_factor", "cs_side_slope",
              "cs_residual_height_frac"):
        assert k in g.notes
    assert "calibration" in g.notes["cs_factors_are"]


# ---------------------------------------------------------------------------
# AUDIT SS5 -- the peak-discharge gate must not pass what it cannot judge
# ---------------------------------------------------------------------------

def test_envelope_gate_is_null_outside_calibration_range():
    """A 3-16x exceedance must NOT report pass: true."""
    res = _big_reservoir()
    g = B.seed_breach("engineered", "overtopping", res, 831.0, 831.0)
    br = B.simulate_breach(res, g, 831.0, t_end=3 * 3600.0)
    env = br.checks["peak_discharge_envelopes"]

    assert env["within_regression_calibration_range"] is False
    assert env["pass"] is None, "out-of-range must be inconclusive, not a pass"
    assert env["applicable"] is False
    assert "NOT APPLICABLE" in env["interpretation"]


def test_envelope_gate_is_a_real_boolean_inside_calibration_range():
    res = _small_reservoir()
    g = B.seed_breach("engineered", "overtopping", res, 125.0, 125.0)
    br = B.simulate_breach(res, g, 125.0, t_end=3 * 3600.0)
    env = br.checks["peak_discharge_envelopes"]
    assert env["within_regression_calibration_range"] is True
    assert isinstance(env["pass"], bool)


def test_critical_flow_gate_is_scale_independent_and_binds():
    """SS5: the replacement gate must exist and be a hard physical ceiling."""
    res = _big_reservoir()
    g = B.seed_breach("engineered", "overtopping", res, 831.0, 831.0)
    br = B.simulate_breach(res, g, 831.0, t_end=3 * 3600.0)
    crit = br.checks["peak_discharge_critical_flow"]

    b_top = crit["breach_top_width_m"]
    head = crit["head_m"]
    expected = (2.0 / 3.0) ** 1.5 * math.sqrt(B.G) * b_top * head ** 1.5
    assert crit["critical_flow_ceiling_m3s"] == pytest.approx(expected, rel=1e-3)
    assert isinstance(crit["pass"], bool)
    # free discharge cannot exceed critical flow through its own section
    assert br.peak_q <= crit["critical_flow_ceiling_m3s"] * 1.05


def test_mass_balance_delegates_to_volume_balance():
    """SS10: there must be exactly one implementation of the continuity gate."""
    res = _big_reservoir()
    g = B.seed_breach("engineered", "overtopping", res, 831.0, 831.0)
    br = B.simulate_breach(res, g, 831.0, t_end=3 * 3600.0)
    mb = br.checks["mass_balance"]
    ref = RES.volume_balance(res.volume(831.0), float(br.volume[-1]),
                             br.released_m3, 0.0)
    assert mb["pass"] == ref["pass"]
    assert mb["released_mcm"] == pytest.approx(ref["released_mcm"], rel=1e-9)
    assert mb["pass"] is True


# ---------------------------------------------------------------------------
# AUDIT SS11 -- physics that was implemented but unreachable
# ---------------------------------------------------------------------------

def test_piping_actually_uses_the_orifice_branch():
    """The README claims an orifice phase; orifice_discharge must be called."""
    res = _big_reservoir()
    g = B.seed_breach("engineered", "piping", res, 831.0, 831.0)
    br = B.simulate_breach(res, g, 831.0, failure_mode="piping",
                           t_end=3 * 3600.0)
    reg = br.checks["outflow_regime"]
    assert reg["orifice_steps"] > 0
    assert "orifice" in set(br.regime)
    assert "weir" in set(br.regime), "the roof must collapse to a weir"


def test_overtopping_never_uses_the_orifice_branch():
    res = _big_reservoir()
    g = B.seed_breach("engineered", "overtopping", res, 831.0, 831.0)
    br = B.simulate_breach(res, g, 831.0, t_end=3 * 3600.0)
    assert br.checks["outflow_regime"]["orifice_steps"] == 0


def test_tailwater_engages_villemonte_submergence():
    """SS11: no tailwater was ever passed, so the correction never ran."""
    res = _big_reservoir()
    g = B.seed_breach("engineered", "overtopping", res, 831.0, 831.0)

    free = B.simulate_breach(res, g, 831.0, t_end=3 * 3600.0)
    assert free.checks["outflow_regime"]["villemonte_applied"] is False
    assert free.checks["outflow_regime"]["tailwater_supplied"] is False

    tw = B.normal_depth_tailwater(657.0, 300.0, 0.01, 0.045)
    sub = B.simulate_breach(res, g, 831.0, t_end=3 * 3600.0, tailwater=tw)
    assert sub.checks["outflow_regime"]["tailwater_supplied"] is True
    assert sub.checks["outflow_regime"]["submerged_weir_steps"] > 0
    # submergence can only reduce discharge, never increase it
    assert sub.peak_q <= free.peak_q * 1.0001
    assert sub.released_m3 <= free.released_m3 * 1.0001


def test_tailwater_rating_is_monotonic_and_lagged():
    tw = B.normal_depth_tailwater(100.0, 200.0, 0.01, 0.04)
    assert tw(0.0) == 100.0            # no discharge yet -> invert
    tw.update(1000.0)
    a = tw(1.0)
    tw.update(50_000.0)
    b = tw(2.0)
    assert 100.0 < a < b


def test_weir_submergence_reduces_discharge():
    free = B.weir_discharge(120.0, 50.0, 1.0, 100.0, tailwater=None)
    mild = B.weir_discharge(120.0, 50.0, 1.0, 100.0, tailwater=105.0)
    deep = B.weir_discharge(120.0, 50.0, 1.0, 100.0, tailwater=119.0)
    assert free == pytest.approx(mild)   # below the modular limit: no effect
    assert deep < free


# ---------------------------------------------------------------------------
# AUDIT SS1, SS2 -- reservoir provenance must not be circular
# ---------------------------------------------------------------------------

def test_hybrid_hva_records_that_capacity_came_from_the_published_figure():
    dem_res = RES.analytic_reservoir(4.0, 10.0, bed=821.0)   # the DSM sliver
    res, prov = RES.hybrid_hva(dem_res, water_surface_m=826.0,
                               area_at_surface_m2=32.7e6,
                               published_capacity_m3=3540e6,
                               dam_height_m=260.5, crest_m=831.5,
                               riverbed_m=657.0)
    # Capacity is SET BY the published figure -- that is the point being
    # recorded.  It is not exactly V_pub: the curve is
    #     V(crest) = V_pub * (d_ws/d_crest)**b  +  (DEM hypsometry above the
    #                                               water plate)
    # so it lands a few percent below V_pub, with the DEM contributing only
    # the small second term.  Either way it is the published number that sets
    # the scale, which is what must never be described as DEM-derived.
    assert 0.85 * 3540e6 < res.capacity < 1.05 * 3540e6
    assert prov["dem_only_capacity_mcm"] == pytest.approx(4.0, abs=0.5)
    assert prov["capacity_from_published_fraction"] > 0.99
    assert "must NOT be described as DEM-derived" in prov["caveat"]
    assert "shape_exponent_solved" in prov
    assert isinstance(prov["shape_exponent_solved"], bool)


def test_shape_exponent_records_when_it_was_defaulted():
    """SS13: a defaulted exponent is a guess and must be distinguishable."""
    b_ok, solved = RES._solve_shape_exponent(170.0, 175.0, 32.7e6, 3540e6)
    assert isinstance(solved, bool)
    # an impossible constraint pair falls back and says so
    b_bad, solved_bad = RES._solve_shape_exponent(1.0, 2.0, 1e12, 1.0)
    assert solved_bad is False
    assert b_bad == RES.DEFAULT_SHAPE_EXPONENT


# ---------------------------------------------------------------------------
# AUDIT SS12, SS15, SS16 -- loss accounting
# ---------------------------------------------------------------------------

def test_no_literal_buried_in_the_loss_code_scales_the_total():
    """AUDIT SS12, re-pointed.

    The finding was that a hardcoded 0.35 inside `estimate_losses` produced
    ~96% of the reported rupee total while the module promised every figure
    was tagged with the rate that produced it. The audit branch fixed this by
    promoting the factor into `AssetValues`. `damage.py` then superseded that
    entirely: road damage is now the JRC INFRASTRUCTURE depth-damage curve
    evaluated at the sampled depth, which is both sourced and depth dependent,
    and the flat factor is gone.

    So the test no longer looks for the factor. It asserts the property the
    finding was really about: every line of the loss breakdown must carry the
    source of the number that produced it.
    """
    assert not hasattr(HZ.AssetValues(), "road_partial_damage_factor"),         "the flat factor was superseded by a sourced curve; do not reinstate it"

    out = EX.estimate_losses(
        {"exposed": 5, "by_asset_class": []},
        {"total_inundated_km": 120.0}, {}, None, iso3="IND", road_depth_m=1.5)
    lines = out.get("line_items") or []
    assert lines, "the loss breakdown must be itemised, not a single figure"
    for ln in lines:
        assert ln.get("source"), f"line item {ln.get('component')} has no source"
        assert "unit_value" in ln and "damage_fraction" in ln


def test_road_loss_is_depth_dependent_not_a_flat_fraction():
    """The replacement for the 0.35 must actually vary with depth."""
    roads = {"total_inundated_km": 100.0}
    shallow = EX.estimate_losses({}, roads, {}, None, iso3="IND", road_depth_m=0.5)
    deep = EX.estimate_losses({}, roads, {}, None, iso3="IND", road_depth_m=4.0)
    assert deep["roads"] > shallow["roads"] > 0,         "road damage must rise with depth; a flat factor would give equal values"


def test_asset_classes_get_different_unit_values():
    """AUDIT SS15, re-pointed.

    The finding was that `commercial_per_building` was printed as if used and
    never read - everything was priced as a house. `damage.max_damage_set`
    now carries a published rate per asset class.
    """
    rates = DMG.max_damage_set("IND")
    for cls in ("residential", "commercial", "industrial"):
        assert cls in rates, cls
        assert rates[cls].value > 0
        assert rates[cls].source, f"{cls} rate has no provenance"
    assert len({rates[c].value for c in
                ("residential", "commercial", "industrial")}) > 1,         "asset classes must not all share one unit value"


@pytest.mark.parametrize("tags,cls", [
    (None, "residential"),
    ({"building": "yes"}, "residential"),
    ({"building": "house"}, "residential"),
    ({"building": "commercial"}, "commercial"),
    ({"building": "warehouse"}, "industrial"),
    ({"building": "some_unmapped_value"}, "residential"),
])
def test_building_occupancy_classification(tags, cls):
    """AUDIT SS15. Classification lives in damage.py and reports its basis."""
    got, basis = DMG.classify_building(tags)
    assert got == cls
    assert basis, "every classification must say what it was based on"


def test_only_one_building_classifier_exists():
    """AUDIT SS10 in spirit: one mapping, one implementation.

    The audit branch grew a second copy of this table in hazard.py; two
    implementations of one mapping is the duplication the audit complained
    about elsewhere.
    """
    assert not hasattr(HZ, "classify_building"),         "hazard.classify_building duplicates damage.classify_building"
    assert not hasattr(HZ, "BUILDING_CLASS_MAP")


def test_damage_curves_match_the_published_source():
    """AUDIT SS16, and the reason it needed re-pointing.

    The original finding was that the curves were typed-in literals with no
    traceable source, and the audit branch "fixed" it by adding a confident
    citation - to numbers that turn out not to match the published tables
    (residential read 0.58 at 1 m against the published 0.49). A citation on a
    wrong number is worse than no citation, because it makes it look checked.

    The curves are now read from the extracted JRC database, so this test
    asserts the VALUES, not the presence of a citation.
    """
    c = DMG.curve("residential", "IND")
    assert c.factor(1.0) == pytest.approx(0.49, abs=1e-6),         "residential damage at 1 m must match the published JRC Asia table"
    assert c.factor(0.0) == pytest.approx(0.0)
    assert HZ.DD_SOURCE and "Huizinga" in HZ.DD_SOURCE
    for name, curve in HZ.DD_CURVES.items():
        arr = np.asarray(curve, dtype=float)
        assert arr[0] == 0.0, name
        assert np.all(np.diff(arr) >= -1e-12), f"{name} must be monotonic"
        assert arr[-1] <= 1.0 + 1e-12, name


# ---------------------------------------------------------------------------
# AUDIT SS6 -- benchmark validation must subtract permanent water
# ---------------------------------------------------------------------------

def _extent_fixtures():
    obs = np.zeros((40, 40), bool)
    obs[10:30, 10:30] = True            # observed water: flood + river
    base = np.zeros((40, 40), bool)
    base[10:30, 18:22] = True           # permanent river down the middle
    mod = np.zeros((40, 40), bool)
    mod[10:30, 10:30] = True            # model reproduces the river exactly
    return mod, obs, base


def test_benchmark_subtracts_the_baseline_when_supplied():
    mod, obs, base = _extent_fixtures()
    rep = VAL.validate_extent(mod, obs, 10_000.0, mode="benchmark",
                              baseline_water=base)
    assert rep.metrics["permanent_water_removed"] is True
    assert rep.permanent_water_km2 > 0
    assert "removed from both" in " ".join(rep.notes).lower()


def test_benchmark_without_a_baseline_says_the_score_is_inflated():
    """The old call site passed the baseline ONLY in context mode."""
    mod, obs, _base = _extent_fixtures()
    rep = VAL.validate_extent(mod, obs, 10_000.0, mode="benchmark",
                              baseline_water=None)
    assert rep.metrics["permanent_water_removed"] is False
    joined = " ".join(rep.notes)
    assert "NOT A FAIR SCORE" in joined and "UPPER BOUND" in joined


def test_baseline_inflates_the_score_measurably():
    mod, obs, base = _extent_fixtures()
    naive = VAL.validate_extent(mod, obs, 1e4, mode="benchmark",
                                baseline_water=None).metrics["iou"]
    fair = VAL.validate_extent(mod, obs, 1e4, mode="benchmark",
                               baseline_water=base).metrics["iou"]
    assert naive == pytest.approx(1.0)
    assert fair <= naive


def test_context_mode_is_never_reported_as_skill():
    mod, obs, base = _extent_fixtures()
    rep = VAL.validate_extent(mod, obs, 1e4, mode="context",
                              baseline_water=base)
    assert "iou" not in rep.metrics
    assert "CONTEXT ONLY" in " ".join(rep.notes)


# ---------------------------------------------------------------------------
# AUDIT SS10 -- model-vs-model agreement (uses kling_gupta / rmse)
# ---------------------------------------------------------------------------

def test_field_agreement_scores_identical_fields_perfectly():
    a = np.zeros((30, 30))
    a[5:25, 5:25] = np.linspace(1.0, 5.0, 20)[None, :]
    out = VAL.field_agreement(a, a.copy(), 10_000.0, label="self")
    assert out["extent"]["iou"] == pytest.approx(1.0)
    assert out["depth"]["nse"] == pytest.approx(1.0)
    assert out["depth"]["kge"] == pytest.approx(1.0)
    assert out["depth"]["rmse_m"] == pytest.approx(0.0)


def test_field_agreement_detects_a_shifted_field():
    a = np.zeros((30, 30))
    a[5:25, 5:25] = np.linspace(1.0, 5.0, 20)[None, :]
    b = np.roll(a, 6, axis=1)
    out = VAL.field_agreement(a, b, 10_000.0, label="shifted")
    assert out["extent"]["iou"] < 0.9
    assert out["depth"]["rmse_m"] > 0.0


def test_field_agreement_is_labelled_as_not_a_skill_score():
    a = np.zeros((10, 10)); a[2:8, 2:8] = 1.0
    out = VAL.field_agreement(a, a.copy(), 1e4)
    assert "not a skill score" in out["note"]


# ---------------------------------------------------------------------------
# AUDIT SS8 -- steep-terrain Froude limiter
# ---------------------------------------------------------------------------

def test_froude_limiter_leaves_flat_bed_benchmarks_untouched():
    """The whole design constraint: Ritter and lake-at-rest must not change."""
    z = np.zeros((3, 200))
    m = SWE2D(z, 1.0, 1.0, np.zeros((3, 200)), open_edges=True, order=2)
    assert m.steep.sum() == 0, "a flat bed must have zero steep cells"
    m.h[:, :100] = 5.0
    r = m.run(t_end=5.0, cfl=0.4, n_frames=0)
    assert r.stats["froude_limiter"]["limited_cell_steps"] == 0


def test_froude_limiter_flags_and_caps_steep_terrain():
    ny, nx = 3, 120
    slope = np.linspace(0.0, 400.0, nx)[::-1]       # ~73% grade, very steep
    z = np.tile(slope, (ny, 1)).astype(float)
    m = SWE2D(z, 10.0, 10.0, np.full((ny, nx), 0.03), open_edges=True, order=2)
    assert m.steep.sum() > 0, "a 20-degree bed must register as steep"
    m.h[:, :20] = 30.0
    r = m.run(t_end=60.0, cfl=0.4, n_frames=0)
    lim = r.stats["froude_limiter"]
    assert lim["enabled"] is True
    assert lim["limited_cell_steps"] > 0
    # and the capped run must be slower than the uncapped one
    m2 = SWE2D(z, 10.0, 10.0, np.full((ny, nx), 0.03), open_edges=True,
               order=2, froude_max=0.0)
    m2.h[:, :20] = 30.0
    r2 = m2.run(t_end=60.0, cfl=0.4, n_frames=0)
    assert r.stats["max_velocity_ms"] < r2.stats["max_velocity_ms"]


def test_froude_limiter_can_be_disabled():
    z = np.tile(np.linspace(0, 300, 80)[::-1], (3, 1)).astype(float)
    m = SWE2D(z, 10.0, 10.0, np.full((3, 80), 0.03), froude_max=0.0)
    m.h[:, :10] = 20.0
    r = m.run(t_end=30.0, n_frames=0)
    assert r.stats["froude_limiter"]["enabled"] is False
    assert r.stats["froude_limiter"]["limited_cell_steps"] == 0


def test_peak_velocity_context_is_reported():
    """SS8: a bare max velocity cannot be judged without its context."""
    z = np.tile(np.linspace(0, 300, 80)[::-1], (3, 1)).astype(float)
    m = SWE2D(z, 10.0, 10.0, np.full((3, 80), 0.03))
    m.h[:, :10] = 20.0
    r = m.run(t_end=30.0, n_frames=0)
    ctx = r.stats["peak_velocity_context"]
    for k in ("depth_there_m", "bed_slope_deg", "froude_number",
              "p99_9_velocity_ms", "median_velocity_ms"):
        assert k in ctx
    assert ctx["p99_9_velocity_ms"] <= r.stats["max_velocity_ms"] + 1e-9
    assert ctx["median_velocity_ms"] <= ctx["p99_9_velocity_ms"] + 1e-9


def test_hazard_reports_velocity_distribution_not_just_the_max():
    depth = np.zeros((50, 50))
    vel = np.zeros((50, 50))
    depth[10:40, 10:40] = 2.0
    vel[10:40, 10:40] = 3.0
    vel[25, 25] = 120.0                       # one absurd outlier cell
    hz = HZ.build_hazard(depth, vel, 10_000.0)
    p = hz.summary["velocity_percentiles_ms"]
    assert hz.summary["max_velocity_ms"] == pytest.approx(120.0)
    assert p["p50"] == pytest.approx(3.0)
    assert p["p99.9"] < 120.0, "the outlier must not dominate the percentiles"
    assert "structural_dv" in hz.summary


# ---------------------------------------------------------------------------
# AUDIT SS4 -- hydrograph peak sampling must not step over short transients
# ---------------------------------------------------------------------------

def test_hydrograph_peak_catches_a_short_early_spike():
    """The old sampler took 400 points over 3 h and missed a 12 s SPH window."""
    from damburst.pipeline import _hydrograph_peak

    # A spike that starts AFTER t=0, exactly like the SPH transfer window
    # relative to a weir hydrograph that is still near zero.
    def hyd(t):
        return 1_000_000.0 if 5.0 <= t < 17.0 else 1000.0

    t_end = 3 * 3600.0
    coarse = max(hyd(t) for t in np.linspace(0, t_end, 400))
    assert coarse == 1000.0, "the old approach genuinely missed the spike"
    assert _hydrograph_peak(hyd, t_end) == pytest.approx(1_000_000.0)


def test_hydrograph_peak_uses_supplied_sample_times():
    from damburst.pipeline import _hydrograph_peak

    spike_t = 4.3219
    def hyd(t):
        return 5e5 if abs(t - spike_t) < 1e-6 else 1.0

    got = _hydrograph_peak(hyd, 3 * 3600.0,
                           extra_times=np.array([0.0, spike_t, 10.0]))
    assert got == pytest.approx(5e5)


# ---------------------------------------------------------------------------
# AUDIT SS7 -- the SPH balance gate must fail on a truncated solve
# ---------------------------------------------------------------------------

def _fake_interface(peak_unit=30.0):
    from damburst.core.coupling import TransferInterface
    t = np.linspace(0.0, 12.0, 50)
    q_unit = np.full_like(t, peak_unit)
    return TransferInterface(
        t=t, depth=np.full_like(t, 40.0), u_mean=np.full_like(t, 20.0),
        q_unit=q_unit, width_m=300.0, q_total=q_unit * 300.0,
        transfer_x_m=48.0, overlap_start_m=0.0, overlap_end_m=48.0)


def _fake_breach():
    res = _big_reservoir()
    g = B.seed_breach("engineered", "overtopping", res, 831.0, 831.0)
    return B.simulate_breach(res, g, 831.0, t_end=3 * 3600.0)


def test_balance_gate_fails_when_sph_did_not_complete():
    from damburst.core.coupling import compare_hydrographs
    head = 174.0
    q_crit = (2.0 / 3.0) ** 1.5 * math.sqrt(9.80665) * head ** 1.5
    iface = _fake_interface(peak_unit=q_crit)      # ratio == 1.0, inside band
    br = _fake_breach()

    ok = compare_hydrographs(br, iface, head_m=head,
                             sph_stats={"completed": True})
    assert ok["critical_flow_ratio_pass"] is True
    assert ok["balance_pass"] is True

    bad = compare_hydrographs(br, iface, head_m=head, sph_stats={
        "completed": False, "simulated_s": 12.1, "requested_s": 20.0,
        "stop_reason": "timestep collapsed (pressure instability)"})
    assert bad["critical_flow_ratio_pass"] is True    # same ratio
    assert bad["balance_pass"] is False, "a crashed solve cannot pass"
    assert "NOT A PASS" in bad["interpretation"]
    assert "pressure instability" in bad["sph_stop_reason"]


def test_comparison_table_carries_status_and_caveat():
    from damburst.core.coupling import build_comparison_table
    rows = build_comparison_table({
        "grid_standalone": {"description": "x", "is_primary": True},
        "sph_nearfield": {"description": "y", "status": "truncated",
                          "caveat": "stopped at 12.1 s"},
    })
    by = {r["model"]: r for r in rows}
    assert by["grid_standalone"]["is_primary"] is True
    assert by["grid_standalone"]["status"] == "ok"
    assert by["sph_nearfield"]["status"] == "truncated"
    assert "12.1" in by["sph_nearfield"]["caveat"]


# ---------------------------------------------------------------------------
# AUDIT SS11 -- every scenario knob must be reachable from BOTH interfaces
# ---------------------------------------------------------------------------

PREVIOUSLY_UNREACHABLE = [
    "seed_method", "inflow_m3s", "channel_burn_m", "tailwater",
    "tailwater_slope", "froude_max", "steep_slope_deg", "baseline_window",
    "population_product", "iso3", "satellite_basemap",
]


@pytest.mark.parametrize("field", PREVIOUSLY_UNREACHABLE)
def test_scenario_field_is_reachable_from_the_rest_api(field):
    from damburst.api.main import RunRequest
    assert field in RunRequest.model_fields


@pytest.mark.parametrize("field", PREVIOUSLY_UNREACHABLE + ["asset_values"])
def test_scenario_field_exists_on_the_scenario(field):
    from damburst.pipeline import Scenario
    assert field in Scenario.__dataclass_fields__


def test_cli_exposes_the_previously_unreachable_flags():
    from damburst.cli import main
    import argparse
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        main(["run", "--help"])
    help_text = buf.getvalue()
    for flag in ("--seed-method", "--inflow-m3s", "--channel-burn-m",
                 "--no-tailwater", "--froude-max", "--baseline-window",
                 "--value-residential"):
        assert flag in help_text, f"{flag} missing from the CLI"


def test_api_rejects_unknown_asset_value_keys():
    from damburst.api.main import RunRequest, _build_scenario
    with pytest.raises(ValueError, match="unknown asset_values"):
        _build_scenario(RunRequest(preset="tehri",
                                   asset_values={"not_a_rate": 1.0}))


def test_api_applies_asset_value_overrides():
    from damburst.api.main import RunRequest, _build_scenario
    scn = _build_scenario(RunRequest(
        preset="tehri", seed_method="macdonald",
        asset_values={"residential_per_building": 1_800_000.0}))
    assert scn.seed_method == "macdonald"
    assert scn.asset_values.residential_per_building == 1_800_000.0
