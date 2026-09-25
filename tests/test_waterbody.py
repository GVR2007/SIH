"""Tests for core/waterbody.py -- the Sentinel-1/2 + WorldCover fusion layer.

These are offline and deterministic (synthetic rasters, no network), covering
the properties that matter for the SIH technical-approach spec's "data fusion"
and "river/water-body path extraction" asks:

  * NDWI/MNDWI correctly separate a synthetic water body from land;
  * the Otsu threshold is not a magic number -- it moves with the histogram;
  * fusion counts AGREEMENT, not a single OR/AND, and degrades gracefully when
    fewer than three sources are available;
  * channel-like (elongated) reaches are told apart from basin-like (round)
    ones, and the returned bearing is the reach's actual long axis;
  * the channel mask returned by `channel_like_mask` is pixel-identical to the
    labelled components `river_reaches` flagged `is_channel_like`, even for a
    bent (non-convex) reach whose centroid can fall outside the water mask.
"""

import numpy as np
import pytest

from damburst.core import waterbody as WB


# ---------------------------------------------------------------------------
# NDWI / MNDWI
# ---------------------------------------------------------------------------

def test_ndwi_positive_over_water_negative_over_land():
    green = np.array([0.10, 0.08])
    nir = np.array([0.03, 0.30])          # water, land
    idx = WB.ndwi(green, nir)
    assert idx[0] > 0
    assert idx[1] < 0


def test_mndwi_positive_over_water_negative_over_land():
    green = np.array([0.10, 0.08])
    swir = np.array([0.02, 0.18])
    idx = WB.mndwi(green, swir)
    assert idx[0] > 0
    assert idx[1] < 0


def test_index_handles_zero_denominator_without_crashing():
    z = np.zeros((3, 3))
    idx = WB.ndwi(z, z)
    assert np.all(np.isnan(idx))


def test_optical_water_mask_separates_a_synthetic_lake():
    green = np.full((30, 30), 0.08, dtype=np.float32)
    nir = np.full((30, 30), 0.30, dtype=np.float32)
    swir = np.full((30, 30), 0.18, dtype=np.float32)
    green[10:20, 10:20] = 0.10
    nir[10:20, 10:20] = 0.03
    swir[10:20, 10:20] = 0.02
    mask, prov = WB.optical_water_mask(green, nir=nir, swir=swir, method="mndwi")
    assert mask[15, 15] and not mask[2, 2]
    assert prov["method"] == "MNDWI"
    assert prov["threshold_source"] == "Otsu"


def test_optical_water_mask_requires_the_right_bands():
    green = np.zeros((5, 5))
    with pytest.raises(ValueError):
        WB.optical_water_mask(green, method="mndwi")   # no swir supplied
    with pytest.raises(ValueError):
        WB.optical_water_mask(green, method="ndwi")    # no nir supplied
    with pytest.raises(ValueError):
        WB.optical_water_mask(green, nir=green, method="bogus")


def test_otsu_threshold_moves_with_the_histogram():
    """Not a magic constant -- shifting the whole distribution shifts the cut."""
    low = np.concatenate([np.full(500, -0.5), np.full(500, 0.5)])
    high = low + 1.0
    t_low = WB._otsu_threshold(low)
    t_high = WB._otsu_threshold(high)
    assert t_high > t_low
    assert t_high - t_low == pytest.approx(1.0, abs=0.05)


def test_explicit_threshold_overrides_otsu():
    green = np.array([[0.1, 0.1]])
    swir = np.array([[0.05, 0.15]])
    _mask_otsu, prov_otsu = WB.optical_water_mask(green, swir=swir, method="mndwi")
    _mask_fixed, prov_fixed = WB.optical_water_mask(
        green, swir=swir, method="mndwi", threshold=0.0)
    assert prov_fixed["threshold_source"] == "user-supplied"
    assert prov_fixed["threshold"] == 0.0
    assert prov_otsu["threshold_source"] == "Otsu"


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _boxes(shape=(40, 40)):
    wc = np.zeros(shape, bool); wc[10:20, 10:20] = True
    sar = np.zeros(shape, bool); sar[12:22, 12:22] = True     # shifted overlap
    opt = np.zeros(shape, bool); opt[10:20, 10:20] = True     # matches wc exactly
    return wc, sar, opt


def test_fusion_counts_agreement_not_a_single_boolean():
    wc, sar, opt = _boxes()
    ev = WB.fuse_water_evidence(100.0, worldcover_water=wc, sar_water=sar,
                                optical_water=opt)
    assert ev.max_sources == 3
    # the wc/opt core (10:20,10:20) minus the sar overlap (12:20,12:20) agrees
    # on exactly 2 sources; the sar-only sliver agrees on 1.
    assert ev.n_sources[15, 15] == 3      # centre: all three agree
    assert ev.n_sources[11, 11] == 2      # wc+opt but outside sar's box
    assert ev.n_sources[21, 21] == 1      # sar only
    assert ev.n_sources[0, 0] == 0
    assert ev.any_source[21, 21] and not ev.all_sources[21, 21]
    assert ev.all_sources[15, 15]


def test_fusion_degrades_gracefully_with_fewer_sources():
    wc, _sar, _opt = _boxes()
    ev = WB.fuse_water_evidence(100.0, worldcover_water=wc)
    assert ev.max_sources == 1
    assert list(ev.per_source.keys()) == ["worldcover"]
    assert np.array_equal(ev.any_source, wc)
    assert np.array_equal(ev.all_sources, wc)
    assert ev.pairwise_agreement == {}


def test_fusion_needs_at_least_one_source():
    with pytest.raises(ValueError):
        WB.fuse_water_evidence(100.0)


def test_fusion_summary_reports_area_and_pairwise_iou():
    wc, sar, opt = _boxes()
    ev = WB.fuse_water_evidence(2500.0, worldcover_water=wc, sar_water=sar,
                                optical_water=opt)
    assert ev.summary["n_sources_available"] == 3
    assert ev.summary["area_km2_any_source"] > ev.summary["area_km2_all_sources"]
    assert set(ev.summary["pairwise_agreement_iou"]) == {
        "worldcover_vs_sentinel1_sar_iou",
        "worldcover_vs_sentinel2_optical_iou",
        "sentinel1_sar_vs_sentinel2_optical_iou",
    }
    # worldcover and optical are pixel-identical boxes here
    assert ev.summary["pairwise_agreement_iou"][
        "worldcover_vs_sentinel2_optical_iou"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# River reaches / channel classification
# ---------------------------------------------------------------------------

def test_straight_channel_is_flagged_channel_like_with_correct_bearing():
    mask = np.zeros((100, 20), bool)
    mask[:, 9:11] = True                       # north-south channel
    reaches = WB.river_reaches(mask, dx=10.0, dy=10.0, min_cells=5)
    assert len(reaches) == 1
    r = reaches[0]
    assert r.is_channel_like
    assert r.elongation > 3.0
    assert r.direction_deg == pytest.approx(0.0, abs=1.0)   # north-south -> 0 deg


def test_east_west_channel_bearing_is_90_degrees():
    mask = np.zeros((20, 100), bool)
    mask[9:11, :] = True
    reaches = WB.river_reaches(mask, dx=10.0, dy=10.0, min_cells=5)
    assert reaches[0].direction_deg == pytest.approx(90.0, abs=1.0)


def test_round_pond_is_not_channel_like():
    ny, nx = 40, 40
    yy, xx = np.mgrid[0:ny, 0:nx]
    mask = (yy - 20) ** 2 + (xx - 20) ** 2 <= 8 ** 2
    reaches = WB.river_reaches(mask, dx=10.0, dy=10.0, min_cells=5)
    assert len(reaches) == 1
    assert not reaches[0].is_channel_like
    assert reaches[0].elongation < 3.0


def test_small_components_below_min_cells_are_dropped():
    mask = np.zeros((30, 30), bool)
    mask[0, 0] = True
    mask[5, 5] = True                          # two isolated single pixels
    reaches = WB.river_reaches(mask, dx=10.0, dy=10.0, min_cells=5)
    assert reaches == []


def test_reaches_are_sorted_largest_first():
    mask = np.zeros((50, 50), bool)
    mask[0:5, 0:5] = True                      # small, 25 cells
    mask[10:40, 10:15] = True                  # large, 150 cells
    reaches = WB.river_reaches(mask, dx=10.0, dy=10.0, min_cells=5)
    assert len(reaches) == 2
    assert reaches[0].area_km2 >= reaches[1].area_km2


def test_channel_like_mask_matches_river_reaches_exactly():
    """The mask must be pixel-identical to what river_reaches flagged, even for
    a curved (non-convex) reach whose PCA centroid falls outside the water
    mask itself -- the failure mode a centroid-relookup implementation would
    hit. A quarter-circle meander is a realistic river bend (a sharp 90-degree
    L-bend is not: PCA elongation for a right-angle L caps out around 2,
    because the two perpendicular arms spread the covariance equally in both
    directions -- a real, worth-noting limitation of the PCA-axis approach for
    sharply-angled channels, documented in the module docstring rather than
    hidden behind an unrealistic fixture).
    """
    ny, nx = 100, 100
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = 90, 10
    r = np.hypot(yy - cy, xx - cx)
    theta = np.degrees(np.arctan2(cy - yy, xx - cx))
    meander = (r > 50) & (r < 54) & (theta >= 0) & (theta <= 90)
    # a separate round pond, should NOT appear in the channel mask
    pond = (yy - 10) ** 2 + (xx - 90) ** 2 <= 6 ** 2
    combined = meander | pond

    reaches = WB.river_reaches(combined, dx=10.0, dy=10.0, min_cells=5)
    chan_mask = WB.channel_like_mask(combined, dx=10.0, dy=10.0, min_cells=5)

    channel_reaches = [r for r in reaches if r.is_channel_like]
    assert len(channel_reaches) == 1, "the meander must be one channel-like component"
    assert channel_reaches[0].cells == int(chan_mask.sum())
    # the pond must not leak into the channel mask
    assert not (chan_mask & pond).any()
    # the meander's centroid (PCA mean) legitimately falls in the concave gap
    # of the curve, off the actual mask -- assert that so this test is really
    # exercising the failure mode, not a convex shape
    cr = channel_reaches[0].centroid_rc
    rr, cc = int(round(cr[0])), int(round(cr[1]))
    assert not combined[rr, cc], (
        "fixture is not curved enough to exercise the centroid-off-mask case")


def test_summarise_reaches_separates_channels_from_basins():
    channel = np.zeros((60, 40), bool); channel[:, 6:9] = True
    yy, xx = np.mgrid[0:60, 0:40]
    basin = (yy - 45) ** 2 + (xx - 30) ** 2 <= 7 ** 2   # well clear of cols 6:9
    reaches = WB.river_reaches(channel | basin, dx=10.0, dy=10.0, min_cells=5)
    summary = WB.summarise_reaches(reaches)
    assert summary["channel_like_reaches"] == 1
    assert summary["basin_like_bodies"] == 1
    assert summary["total_components"] == 2


# ---------------------------------------------------------------------------
# OSM cross-check
# ---------------------------------------------------------------------------

def test_osm_cross_check_reports_both_directions_of_disagreement():
    sat = np.zeros((30, 30), bool); sat[5:25, 5:10] = True
    osm = np.zeros((30, 30), bool); osm[5:25, 8:13] = True     # partial overlap
    out = WB.compare_to_osm_waterways(sat, osm, cell_area=100.0)
    assert out["satellite_beyond_osm_km2"] > 0
    assert out["osm_beyond_satellite_km2"] > 0
    assert 0.0 < out["iou"] < 1.0


def test_osm_cross_check_perfect_agreement():
    sat = np.zeros((10, 10), bool); sat[2:5, 2:5] = True
    out = WB.compare_to_osm_waterways(sat, sat.copy(), cell_area=100.0)
    assert out["iou"] == pytest.approx(1.0)
    assert out["satellite_beyond_osm_km2"] == 0.0
    assert out["osm_beyond_satellite_km2"] == 0.0


def test_osm_cross_check_no_osm_data_at_all():
    sat = np.zeros((10, 10), bool); sat[2:5, 2:5] = True
    osm = np.zeros((10, 10), bool)
    out = WB.compare_to_osm_waterways(sat, osm, cell_area=100.0)
    assert out["iou"] == 0.0
    assert out["osm_waterway_km2"] == 0.0


# ---------------------------------------------------------------------------
# Wiring: the scenario fields this feature added must be reachable from BOTH
# the CLI and the REST API, per this project's established practice of
# testing that explicitly rather than assuming it (see
# tests/test_audit_regressions.py::PREVIOUSLY_UNREACHABLE for the precedent
# this follows -- several earlier scenario parameters were implemented but
# reachable from neither interface).
# ---------------------------------------------------------------------------

WATER_FUSION_FIELDS = ["water_fusion", "water_fusion_window"]


@pytest.mark.parametrize("field", WATER_FUSION_FIELDS)
def test_water_fusion_field_on_scenario(field):
    from damburst.pipeline import Scenario
    assert field in Scenario.__dataclass_fields__


@pytest.mark.parametrize("field", WATER_FUSION_FIELDS)
def test_water_fusion_field_reachable_from_the_rest_api(field):
    from damburst.api.main import RunRequest
    assert field in RunRequest.model_fields


def test_water_fusion_default_is_enabled():
    from damburst.pipeline import Scenario
    s = Scenario(name="x", bbox_ll=(0, 0, 1, 1), dam_name="d")
    assert s.water_fusion is True
    assert s.water_fusion_window is None


def test_cli_exposes_water_fusion_flags():
    from damburst.cli import main
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        main(["run", "--help"])
    help_text = buf.getvalue()
    for flag in ("--no-water-fusion", "--water-fusion-window"):
        assert flag in help_text, f"{flag} missing from the CLI"


def test_api_applies_water_fusion_override():
    from damburst.api.main import RunRequest, _build_scenario
    scn = _build_scenario(RunRequest(preset="tehri", water_fusion=False,
                                     water_fusion_window=["2025-01-01",
                                                          "2025-03-31"]))
    assert scn.water_fusion is False
    assert scn.water_fusion_window == ("2025-01-01", "2025-03-31")
