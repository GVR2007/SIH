"""Tests for the terrain morphometry products added to core/dem.py: slope
(refactored out of qc_report), aspect, D8 flow direction, watershed
delineation, and contour extraction.

These were the terrain products the SIH technical-approach spec names and the
codebase was missing entirely (only D8 flow ACCUMULATION and a single traced
thalweg existed; no aspect, no formal watershed delineation, no contours).
"""

import math

import numpy as np
import pytest

from damburst.core import dem as D


# ---------------------------------------------------------------------------
# slope() -- refactor, must not change qc_report's numbers
# ---------------------------------------------------------------------------

def test_slope_zero_on_flat_terrain():
    z = np.full((20, 20), 500.0)
    s = D.slope(z, dx=10.0, dy=10.0)
    assert np.allclose(s, 0.0)


def test_slope_matches_known_grade():
    """A uniform 1-in-10 grade (dz/dx = 0.1) is atan(0.1) ~ 5.71 degrees."""
    nx = 40
    z = np.tile(np.arange(nx) * 1.0, (10, 1))     # 1 m rise per 10 m cell
    s = D.slope(z, dx=10.0, dy=10.0)
    interior = s[2:-2, 2:-2]
    assert interior.mean() == pytest.approx(math.degrees(math.atan(0.1)), abs=0.2)


def test_qc_report_slope_unaffected_by_the_refactor():
    """slope() must produce the same field qc_report used to compute inline."""
    rng = np.random.default_rng(0)
    ny, nx = 25, 25

    class FakeDEM:
        pass
    fd = FakeDEM()
    fd.z = 500.0 + rng.random((ny, nx)) * 50.0
    fd.dx = fd.dy = 30.0
    fd.ny, fd.nx = ny, nx
    fd.source = "synthetic"
    fd.crs = "EPSG:32644"

    report = D.qc_report(fd)
    finite = np.isfinite(fd.z)
    expected = D.slope(np.where(finite, fd.z, np.nan), fd.dx, fd.dy)
    # qc_report rounds to 2 dp before returning, so compare at that precision
    assert report["max_slope_deg"] == pytest.approx(float(np.nanmax(expected)), abs=5e-3)
    assert report["mean_slope_deg"] == pytest.approx(float(np.nanmean(expected)), abs=5e-3)


# ---------------------------------------------------------------------------
# aspect()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grade_fn,expect_deg", [
    (lambda row, col: -col, 90.0),      # z decreases eastward -> downslope east
    (lambda row, col: row, 0.0),        # z decreases northward (higher south) -> downslope north
    (lambda row, col: -row, 180.0),     # z decreases southward -> downslope south
    (lambda row, col: col, 270.0),      # z decreases westward -> downslope west
])
def test_aspect_matches_known_cardinal_directions(grade_fn, expect_deg):
    ny, nx = 20, 20
    row, col = np.mgrid[0:ny, 0:nx]
    z = grade_fn(row, col).astype(float)
    a = D.aspect(z, dx=10.0, dy=10.0)
    assert a[10, 10] == pytest.approx(expect_deg, abs=1.0)


def test_aspect_is_nan_on_flat_terrain():
    z = np.zeros((10, 10))
    a = D.aspect(z, dx=10.0, dy=10.0)
    assert np.all(np.isnan(a))


def test_aspect_stays_in_0_360_range():
    rng = np.random.default_rng(1)
    z = rng.random((30, 30)) * 100.0
    a = D.aspect(z, dx=15.0, dy=15.0)
    finite = a[np.isfinite(a)]
    assert (finite >= 0).all() and (finite < 360).all()


# ---------------------------------------------------------------------------
# D8 flow direction / accumulation
# ---------------------------------------------------------------------------

def _bowl(ny=20, nx=20, outlet_col=10):
    row, col = np.mgrid[0:ny, 0:nx]
    return (np.abs(col - outlet_col).astype(float) * 2.0
           + (ny - 1 - row).astype(float) * 1.0)


def test_flow_direction_targets_are_within_grid_or_self():
    z = _bowl()
    ny, nx = z.shape
    fd = D.d8_flow_direction(z, dx=10.0)
    assert fd.min() >= 0
    assert fd.max() < z.size


def test_accumulation_conserves_total_cell_count():
    """Every cell contributes exactly 1 unit of flow somewhere -- the sum over
    all local maxima (cells nothing flows further from) must equal the total
    cell count, since D8 routes every cell's unit contribution to exactly one
    sink eventually."""
    z = _bowl()
    fd = D.d8_flow_direction(z, dx=10.0)
    acc = D.d8_flow_accumulation(z, dx=10.0, flow_dir=fd)
    sinks = np.arange(fd.size)[fd == np.arange(fd.size)]
    total_at_sinks = acc.ravel()[sinks].sum()
    assert total_at_sinks == pytest.approx(float(z.size), rel=1e-9)


def test_accumulation_matches_with_and_without_precomputed_direction():
    z = _bowl()
    fd = D.d8_flow_direction(z, dx=10.0)
    a1 = D.d8_flow_accumulation(z, dx=10.0, flow_dir=fd)
    a2 = D.d8_flow_accumulation(z, dx=10.0)
    assert np.array_equal(a1, a2)


def test_single_outlet_bowl_drains_everything_to_one_cell():
    z = _bowl(ny=15, nx=15, outlet_col=7)
    acc = D.d8_flow_accumulation(z, dx=10.0)
    assert float(acc.max()) == pytest.approx(float(z.size), rel=1e-9)


# ---------------------------------------------------------------------------
# Watershed delineation
# ---------------------------------------------------------------------------

def test_watershed_of_a_single_outlet_bowl_is_the_whole_grid():
    z = _bowl(ny=18, nx=18, outlet_col=9)
    fd = D.d8_flow_direction(z, dx=10.0)
    acc = D.d8_flow_accumulation(z, dx=10.0, flow_dir=fd)
    outlet = np.unravel_index(np.argmax(acc), acc.shape)
    ws = D.delineate_watershed(z, dx=10.0, pour_point_rc=outlet, flow_dir=fd)
    assert ws.sum() == z.size


def test_watershed_upstream_of_an_interior_point_is_smaller_than_the_whole_grid():
    z = _bowl(ny=20, nx=20, outlet_col=10)
    ws_mid = D.delineate_watershed(z, dx=10.0, pour_point_rc=(10, 10))
    ws_outlet = D.delineate_watershed(z, dx=10.0, pour_point_rc=(19, 10))
    assert 0 < ws_mid.sum() < ws_outlet.sum()
    # everything draining through the midpoint must also drain through the
    # (downstream) outlet
    assert np.all(ws_mid <= ws_outlet)


def test_watershed_rejects_an_out_of_bounds_pour_point():
    z = _bowl()
    with pytest.raises(ValueError):
        D.delineate_watershed(z, dx=10.0, pour_point_rc=(999, 999))


def test_two_separate_valleys_have_disjoint_watersheds():
    """Two independent V-valleys side by side must not leak into each other."""
    ny, nx = 20, 40
    row, col = np.mgrid[0:ny, 0:nx]
    left = np.abs(col - 10).astype(float) * 2.0 + (ny - 1 - row).astype(float)
    right = np.abs(col - 30).astype(float) * 2.0 + (ny - 1 - row).astype(float)
    z = np.minimum(left, right) + 100.0     # a ridge separates the two valleys

    fd = D.d8_flow_direction(z, dx=10.0)
    ws_left = D.delineate_watershed(z, dx=10.0, pour_point_rc=(19, 10), flow_dir=fd)
    ws_right = D.delineate_watershed(z, dx=10.0, pour_point_rc=(19, 30), flow_dir=fd)
    assert not (ws_left & ws_right).any()


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------

def test_contours_empty_on_a_flat_domain():
    z = np.full((20, 20), 500.0)
    contours = D.extract_contours(z, dx=10.0, dy=10.0, interval=10.0)
    assert contours == []


def test_contours_found_on_a_sloped_domain():
    nx = 40
    z = np.tile(np.arange(nx) * 5.0, (10, 1)).astype(float)   # 0..195 m relief
    contours = D.extract_contours(z, dx=10.0, dy=10.0, interval=25.0)
    assert len(contours) > 0
    for c in contours:
        assert "elevation_m" in c and "coords" in c
        assert len(c["coords"]) >= 2
        for pt in c["coords"]:
            assert len(pt) == 2


def test_contour_coordinates_respect_origin_offset():
    nx = 30
    z = np.tile(np.arange(nx) * 5.0, (10, 1)).astype(float)
    c0 = D.extract_contours(z, dx=10.0, dy=10.0, interval=20.0, origin_xy=(0.0, 0.0))
    c1 = D.extract_contours(z, dx=10.0, dy=10.0, interval=20.0,
                            origin_xy=(1000.0, 2000.0))
    assert len(c0) == len(c1)
    x0 = c0[0]["coords"][0][0]
    x1 = c1[0]["coords"][0][0]
    assert x1 - x0 == pytest.approx(1000.0, abs=1e-6)


def test_no_contours_when_relief_is_smaller_than_the_interval():
    z = np.tile(np.arange(10) * 0.1, (10, 1)).astype(float)   # <1 m relief
    contours = D.extract_contours(z, dx=10.0, dy=10.0, interval=25.0)
    assert contours == []
