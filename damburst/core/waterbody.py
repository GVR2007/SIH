"""Multi-source water-body and river-network detection.

WHY THIS MODULE EXISTS
----------------------
Before this module, the framework's river/reservoir extent came from exactly
two places: DEM hypsometry (a pure elevation flood-fill) and the ESA WorldCover
permanent-water class. Sentinel-1 was fetched and Otsu-thresholded, but its
water mask was used ONLY inside `validation.py`, to score a hypothetical flood
against an observed one -- never as an input to what the pre-event river or
reservoir actually looks like. Sentinel-2 was fetched only as a 3D/2D basemap
TEXTURE. Neither satellite carried any analytical weight before a run started.

That is the DEM-plus-cross-check architecture the SIH technical-approach spec
explicitly asks to move past: "the combined [DEM + Sentinel-1 + Sentinel-2]
dataset should be used rather than relying exclusively on DEM."

This module is the fusion layer. It:

  1. computes real optical water indices (NDWI, MNDWI) from Sentinel-2
     reflectance bands -- not the RGB basemap texture, which is visually
     stretched and gamma-corrected and would corrupt an index computation;
  2. reuses the existing Otsu SAR water mask from Sentinel-1;
  3. combines those two with the ESA WorldCover permanent-water class into one
     evidenced water-body mask, where every cell records HOW MANY independent
     sources called it water (0-3), not just a boolean;
  4. estimates the river's flow direction and approximate course per reach via
     a per-connected-component principal-axis (PCA) fit over the water mask --
     stated as an approximation, not a claim of a true hydrological centreline;
  5. cross-validates the fused mask against the OSM-mapped waterway/dam
     geometry, so the manifest can report whether the satellite evidence and
     the crowd-mapped vector data actually agree over this domain.

WHAT THIS DOES NOT CLAIM
------------------------
* NDWI/MNDWI are computed on a DECIMATED Sentinel-2 mosaic (see
  `datasources.fetch_sentinel2_water_bands`), not at native 10 m, for the same
  reason the RGB texture is: full-resolution windowed COG reads over a
  basin-sized bbox are minutes per granule. The read resolution is carried in
  the provenance dict and printed alongside every mask this module returns.
* The flow-direction estimate is a per-reach straight-line PCA fit, not a
  traced hydrological centreline (that already exists, separately, as
  `dem.steepest_descent_path` on the conditioned DEM). Where the two disagree
  it is worth investigating the water mask, not assuming the PCA fit is wrong.
* Fusion is agreement-counting, not a learned classifier. A cell that only one
  source calls water is not discarded, it is reported at confidence 1/3 -- the
  caller decides the threshold that matters for what it is doing (a reservoir
  pool boundary wants stricter agreement than a coarse cross-validation).

REFERENCES
----------
* McFeeters, S.K. (1996). The use of the Normalized Difference Water Index
  (NDWI) in the delineation of open water features. Int. J. Remote Sens.
  17(7), 1425-1432.
* Xu, H. (2006). Modification of normalised difference water index (NDWI) to
  enhance open water features in remotely sensed imagery. Int. J. Remote Sens.
  27(14), 3025-3033.  (MNDWI, using SWIR instead of NIR to suppress built-up
  false positives.)
* Otsu, N. (1979). A threshold selection method from gray-level histograms.
  IEEE Trans. Syst. Man Cybern. 9(1), 62-66.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Optical water indices
# ---------------------------------------------------------------------------

def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters (1996) NDWI = (Green - NIR) / (Green + NIR)."""
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 1e-6, (green - nir) / denom, np.nan)
    return out


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Xu (2006) MNDWI = (Green - SWIR) / (Green + SWIR).

    Preferred over NDWI where built-up areas are present: SWIR responds more
    strongly to construction materials than NIR does, so MNDWI suppresses the
    built-up false positives NDWI is known to produce.
    """
    denom = green + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 1e-6, (green - swir) / denom, np.nan)
    return out


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on a 1D sample -- the same unsupervised threshold already
    used for the SAR water mask, applied here to an optical index histogram."""
    v = values[np.isfinite(values)]
    if v.size < 16:
        return float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
    hist, edges = np.histogram(v, bins=bins)
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist)
    w = np.where(w == 0, np.nan, w)
    total = hist.sum()
    w_b = w
    w_f = total - w
    mu_b = np.cumsum(hist * centers) / w_b
    cum_all = float(np.nansum(hist * centers))
    mu_f = (cum_all - np.nancumsum(hist * centers)) / np.where(w_f == 0, np.nan, w_f)
    variance = w_b * w_f * (mu_b - mu_f) ** 2
    if not np.isfinite(variance).any():
        return float(np.nanmedian(v))
    k = int(np.nanargmax(variance))
    return float(centers[k])


def optical_water_mask(green: np.ndarray, nir: Optional[np.ndarray] = None,
                       swir: Optional[np.ndarray] = None,
                       method: str = "mndwi",
                       threshold: Optional[float] = None
                       ) -> Tuple[np.ndarray, dict]:
    """Threshold an optical water index into a boolean water mask.

    `method="mndwi"` (default, needs `swir`) suppresses built-up false
    positives; `method="ndwi"` (needs `nir`) is the classic McFeeters index,
    kept for cross-checking or where SWIR is unavailable. Otsu's method picks
    the threshold unless one is supplied explicitly.
    """
    if method == "mndwi":
        if swir is None:
            raise ValueError("mndwi needs the swir band")
        idx = mndwi(green, swir)
    elif method == "ndwi":
        if nir is None:
            raise ValueError("ndwi needs the nir band")
        idx = ndwi(green, nir)
    else:
        raise ValueError(f"unknown method {method!r}; use 'mndwi' or 'ndwi'")

    thr = threshold if threshold is not None else _otsu_threshold(idx)
    mask = np.where(np.isfinite(idx), idx > thr, False)
    return mask, {
        "method": method.upper(),
        "threshold": round(thr, 4),
        "threshold_source": "user-supplied" if threshold is not None else "Otsu",
        "valid_fraction": round(float(np.isfinite(idx).mean()), 4),
        "water_fraction": round(float(mask.mean()), 4),
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

@dataclass
class WaterEvidence:
    """Per-cell agreement across every water-detection source that ran.

    `n_sources` is an integer 0..3 count, not a boolean -- the caller decides
    what confidence threshold its use case needs.
    """
    n_sources: np.ndarray             # int8, count of sources agreeing per cell
    max_sources: int                  # how many sources were actually available
    any_source: np.ndarray            # bool, n_sources >= 1
    majority: np.ndarray              # bool, n_sources >= ceil(max_sources/2)
    all_sources: np.ndarray           # bool, n_sources == max_sources
    per_source: Dict[str, np.ndarray] = field(default_factory=dict)
    pairwise_agreement: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, object] = field(default_factory=dict)


def fuse_water_evidence(cell_area: float,
                        worldcover_water: Optional[np.ndarray] = None,
                        sar_water: Optional[np.ndarray] = None,
                        optical_water: Optional[np.ndarray] = None
                        ) -> WaterEvidence:
    """Combine up to three independent water-detection sources.

    Any subset may be `None` (e.g. no cloud-free Sentinel-2 scene in the
    window); `max_sources` and every derived mask account for exactly the
    sources actually supplied, so a two-source fusion is not silently graded
    against a three-source bar.
    """
    sources: Dict[str, np.ndarray] = {}
    if worldcover_water is not None:
        sources["worldcover"] = worldcover_water.astype(bool)
    if sar_water is not None:
        sources["sentinel1_sar"] = sar_water.astype(bool)
    if optical_water is not None:
        sources["sentinel2_optical"] = optical_water.astype(bool)

    if not sources:
        raise ValueError("fuse_water_evidence needs at least one source")

    shape = next(iter(sources.values())).shape
    n = np.zeros(shape, dtype=np.int8)
    for m in sources.values():
        n += m.astype(np.int8)
    max_n = len(sources)

    pairwise: Dict[str, float] = {}
    keys = list(sources.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = sources[keys[i]], sources[keys[j]]
            union = (a | b).sum()
            pairwise[f"{keys[i]}_vs_{keys[j]}_iou"] = (
                round(float((a & b).sum()) / union, 4) if union else None)

    any_src = n >= 1
    majority = n >= int(np.ceil(max_n / 2.0))
    all_src = n >= max_n

    summary = {
        "sources_used": keys,
        "n_sources_available": max_n,
        "area_km2_any_source": round(float(any_src.sum()) * cell_area / 1e6, 4),
        "area_km2_majority": round(float(majority.sum()) * cell_area / 1e6, 4),
        "area_km2_all_sources": round(float(all_src.sum()) * cell_area / 1e6, 4),
        "pairwise_agreement_iou": pairwise,
        "note": ("n_sources is a per-cell agreement COUNT across independent "
                 "water detections, not a single boolean. A cell only one "
                 "source calls water is reported, not discarded -- use "
                 "'majority' or 'all_sources' where higher confidence is "
                 "needed, 'any_source' where recall matters more (e.g. never "
                 "missing part of a reservoir)."),
    }
    return WaterEvidence(n_sources=n, max_sources=max_n, any_source=any_src,
                         majority=majority, all_sources=all_src,
                         per_source=sources, pairwise_agreement=pairwise,
                         summary=summary)


# ---------------------------------------------------------------------------
# River network: connected reaches + approximate flow direction
# ---------------------------------------------------------------------------

@dataclass
class WaterReach:
    """One connected component of a water mask, with its approximate axis."""
    cells: int
    area_km2: float
    centroid_rc: Tuple[float, float]
    length_m: float                   # long-axis extent (2 std devs)
    width_m: float                     # short-axis extent
    direction_deg: float               # compass bearing of the long axis, 0=+row
    elongation: float                  # length / max(width, 1) -- river-like if >> 1
    is_channel_like: bool


def _label_and_fit(water_mask: np.ndarray, dx: float, dy: float,
                   min_cells: int, elongation_threshold: float
                   ) -> Tuple[np.ndarray, List[Tuple[int, WaterReach]]]:
    """Connected-component labelling + PCA fit, shared by the public functions
    below so the label array and the fitted reaches never disagree with each
    other (no second, independent re-labelling / centroid re-lookup)."""
    labels, n = ndi.label(water_mask)
    cell_area = dx * dy
    out: List[Tuple[int, WaterReach]] = []
    for lbl in range(1, n + 1):
        rows, cols = np.nonzero(labels == lbl)
        if rows.size < min_cells:
            continue
        # PCA on physical (x, y) coordinates, not row/col, so length/width and
        # the bearing are in real metres and compass degrees.
        xs = cols.astype(np.float64) * dx
        ys = -rows.astype(np.float64) * dy      # row increases southward
        pts = np.stack([xs, ys], axis=1)
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        if not np.all(np.isfinite(cov)):
            continue
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]
        # length/width as +-2 std dev along each principal axis (~95% extent)
        length = 4.0 * float(np.sqrt(max(eigval[0], 0.0)))
        width = 4.0 * float(np.sqrt(max(eigval[1], 0.0))) if eigval.size > 1 else 0.0
        long_axis = eigvec[:, 0]
        bearing = float(np.degrees(np.arctan2(long_axis[0], long_axis[1])) % 180.0)
        elong = length / max(width, dx)
        out.append((lbl, WaterReach(
            cells=int(rows.size),
            area_km2=round(rows.size * cell_area / 1e6, 5),
            centroid_rc=(float(rows.mean()), float(cols.mean())),
            length_m=round(length, 1),
            width_m=round(width, 1),
            direction_deg=round(bearing, 1),
            elongation=round(elong, 2),
            is_channel_like=bool(elong >= elongation_threshold),
        )))
    out.sort(key=lambda pair: -pair[1].area_km2)
    return labels, out


def river_reaches(water_mask: np.ndarray, dx: float, dy: float,
                  min_cells: int = 20,
                  elongation_threshold: float = 3.0) -> List[WaterReach]:
    """Label connected water regions and fit a principal axis to each.

    This is an APPROXIMATION of flow direction and course, stated as such in
    `WaterReach.direction_deg`'s docstring above: it is the long axis of a
    PCA fit over the component's pixel coordinates, not a traced centreline.
    A traced hydrological centreline already exists separately as
    `dem.steepest_descent_path` on the conditioned DEM; comparing the two is
    a legitimate cross-check, not a duplicate of the same computation.

    `is_channel_like` flags components whose elongation exceeds
    `elongation_threshold` -- a river reach is long and narrow, a reservoir or
    lake is comparatively round, and conflating the two would mislabel a pond
    as a stream reach.

    KNOWN LIMITATION.  A single global PCA axis per connected component
    under-measures elongation for a SHARPLY bent reach: a right-angle L-shape
    spreads the covariance roughly equally along both arms, capping the
    measured elongation near 2 regardless of how long each arm actually is.
    Gentle, gradually-curving meanders (the common case for a real river) are
    unaffected -- their principal axis still tracks the dominant direction of
    flow well. A component this function calls basin-like purely because of a
    sharp bend is a case worth checking against `dem.steepest_descent_path` or
    the raw connected-component shape before accepting the label.
    """
    _labels, pairs = _label_and_fit(water_mask, dx, dy, min_cells,
                                    elongation_threshold)
    return [r for _lbl, r in pairs]


def channel_like_mask(water_mask: np.ndarray, dx: float, dy: float,
                      min_cells: int = 20,
                      elongation_threshold: float = 3.0) -> np.ndarray:
    """Boolean raster of just the channel-like (not basin-like) components.

    Same labelling and PCA-elongation test as `river_reaches`, returned as a
    mask instead of a list -- for callers that want to burn or union the
    satellite-detected channel geometry rather than report on it. Uses the
    label ids directly rather than re-locating each component from its
    centroid, which would misfire on non-convex (bent) reaches.
    """
    labels, pairs = _label_and_fit(water_mask, dx, dy, min_cells,
                                   elongation_threshold)
    out = np.zeros_like(water_mask, dtype=bool)
    for lbl, r in pairs:
        if r.is_channel_like:
            out |= (labels == lbl)
    return out


def summarise_reaches(reaches: List[WaterReach]) -> dict:
    channels = [r for r in reaches if r.is_channel_like]
    basins = [r for r in reaches if not r.is_channel_like]
    return {
        "total_components": len(reaches),
        "channel_like_reaches": len(channels),
        "basin_like_bodies": len(basins),
        "largest_channel_km2": (max((r.area_km2 for r in channels), default=0.0)),
        "largest_basin_km2": (max((r.area_km2 for r in basins), default=0.0)),
        "reaches": [r.__dict__ for r in reaches[:25]],   # cap payload size
        "method": ("connected-component labelling of the fused water mask, "
                  "PCA principal-axis per component for an approximate "
                  "length/width/bearing; elongation >= 3.0 is called "
                  "channel-like, otherwise basin-like (reservoir/lake/pond)"),
    }


# ---------------------------------------------------------------------------
# Cross-validation against OSM
# ---------------------------------------------------------------------------

def compare_to_osm_waterways(fused_mask: np.ndarray, osm_waterway_mask: np.ndarray,
                             cell_area: float) -> dict:
    """How much does the satellite-evidenced water mask agree with the
    crowd-mapped OSM waterway/water-body geometry over this domain?

    A low IoU is not necessarily an error in either source: OSM waterway
    mapping is known to be sparse in rural India (the same caveat the exposure
    QC panel raises for buildings), so a satellite mask can legitimately show
    far more water than OSM has mapped. Both directions are reported so a
    reviewer can tell which case applies.
    """
    fused = fused_mask.astype(bool)
    osm = osm_waterway_mask.astype(bool)
    inter = (fused & osm).sum()
    union = (fused | osm).sum()
    return {
        "satellite_water_km2": round(float(fused.sum()) * cell_area / 1e6, 4),
        "osm_waterway_km2": round(float(osm.sum()) * cell_area / 1e6, 4),
        "overlap_km2": round(float(inter) * cell_area / 1e6, 4),
        "iou": round(float(inter) / union, 4) if union else None,
        "satellite_beyond_osm_km2": round(
            float((fused & ~osm).sum()) * cell_area / 1e6, 4),
        "osm_beyond_satellite_km2": round(
            float((osm & ~fused).sum()) * cell_area / 1e6, 4),
        "note": ("A low IoU does not by itself mean either source is wrong -- "
                 "OSM waterway mapping is frequently sparse in rural areas. "
                 "'satellite_beyond_osm' flags water the satellites see that "
                 "is not mapped; 'osm_beyond_satellite' flags mapped "
                 "waterways with no current satellite water signature "
                 "(seasonal/dry channels, or a cloud/coverage gap)."),
    }
