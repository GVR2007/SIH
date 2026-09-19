"""Exposure, impact and loss accounting.

Implements the right-hand column of the technical approach:

    Exposure Data -> Zonal Intersection
      -> critical facilities / crop raster / road lines / building polygons /
         population raster
      -> QC checks -> hazard map, evacuation timeline table,
         district breakdown, impact summary

All exposure inputs are real: OpenStreetMap features (ODbL), WorldPop 2020
population counts (CC-BY), ESA WorldCover land cover (CC-BY) and geoBoundaries
ADM2 districts (CC-BY).  Nothing here invents an asset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import features as rfeatures
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_geom

from .dem import DEM
from .hazard import (DAMAGE_CLASS_NAMES, HAZARD_CLASSES, AssetValues,
                     damage_class, damage_fraction)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def ll_to_rowcol(dem: DEM, lons: Sequence[float],
                 lats: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project lon/lat to the model grid; returns (row, col, inside_mask)."""
    if len(lons) == 0:
        e = np.array([], dtype=int)
        return e, e, np.array([], dtype=bool)
    xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_string(dem.crs),
                            list(lons), list(lats))
    rows, cols = dem.rowcol(np.array(xs), np.array(ys))
    inside = (rows >= 0) & (rows < dem.ny) & (cols >= 0) & (cols < dem.nx)
    return rows, cols, inside


def sample_raster(dem: DEM, raster: np.ndarray, lons, lats,
                  default=0.0) -> np.ndarray:
    rows, cols, inside = ll_to_rowcol(dem, lons, lats)
    out = np.full(len(lons), default, dtype=float)
    if inside.any():
        out[inside] = raster[rows[inside], cols[inside]]
    return out


def polyline_length_in_mask(dem: DEM, coords_ll: Sequence[Sequence[float]],
                            mask: np.ndarray, step_m: float = 20.0) -> float:
    """Length of an OSM way that falls inside a boolean raster mask."""
    if len(coords_ll) < 2:
        return 0.0
    lons = [c[0] for c in coords_ll]
    lats = [c[1] for c in coords_ll]
    xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_string(dem.crs),
                            lons, lats)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    total = 0.0
    for k in range(len(xs) - 1):
        seg = math.hypot(xs[k + 1] - xs[k], ys[k + 1] - ys[k])
        if seg <= 0:
            continue
        n = max(int(seg / step_m), 1)
        for m in range(n):
            f = (m + 0.5) / n
            x = xs[k] + f * (xs[k + 1] - xs[k])
            y = ys[k] + f * (ys[k + 1] - ys[k])
            r, c = dem.rowcol(x, y)
            if 0 <= r < dem.ny and 0 <= c < dem.nx and mask[r, c]:
                total += seg / n
    return total


def rasterize_districts(dem: DEM, geojson: dict,
                        name_key: str = "shapeName") -> Tuple[np.ndarray, Dict[int, str]]:
    """Burn ADM2 polygons onto the model grid (only those intersecting it)."""
    west, south, east, north = dem.bounds()
    shapes = []
    names: Dict[int, str] = {}
    idx = 1
    for feat in geojson.get("features", []):
        try:
            geom = transform_geom("EPSG:4326", dem.crs, feat["geometry"])
        except Exception:
            continue
        shapes.append((geom, idx))
        names[idx] = feat.get("properties", {}).get(name_key, f"district_{idx}")
        idx += 1
    if not shapes:
        return np.zeros(dem.shape, dtype=np.int32), {}
    arr = rfeatures.rasterize(shapes, out_shape=dem.shape,
                              transform=dem.transform, fill=0,
                              dtype="int32", all_touched=False)
    present = set(np.unique(arr).tolist()) - {0}
    return arr, {k: v for k, v in names.items() if k in present}


# ---------------------------------------------------------------------------
# Impact accounting
# ---------------------------------------------------------------------------

@dataclass
class ImpactReport:
    population: Dict[str, object] = field(default_factory=dict)
    settlements: List[dict] = field(default_factory=list)
    buildings: Dict[str, object] = field(default_factory=dict)
    facilities: List[dict] = field(default_factory=list)
    roads: Dict[str, object] = field(default_factory=dict)
    landcover: Dict[str, object] = field(default_factory=dict)
    districts: List[dict] = field(default_factory=list)
    evacuation: List[dict] = field(default_factory=list)
    losses: Dict[str, object] = field(default_factory=dict)
    qc: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "population": self.population,
            "settlements": self.settlements,
            "buildings": self.buildings,
            "facilities": self.facilities,
            "roads": self.roads,
            "landcover": self.landcover,
            "districts": self.districts,
            "evacuation": self.evacuation,
            "losses": self.losses,
            "qc": self.qc,
        }


def _class_name(idx: int) -> str:
    if idx <= 0:
        return "None"
    return HAZARD_CLASSES[idx - 1][2]


def population_impact(pop: np.ndarray, hazard_cls: np.ndarray,
                      depth: np.ndarray) -> Dict[str, object]:
    """Population inside each hazard class (WorldPop counts, real data)."""
    out = {"total_in_domain": round(float(pop.sum()), 0), "by_class": []}
    exposed = 0.0
    for idx in range(1, 5):
        m = hazard_cls == idx
        n = float(pop[m].sum())
        exposed += n
        out["by_class"].append({"class": idx, "name": _class_name(idx),
                                "population": round(n, 0)})
    out["total_exposed"] = round(exposed, 0)
    out["exposed_over_1m_depth"] = round(float(pop[depth > 1.0].sum()), 0)
    out["exposed_over_3m_depth"] = round(float(pop[depth > 3.0].sum()), 0)
    return out


def settlement_impact(dem: DEM, settlements: List[dict], depth: np.ndarray,
                      hazard_cls: np.ndarray, arrival_s: np.ndarray,
                      velocity: np.ndarray) -> List[dict]:
    """Per-settlement depth, hazard class and warning time."""
    if not settlements:
        return []
    lons = [s["lon"] for s in settlements]
    lats = [s["lat"] for s in settlements]
    d = sample_raster(dem, depth, lons, lats)
    c = sample_raster(dem, hazard_cls.astype(float), lons, lats)
    a = sample_raster(dem, arrival_s, lons, lats, default=-1.0)
    v = sample_raster(dem, velocity, lons, lats)

    out = []
    for k, s in enumerate(settlements):
        if d[k] <= 0.05:
            continue
        out.append({
            "name": s.get("name") or "(unnamed)",
            "place": s.get("place"),
            "osm_id": s.get("osm_id"),
            "lon": s["lon"], "lat": s["lat"],
            "osm_population_tag": s.get("population"),
            "depth_m": round(float(d[k]), 2),
            "velocity_ms": round(float(v[k]), 2),
            "hazard_class": int(c[k]),
            "hazard_name": _class_name(int(c[k])),
            "arrival_s": None if a[k] < 0 else round(float(a[k]), 1),
            "arrival_min": None if a[k] < 0 else round(float(a[k]) / 60.0, 1),
        })
    out.sort(key=lambda r: (r["arrival_min"] is None, r["arrival_min"]))
    return out


def facility_impact(dem: DEM, facilities: List[dict], depth: np.ndarray,
                    hazard_cls: np.ndarray, arrival_s: np.ndarray) -> List[dict]:
    if not facilities:
        return []
    lons = [f["lon"] for f in facilities]
    lats = [f["lat"] for f in facilities]
    d = sample_raster(dem, depth, lons, lats)
    c = sample_raster(dem, hazard_cls.astype(float), lons, lats)
    a = sample_raster(dem, arrival_s, lons, lats, default=-1.0)
    out = []
    for k, f in enumerate(facilities):
        if d[k] <= 0.05:
            continue
        out.append({
            "name": f.get("name") or "(unnamed)",
            "kind": f.get("kind"),
            "osm_id": f.get("osm_id"),
            "lon": f["lon"], "lat": f["lat"],
            "depth_m": round(float(d[k]), 2),
            "hazard_class": int(c[k]),
            "hazard_name": _class_name(int(c[k])),
            "arrival_min": None if a[k] < 0 else round(float(a[k]) / 60.0, 1),
        })
    out.sort(key=lambda r: -r["depth_m"])
    return out


def building_impact(dem: DEM, buildings: List[dict], depth: np.ndarray,
                    hazard_cls: np.ndarray, dv: np.ndarray) -> Dict[str, object]:
    """Buildings by hazard class and by depth-damage class."""
    if not buildings:
        return {"total_in_domain": 0, "exposed": 0, "by_hazard_class": [],
                "by_damage_class": [], "structural_dv": {}}
    lons = [b["lon"] for b in buildings]
    lats = [b["lat"] for b in buildings]
    d = sample_raster(dem, depth, lons, lats)
    c = sample_raster(dem, hazard_cls.astype(float), lons, lats).astype(int)
    q = sample_raster(dem, dv, lons, lats)

    wet = d > 0.05
    frac = damage_fraction(d, "residential")
    dcls = damage_class(frac)

    by_haz = [{"class": i, "name": _class_name(i),
               "buildings": int(((c == i) & wet).sum())} for i in range(1, 5)]
    by_dmg = [{"class": i, "name": DAMAGE_CLASS_NAMES[i],
               "buildings": int(((dcls == i) & wet).sum())} for i in range(1, 5)]

    return {
        "total_in_domain": len(buildings),
        "exposed": int(wet.sum()),
        "by_hazard_class": by_haz,
        "by_damage_class": by_dmg,
        "mean_damage_fraction": round(float(frac[wet].mean()), 4) if wet.any() else 0.0,
        "structural_dv": {
            "dv_lt_3_inundation_only": int(((q > 0) & (q < 3) & wet).sum()),
            "dv_3_to_7_partial_collapse": int(((q >= 3) & (q < 7) & wet).sum()),
            "dv_gt_7_total_destruction": int(((q >= 7) & wet).sum()),
            "reference": "Clausen & Clark (1990) masonry stability thresholds",
        },
        "damage_curve": "JRC Huizinga et al. (2017), Asia residential",
    }


def road_impact(dem: DEM, roads_geom: List[dict], wet: np.ndarray) -> Dict[str, object]:
    """Inundated length of the lifeline road/rail network, by class."""
    by_class: Dict[str, float] = {}
    total = 0.0
    for w in roads_geom:
        L = polyline_length_in_mask(dem, w.get("coords_ll", []), wet)
        if L <= 0:
            continue
        key = w.get("highway") or w.get("railway") or "other"
        by_class[key] = by_class.get(key, 0.0) + L
        total += L
    return {
        "total_inundated_km": round(total / 1000.0, 3),
        "by_class_km": {k: round(v / 1000.0, 3)
                        for k, v in sorted(by_class.items(), key=lambda x: -x[1])},
        "ways_considered": len(roads_geom),
    }


def landcover_impact(landcover: Optional[np.ndarray], wet: np.ndarray,
                     cell_area: float) -> Dict[str, object]:
    """Inundated area per ESA WorldCover class -- gives cropland loss directly."""
    from .dem import WORLDCOVER_CLASSES
    if landcover is None:
        return {}
    out = {}
    for code, (label, _n) in WORLDCOVER_CLASSES.items():
        n = int(((landcover == code) & wet).sum())
        if n:
            out[label] = {"area_km2": round(n * cell_area / 1e6, 4),
                          "area_ha": round(n * cell_area / 1e4, 2)}
    return out


def district_breakdown(dem: DEM, district_raster: np.ndarray,
                       district_names: Dict[int, str], pop: np.ndarray,
                       hazard_cls: np.ndarray, depth: np.ndarray,
                       cell_area: float) -> List[dict]:
    rows = []
    wet = depth > 0.05
    for idx, name in district_names.items():
        m = district_raster == idx
        mw = m & wet
        if not mw.any():
            continue
        rows.append({
            "district": name,
            "inundated_km2": round(int(mw.sum()) * cell_area / 1e6, 4),
            "population_exposed": round(float(pop[mw].sum()), 0),
            "max_depth_m": round(float(depth[mw].max()), 2),
            "extreme_hazard_km2": round(int((m & (hazard_cls == 4)).sum())
                                        * cell_area / 1e6, 4),
        })
    rows.sort(key=lambda r: -r["population_exposed"])
    return rows


def evacuation_timeline(settlements: List[dict],
                        bands_min=(15, 30, 60, 120, 240)) -> List[dict]:
    """Group exposed settlements into warning-time bands for the HADR table."""
    out = []
    prev = 0
    for b in bands_min:
        sel = [s for s in settlements
               if s["arrival_min"] is not None and prev <= s["arrival_min"] < b]
        out.append({
            "band": f"{prev}-{b} min",
            "from_min": prev, "to_min": b,
            "settlements": len(sel),
            "names": [s["name"] for s in sel if s["name"] != "(unnamed)"][:12],
            "max_depth_m": round(max([s["depth_m"] for s in sel], default=0.0), 2),
        })
        prev = b
    late = [s for s in settlements
            if s["arrival_min"] is not None and s["arrival_min"] >= prev]
    out.append({
        "band": f">{prev} min", "from_min": prev, "to_min": None,
        "settlements": len(late),
        "names": [s["name"] for s in late if s["name"] != "(unnamed)"][:12],
        "max_depth_m": round(max([s["depth_m"] for s in late], default=0.0), 2),
    })
    return out


def estimate_losses(buildings: Dict[str, object], roads: Dict[str, object],
                    landcover: Dict[str, object],
                    values: AssetValues) -> Dict[str, object]:
    """Monetary loss.  Every figure is tagged with the unit rates used."""
    n_dmg = buildings.get("exposed", 0)
    mean_frac = buildings.get("mean_damage_fraction", 0.0) or 0.0
    building_loss = n_dmg * mean_frac * values.residential_per_building

    road_km = roads.get("total_inundated_km", 0.0) or 0.0
    road_loss = road_km * values.road_per_km * 0.35   # partial-damage factor

    crop_ha = 0.0
    for label, v in (landcover or {}).items():
        if "Cropland" in label:
            crop_ha += v.get("area_ha", 0.0)
    crop_loss = crop_ha * values.cropland_per_hectare

    total = building_loss + road_loss + crop_loss
    return {
        "currency": values.currency,
        "buildings": round(building_loss, 0),
        "roads": round(road_loss, 0),
        "cropland": round(crop_loss, 0),
        "total": round(total, 0),
        "total_crore": round(total / 1e7, 2),
        "components": {
            "buildings_exposed": n_dmg,
            "mean_damage_fraction": mean_frac,
            "road_km_inundated": road_km,
            "cropland_ha_inundated": round(crop_ha, 2),
        },
        "unit_values_used": values.to_dict(),
        "caveat": ("Monetary figures are derived from the unit rates above, "
                   "which are assumptions, not measurements. Physical exposure "
                   "counts are from OSM/WorldPop/WorldCover and are the "
                   "defensible output."),
    }


# ---------------------------------------------------------------------------
# QC gate
# ---------------------------------------------------------------------------

def qc_checks(dem: DEM, depth: np.ndarray, pop: np.ndarray,
              settlements: List[dict], exposure_counts: Dict[str, int],
              population_exposed: Optional[float] = None,
              buildings_exposed: Optional[int] = None) -> dict:
    wet = depth > 0.05
    edge_wet = bool(wet[0, :].any() or wet[-1, :].any() or
                    wet[:, 0].any() or wet[:, -1].any())
    msgs = []
    warnings = []
    if edge_wet:
        msgs.append("Inundation reaches the domain edge - extend the bbox to "
                    "capture the full downstream footprint.")
    if not settlements:
        msgs.append("No settlement intersects the flood extent; check the bbox.")
    if pop.sum() <= 0:
        msgs.append("Population raster is empty over this domain.")

    # Cross-check two independent exposure sources.  WorldPop is a modelled
    # 1 km surface; OSM buildings are volunteer-mapped and very incomplete in
    # rural India.  A large mismatch does not mean the run is wrong, but the
    # building count is then a severe lower bound and must not be read as an
    # asset inventory -- say so before anyone quotes it.
    ppb = None
    if population_exposed and buildings_exposed is not None:
        if buildings_exposed <= 0:
            warnings.append(
                f"{population_exposed:,.0f} people exposed but ZERO OSM "
                "buildings inundated: building footprints are unmapped here. "
                "Use the population figure, not the building count.")
        else:
            ppb = population_exposed / buildings_exposed
            if ppb > 50:
                warnings.append(
                    f"{population_exposed:,.0f} people exposed against only "
                    f"{buildings_exposed} inundated OSM buildings "
                    f"({ppb:,.0f} people per mapped building). OSM building "
                    "coverage is sparse in this area, so the building count is "
                    "a lower bound and damage-to-buildings is understated.")

    return {
        "inundation_touches_domain_edge": edge_wet,
        "exposure_feature_counts": exposure_counts,
        "population_total_in_domain": round(float(pop.sum()), 0),
        "population_exposed": population_exposed,
        "buildings_exposed": buildings_exposed,
        "people_per_inundated_building": round(ppb, 1) if ppb else None,
        "messages": msgs,
        "warnings": warnings,
        "pass": len(msgs) == 0,
    }
