"""Google Earth Engine bridge.

Two directions, deliberately separated because they have very different costs:

  EXPORT (always available, no credentials)
      Write a self-contained Earth Engine script plus a GeoJSON asset for a
      finished run, so the modelled flood can be opened in the Code Editor and
      draped over GEE's own imagery catalogue.  Nothing is uploaded and nothing
      needs an account -- the script is a text file the user pastes in.

  INGEST (optional, needs credentials)
      Use `ee` as a data source for imagery and terrain instead of the
      Microsoft Planetary Computer.  This requires a Google account, a
      registered Cloud project and `earthengine authenticate`.

The default path is EXPORT.  The rest of this framework deliberately runs with
no credentials at all -- Copernicus DEM from AWS Open Data, Sentinel-1/2 from
the Planetary Computer with a free anonymous SAS token, OSM from Overpass -- so
making GEE a hard dependency would take a pipeline anyone can run and put it
behind a Google login.  `ingest_available()` reports honestly whether the
optional path is usable on this machine rather than failing deep in a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Optional ingest path
# ---------------------------------------------------------------------------

def ingest_available() -> Tuple[bool, str]:
    """Is the authenticated `ee` path usable here?  Never raises."""
    try:
        import ee                                       # noqa: F401
    except ImportError:
        return False, ("earthengine-api is not installed. "
                       "`pip install earthengine-api` to enable the GEE "
                       "ingest path; the export path below needs nothing.")
    try:
        import ee
        ee.Initialize()
        return True, "earthengine-api initialised"
    except Exception as exc:                            # noqa: BLE001
        return False, (f"earthengine-api is installed but not authenticated "
                       f"({type(exc).__name__}: {exc}). Run "
                       f"`earthengine authenticate` and set a Cloud project.")


# GEE collection ids matching the products this framework already uses, so the
# two paths are like for like rather than quietly different data.
GEE_EQUIVALENTS = {
    "dem": {
        "collection": "COPERNICUS/DEM/GLO30",
        "band": "DEM",
        "matches": "Copernicus GLO-30 DSM, the same product read from AWS",
    },
    "sentinel2": {
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": ["B4", "B3", "B2"],
        "matches": "Sentinel-2 L2A surface reflectance, as used for the basemap",
    },
    "sentinel1": {
        "collection": "COPERNICUS/S1_GRD",
        "bands": ["VV"],
        "matches": ("Sentinel-1 GRD. NOTE: this is NOT the same product as the "
                    "sentinel-1-rtc used for validation -- GEE's S1_GRD is "
                    "terrain-flattened differently. Compare with care."),
    },
    "landcover": {
        "collection": "ESA/WorldCover/v200",
        "band": "Map",
        "matches": "ESA WorldCover v200, the same product used for roughness",
    },
    "population": {
        "collection": "WorldPop/GP/100m/pop",
        "matches": "WorldPop 100 m, the same family as the constrained product",
    },
}


# ---------------------------------------------------------------------------
# Export path -- no credentials needed
# ---------------------------------------------------------------------------

_SCRIPT = """// ---------------------------------------------------------------------------
// DamBurst run: %(run_id)s
// Dam: %(dam)s   Generated: %(created)s
//
// Paste into https://code.earthengine.google.com and press Run.
// The modelled flood extent is embedded below, so this script is
// self-contained: nothing needs to be uploaded as an asset first.
//
// The flood polygon is the output of the 2D shallow-water model in this
// framework. Everything it is drawn over is Earth Engine's own data, which
// makes this a genuine cross-check: if the modelled extent runs uphill or
// across a ridge it will be obvious against GEE's terrain and imagery.
// ---------------------------------------------------------------------------

var AOI = ee.Geometry.Rectangle(%(bbox)s);
Map.centerObject(AOI, %(zoom)d);

// -- modelled flood extent (from the DamBurst run) --------------------------
var flood = ee.FeatureCollection(%(flood)s);

// -- Copernicus GLO-30, the same DEM the model was built on -----------------
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30')
            .select('DEM').mosaic().clip(AOI);
var hill = ee.Terrain.hillshade(dem.multiply(%(vex).1f));
Map.addLayer(hill, {min: 0, max: 255}, 'GLO-30 hillshade', true, 0.85);

// -- Sentinel-2 true colour, least-cloudy scene in the window ---------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(AOI)
           .filterDate('%(s2_start)s', '%(s2_end)s')
           .sort('CLOUDY_PIXEL_PERCENTAGE')
           .first();
Map.addLayer(s2, {bands: ['B4','B3','B2'], min: 0, max: 3000},
             'Sentinel-2 true colour', false);

// -- Sentinel-1 VV, and a water mask by the same thresholding idea ----------
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterBounds(AOI)
           .filterDate('%(s1_start)s', '%(s1_end)s')
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .select('VV').median().clip(AOI);
Map.addLayer(s1, {min: -25, max: 0}, 'Sentinel-1 VV (dB)', false);

// Open water is low backscatter. The model run used an Otsu threshold; this
// fixed cut is only a visual sanity layer, not the validation number.
var slope = ee.Terrain.slope(dem);
var water = s1.lt(%(s1_thresh).1f).and(slope.lt(12)).selfMask();
Map.addLayer(water, {palette: ['00b4ff']}, 'S1 low-backscatter water', false);

// -- ESA WorldCover, the land cover behind the Manning field ----------------
var lc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(AOI);
Map.addLayer(lc, {}, 'ESA WorldCover v200', false);

// -- the modelled flood, on top ---------------------------------------------
Map.addLayer(flood.style({color: 'ff2d55', fillColor: '00000000', width: 2}),
             {}, 'DamBurst modelled flood extent');

print('DamBurst run', '%(run_id)s');
print('Modelled inundated area (km2)', %(area_km2).3f);
print('Peak breach discharge (m3/s)', %(peak_q).1f);
"""


def export_script(run_dir: Path, out_path: Optional[Path] = None,
                  max_vertices: int = 4000) -> Path:
    """Write a ready-to-paste Earth Engine script for a finished run.

    The flood polygon is inlined, so the script works without the user first
    uploading an asset -- which would need credentials and a Cloud project.
    """
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "results.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())

    bbox = manifest["scenario"]["bbox_ll"]
    flood = _flood_featurecollection(run_dir, max_vertices)

    s1 = (manifest.get("data_sources", {}).get("sentinel1") or {})
    s1_date = (s1.get("datetime") or "2024-01-01T00:00:00Z")[:10]
    s1_start, s1_end = _window_around(s1_date, days=30)
    val = results.get("validation") or {}
    thresh = val.get("otsu_threshold_db")
    if not isinstance(thresh, (int, float)):
        thresh = -16.0

    hz = results.get("hazard") or {}
    br = results.get("breach") or {}
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    zoom = 11 if span < 0.4 else 10 if span < 0.8 else 9

    script = _SCRIPT % {
        "run_id": results.get("run_id", run_dir.name),
        # The dossier in the manifest nests the OSM record; results.map.dam
        # carries the resolved display name.
        "dam": ((results.get("map") or {}).get("dam") or {}).get(
            "name", manifest["scenario"].get("dam_name", "unknown")),
        "created": manifest.get("created_utc", ""),
        "bbox": json.dumps([bbox[0], bbox[1], bbox[2], bbox[3]]),
        "zoom": zoom,
        "flood": json.dumps(flood),
        "vex": 1.0,
        "s2_start": "2023-10-01", "s2_end": "2024-05-31",
        "s1_start": s1_start, "s1_end": s1_end,
        "s1_thresh": float(thresh),
        "area_km2": float(hz.get("inundated_area_km2") or 0.0),
        "peak_q": float(br.get("peak_q") or 0.0),
    }
    out_path = Path(out_path or (run_dir / "earthengine" / "damburst_gee.js"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)

    # The same geometry as a plain file, for users who would rather upload it
    # as a table asset than carry it inline.
    (out_path.parent / "flood_extent_ee.geojson").write_text(json.dumps(flood))
    (out_path.parent / "README.txt").write_text(_README % {
        "run_id": results.get("run_id", run_dir.name),
        "script": out_path.name,
    })
    return out_path


_README = """DamBurst -> Google Earth Engine
================================

Run: %(run_id)s

1. Open https://code.earthengine.google.com
2. Paste the contents of %(script)s into the editor.
3. Press Run.

No asset upload and no authentication beyond your normal Earth Engine login is
needed: the modelled flood extent is embedded in the script.

What you are looking at
-----------------------
Every raster layer is Earth Engine's own copy of the data -- Copernicus GLO-30,
Sentinel-2 L2A, Sentinel-1 GRD, ESA WorldCover. The only DamBurst output is the
red flood outline. That makes this an independent check: the model was built on
data fetched from AWS and the Microsoft Planetary Computer, so if the extent
lines up with GEE's terrain and imagery, two separate copies of the world agree.

One caveat worth knowing: GEE's COPERNICUS/S1_GRD is not the same product as
the sentinel-1-rtc collection used for the validation score in this run. It is
terrain-flattened differently, so the water mask in the script is a visual
sanity layer only. The number in the Validation tab comes from the RTC product.

flood_extent_ee.geojson is the same geometry as a standalone file, if you would
rather upload it as a table asset.
"""


def _window_around(date_str: str, days: int = 30) -> Tuple[str, str]:
    import datetime
    try:
        d = datetime.date.fromisoformat(date_str)
    except ValueError:
        d = datetime.date(2024, 1, 1)
    return ((d - datetime.timedelta(days=days)).isoformat(),
            (d + datetime.timedelta(days=days)).isoformat())


def _flood_featurecollection(run_dir: Path, max_vertices: int) -> dict:
    """The run's flood extent as a GeoJSON FeatureCollection, simplified.

    Inlining a full-resolution polygon would make the script megabytes long and
    slow to paste, so the geometry is thinned. The thinning is reported in the
    feature properties rather than hidden.
    """
    src = run_dir / "vectors" / "flood_extent.geojson"
    if not src.exists():
        return {"type": "FeatureCollection", "features": []}
    gj = json.loads(src.read_text())

    total = _count_vertices(gj)
    step = max(1, total // max_vertices) if total > max_vertices else 1
    if step > 1:
        for f in gj.get("features", []):
            f["geometry"] = _thin_geometry(f["geometry"], step)
            f.setdefault("properties", {})["_simplified"] = (
                f"every {step}th vertex kept for inline embedding "
                f"({total} -> ~{total // step}); use the exported .shp for "
                f"any measurement")
    return gj


def _count_vertices(gj: dict) -> int:
    n = 0
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        for ring in _rings(g):
            n += len(ring)
    return n


def _rings(geom: dict) -> List[list]:
    t = geom.get("type")
    c = geom.get("coordinates") or []
    if t == "Polygon":
        return list(c)
    if t == "MultiPolygon":
        return [r for poly in c for r in poly]
    return []


def _thin_geometry(geom: dict, step: int) -> dict:
    def thin(ring: list) -> list:
        if len(ring) <= 5:
            return ring
        out = ring[::step]
        # A ring must stay closed or Earth Engine rejects it.
        if out[0] != ring[-1]:
            out.append(ring[-1])
        return out if len(out) >= 4 else ring

    t = geom.get("type")
    if t == "Polygon":
        return {"type": t, "coordinates": [thin(r) for r in geom["coordinates"]]}
    if t == "MultiPolygon":
        return {"type": t,
                "coordinates": [[thin(r) for r in poly]
                                for poly in geom["coordinates"]]}
    return geom
