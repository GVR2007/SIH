"""Output products: GeoTIFF, ESRI Shapefile, KML/KMZ, GeoJSON, CSV, PNG.

Covers the deliverable "Output should be converted to .shp or .Kml file" and
the dashboard's "Export - .shp / .kml" node.

Rasters are written in the model's projected CRS; vector products are written
in both the projected CRS and WGS84 (KML requires WGS84).
"""

from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import features as rfeatures
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import (calculate_default_transform, reproject,
                           transform as transform_coords, transform_geom)

import shapefile  # pyshp

from .dem import DEM
from .hazard import HAZARD_CLASSES


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------

def write_geotiff(path: Path, arr: np.ndarray, dem: DEM,
                  nodata: float = -9999.0, dtype: str = "float32",
                  description: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(np.isfinite(arr), arr, nodata).astype(dtype)
    prof = dict(driver="GTiff", height=dem.ny, width=dem.nx, count=1,
                dtype=dtype, crs=dem.crs, transform=dem.transform,
                nodata=nodata, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data, 1)
        if description:
            dst.update_tags(1, DESCRIPTION=description)
    return path


def raster_to_png_4326(arr: np.ndarray, dem: DEM, path: Path,
                       vmin: float, vmax: float, cmap: str = "depth",
                       mask_below: float = 0.0,
                       max_px: int = 1600) -> dict:
    """Reproject to WGS84 and write an RGBA PNG for a Leaflet image overlay.

    Returns the lat/lon bounds the front end needs to place the image.
    """
    from PIL import Image

    dst_crs = CRS.from_epsg(4326)
    src_crs = CRS.from_string(dem.crs)
    left, bottom, right, top = dem.bounds()
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, dem.nx, dem.ny, left, bottom, right, top)

    scale = min(1.0, max_px / max(width, height))
    if scale < 1.0:
        width = max(int(width * scale), 1)
        height = max(int(height * scale), 1)
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, dem.nx, dem.ny, left, bottom, right, top,
            dst_width=width, dst_height=height)

    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(arr.astype("float32"), dst,
              src_transform=dem.transform, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=transform, dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)

    rgba = _colourise(dst, vmin, vmax, cmap, mask_below)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)

    w = transform.c
    n = transform.f
    e = w + width * transform.a
    s = n + height * transform.e
    return {"file": path.name, "bounds_ll": [[s, w], [n, e]],
            "vmin": vmin, "vmax": vmax, "cmap": cmap,
            "width": width, "height": height}


def hillshade_png_4326(dem: DEM, path: Path, azimuth: float = 315.0,
                       altitude: float = 45.0, z_factor: float = 2.0,
                       max_px: int = 1600,
                       rgb: Optional[np.ndarray] = None) -> dict:
    """Render the DEM as a shaded-relief basemap PNG in WGS84.

    The dashboard must not depend on an external tile server: a judged demo may
    run with no internet, and a sandboxed browser may refuse third-party tiles
    outright -- either way the map goes black and the flood ribbon, which
    occupies only a few percent of the frame in a gorge, becomes invisible.
    Shaded relief from the DEM we already hold is always available, and for a
    dam-break study it carries more information than a road map: the valley the
    wave travels down is the whole story.
    """
    from PIL import Image

    dst_crs = CRS.from_epsg(4326)
    src_crs = CRS.from_string(dem.crs)
    left, bottom, right, top = dem.bounds()
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, dem.nx, dem.ny, left, bottom, right, top)
    scale = min(1.0, max_px / max(width, height))
    if scale < 1.0:
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, dem.nx, dem.ny, left, bottom, right, top,
            dst_width=max(int(width * scale), 1),
            dst_height=max(int(height * scale), 1))

    ny, nx = dem.shape
    z = np.asarray(dem.z, dtype="float64")
    gy, gx = np.gradient(z, dem.dy, dem.dx)
    slope = np.arctan(z_factor * np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    shade = (np.sin(alt) * np.cos(slope) +
             np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    shade = np.clip(shade, 0.0, 1.0)

    # tint by elevation so valleys read dark and ridges light, then modulate
    # with the hillshade: flat greyscale relief is hard to read on a dark UI
    lo, hi = np.nanpercentile(z, [2, 98])
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    nsrc = 2 if rgb is None else 5
    src = np.zeros((nsrc, ny, nx), dtype="float32")
    src[0], src[1] = shade, t
    if rgb is not None:
        src[2:5] = rgb.astype("float32")
    dst = np.full((nsrc, height, width), np.nan, dtype="float32")
    reproject(src, dst, src_transform=dem.transform, src_crs=src_crs,
              src_nodata=np.nan, dst_transform=transform, dst_crs=dst_crs,
              dst_nodata=np.nan, resampling=Resampling.bilinear)
    sh, tt = dst[0], dst[1]
    valid = np.isfinite(sh) & np.isfinite(tt)
    sh = np.nan_to_num(sh, nan=0.5)
    tt = np.nan_to_num(tt, nan=0.0)

    # Deliberately near-neutral: the basemap must not compete with the flood
    # ramps drawn on top of it. A saturated terrain tint makes a blue depth
    # overlay unreadable over a blue valley floor. Slight warmth with
    # elevation is enough to read relief without claiming any hue.
    base = np.stack([
        0.30 + 0.46 * tt,
        0.30 + 0.44 * tt,
        0.31 + 0.40 * tt,
    ], axis=-1)

    if rgb is not None:
        # Real imagery where we have it, relief-shaded so the terrain still
        # reads. Sentinel-2 granules rarely tile a whole study bbox, so the
        # synthetic relief fills the gaps instead of leaving them black.
        sat = np.moveaxis(np.nan_to_num(dst[2:5], nan=-1.0), 0, -1)
        have = sat.max(axis=-1) > 0.0
        base = np.where(have[..., None], np.clip(sat, 0.0, 1.0), base)

    out = np.clip(base * (0.30 + 0.80 * sh[..., None]), 0.0, 1.0)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = (out * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)

    w, n = transform.c, transform.f
    e, s = w + width * transform.a, n + height * transform.e
    return {"file": path.name, "bounds_ll": [[s, w], [n, e]],
            "width": width, "height": height,
            "satellite": rgb is not None,
            "description": ("Sentinel-2 true colour, relief-shaded, with DEM "
                            "hillshade filling granule gaps" if rgb is not None
                            else "Shaded relief from the Copernicus GLO-30 DEM")}


def export_terrain_3d(dem: DEM, depth_frames: Sequence[np.ndarray],
                      frame_times: Sequence[float], path: Path,
                      max_dim: int = 224, depth_cap: Optional[float] = None,
                      hazard: Optional[np.ndarray] = None,
                      dam_cells: Optional[np.ndarray] = None,
                      rgb: Optional[np.ndarray] = None) -> dict:
    """Write the terrain + time-varying water surface for the 3D viewer.

    The browser needs the actual elevation grid, not a picture of one, so this
    ships numbers.  Two encodings keep it small enough to fetch comfortably:

      * elevation as Float32 (exact -- terrain relief is the whole point), and
      * each depth frame as UInt8 quantised against a shared cap, because the
        water surface only needs enough precision to look right, and 36 frames
        of Float32 at full model resolution would be tens of megabytes.

    Both are base64'd into one JSON document so the viewer needs a single
    request and no binary-range plumbing.
    """
    import base64

    ny, nx = dem.shape
    step = max(1, int(math.ceil(max(ny, nx) / max_dim)))
    z = np.asarray(dem.z, dtype="float32")[::step, ::step]
    sy, sx = z.shape

    cap = float(depth_cap) if depth_cap else 0.0
    if not cap:
        for h in depth_frames:
            if h is not None and h.size:
                cap = max(cap, float(np.nanpercentile(h[h > 0.05], 99))
                          if (h > 0.05).any() else 0.0)
    cap = max(cap, 1.0)

    frames_b64 = []
    for h in depth_frames:
        if h is None:
            continue
        hh = np.asarray(h, dtype="float32")[::step, ::step]
        q = np.clip(hh / cap, 0.0, 1.0) * 255.0
        # keep a wetted cell visible even when it quantises to zero
        q = np.where((hh > 0.05) & (q < 1.0), 1.0, q)
        frames_b64.append(base64.b64encode(
            q.astype(np.uint8).tobytes()).decode("ascii"))

    payload = {
        "nx": int(sx), "ny": int(sy),
        "dx_m": float(dem.dx * step), "dy_m": float(dem.dy * step),
        "z_min": float(np.nanmin(z)), "z_max": float(np.nanmax(z)),
        "elevation_b64": base64.b64encode(
            np.nan_to_num(z, nan=float(np.nanmin(z))).tobytes()).decode("ascii"),
        "depth_cap_m": cap,
        "frames_b64": frames_b64,
        "times_s": [round(float(t), 1) for t in frame_times][:len(frames_b64)],
        "downsample_step": step,
        "encoding": ("elevation: float32 little-endian, row-major, ny*nx; "
                     "frames: uint8, depth_m = value/255 * depth_cap_m"),
    }
    # Per-vertex satellite colour, so the 3D terrain wears real imagery rather
    # than a synthetic ramp. Shipped as uint8 RGB triples alongside the grid.
    if rgb is not None:
        sat = (np.clip(np.asarray(rgb, dtype="float32"), 0, 1)
               [:, ::step, ::step] * 255).astype(np.uint8)
        payload["satellite_rgb_b64"] = base64.b64encode(
            np.ascontiguousarray(np.moveaxis(sat, 0, -1)).tobytes()).decode("ascii")

    if hazard is not None:
        hz = np.asarray(hazard, dtype="uint8")[::step, ::step]
        payload["hazard_b64"] = base64.b64encode(hz.tobytes()).decode("ascii")

    # The dam crest, in the SAME downsampled index space as the meshes. Doing
    # the projection here rather than re-deriving it in the browser from
    # lat/lon keeps the marker exactly on the barrier the solver used.
    if dam_cells is not None and len(dam_cells):
        pts = []
        for r, c in np.asarray(dam_cells):
            i, j = int(c) // step, int(r) // step
            if 0 <= i < sx and 0 <= j < sy:
                pts.append([i, j, float(z[j, i])])
        if pts:
            payload["dam_crest"] = pts
            payload["dam_crest_note"] = "[ix, iy, elevation_m] in the mesh grid"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return {"file": path.name, "grid": [int(sy), int(sx)],
            "frames": len(frames_b64), "depth_cap_m": round(cap, 2),
            "downsample_step": step,
            "bytes": path.stat().st_size}


def build_terrain3d_from_run(run_dir: Path, satellite: bool = True) -> dict:
    """Derive the 3D payload from a finished run's own artefacts.

    A run directory already contains everything the 3D view needs -- the
    conditioned DEM, the hazard raster and the solver's depth frames -- so the
    payload is reconstructed on demand rather than only existing for runs that
    happened to be produced after the 3D exporter was added. Older runs would
    otherwise be permanently 3D-less unless the whole simulation were repeated,
    which is minutes of compute to regenerate data already sitting on disk.
    """
    run_dir = Path(run_dir)
    dem_path = run_dir / "rasters" / "dem_conditioned.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"{dem_path} missing; cannot rebuild 3D data")

    with rasterio.open(dem_path) as src:
        z = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            z = np.where(z == nodata, np.nan, z)
        z = np.where(np.isfinite(z), z, np.nanmin(z[np.isfinite(z)]))
        dem = DEM(z=z, transform=src.transform, crs=str(src.crs),
                  dx=abs(src.transform.a), dy=abs(src.transform.e),
                  name=run_dir.name, source=f"rebuilt from {dem_path.name}")

    # depth frames, newest layout first, then any single-level fallback
    frames, times = [], []
    results = {}
    rj = run_dir / "results.json"
    if rj.exists():
        try:
            results = json.loads(rj.read_text())
        except Exception:                            # noqa: BLE001
            results = {}
    meta_times = ((results.get("exports") or {}).get("frames") or {}).get("times_s") or []

    frame_dirs = [run_dir / "frames" / "grid_standalone",
                  run_dir / "frames" / "coupled", run_dir / "frames"]
    for fd in frame_dirs:
        if not fd.is_dir():
            continue
        files = sorted(fd.glob("frame_*.npz"))
        if not files:
            continue
        for i, f in enumerate(files):
            try:
                frames.append(np.load(f)["h"].astype("float32"))
            except Exception:                        # noqa: BLE001
                continue
            times.append(float(meta_times[i]) if i < len(meta_times) else float(i))
        break

    if not frames:
        dmax = run_dir / "rasters" / "depth_max.tif"
        if not dmax.exists():
            raise FileNotFoundError("no depth frames and no depth_max.tif")
        with rasterio.open(dmax) as src:
            d = src.read(1).astype("float32")
            if src.nodata is not None:
                d = np.where(d == src.nodata, 0.0, d)
        frames, times = [np.nan_to_num(d)], [0.0]

    hazard = None
    hz_path = run_dir / "rasters" / "hazard_class.tif"
    if hz_path.exists():
        with rasterio.open(hz_path) as src:
            h = src.read(1)
            hazard = np.nan_to_num(h, nan=0.0).astype("uint8")

    # The dam crest is stored in lat/lon in results.json; project it back into
    # grid cells so the marker lands on the barrier the solver actually used.
    dam_cells = None
    dam = (results.get("map") or {}).get("dam") or {}
    geom = dam.get("geometry_ll") or []
    if geom:
        lons = [p[0] for p in geom]
        lats = [p[1] for p in geom]
        xs, ys = transform_coords(CRS.from_epsg(4326),
                                  CRS.from_string(dem.crs), lons, lats)
        inv = ~dem.transform
        rc = []
        for x, y in zip(xs, ys):
            col, row = inv * (x, y)
            r, c = int(row), int(col)
            if 0 <= r < dem.ny and 0 <= c < dem.nx:
                rc.append((r, c))
        if rc:
            dam_cells = np.array(rc)

    rgb = None
    if satellite:
        bbox = (results.get("map") or {}).get("bbox_ll")
        if bbox:
            try:
                # cache_only: this runs inside a web request, and mosaicking
                # granules on a miss would block the client for minutes. If the
                # imagery was never fetched for this area the terrain simply
                # falls back to elevation-tinted relief.
                from . import datasources as _ds
                rgb, _ = _ds.fetch_sentinel2_rgb(tuple(bbox), dem,
                                                 cache_only=True)
            except Exception:                        # noqa: BLE001 - optional
                rgb = None

    return export_terrain_3d(dem, frames, times, run_dir / "terrain3d.json",
                             hazard=hazard, dam_cells=dam_cells, rgb=rgb)


def wet_bounds_ll(wet: np.ndarray, dem: DEM, pad_frac: float = 0.12
                  ) -> Optional[List[List[float]]]:
    """Lat/lon bounds of the inundated cells, for zooming the map.

    Fitting the view to the whole model domain leaves the flood as a hairline:
    the wetted area is a few percent of a bounding box sized for the catchment.
    """
    if not wet.any():
        return None
    rows, cols = np.where(wet)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    left, top = dem.xy(r0, c0)
    right, bottom = dem.xy(r1, c1)
    xs = [min(left, right), max(left, right)]
    ys = [min(top, bottom), max(top, bottom)]
    padx = (xs[1] - xs[0]) * pad_frac
    pady = (ys[1] - ys[0]) * pad_frac
    corners_x = [xs[0] - padx, xs[1] + padx, xs[0] - padx, xs[1] + padx]
    corners_y = [ys[0] - pady, ys[0] - pady, ys[1] + pady, ys[1] + pady]
    lon, lat = transform_coords(CRS.from_string(dem.crs), CRS.from_epsg(4326),
                                corners_x, corners_y)
    return [[float(min(lat)), float(min(lon))],
            [float(max(lat)), float(max(lon))]]


_RAMPS = {
    # perceptually ordered, colour-blind safe ramps
    "depth": [(0.00, (222, 235, 247)), (0.25, (158, 202, 225)),
              (0.50, (66, 146, 198)), (0.75, (8, 81, 156)), (1.00, (4, 35, 92))],
    "speed": [(0.00, (255, 255, 204)), (0.25, (161, 218, 180)),
              (0.50, (65, 182, 196)), (0.75, (34, 94, 168)), (1.00, (12, 44, 132))],
    "hazard": [(0.00, (44, 127, 184)), (0.34, (127, 205, 187)),
               (0.67, (253, 174, 97)), (1.00, (215, 25, 28))],
    "arrival": [(0.00, (215, 25, 28)), (0.25, (253, 174, 97)),
                (0.50, (255, 255, 191)), (0.75, (171, 217, 233)),
                (1.00, (44, 123, 182))],
}


def _colourise(a: np.ndarray, vmin: float, vmax: float, cmap: str,
               mask_below: float) -> np.ndarray:
    ramp = _RAMPS.get(cmap, _RAMPS["depth"])
    h, w = a.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    valid = np.isfinite(a) & (a > mask_below)
    if not valid.any():
        return out
    t = np.clip((a - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    stops = np.array([s for s, _ in ramp])
    cols = np.array([c for _, c in ramp], dtype=float)
    for ch in range(3):
        out[..., ch] = np.where(valid,
                                np.interp(t, stops, cols[:, ch]).astype(np.uint8), 0)
    alpha = np.clip(60 + 195 * t, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(valid, alpha, 0)
    return out


# ---------------------------------------------------------------------------
# Vector: polygonise
# ---------------------------------------------------------------------------

def polygonise(mask: np.ndarray, dem: DEM, value_raster: Optional[np.ndarray] = None,
               min_area_m2: float = 0.0) -> List[dict]:
    """Vectorise a boolean/class raster into polygon features (projected CRS)."""
    src = (value_raster if value_raster is not None else mask.astype(np.int16))
    src = src.astype(np.int32)
    valid = mask.astype(np.uint8)
    feats = []
    for geom, val in rfeatures.shapes(src, mask=valid.astype(bool),
                                      transform=dem.transform, connectivity=8):
        area = _ring_area(geom)
        if area < min_area_m2:
            continue
        feats.append({"geometry": geom, "value": int(val), "area_m2": area})
    return feats


def _ring_area(geom: dict) -> float:
    total = 0.0
    for ring_i, ring in enumerate(geom.get("coordinates", [])):
        a = 0.0
        for k in range(len(ring) - 1):
            x1, y1 = ring[k][0], ring[k][1]
            x2, y2 = ring[k + 1][0], ring[k + 1][1]
            a += x1 * y2 - x2 * y1
        a = abs(a) / 2.0
        total += a if ring_i == 0 else -a
    return max(total, 0.0)


# ---------------------------------------------------------------------------
# Shapefile
# ---------------------------------------------------------------------------

def write_shapefile(path: Path, feats: List[dict], dem: DEM,
                    fields: Sequence[Tuple[str, str, int, int]],
                    attr_fn) -> Path:
    """Write polygons to an ESRI Shapefile (+ .prj) using pyshp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = str(path.with_suffix(""))

    w = shapefile.Writer(base, shapeType=shapefile.POLYGON)
    for name, ftype, size, dec in fields:
        w.field(name, ftype, size, dec)
    for f in feats:
        coords = f["geometry"]["coordinates"]
        parts = [[[float(x), float(y)] for x, y in ring] for ring in coords]
        w.poly(parts)
        w.record(*attr_fn(f))
    w.close()

    try:
        wkt = CRS.from_string(dem.crs).to_wkt()
        Path(base + ".prj").write_text(wkt)
    except Exception:
        pass
    return Path(base + ".shp")


def export_flood_extent_shp(path: Path, depth: np.ndarray, hazard_cls: np.ndarray,
                            dem: DEM, threshold: float = 0.05) -> Path:
    mask = depth > threshold
    feats = polygonise(mask, dem, hazard_cls, min_area_m2=dem.cell_area * 2)
    fields = [("haz_class", "N", 4, 0), ("haz_name", "C", 20, 0),
              ("area_m2", "N", 18, 2), ("area_ha", "N", 16, 3)]

    def attrs(f):
        v = int(f["value"])
        name = HAZARD_CLASSES[v - 1][2] if 1 <= v <= 4 else "None"
        return [v, name, round(f["area_m2"], 2), round(f["area_m2"] / 1e4, 3)]

    return write_shapefile(path, feats, dem, fields, attrs)


# ---------------------------------------------------------------------------
# KML / KMZ
# ---------------------------------------------------------------------------

KML_COLOURS = {1: "ffb87f2c", 2: "ffbbcd7f", 3: "ff61aefd", 4: "ff1c19d7"}


def export_kml(path: Path, depth: np.ndarray, hazard_cls: np.ndarray, dem: DEM,
               name: str = "Dam-break inundation",
               threshold: float = 0.05, kmz: bool = True) -> Path:
    """Write the hazard-classified extent as KML (WGS84), optionally zipped."""
    mask = depth > threshold
    feats = polygonise(mask, dem, hazard_cls, min_area_m2=dem.cell_area * 4)

    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f"<name>{_esc(name)}</name>"]
    for cid, colour in KML_COLOURS.items():
        parts.append(
            f'<Style id="haz{cid}"><LineStyle><color>ff333333</color>'
            f"<width>1</width></LineStyle>"
            f"<PolyStyle><color>{colour[:2]}99{colour[2:]}</color>"
            f"<fill>1</fill><outline>1</outline></PolyStyle></Style>")

    for f in feats:
        v = int(f["value"])
        if not 1 <= v <= 4:
            continue
        hname = HAZARD_CLASSES[v - 1][2]
        geom = transform_geom(dem.crs, "EPSG:4326", f["geometry"])
        rings = geom["coordinates"]
        outer = _kml_ring(rings[0])
        inners = "".join(
            f"<innerBoundaryIs><LinearRing><coordinates>{_kml_ring(r)}"
            "</coordinates></LinearRing></innerBoundaryIs>" for r in rings[1:])
        parts.append(
            f"<Placemark><name>{hname}</name><styleUrl>#haz{v}</styleUrl>"
            f"<ExtendedData>"
            f'<Data name="hazard_class"><value>{v}</value></Data>'
            f'<Data name="area_ha"><value>{f["area_m2"] / 1e4:.3f}</value></Data>'
            f"</ExtendedData>"
            f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{outer}"
            f"</coordinates></LinearRing></outerBoundaryIs>{inners}</Polygon>"
            f"</Placemark>")
    parts.append("</Document></kml>")
    kml = "\n".join(parts)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kmz:
        out = path.with_suffix(".kmz")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("doc.kml", kml)
        return out
    out = path.with_suffix(".kml")
    out.write_text(kml)
    return out


def _kml_ring(ring) -> str:
    return " ".join(f"{x:.6f},{y:.6f},0" for x, y in ring)


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# GeoJSON / CSV
# ---------------------------------------------------------------------------

def export_geojson(path: Path, depth: np.ndarray, hazard_cls: np.ndarray,
                   dem: DEM, threshold: float = 0.05,
                   simplify_min_ha: float = 0.05) -> Path:
    mask = depth > threshold
    feats = polygonise(mask, dem, hazard_cls,
                       min_area_m2=max(dem.cell_area * 4, simplify_min_ha * 1e4))
    out = {"type": "FeatureCollection", "features": []}
    for f in feats:
        v = int(f["value"])
        if not 1 <= v <= 4:
            continue
        out["features"].append({
            "type": "Feature",
            "geometry": transform_geom(dem.crs, "EPSG:4326", f["geometry"]),
            "properties": {"hazard_class": v,
                           "hazard_name": HAZARD_CLASSES[v - 1][2],
                           "area_ha": round(f["area_m2"] / 1e4, 4)},
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return path


def export_csv(path: Path, rows: List[dict],
               columns: Optional[Sequence[str]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    cols = list(columns or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    return path


def export_hydrograph_csv(path: Path, t: np.ndarray, series: Dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(series)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "t_min"] + keys)
        for k in range(len(t)):
            wr.writerow([f"{t[k]:.3f}", f"{t[k] / 60:.4f}"] +
                        [f"{series[key][k]:.6g}" for key in keys])
    return path
