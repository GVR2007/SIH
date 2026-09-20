"""Real open-data acquisition.

Every raster and vector layer the framework consumes is fetched live from a
public, no-registration source.  Nothing in this module fabricates data; if a
source is unreachable the call raises rather than substituting a guess.

    layer                source                                      licence
    -------------------  ------------------------------------------  -----------
    elevation            Copernicus GLO-30 DSM COGs on AWS S3        free/open
    land cover           ESA WorldCover v200 (10 m) via MS PC STAC   CC-BY 4.0
    population           WorldPop 2020 constrained/1 km, India       CC-BY 4.0
    dam + exposure       OpenStreetMap via Overpass API              ODbL
    dam attributes       Wikidata SPARQL + Wikipedia REST            CC0 / CC-BY-SA
    districts            geoBoundaries gbOpen ADM2                   CC-BY 4.0
    observed flood       Sentinel-1 GRD via MS Planetary Computer    free/open

All responses are cached under `data/cache` keyed by request hash so a demo
runs offline once primed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import windows as rwindows
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from .dem import DEM, roughness_from_worldcover

USER_AGENT = "damburst-sih2026/0.1 (dam-break inundation prototype; contact: sih-team)"

CACHE_DIR = Path(os.environ.get("DAMBURST_CACHE",
                                Path(__file__).resolve().parents[2] / "data" / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    # Deliberately NOT setting CPL_VSIL_CURL_ALLOWED_EXTENSIONS: Sentinel-1 GRD
    # assets are `.tiff` carrying a SAS query string, and the extension filter
    # rejects them outright ("does not exist in the file system"). Every URL
    # here is constructed explicitly, so the filter buys nothing.
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="134217728",
    GDAL_NUM_THREADS="ALL_CPUS",
)
os.environ.update(GDAL_ENV)

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

COP_DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
WORLDPOP_100M = ("https://data.worldpop.org/GIS/Population/"
                 "Global_2000_2020_Constrained/2020/BSGM/{iso}/{iso_l}_ppp_2020_constrained.tif")
WORLDPOP_1KM = ("https://data.worldpop.org/GIS/Population/"
                "Global_2000_2020_1km/2020/{iso}/{iso_l}_ppp_2020_1km_Aggregated.tif")


class SourceUnavailable(RuntimeError):
    """Raised when a live source cannot be reached.  Never silently substituted."""


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

def _cache_path(tag: str, key: str, ext: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{tag}_{h}{ext}"


def http_get(url: str, data: Optional[bytes] = None,
             headers: Optional[dict] = None, timeout: int = 120,
             retries: int = 3) -> bytes:
    hdr = {"User-Agent": USER_AGENT}
    if headers:
        hdr.update(headers)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise SourceUnavailable(f"{url}: {last}")


def http_json(url: str, payload: Optional[dict] = None, timeout: int = 120) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    return json.loads(http_get(url, data, headers, timeout))


def download_file(url: str, dest: Path, timeout: int = 900) -> Path:
    """Stream a large file to disk once, then reuse the cached copy."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


@dataclass
class Grid:
    """Target computational grid in a projected CRS."""
    crs: str
    transform: object
    width: int
    height: int
    res: float

    def bounds(self):
        w, n = self.transform.c, self.transform.f
        return (w, n - self.height * self.res, w + self.width * self.res, n)


def make_grid(bbox_ll: Sequence[float], res_m: float) -> Grid:
    """Build the computational domain/mesh from a lon/lat bbox."""
    west, south, east, north = bbox_ll
    crs = utm_epsg((west + east) / 2, (south + north) / 2)
    l, b, r, t = transform_bounds(CRS.from_epsg(4326), CRS.from_string(crs),
                                  west, south, east, north, densify_pts=64)
    l = math.floor(l / res_m) * res_m
    b = math.floor(b / res_m) * res_m
    r = math.ceil(r / res_m) * res_m
    t = math.ceil(t / res_m) * res_m
    width = int(round((r - l) / res_m))
    height = int(round((t - b) / res_m))
    return Grid(crs=crs, transform=from_origin(l, t, res_m, res_m),
                width=width, height=height, res=res_m)


def _warp_into(src_path: str, grid: Grid, band: int = 1,
               src_nodata: Optional[float] = None,
               resampling: Resampling = Resampling.bilinear,
               dtype: str = "float32",
               bbox_ll: Optional[Sequence[float]] = None) -> np.ndarray:
    """Windowed read of a remote raster reprojected onto the target grid."""
    dest = np.full((grid.height, grid.width), np.nan, dtype=dtype)
    with rasterio.open(src_path) as src:
        if bbox_ll is not None:
            sb = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox_ll, densify_pts=32)
            win = rwindows.from_bounds(*sb, transform=src.transform)
            win = win.round_offsets().round_lengths()
            win = win.intersection(rwindows.Window(0, 0, src.width, src.height))
            pad = 8
            win = rwindows.Window(max(0, win.col_off - pad), max(0, win.row_off - pad),
                                  min(src.width, win.width + 2 * pad),
                                  min(src.height, win.height + 2 * pad))
            if win.width <= 0 or win.height <= 0:
                return dest
            arr = src.read(band, window=win).astype(dtype)
            src_tr = src.window_transform(win)
        else:
            arr = src.read(band).astype(dtype)
            src_tr = src.transform
        nod = src_nodata if src_nodata is not None else src.nodata
        reproject(arr, dest,
                  src_transform=src_tr, src_crs=src.crs, src_nodata=nod,
                  dst_transform=grid.transform, dst_crs=CRS.from_string(grid.crs),
                  dst_nodata=np.nan, resampling=resampling)
    return dest


# ---------------------------------------------------------------------------
# 1. Elevation -- Copernicus GLO-30
# ---------------------------------------------------------------------------

def _cop_dem_tiles(bbox_ll: Sequence[float], pad_deg: float = 0.12) -> List[str]:
    # Pad before tile selection: the UTM target grid is a rotated superset of the
    # lon/lat bbox, so its corners can fall into the neighbouring 1-degree tile.
    west, south, east, north = (bbox_ll[0] - pad_deg, bbox_ll[1] - pad_deg,
                                bbox_ll[2] + pad_deg, bbox_ll[3] + pad_deg)
    urls = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            name = (f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
                    f"{ew}{abs(lon):03d}_00_DEM")
            urls.append(f"/vsicurl/{COP_DEM_BUCKET}/{name}/{name}.tif")
    return urls


def fetch_dem(bbox_ll: Sequence[float], res_m: float = 60.0,
              name: str = "dem") -> DEM:
    """Copernicus GLO-30 DSM, mosaicked and reprojected to UTM at `res_m`.

    GLO-30 is the ESA/Airbus WorldDEM-derived global 30 m DSM, released for free
    public use.  Heights are orthometric (EGM2008).
    """
    grid = make_grid(bbox_ll, res_m)
    cache = _cache_path("copdem", f"{bbox_ll}|{res_m}", ".tif")

    if cache.exists():
        with rasterio.open(cache) as src:
            z = src.read(1).astype(np.float64)
            tr, crs = src.transform, str(src.crs)
    else:
        mosaic = np.full((grid.height, grid.width), np.nan, dtype="float32")
        got = 0
        errors = []
        pad = 0.12
        read_bbox = (bbox_ll[0] - pad, bbox_ll[1] - pad,
                     bbox_ll[2] + pad, bbox_ll[3] + pad)
        for url in _cop_dem_tiles(bbox_ll):
            try:
                part = _warp_into(url, grid, bbox_ll=read_bbox)
            except Exception as exc:                  # tile may not exist (ocean)
                errors.append(f"{url.rsplit('/', 1)[-1]}: {exc}")
                continue
            fill = np.isnan(mosaic) & np.isfinite(part)
            mosaic[fill] = part[fill]
            got += 1
        if got == 0:
            raise SourceUnavailable(
                "No Copernicus GLO-30 tile could be read for bbox "
                f"{bbox_ll}. Attempts:\n  " + "\n  ".join(errors))
        z = mosaic.astype(np.float64)
        tr, crs = grid.transform, grid.crs
        prof = dict(driver="GTiff", height=grid.height, width=grid.width, count=1,
                    dtype="float32", crs=grid.crs, transform=grid.transform,
                    nodata=np.nan, compress="deflate", tiled=True)
        with rasterio.open(cache, "w", **prof) as dst:
            dst.write(mosaic, 1)

    return DEM(z=z, transform=tr, crs=crs, dx=res_m, dy=res_m, name=name,
               source=("Copernicus GLO-30 DSM (ESA/Airbus), AWS Open Data "
                       "bucket copernicus-dem-30m"),
               meta={"bbox_ll": list(bbox_ll), "res_m": res_m})


# ---------------------------------------------------------------------------
# 2. Land cover -- ESA WorldCover via Planetary Computer
# ---------------------------------------------------------------------------

_sas_cache: Dict[str, Tuple[float, str]] = {}


def pc_sign(href: str, collection: str) -> str:
    """Attach a Planetary Computer SAS token (free, no account needed)."""
    now = time.time()
    tok = _sas_cache.get(collection)
    if tok is None or tok[0] < now:
        j = http_json(f"{PC_SAS}/{collection}")
        _sas_cache[collection] = (now + 1800, j["token"])
        tok = _sas_cache[collection]
    sep = "&" if "?" in href else "?"
    return f"{href}{sep}{tok[1]}"


def pc_search(collection: str, bbox_ll: Sequence[float],
              datetime_range: Optional[str] = None, limit: int = 12,
              query: Optional[dict] = None) -> List[dict]:
    payload = {"collections": [collection], "bbox": list(bbox_ll), "limit": limit}
    if datetime_range:
        payload["datetime"] = datetime_range
    if query:
        payload["query"] = query
    j = http_json(f"{PC_STAC}/search", payload)
    return j.get("features", [])


def fetch_landcover(bbox_ll: Sequence[float], dem: DEM) -> Tuple[np.ndarray, str]:
    """ESA WorldCover v200 (2021, 10 m) resampled to the model grid."""
    grid = Grid(crs=dem.crs, transform=dem.transform, width=dem.nx,
                height=dem.ny, res=dem.dx)
    items = pc_search("esa-worldcover", bbox_ll, limit=6)
    if not items:
        raise SourceUnavailable("No ESA WorldCover tile returned for this bbox")

    out = np.full((dem.ny, dem.nx), np.nan, dtype="float32")
    used = []
    for it in items:
        asset = it["assets"].get("map")
        if asset is None:
            continue
        href = pc_sign(asset["href"], "esa-worldcover")
        try:
            part = _warp_into(href, grid, resampling=Resampling.nearest,
                              bbox_ll=bbox_ll)
        except Exception:
            continue
        fill = np.isnan(out) & np.isfinite(part)
        out[fill] = part[fill]
        used.append(it["id"])
    if not used:
        raise SourceUnavailable("ESA WorldCover assets could not be read")
    lc = np.nan_to_num(out, nan=30).astype(np.int16)   # unknown -> grassland
    return lc, f"ESA WorldCover v200 via MS Planetary Computer ({', '.join(used[:3])})"


def fetch_sentinel2_rgb(bbox_ll: Sequence[float], dem: DEM,
                        start: str = "2023-10-01", end: str = "2024-05-31",
                        max_cloud: int = 15, max_px: int = 700,
                        budget_s: float = 150.0,
                        cache_only: bool = False) -> Tuple[np.ndarray, dict]:
    """Real Sentinel-2 L2A true-colour composite on the model grid.

    Returns (rgb float32 in 0..1 with shape (3, ny, nx), provenance).

    Picks the least-cloudy scene in the window rather than the most recent: for
    a Himalayan study area the difference between a 3% and a 40% cloud scene is
    the difference between usable imagery and a white sheet. Winter/spring is
    the default window because the monsoon makes summer scenes unusable.
    """
    # Mosaicking 10 m granules over a basin is the slowest step in the whole
    # pipeline (minutes), and it is pure input data -- cache it so the second
    # and subsequent demo runs over the same area are instant.
    ckey = f"s2rgb|{tuple(np.round(bbox_ll,5))}|{dem.crs}|{dem.nx}x{dem.ny}|{dem.dx}|{start}|{end}|{max_cloud}|{max_px}"
    cpath = _cache_path("s2rgb", ckey, ".npz")
    if cpath.exists():
        try:
            z = np.load(cpath, allow_pickle=False)
            prov = json.loads(str(z["prov"]))
            prov["cached"] = True
            return z["rgb"].astype("float32"), prov
        except Exception:                            # noqa: BLE001 - refetch
            pass
    if cache_only:
        # Used on request-serving paths: a cache miss must not turn into
        # minutes of granule mosaicking while an HTTP client waits.
        raise SourceUnavailable("Sentinel-2 mosaic not cached for this area")

    # A cloud-sorted page of 40 can easily be 40 revisits of the SAME granule,
    # leaving the rest of the bbox uncovered, so ask for a deep page.
    items = pc_search("sentinel-2-l2a", bbox_ll,
                      f"{start}T00:00:00Z/{end}T23:59:59Z", limit=200,
                      query={"eo:cloud_cover": {"lt": max_cloud}})
    if not items:
        items = pc_search("sentinel-2-l2a", bbox_ll,
                          f"{start}T00:00:00Z/{end}T23:59:59Z", limit=200)
    if not items:
        raise SourceUnavailable(
            f"No Sentinel-2 L2A scene over {bbox_ll} in {start}..{end}")

    # A Sentinel-2 granule is ~110 km square and a study bbox routinely straddles
    # several, so one scene covers only a fraction of the grid. Mosaic the
    # least-cloudy scenes, each filling only the gaps its predecessors left.
    items.sort(key=lambda it: it["properties"].get("eo:cloud_cover", 100.0))

    # Mosaic onto a COARSE grid, then upsample. This is a basemap texture, not
    # an analysis layer: reading 10 m bands at full model resolution over a
    # basin-sized bbox is minutes of windowed COG traffic per granule, which is
    # far too slow to sit in a demo path. A few hundred pixels is plenty once
    # it is draped on terrain, and the DEM still carries the geometry.
    dec = max(1, int(math.ceil(max(dem.ny, dem.nx) / max_px)))
    gh, gw = max(dem.ny // dec, 1), max(dem.nx // dec, 1)
    coarse = Grid(crs=dem.crs,
                  transform=from_origin(dem.transform.c, dem.transform.f,
                                        dem.dx * dec, dem.dy * dec),
                  width=gw, height=gh, res=dem.dx * dec)
    grid = coarse

    t_start = time.time()
    rgb = np.full((3, gh, gw), np.nan, dtype="float32")
    used, tiles = [], set()
    for item in items:
        if np.isfinite(rgb[0]).mean() > 0.995:
            break
        if time.time() - t_start > budget_s:          # never hang the demo
            break
        tile = item["properties"].get("s2:mgrs_tile") or item["id"][38:44]
        if tile in tiles:
            continue                                 # best scene per tile only
        try:
            part = np.stack([
                _warp_into(pc_sign(item["assets"][b]["href"], "sentinel-2-l2a"),
                           grid, resampling=Resampling.bilinear, bbox_ll=bbox_ll)
                for b in ("B04", "B03", "B02")])
        except Exception:                            # noqa: BLE001 - skip bad granule
            continue
        gap = ~np.isfinite(rgb[0]) & np.isfinite(part[0])
        if not gap.any():
            continue
        for k in range(3):
            rgb[k][gap] = part[k][gap]
        tiles.add(tile)
        used.append({"id": item["id"], "tile": tile,
                     "datetime": item["properties"].get("datetime"),
                     "cloud_cover_pct": round(
                         float(item["properties"].get("eo:cloud_cover", -1)), 2)})
        if len(used) >= 12:
            break

    # back up to the model grid (nearest: it is a texture, not a measurement)
    if dec > 1:
        yi = np.minimum((np.arange(dem.ny) // dec), gh - 1)
        xi = np.minimum((np.arange(dem.nx) // dec), gw - 1)
        rgb = rgb[:, yi][:, :, xi]

    coverage = float(np.isfinite(rgb[0]).mean())
    if coverage < 0.05:
        raise SourceUnavailable(
            f"Sentinel-2 mosaic covered only {coverage:.1%} of the domain")

    # L2A is scaled surface reflectance (10000 = 1.0). Stretch on VALID pixels
    # only -- including the nodata gaps would drag the low percentile to zero
    # and crush the whole image dark.
    out = np.zeros_like(rgb)
    for k in range(3):
        b = rgb[k]
        good = np.isfinite(b) & (b > 0)
        if not good.any():
            continue
        lo, hi = np.percentile(b[good], [2, 96])
        out[k] = np.clip((b - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0)
    out = np.power(out, 0.85)                        # mild gamma lift

    prov = {
        "collection": "sentinel-2-l2a via Microsoft Planetary Computer",
        "scenes": used,
        "granules_used": len(used),
        "grid_coverage_pct": round(100.0 * coverage, 1),
        "read_decimation": dec,
        "seconds": round(time.time() - t_start, 1),
        "bands": "B04/B03/B02 true colour, 2-96% per-band stretch, gamma 0.85",
        "licence": "Copernicus Sentinel data, free and open",
    }
    try:
        np.savez_compressed(cpath, rgb=out.astype("float32"),
                            prov=json.dumps(prov))
    except Exception:                                # noqa: BLE001 - cache is optional
        pass
    return out, prov


def attach_roughness(dem: DEM, bbox_ll: Sequence[float]) -> DEM:
    """Roughness-map node: land cover -> spatially varying Manning n."""
    lc, src = fetch_landcover(bbox_ll, dem)
    dem.landcover = lc
    dem.manning = roughness_from_worldcover(lc)
    dem.meta = dict(dem.meta)
    dem.meta["landcover_source"] = src
    return dem


# ---------------------------------------------------------------------------
# 3. Population -- WorldPop
# ---------------------------------------------------------------------------

def fetch_population(bbox_ll: Sequence[float], dem: DEM, iso: str = "IND",
                     product: str = "1km") -> Tuple[np.ndarray, str]:
    """WorldPop 2020 population counts resampled to the model grid.

    `1km` (19 MB for India) is cached whole; `100m` is the 531 MB constrained
    product and is only fetched on request.  Counts are rescaled by the cell-area
    ratio so the total population in the domain is preserved.
    """
    # The 100 m constrained product is the right resolution to intersect with a
    # 60-180 m model grid, but it is a ~500 MB national raster. Making that a
    # blocking first-run download is a bad default for a tool people demo, so
    # "auto" uses 100 m only when it is already cached and otherwise runs on
    # 1 km immediately, saying so in the provenance string. Ask for "100m"
    # explicitly to accept the download.
    if product == "auto":
        cached = CACHE_DIR / Path(
            WORLDPOP_100M.format(iso=iso, iso_l=iso.lower())).name
        product = "100m" if (cached.exists() and cached.stat().st_size > 0) \
            else "1km"

    # Fall back rather than fail the whole run, and say which product actually
    # got used in the returned provenance string.
    order = (["100m", "1km"] if product == "100m" else ["1km"])
    errors = []
    for attempt in order:
        tmpl = WORLDPOP_1KM if attempt == "1km" else WORLDPOP_100M
        url = tmpl.format(iso=iso, iso_l=iso.lower())
        local = CACHE_DIR / Path(url).name
        try:
            download_file(url, local)
            product = attempt
            break
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"{attempt}: {exc}")
    else:
        raise SourceUnavailable(
            f"No WorldPop product available for {iso}. Tried: "
            + "; ".join(errors))

    grid = Grid(crs=dem.crs, transform=dem.transform, width=dem.nx,
                height=dem.ny, res=dem.dx)
    with rasterio.open(local) as src:
        src_cell_deg = abs(src.transform.a)
        lat = (bbox_ll[1] + bbox_ll[3]) / 2
        src_cell_m2 = (src_cell_deg * 111_320.0) * \
                      (src_cell_deg * 111_320.0 * math.cos(math.radians(lat)))
    dens = _warp_into(str(local), grid, resampling=Resampling.bilinear,
                      bbox_ll=bbox_ll)
    dens = np.nan_to_num(dens, nan=0.0)
    dens[dens < 0] = 0.0
    # counts per source cell -> counts per model cell
    pop = dens * (dem.cell_area / src_cell_m2)
    return pop.astype(np.float64), f"WorldPop 2020 {product} ({iso}), {url}"


# ---------------------------------------------------------------------------
# 4. OpenStreetMap -- dam, exposure, channel network
# ---------------------------------------------------------------------------

def overpass(query: str, cache_tag: str = "osm", timeout: int = 180) -> dict:
    cache = _cache_path(cache_tag, query, ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for ep in OVERPASS_ENDPOINTS:
        try:
            raw = http_get(ep, data=data,
                           headers={"Content-Type": "application/x-www-form-urlencoded"},
                           timeout=timeout, retries=2)
            j = json.loads(raw)
            if "elements" in j:
                cache.write_text(json.dumps(j))
                return j
            last = f"{ep}: no elements key"
        except Exception as exc:                      # noqa: BLE001
            last = f"{ep}: {exc}"
    raise SourceUnavailable(f"All Overpass endpoints failed. Last: {last}")


def _bbox_str(bbox_ll: Sequence[float]) -> str:
    w, s, e, n = bbox_ll
    return f"{s},{w},{n},{e}"


def fetch_dams(bbox_ll: Sequence[float]) -> List[dict]:
    """All mapped dams / weirs / reservoirs in the bbox, with OSM + Wikidata ids."""
    bb = _bbox_str(bbox_ll)
    q = f"""[out:json][timeout:120];
(
  way["waterway"="dam"]({bb});
  relation["waterway"="dam"]({bb});
  way["man_made"="dyke"]({bb});
);
out tags geom;"""
    j = overpass(q, "osm_dams")
    dams = []
    for el in j["elements"]:
        tags = el.get("tags", {})
        geom = el.get("geometry") or []
        if not geom:
            continue
        lons = [p["lon"] for p in geom]
        lats = [p["lat"] for p in geom]
        dams.append({
            "osm_type": el["type"], "osm_id": el["id"],
            "name": tags.get("name") or tags.get("name:en") or "unnamed",
            "tags": tags,
            "wikidata": tags.get("wikidata"),
            "geometry_ll": [[p["lon"], p["lat"]] for p in geom],
            "center_ll": [float(np.mean(lons)), float(np.mean(lats))],
            "crest_length_m": _polyline_length_m(lons, lats),
        })
    dams.sort(key=lambda d: -d["crest_length_m"])
    return dams


def _polyline_length_m(lons, lats) -> float:
    if len(lons) < 2:
        return 0.0
    lat0 = math.radians(float(np.mean(lats)))
    x = np.asarray(lons) * 111_320.0 * math.cos(lat0)
    y = np.asarray(lats) * 110_540.0
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def fetch_waterways(bbox_ll: Sequence[float]) -> List[dict]:
    """River/stream centrelines -- drives channel burning and the long profile."""
    bb = _bbox_str(bbox_ll)
    q = f"""[out:json][timeout:180];
(
  way["waterway"~"^(river|stream)$"]({bb});
);
out geom;"""
    j = overpass(q, "osm_waterways")
    return [{"osm_id": el["id"], "name": el.get("tags", {}).get("name", ""),
             "waterway": el.get("tags", {}).get("waterway"),
             "coords_ll": [[p["lon"], p["lat"]] for p in el.get("geometry", [])]}
            for el in j["elements"] if el.get("geometry")]


def fetch_building_footprints(bbox_ll: Sequence[float]) -> Dict[int, float]:
    """Building footprint areas in m2, keyed by OSM way id.

    The JRC maximum-damage values are per square metre of floor area, so a
    building count alone cannot be converted to currency.  This is a separate
    Overpass call because it needs `out geom` -- pulling full geometry for
    every element in the exposure query would multiply that response for no
    benefit to the settlement/facility/road layers.
    """
    # `out geom` on every building in a large, densely-mapped domain is the
    # heaviest request this framework makes -- hundreds of MB over parts of
    # Kerala or the Gangetic plain. Refuse rather than stall: the caller treats
    # this as optional and falls back to counts without floor area.
    west, south, east, north = bbox_ll
    area_km2 = (abs(east - west) * 111.0 * math.cos(math.radians((south + north) / 2))
                * abs(north - south) * 111.0)
    if area_km2 > 12_000:
        raise SourceUnavailable(
            f"Domain is {area_km2:,.0f} km2; a full building-footprint query "
            f"over an area this size can return hundreds of MB from Overpass. "
            f"Narrow the bbox to get floor-area-based losses.")

    bb = _bbox_str(bbox_ll)
    q = f"""[out:json][timeout:240];
way["building"]({bb});
out geom;"""
    j = overpass(q, "osm_building_geom")
    out: Dict[int, float] = {}
    for el in j["elements"]:
        g = el.get("geometry")
        if not g or len(g) < 4:
            continue
        area = _ring_area_m2([(p["lon"], p["lat"]) for p in g])
        if area > 0:
            out[int(el["id"])] = area
    return out


def _ring_area_m2(coords: Sequence[Tuple[float, float]]) -> float:
    """Planar shoelace area of a small lon/lat ring, metres squared.

    Over a building footprint the local scale factor is effectively constant,
    so a local equirectangular projection about the ring centroid is accurate
    to far better than the precision of the OSM geometry itself.
    """
    if len(coords) < 4:
        return 0.0
    lat0 = math.radians(sum(c[1] for c in coords) / len(coords))
    kx = 111320.0 * math.cos(lat0)
    ky = 110540.0
    xs = [c[0] * kx for c in coords]
    ys = [c[1] * ky for c in coords]
    s = 0.0
    for i in range(len(coords) - 1):
        s += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return abs(s) * 0.5


def fetch_roads_geom(bbox_ll: Sequence[float]) -> List[dict]:
    """Lifeline road and rail centrelines WITH geometry, for length statistics."""
    bb = _bbox_str(bbox_ll)
    q = f"""[out:json][timeout:180];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bb});
  way["railway"="rail"]({bb});
);
out geom;"""
    j = overpass(q, "osm_roads_geom")
    out = []
    for el in j["elements"]:
        g = el.get("geometry")
        if not g:
            continue
        t = el.get("tags", {})
        out.append({"osm_id": el["id"], "name": t.get("name", ""),
                    "highway": t.get("highway"), "railway": t.get("railway"),
                    "coords_ll": [[p["lon"], p["lat"]] for p in g]})
    return out


EXPOSURE_QUERY = """[out:json][timeout:240];
(
  node["place"~"^(city|town|village|hamlet|suburb)$"]({bb});
  way["building"]({bb});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bb});
  way["railway"="rail"]({bb});
  node["amenity"~"^(hospital|clinic|school|college|police|fire_station)$"]({bb});
  way["amenity"~"^(hospital|clinic|school|college)$"]({bb});
  node["man_made"="bridge"]({bb});
  way["man_made"="bridge"]({bb});
  node["power"="substation"]({bb});
  way["power"="substation"]({bb});
);
out tags center;"""


def fetch_exposure_osm(bbox_ll: Sequence[float]) -> Dict[str, list]:
    """Settlements, buildings, lifeline roads/rail and critical facilities."""
    j = overpass(EXPOSURE_QUERY.replace("{bb}", _bbox_str(bbox_ll)), "osm_exposure")
    out: Dict[str, list] = {"settlements": [], "buildings": [], "roads": [],
                            "railways": [], "facilities": [], "bridges": [],
                            "power": []}
    for el in j["elements"]:
        t = el.get("tags", {})
        c = el.get("center") or ({"lat": el.get("lat"), "lon": el.get("lon")}
                                 if el.get("lat") is not None else None)
        if not c or c.get("lat") is None:
            continue
        rec = {"osm_id": el["id"], "name": t.get("name", ""),
               "lon": c["lon"], "lat": c["lat"], "tags": t}
        if t.get("place"):
            rec["place"] = t["place"]
            rec["population"] = _safe_int(t.get("population"))
            out["settlements"].append(rec)
        elif t.get("amenity") in ("hospital", "clinic", "school", "college",
                                  "police", "fire_station"):
            rec["kind"] = t["amenity"]
            out["facilities"].append(rec)
        elif t.get("man_made") == "bridge":
            out["bridges"].append(rec)
        elif t.get("power") == "substation":
            out["power"].append(rec)
        elif t.get("railway") == "rail":
            out["railways"].append(rec)
        elif t.get("highway"):
            rec["highway"] = t["highway"]
            out["roads"].append(rec)
        elif t.get("building"):
            rec["building"] = t["building"]
            out["buildings"].append(rec)
    return out


def _safe_int(v) -> Optional[int]:
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. Dam attributes -- Wikidata / Wikipedia
# ---------------------------------------------------------------------------

WD_PROPS = {
    "P2048": "height_m", "P2043": "length_m", "P2234": "volume_m3",
    "P625": "coordinate", "P2661": "capacity_m3", "P571": "inception",
    "P2109": "installed_capacity_W", "P4511": "vertical_depth_m",
}


def wikidata_entity(qid: str) -> dict:
    """Structured dam attributes from Wikidata (CC0)."""
    cache = _cache_path("wikidata", qid, ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    j = http_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
    ent = j["entities"][qid]
    out = {"qid": qid,
           "label": ent.get("labels", {}).get("en", {}).get("value"),
           "description": ent.get("descriptions", {}).get("en", {}).get("value"),
           "sitelink": ent.get("sitelinks", {}).get("enwiki", {}).get("title")}
    for pid, key in WD_PROPS.items():
        claims = ent.get("claims", {}).get(pid)
        if not claims:
            continue
        dv = claims[0]["mainsnak"].get("datavalue", {}).get("value")
        if isinstance(dv, dict) and "amount" in dv:
            out[key] = float(dv["amount"])
        elif isinstance(dv, dict) and "latitude" in dv:
            out[key] = [dv["longitude"], dv["latitude"]]
        elif isinstance(dv, dict) and "time" in dv:
            out[key] = dv["time"]
        else:
            out[key] = dv
    cache.write_text(json.dumps(out))
    return out


WDQS = "https://query.wikidata.org/sparql"

# Q12323 = dam.  P17 country, P625 coordinate, P2048 height, P2234 reservoir
# volume, P4614 drainage basin, P131 administrative unit.
_DAM_INDEX_SPARQL = """
SELECT ?dam ?damLabel ?coord ?height ?cap ?riverLabel ?admLabel WHERE {
  ?dam wdt:P31/wdt:P279* wd:Q12323 .
  ?dam wdt:P17 wd:%(country)s .
  ?dam wdt:P625 ?coord .
  OPTIONAL { ?dam wdt:P2048 ?height }
  OPTIONAL { ?dam wdt:P2234 ?cap }
  OPTIONAL { ?dam wdt:P4614 ?river }
  OPTIONAL { ?dam wdt:P131 ?adm }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
"""

# Wikidata country items for the countries the damage module can price.
WD_COUNTRY = {"IND": "Q668", "CHN": "Q148", "USA": "Q30", "BRA": "Q155",
              "PAK": "Q843", "NPL": "Q837", "BGD": "Q902", "LKA": "Q854",
              "VNM": "Q881", "IDN": "Q252", "ZAF": "Q258", "GBR": "Q145"}


def fetch_dam_index(iso3: str = "IND") -> List[dict]:
    """Every dam in a country that Wikidata knows the coordinates of.

    This is what makes the framework national rather than a five-preset demo:
    the DEM, land cover, population and imagery are all fetched on demand for
    whatever bbox a dam implies, so the only thing that ever limited coverage
    was knowing where the dams are.

    Wikidata rather than a nationwide Overpass sweep: a country-wide
    `nwr["waterway"="dam"]` query is a heavy request against a shared public
    endpoint, while WDQS answers this one in seconds and carries the published
    height and storage alongside the coordinate.  Use `fetch_dams` for the
    authoritative mapped geometry once a dam has been picked.
    """
    qid = WD_COUNTRY.get(iso3.upper())
    if qid is None:
        raise SourceUnavailable(
            f"No Wikidata country item mapped for {iso3}; add one to "
            f"WD_COUNTRY. Known: {sorted(WD_COUNTRY)}")
    cache = _cache_path("damindex", iso3.upper(), ".json")
    if cache.exists():
        return json.loads(cache.read_text())

    query = _DAM_INDEX_SPARQL % {"country": qid}
    url = f"{WDQS}?{urllib.parse.urlencode({'query': query})}"
    raw = http_get(url, headers={"Accept": "application/sparql-results+json"},
                   timeout=180)
    j = json.loads(raw)

    # The tallest dam ever built is Jinping-I at 305 m. Anything above that in
    # P2048 is a mis-tagged crest ELEVATION, which is a common Wikidata error
    # for Indian dams -- carry the value through but mark it, so a screening
    # domain is never sized from a number that cannot be a dam height.
    tallest_built_m = 305.0

    by_qid: Dict[str, dict] = {}
    for r in j["results"]["bindings"]:
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r["coord"]["value"])
        if not m:
            continue
        lon, lat = float(m.group(1)), float(m.group(2))
        name = r["damLabel"]["value"]
        # An unresolved label comes back as the bare Q-id; those carry no
        # usable dam name, so they cannot be matched against OSM later.
        if re.fullmatch(r"Q\d+", name):
            continue
        qid = r["dam"]["value"].rsplit("/", 1)[-1]
        h = _safe_float(r.get("height", {}).get("value"))
        suspect = bool(h and h > tallest_built_m)
        rec = {
            "name": name, "qid": qid, "lon": lon, "lat": lat,
            "height_m": None if suspect else h,
            "height_suspect_m": h if suspect else None,
            "capacity_m3": _safe_float(r.get("cap", {}).get("value")),
            "river": r.get("riverLabel", {}).get("value"),
            "admin": r.get("admLabel", {}).get("value"),
        }
        # P131 is multi-valued, so WDQS returns one row per administrative
        # unit. Keep the first and do not let a dam appear several times.
        prev = by_qid.get(qid)
        if prev is None:
            by_qid[qid] = rec
        elif prev.get("river") is None and rec.get("river"):
            by_qid[qid] = rec

    out = list(by_qid.values())
    out.sort(key=lambda d: (-(d["height_m"] or 0), d["name"]))
    cache.write_text(json.dumps(out))
    return out


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def dams_near(lat: float, lon: float, radius_deg: float = 0.25,
              iso3: str = "IND") -> List[dict]:
    """Dams within a box around a point, nearest first.

    Backs "click anywhere on the map and model that dam" -- the national index
    is consulted first because it is cached and carries published attributes,
    and OSM is queried as well so a dam that is mapped but not in Wikidata is
    still offered.
    """
    out: List[dict] = []
    try:
        for d in fetch_dam_index(iso3):
            if (abs(d["lat"] - lat) <= radius_deg
                    and abs(d["lon"] - lon) <= radius_deg):
                out.append(dict(d, source="Wikidata"))
    except SourceUnavailable:
        pass

    bbox = (lon - radius_deg, lat - radius_deg,
            lon + radius_deg, lat + radius_deg)
    try:
        known = {d["name"].lower() for d in out if d.get("name")}
        for d in fetch_dams(bbox):
            nm = (d.get("name") or "").strip()
            if not nm or nm.lower() in known:
                continue
            # fetch_dams returns the crest centroid as center_ll, not lon/lat.
            centre = d.get("center_ll") or []
            if len(centre) != 2:
                continue
            dlon, dlat = centre
            out.append({"name": nm, "qid": d.get("wikidata"),
                        "lon": float(dlon), "lat": float(dlat),
                        "height_m": None, "capacity_m3": None,
                        "river": None, "admin": None, "source": "OpenStreetMap"})
    except Exception:                                  # noqa: BLE001
        pass

    out.sort(key=lambda d: (d["lat"] - lat) ** 2 + (d["lon"] - lon) ** 2)
    return out


def auto_bbox(lat: float, lon: float, height_m: Optional[float] = None,
              ) -> Tuple[Tuple[float, float, float, float], str]:
    """A screening domain around a dam, sized from its height.

    SCREENING HEURISTIC, not a derived quantity.  Taller dams release more head
    and flood further, so the half-extent scales with dam height, calibrated
    against the hand-picked preset domains:

        Koteshwar   ~60 m    half 0.235 deg
        Bhakra      226 m    half 0.250 deg
        Srisailam   145 m    half 0.250 deg
        Tehri       260 m    half 0.480 deg

    The slope and clamps below reproduce that range.  Do not widen them
    casually: domain area is the single biggest driver of run cost, and not
    because of the solver.  Every exposure layer is an Overpass query over the
    whole box, and OSM density varies enormously -- a 1.2 degree box over
    Kerala returns a 76 MB exposure response and effectively stalls the run,
    while the same box over the Himalaya returns a few MB.

    For a reportable run, set an explicit bbox: a domain that clips the flood
    shows up in the QC gate as wetted cells touching the boundary.
    """
    h = height_m if (height_m and height_m > 0) else 60.0
    half = min(max(h * 0.0015, 0.12), 0.35)
    bbox = (round(lon - half, 4), round(lat - half, 4),
            round(lon + half, 4), round(lat + half, 4))
    area_km2 = (2 * half * 111.0) ** 2 * math.cos(math.radians(lat))
    note = (f"Screening domain: half-extent {half:.3f} deg (~{area_km2:,.0f} "
            f"km2) scaled from a dam height of {h:.0f} m, clamped to "
            f"0.12-0.35 deg to match the calibrated preset domains. "
            f"HEURISTIC - set an explicit bbox for a reportable run, and "
            f"check the domain-edge QC flag for a clipped flood.")
    return bbox, note


def wikipedia_infobox(title: str) -> dict:
    """Published engineering figures (height, gross storage) for cross-check.

    Returned values are labelled `published_*` and are never fed into the solver
    -- the solver uses DEM-derived hypsometry.  They exist so the QC panel can
    show model-vs-record agreement.
    """
    cache = _cache_path("wikipedia", title, ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=revisions"
           "&rvprop=content&rvslots=main&format=json&titles="
           + urllib.parse.quote(title))
    j = http_json(url)
    pages = j["query"]["pages"]
    text = ""
    for p in pages.values():
        try:
            text = p["revisions"][0]["slots"]["main"]["*"]
        except Exception:
            pass
    out = {"title": title, "fields": {}}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and "=" in line:
            k, _, v = line[1:].partition("=")
            k, v = k.strip().lower(), v.strip()
            if k.startswith(("dam_height", "dam_length", "res_capacity",
                             "dam_crosses", "res_surface", "dam_type",
                             "plant_capacity", "res_catchment")) and v:
                out["fields"][k] = v[:200]
    out["source"] = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    cache.write_text(json.dumps(out))
    return out


def dam_dossier(dam: dict) -> dict:
    """Merge OSM tags + Wikidata + Wikipedia into one provenance-tagged record."""
    d = {"osm": {"id": dam["osm_id"], "name": dam["name"],
                 "crest_length_m": round(dam["crest_length_m"], 1),
                 "tags": dam["tags"]},
         "center_ll": dam["center_ll"], "sources": ["OpenStreetMap (ODbL)"]}
    qid = dam.get("wikidata")
    if qid:
        try:
            wd = wikidata_entity(qid)
            d["wikidata"] = wd
            d["sources"].append(f"Wikidata {qid} (CC0)")
            if wd.get("sitelink"):
                d["wikipedia"] = wikipedia_infobox(wd["sitelink"])
                d["sources"].append(d["wikipedia"]["source"])
        except SourceUnavailable:
            pass
    return d


# ---------------------------------------------------------------------------
# 6. Administrative boundaries -- geoBoundaries
# ---------------------------------------------------------------------------

def fetch_districts(iso: str = "IND", adm: str = "ADM2") -> dict:
    """geoBoundaries gbOpen district polygons (CC-BY 4.0)."""
    cache = _cache_path("geoboundaries", f"{iso}{adm}", ".geojson")
    if cache.exists():
        return json.loads(cache.read_text())
    meta = http_json(f"https://www.geoboundaries.org/api/current/gbOpen/{iso}/{adm}/")
    gj = json.loads(http_get(meta["gjDownloadURL"], timeout=300))
    cache.write_text(json.dumps(gj))
    return gj


# ---------------------------------------------------------------------------
# 7. Observed flood extent -- Sentinel-1 GRD
# ---------------------------------------------------------------------------

# Sentinel-1 RTC, not GRD.  GRD assets are ground-range detected: they carry
# GCPs but no CRS at all, so they cannot be warped onto a projected model grid
# ("CRS is invalid: None").  RTC is radiometrically terrain corrected and
# delivered as projected UTM COGs -- which is also the right product for flood
# mapping in mountainous terrain, since it removes topographic backscatter bias.
S1_COLLECTION = "sentinel-1-rtc"


def find_sentinel1(bbox_ll: Sequence[float], start: str, end: str,
                   polarisation: str = "vv") -> List[dict]:
    items = pc_search(S1_COLLECTION, bbox_ll,
                      f"{start}T00:00:00Z/{end}T23:59:59Z", limit=20)
    return [it for it in items if polarisation in it["assets"]]


def sentinel1_backscatter(item: dict, dem: DEM, bbox_ll: Sequence[float],
                          polarisation: str = "vv") -> np.ndarray:
    """Read a Sentinel-1 RTC band onto the model grid and convert to dB.

    RTC pixels are gamma0 in linear power, so dB = 10*log10(gamma0) directly
    (no amplitude squaring, which is what a DN-valued GRD would need).
    """
    href = pc_sign(item["assets"][polarisation]["href"], S1_COLLECTION)
    grid = Grid(crs=dem.crs, transform=dem.transform, width=dem.nx,
                height=dem.ny, res=dem.dx)
    g0 = _warp_into(href, grid, resampling=Resampling.bilinear, bbox_ll=bbox_ll)
    g0 = np.where(np.isfinite(g0) & (g0 > 0), g0, np.nan)
    return 10.0 * np.log10(g0.astype(np.float64) + 1e-9)


def sar_water_mask(db: np.ndarray, threshold_db: Optional[float] = None,
                   slope_deg: Optional[np.ndarray] = None,
                   slope_limit: float = 12.0) -> Tuple[np.ndarray, float]:
    """Threshold SAR backscatter into open water.

    Uses Otsu's method on the valid-pixel histogram (the standard unsupervised
    approach for SAR flood mapping) and masks steep terrain, where radar shadow
    mimics the low-backscatter signature of water.
    """
    valid = np.isfinite(db)
    if threshold_db is None:
        v = db[valid]
        v = v[(v > -35) & (v < 10)]
        if v.size < 100:
            raise SourceUnavailable("Not enough valid SAR pixels to threshold")
        hist, edges = np.histogram(v, bins=256)
        p = hist.astype(np.float64) / hist.sum()
        omega = np.cumsum(p)
        mu = np.cumsum(p * ((edges[:-1] + edges[1:]) / 2))
        mu_t = mu[-1]
        denom = omega * (1 - omega)
        denom[denom == 0] = 1e-12
        sigma_b = (mu_t * omega - mu) ** 2 / denom
        threshold_db = float((edges[:-1] + edges[1:])[np.argmax(sigma_b)] / 2)
    water = valid & (db < threshold_db)
    if slope_deg is not None:
        water &= slope_deg < slope_limit
    return water, float(threshold_db)
