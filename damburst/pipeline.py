"""End-to-end dam-break simulation pipeline.

Walks the whole technical approach for one scenario and writes a self-describing
run directory:

    runs/<run_id>/
        manifest.json          provenance, QC gates, every parameter used
        results.json           hydrographs, hazard summary, impact, comparison
        rasters/*.tif          depth, velocity, hazard, arrival, duration
        vectors/*.shp|.kmz|.geojson
        overlays/*.png         WGS84 image overlays for the dashboard
        frames/                time-series depth/speed frames for animation
        tables/*.csv           evacuation timeline, districts, settlements
"""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import breach as B
from .core import coupling as C
from .core import damage as DMG
from .core import datasources as ds
from .core import dem as D
from .core import exposure as EX
from .core import export as OUT
from .core import gee as GEE
from .core import hazard as HZ
from .core import reservoir as RES
from .core import sph as SPH
from .core import swe2d as SW
from .core import validation as VAL

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    bbox_ll: Tuple[float, float, float, float]
    dam_name: str                        # substring matched against OSM names
    barrier_type: str = "engineered"     # engineered | natural
    failure_mode: str = "overtopping"    # overtopping|piping|progressive_erosion|instantaneous
    growth_law: str = "sine"             # linear|sine|erosion
    seed_method: str = "auto"
    loading: str = "FRL"                 # FRL (full) | fraction of depth, e.g. "0.8"
    res_m: float = 90.0
    sim_hours: float = 4.0
    n_frames: int = 48
    breach_hours: float = 4.0
    inflow_m3s: float = 0.0
    manning_scale: float = 1.0
    channel_burn_m: float = 0.0
    run_sph: bool = True
    sph_dp: float = 3.0
    sph_seconds: float = 30.0
    sentinel1_window: Optional[Tuple[str, str]] = None
    validation_mode: str = "context"
    satellite_basemap: bool = True
    # WorldPop 100 m constrained, not 1 km aggregated: a 1 km cell assigned
    # uniformly is coarser than the model grid, so a flood edge slices a whole
    # square kilometre of population proportionally. `fetch_population` falls
    # back to the 1 km product, and records which one it used, if the
    # constrained raster is unavailable for the country.
    population_product: str = "100m"
    iso3: str = "IND"
    # Set when the domain was derived from a dam coordinate rather than
    # supplied, so the manifest records that it is a screening box.
    domain_note: Optional[str] = None
    # Catalogue coordinate of the dam, when it came from the national
    # index. Used to resolve it in OSM when the two spell the name
    # differently, which they often do.
    dam_lonlat: Optional[Tuple[float, float]] = None
    asset_values: HZ.AssetValues = field(default_factory=HZ.AssetValues)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_ll"] = list(self.bbox_ll)
        d["asset_values"] = self.asset_values.to_dict()
        return d


def _log(cb, stage: str, msg: str, pct: float = None, **extra):
    payload = {"stage": stage, "message": msg, "pct": pct, **extra}
    if cb:
        cb(payload)
    else:
        print(f"[{stage:14s}] {msg}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_scenario(scn: Scenario, run_id: Optional[str] = None,
                 progress: Optional[Callable[[dict], None]] = None,
                 out_root: Optional[Path] = None) -> dict:
    t_start = time.time()
    run_id = run_id or f"{scn.name}-{int(time.time())}"
    out = Path(out_root or RUNS) / run_id
    for sub in ("rasters", "vectors", "overlays", "frames", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario": scn.to_dict(),
        "data_sources": {},
        "qc": {},
        "timings": {},
    }
    results: Dict[str, object] = {"run_id": run_id}

    # -- 1. Input data ----------------------------------------------------
    _log(progress, "ingest", "Fetching Copernicus GLO-30 DEM", 2)
    t0 = time.time()
    dem = ds.fetch_dem(scn.bbox_ll, res_m=scn.res_m, name=scn.name)
    manifest["data_sources"]["elevation"] = dem.source
    manifest["timings"]["dem_s"] = round(time.time() - t0, 1)

    _log(progress, "ingest", "Fetching ESA WorldCover -> Manning roughness", 8)
    t0 = time.time()
    dem = ds.attach_roughness(dem, scn.bbox_ll)
    if scn.manning_scale != 1.0:
        dem.manning = dem.manning * scn.manning_scale
    manifest["data_sources"]["landcover"] = dem.meta.get("landcover_source")
    manifest["timings"]["landcover_s"] = round(time.time() - t0, 1)

    _log(progress, "ingest", "Locating dam in OpenStreetMap", 12)
    dams = ds.fetch_dams(scn.bbox_ll)
    dam, how = _resolve_dam(dams, scn.dam_name, scn.dam_lonlat)
    if dam is None:
        raise ds.SourceUnavailable(
            f"No OSM dam matching {scn.dam_name!r} inside {scn.bbox_ll}. "
            f"Found: {sorted({d['name'] for d in dams})}")
    manifest["data_sources"]["dam_match"] = how
    dossier = ds.dam_dossier(dam)
    manifest["dam"] = dossier
    manifest["data_sources"]["dam"] = dossier["sources"]

    _log(progress, "ingest", "Fetching OSM exposure + WorldPop", 16)
    exposure = ds.fetch_exposure_osm(scn.bbox_ll)
    roads_geom = ds.fetch_roads_geom(scn.bbox_ll)
    try:
        footprints = ds.fetch_building_footprints(scn.bbox_ll)
        manifest["data_sources"]["building_footprints"] = (
            f"OpenStreetMap building polygons via Overpass (ODbL), "
            f"{len(footprints)} footprints with computed area")
    except Exception as exc:                           # noqa: BLE001
        footprints = {}
        manifest["data_sources"]["building_footprints"] = f"unavailable: {exc}"
        _log(progress, "ingest", f"Building footprints unavailable: {exc}", 16)
    pop, pop_src = ds.fetch_population(scn.bbox_ll, dem, iso=scn.iso3,
                                       product=scn.population_product)
    manifest["data_sources"]["population"] = pop_src
    manifest["data_sources"]["exposure"] = "OpenStreetMap via Overpass API (ODbL)"

    sat_rgb = None
    if scn.satellite_basemap:
        _log(progress, "ingest", "Fetching Sentinel-2 true-colour imagery", 19)
        try:
            sat_rgb, sat_prov = ds.fetch_sentinel2_rgb(scn.bbox_ll, dem)
            manifest["data_sources"]["satellite_imagery"] = sat_prov
        except Exception as exc:                       # noqa: BLE001
            manifest["data_sources"]["satellite_imagery"] = f"unavailable: {exc}"
            _log(progress, "ingest", f"Sentinel-2 unavailable: {exc}", 19)

    # -- 2. QC + conditioning --------------------------------------------
    _log(progress, "condition", "DEM QC and conditioning", 22)
    qc_dem = D.qc_report(dem)
    manifest["qc"]["dem"] = qc_dem

    dam_rc, dam_line = _dam_cells(dem, dam)
    dem_c = D.condition(dem, channel_mask=None, burn=scn.channel_burn_m,
                        fill_sinks=True)
    dem_c.manning = dem.manning
    dem_c.landcover = dem.landcover
    manifest["qc"]["conditioning"] = dem_c.meta.get("conditioning")

    # -- 3. Reservoir H-V-A ----------------------------------------------
    _log(progress, "reservoir", "Deriving H-V-A curve from the DEM", 28)
    barrier_mask = np.zeros(dem_c.shape, dtype=bool)
    barrier_mask[dam_line[:, 0], dam_line[:, 1]] = True

    crest = float(np.percentile(dem.z[dam_line[:, 0], dam_line[:, 1]], 75))
    frame = _dam_frame(dem_c, dam_line)
    reservoir = RES.hva_from_dem(dem_c.z, dem_c.cell_area, frame.seed_rc, crest,
                                 barrier_mask, frame.upstream_mask, n_levels=40)
    hva_provenance = {"method": "DEM hypsometry only (no existing reservoir "
                                "detected in the DSM)"}

    # If the DSM already contains a filled reservoir, its bathymetry is hidden
    # beneath the water plate.  Reconstruct it from the published record.
    water_mask = (dem_c.landcover == 80) if dem_c.landcover is not None else None
    ws = (RES.detect_water_surface(dem_c.z, water_mask, frame.upstream_mask)
          if water_mask is not None else None)
    pub_cap, pub_h = _published_reservoir(dossier)
    if ws and pub_cap and pub_h:
        area_ws = ws["cells"] * dem_c.cell_area
        reservoir, hva_provenance = RES.hybrid_hva(
            reservoir, ws["elevation_m"], area_ws, pub_cap * 1e6, pub_h, crest,
            riverbed_m=float(dem_c.z[frame.tailwater_rc]))
        _log(progress, "reservoir",
             f"DSM water plate at {ws['elevation_m']:.1f} m; bathymetry "
             f"reconstructed against published {pub_cap:.0f} MCM", 30)
    elif ws:
        hva_provenance = {
            "method": "DEM hypsometry only",
            "warning": ("An existing reservoir was detected in the DSM at "
                        f"{ws['elevation_m']:.1f} m but no published capacity "
                        "was available, so storage below the water surface is "
                        "INVISIBLE and this run understates the release."),
            "dsm_water_surface_m": round(ws["elevation_m"], 2),
        }

    h_init = crest if scn.loading == "FRL" else \
        reservoir.bed + float(scn.loading) * (crest - reservoir.bed)
    res_summary = reservoir.summary()
    res_summary["initial_level_m"] = round(h_init, 2)
    res_summary["initial_storage_mcm"] = round(reservoir.volume(h_init) / 1e6, 3)
    res_summary["derivation"] = hva_provenance
    results["reservoir"] = res_summary
    manifest["qc"]["reservoir"] = _reservoir_cross_check(res_summary, dossier)

    # -- 4. Breach + hydrograph ------------------------------------------
    _log(progress, "breach", "Breach growth and outflow hydrograph", 34)
    geom = B.seed_breach(scn.barrier_type, scn.failure_mode, reservoir,
                         h_init, crest, method=scn.seed_method)
    inflow = (RES.inflow_hydrograph("constant", base=scn.inflow_m3s)
              if scn.inflow_m3s else None)
    br = B.simulate_breach(reservoir, geom, h_init,
                           failure_mode=scn.failure_mode,
                           growth_law=scn.growth_law, inflow=inflow,
                           t_end=scn.breach_hours * 3600.0)
    results["breach"] = br.to_dict()
    manifest["qc"]["breach"] = br.checks
    # The empirical spread across every applicable regression. For a structure
    # outside any single regression's fitted range this band, not a point
    # value, is the defensible statement of breach-parameter uncertainty.
    results["breach"]["regression_ensemble"] = B.regression_ensemble(
        volume_m3=reservoir.volume(h_init), h_dam=crest - reservoir.bed,
        h_water=h_init - geom.invert, h_breach=geom.height,
        mode=scn.failure_mode,
        dam_type="core" if scn.barrier_type == "engineered" else "homogeneous",
        erodibility="medium" if scn.barrier_type == "engineered" else "high")

    OUT.export_hydrograph_csv(out / "tables" / "breach_hydrograph.csv", br.t,
                              {"Q_m3s": br.q, "reservoir_level_m": br.h_res,
                               "breach_top_width_m": br.b_top,
                               "breach_invert_m": br.invert,
                               "storage_m3": br.volume})

    # -- 5. SPH near field -----------------------------------------------
    sph_res = None
    iface = None
    if scn.run_sph:
        _log(progress, "sph", "SPH near-field breach jet", 42)
        t0 = time.time()
        try:
            cfg = SPH.SPHConfig(dp=scn.sph_dp, t_end=scn.sph_seconds)
            pool = reservoir.mask if reservoir.mask is not None else \
                np.zeros(dem_c.shape, bool)
            up_dir = frame.upstream_sign * frame.perp
            dn_start = frame.tailwater_rc
            sph_res, iface = C.run_near_field(
                dem_c, frame.centre_rc, pool, h_init, br,
                upstream_dir=up_dir, along_dir=frame.along,
                start_rc=dn_start, cfg=cfg,
                reservoir_base_m=reservoir.bed,
                water_surface_m=(ws or {}).get("elevation_m"))
            results["sph"] = sph_res.stats
            results["transfer_interface"] = iface.to_dict()
            results["sph_vs_weir"] = C.compare_hydrographs(
                br, iface, head_m=h_init - geom.invert)
            manifest["qc"]["sph_balance"] = results["sph_vs_weir"]
            np.savez_compressed(out / "frames" / "sph_snapshots.npz",
                                meta=json.dumps(sph_res.stats))
            (out / "tables" / "sph_snapshots.json").write_text(
                json.dumps(sph_res.snapshots))
            OUT.export_hydrograph_csv(
                out / "tables" / "sph_transfer_section.csv", sph_res.t,
                {"depth_m": sph_res.depth, "u_mean_ms": sph_res.u_mean,
                 "q_unit_m2s": sph_res.q_unit, "front_x_m": sph_res.front_x})
        except Exception as exc:                       # noqa: BLE001
            results["sph_error"] = f"{type(exc).__name__}: {exc}"
            manifest["qc"]["sph_balance"] = {"pass": False,
                                             "error": results["sph_error"]}
            _log(progress, "sph", f"SPH failed: {exc}", 42)
        manifest["timings"]["sph_s"] = round(time.time() - t0, 1)

    # -- 6. Far-field 2D runs --------------------------------------------
    breach_cells = _breach_cells(dem_c, frame, geom)
    flow_dir = frame.downstream_dir

    model_runs: Dict[str, dict] = {}
    swe_products: Dict[str, SW.SWEResult] = {}

    configs = [("grid_standalone", br.hydrograph(),
                "2D shallow-water (Delft3D-FM class) driven by the empirical "
                "breach weir hydrograph")]
    if iface is not None:
        configs.append(("coupled", _blended_hydrograph(br, iface),
                        "2D shallow-water driven by the SPH transfer-section "
                        "hydrograph in the near field, weir closure thereafter"))

    for k, (key, hyd, desc) in enumerate(configs):
        _log(progress, "swe", f"Running 2D model: {key}", 50 + 20 * k)
        t0 = time.time()
        model = SW.SWE2D(dem_c.z, dem_c.dx, dem_c.dy, dem_c.manning,
                         open_edges=True, order=2)
        model.set_still_water(h_init, mask=_pool_mask(reservoir, dem_c))
        src = SW.PointSource(rows=breach_cells[:, 0], cols=breach_cells[:, 1],
                             hydrograph=hyd, name="breach")
        fdir = out / "frames" / key if key == "grid_standalone" else None

        def _p(t, tend, info, key=key):
            _log(progress, "swe", f"{key}: t={t / 60:.1f}/{tend / 60:.0f} min",
                 None, **info)

        r = model.run(t_end=scn.sim_hours * 3600.0, sources=[src],
                      source_direction=flow_dir, n_frames=scn.n_frames if fdir else 0,
                      frame_dir=fdir, progress=_p, cfl=0.40, dt_max=15.0)
        swe_products[key] = r
        model_runs[key] = {
            "description": desc,
            "peak_inflow_m3s": round(float(max(hyd(tt) for tt in
                                               np.linspace(0, scn.sim_hours * 3600, 400))), 1),
            "inundated_km2": r.stats["inundated_km2"],
            "max_depth_m": r.stats["max_depth_m"],
            "max_velocity_ms": r.stats["max_velocity_ms"],
            "volume_routed_mcm": r.stats["volume_injected_mcm"],
            "wallclock_s": r.stats["wallclock_s"],
            "resolution": f"{r.stats['cells']} cells @ {dem_c.dx:g} m",
            "solver": r.stats,
        }
        manifest["timings"][f"swe_{key}_s"] = round(time.time() - t0, 1)

    if sph_res is not None:
        model_runs["sph_nearfield"] = {
            "description": ("Weakly-compressible SPH vertical slice of the "
                            "breach jet; resolves non-hydrostatic near field"),
            "peak_inflow_m3s": results.get("transfer_interface", {}).get("peak_q_total_m3s"),
            "inundated_km2": None,
            "max_depth_m": sph_res.stats.get("peak_depth_m"),
            "max_velocity_ms": sph_res.stats.get("peak_velocity_ms"),
            "volume_routed_mcm": None,
            "wallclock_s": sph_res.stats.get("wallclock_s"),
            "resolution": f"{sph_res.stats.get('n_particles_final')} particles "
                          f"@ dp={sph_res.stats.get('dp_m')} m",
        }
    results["model_comparison"] = C.build_comparison_table(model_runs)

    primary = swe_products.get("coupled") or swe_products["grid_standalone"]
    results["primary_model"] = "coupled" if "coupled" in swe_products else "grid_standalone"

    # -- 7. Hazard --------------------------------------------------------
    _log(progress, "hazard", "Hazard rating and classification", 82)
    hz = HZ.build_hazard(primary.h_max, primary.v_max, dem_c.cell_area,
                         landcover=dem_c.landcover,
                         arrival_s=primary.arrival_s,
                         duration_s=primary.duration_s)
    results["hazard"] = hz.summary

    # -- 8. Exposure and impact -------------------------------------------
    _log(progress, "impact", "Zonal intersection with exposure layers", 88)
    wet = primary.h_max > 0.05
    settlements = EX.settlement_impact(dem_c, exposure["settlements"],
                                       primary.h_max, hz.hazard_class,
                                       primary.arrival_s, primary.v_max)
    facilities = EX.facility_impact(dem_c, exposure["facilities"],
                                    primary.h_max, hz.hazard_class,
                                    primary.arrival_s)
    buildings = EX.building_impact(dem_c, exposure["buildings"], primary.h_max,
                                   hz.hazard_class, hz.dv,
                                   footprints=footprints, iso3=scn.iso3)
    roads = EX.road_impact(dem_c, roads_geom, wet)
    lc_impact = EX.landcover_impact(dem_c.landcover, wet, dem_c.cell_area)
    pop_impact = EX.population_impact(pop, hz.hazard_class, primary.h_max)

    try:
        adm = ds.fetch_districts(scn.iso3, "ADM2")
        draster, dnames = EX.rasterize_districts(dem_c, adm)
        districts = EX.district_breakdown(dem_c, draster, dnames, pop,
                                          hz.hazard_class, primary.h_max,
                                          dem_c.cell_area)
        manifest["data_sources"]["districts"] = "geoBoundaries gbOpen ADM2 (CC-BY 4.0)"
    except Exception as exc:                           # noqa: BLE001
        districts = []
        manifest["data_sources"]["districts"] = f"unavailable: {exc}"

    evac = EX.evacuation_timeline(settlements)
    # The JRC road and cropland curves are depth dependent, so they need a
    # representative depth rather than a flat partial-damage factor.  The mean
    # depth over inundated cells is the like-for-like quantity: both layers are
    # scored over the same wetted footprint.
    mean_wet_depth = (float(np.nanmean(primary.h_max[wet]))
                      if wet.any() else None)
    losses = EX.estimate_losses(buildings, roads, lc_impact,
                                values=scn.asset_values, iso3=scn.iso3,
                                road_depth_m=mean_wet_depth)
    manifest["data_sources"]["damage_model"] = DMG.citation()

    impact = EX.ImpactReport(
        population=pop_impact, settlements=settlements, buildings=buildings,
        facilities=facilities, roads=roads, landcover=lc_impact,
        districts=districts, evacuation=evac, losses=losses,
        qc=EX.qc_checks(dem_c, primary.h_max, pop, settlements,
                        {k: len(v) for k, v in exposure.items()},
                        population_exposed=pop_impact.get("total_exposed"),
                        buildings_exposed=buildings.get("exposed")),
    )
    results["impact"] = impact.to_dict()
    manifest["qc"]["exposure"] = impact.qc

    OUT.export_csv(out / "tables" / "settlements.csv", settlements)
    OUT.export_csv(out / "tables" / "evacuation_timeline.csv", evac)
    OUT.export_csv(out / "tables" / "districts.csv", districts)
    OUT.export_csv(out / "tables" / "critical_facilities.csv", facilities)

    # -- 9. Validation against Sentinel-1 ---------------------------------
    _log(progress, "validate", "Sentinel-1 observed water extent", 92)
    try:
        results["validation"] = _validate(scn, dem_c, primary, hz,
                                          manifest).to_dict()
    except Exception as exc:                           # noqa: BLE001
        results["validation"] = {"error": f"{type(exc).__name__}: {exc}",
                                 "mode": scn.validation_mode}

    # -- 10. Exports -------------------------------------------------------
    _log(progress, "export", "Writing GeoTIFF / SHP / KMZ / GeoJSON", 95)
    rasters = out / "rasters"
    OUT.write_geotiff(rasters / "depth_max.tif", primary.h_max, dem_c,
                      description="Maximum inundation depth (m)")
    OUT.write_geotiff(rasters / "velocity_max.tif", primary.v_max, dem_c,
                      description="Maximum depth-averaged velocity (m/s)")
    OUT.write_geotiff(rasters / "hazard_rating.tif", hz.hr, dem_c,
                      description="Defra FD2321 hazard rating")
    OUT.write_geotiff(rasters / "hazard_class.tif", hz.hazard_class.astype(float),
                      dem_c, dtype="float32", description="Hazard class 1-4")
    OUT.write_geotiff(rasters / "arrival_time_s.tif", primary.arrival_s, dem_c,
                      description="Time of first arrival (s), -1 = never")
    OUT.write_geotiff(rasters / "duration_s.tif", primary.duration_s, dem_c,
                      description="Duration above arrival threshold (s)")
    OUT.write_geotiff(rasters / "dem_conditioned.tif", dem_c.z, dem_c,
                      description="Conditioned DEM (m)")

    vectors = out / "vectors"
    shp = OUT.export_flood_extent_shp(vectors / "flood_extent.shp",
                                      primary.h_max, hz.hazard_class, dem_c)
    kmz = OUT.export_kml(vectors / "flood_extent", primary.h_max,
                         hz.hazard_class, dem_c,
                         name=f"{scn.name} - {scn.failure_mode}")
    gj = OUT.export_geojson(vectors / "flood_extent.geojson", primary.h_max,
                            hz.hazard_class, dem_c)

    overlays = {}
    ov = out / "overlays"
    overlays["depth"] = OUT.raster_to_png_4326(
        primary.h_max, dem_c, ov / "depth_max.png", 0.0,
        max(float(np.percentile(primary.h_max[wet], 97)) if wet.any() else 1.0, 1.0),
        "depth", mask_below=0.05)
    overlays["velocity"] = OUT.raster_to_png_4326(
        primary.v_max, dem_c, ov / "velocity_max.png", 0.0,
        max(float(np.percentile(primary.v_max[wet], 97)) if wet.any() else 1.0, 0.5),
        "speed", mask_below=0.05)
    overlays["hazard"] = OUT.raster_to_png_4326(
        hz.hr, dem_c, ov / "hazard.png", 0.0, 4.0, "hazard", mask_below=0.05)
    arr_min = np.where(primary.arrival_s >= 0, primary.arrival_s / 60.0, np.nan)
    overlays["arrival"] = OUT.raster_to_png_4326(
        arr_min, dem_c, ov / "arrival_min.png", 0.0,
        float(np.nanpercentile(arr_min, 95)) if np.isfinite(arr_min).any() else 60.0,
        "arrival", mask_below=-1e9)

    # Shaded relief, so the dashboard has a basemap even with no internet.
    overlays["basemap"] = OUT.hillshade_png_4326(dem_c, ov / "basemap.png",
                                                 rgb=sat_rgb)

    # Terrain + time-varying water surface for the 3D viewer.
    _log(progress, "export", "Writing 3D terrain and water surface", 97)
    try:
        results["terrain3d"] = _write_terrain_3d(
            swe_products.get("grid_standalone") or primary, dem_c, primary,
            hz, out, dam_line=dam_line, rgb=sat_rgb)
    except Exception as exc:                           # noqa: BLE001
        results["terrain3d"] = {"error": f"{type(exc).__name__}: {exc}"}
        _log(progress, "export", f"3D export failed: {exc}", 97)

    frame_meta = _write_frame_overlays(swe_products.get("grid_standalone"),
                                       dem_c, out)

    results["exports"] = {
        "rasters": sorted(p.name for p in rasters.glob("*.tif")),
        "shapefile": shp.name,
        "kmz": kmz.name,
        "geojson": gj.name,
        "tables": sorted(p.name for p in (out / "tables").glob("*.csv")),
        "overlays": overlays,
        "frames": frame_meta,
    }

    # -- 11. Map metadata for the dashboard -------------------------------
    results["map"] = {
        "bbox_ll": list(scn.bbox_ll),
        # Zoom target: the wetted area, not the whole catchment-sized domain.
        "flood_bounds_ll": OUT.wet_bounds_ll(wet, dem_c),
        "dam": {"name": dam["name"], "center_ll": dam["center_ll"],
                "geometry_ll": dam["geometry_ll"]},
        "crs": dem_c.crs, "res_m": dem_c.dx,
        "grid": [dem_c.ny, dem_c.nx],
    }
    results["hydrographs"] = {
        "breach": {"t_s": br.t[::max(len(br.t) // 600, 1)].round(1).tolist(),
                   "q_m3s": br.q[::max(len(br.q) // 600, 1)].round(1).tolist(),
                   "level_m": br.h_res[::max(len(br.h_res) // 600, 1)].round(2).tolist()},
    }
    if sph_res is not None:
        st = max(len(sph_res.t) // 600, 1)
        results["hydrographs"]["sph_transfer"] = {
            "t_s": sph_res.t[::st].round(2).tolist(),
            "q_unit_m2s": sph_res.q_unit[::st].round(3).tolist(),
            "depth_m": sph_res.depth[::st].round(3).tolist(),
            "u_ms": sph_res.u_mean[::st].round(3).tolist(),
        }

    manifest["timings"]["total_s"] = round(time.time() - t_start, 1)
    # Roll the gates up explicitly. Scanning for a top-level "pass" key misses
    # the nested breach gates entirely and silently treats them as passing.
    gates = {
        "dem": manifest["qc"].get("dem", {}).get("pass"),
        "reservoir": manifest["qc"].get("reservoir", {}).get("pass"),
        "breach_mass_balance": manifest["qc"].get("breach", {})
            .get("mass_balance", {}).get("pass"),
        "breach_peak_envelope": manifest["qc"].get("breach", {})
            .get("peak_discharge_envelopes", {}).get("pass"),
        "breach_geometry": manifest["qc"].get("breach", {})
            .get("geometry", {}).get("pass"),
        "sph_balance": manifest["qc"].get("sph_balance", {}).get("balance_pass"),
        "exposure": manifest["qc"].get("exposure", {}).get("pass"),
    }
    manifest["qc"]["gates"] = gates
    manifest["qc"]["failed_gates"] = [k for k, v in gates.items() if v is False]
    manifest["qc"]["overall_pass"] = not manifest["qc"]["failed_gates"]

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Earth Engine hand-off. Written last because it reads the two JSON files
    # back; it needs no credentials, so it must never be able to fail a run.
    try:
        script = GEE.export_script(out)
        results["earthengine"] = {
            "script": str(script.relative_to(out)),
            "geojson": "earthengine/flood_extent_ee.geojson",
            "how": ("Paste the script into code.earthengine.google.com. The "
                    "flood extent is embedded, so no asset upload and no "
                    "extra credentials are needed."),
        }
        (out / "results.json").write_text(
            json.dumps(results, indent=2, default=str))
    except Exception as exc:                           # noqa: BLE001
        results["earthengine"] = {"error": f"{type(exc).__name__}: {exc}"}

    _log(progress, "done", f"Run complete in {manifest['timings']['total_s']}s", 100)

    return {"run_id": run_id, "dir": str(out), "manifest": manifest,
            "results": results}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Words that carry no identifying information in a dam name. Wikidata and OSM
# routinely disagree on them ("Idukki Dam" vs "Idukki Arch Dam") and on
# transliteration ("Cheruthoni" vs "Cheruthony"), so an exact substring match
# between the two catalogues fails far more often than it should.
_DAM_STOPWORDS = {"dam", "the", "of", "reservoir", "barrage", "weir", "bund",
                  "anicut", "arch", "project", "hydroelectric", "plant",
                  "saddle", "main", "major", "left", "right", "bank"}


def _dam_tokens(name: str) -> set:
    return {w for w in ''.join(c if c.isalnum() else ' '
                               for c in name.lower()).split()
            if w and w not in _DAM_STOPWORDS}


def _resolve_dam(dams: List[dict], name: str,
                 lonlat: Optional[Sequence[float]] = None
                 ) -> Tuple[Optional[dict], str]:
    """Find the OSM dam a scenario means, by name then by coordinate.

    Name first, because a named match is unambiguous. But a dam picked from the
    Wikidata index arrives with a Wikidata spelling, and OSM may hold another
    ("Cheruthoni" / "Cheruthony") or a longer official one. So: exact substring,
    then distinctive-token overlap, then -- when the caller knows where the dam
    is -- simply the nearest mapped dam to that coordinate, which is the most
    reliable signal of the three. Which route was used is recorded.
    """
    if not dams:
        return None, "no dams mapped in domain"

    needle = (name or "").strip().lower()
    if needle:
        exact = [d for d in dams if needle in d["name"].lower()]
        if exact:
            return exact[0], f"exact name substring {name!r}"

        want = _dam_tokens(name)
        if want:
            scored = []
            for d in dams:
                have = _dam_tokens(d["name"])
                if want & have:
                    scored.append((len(want & have), d["crest_length_m"], d))
            if scored:
                scored.sort(key=lambda s: (-s[0], -s[1]))
                best = scored[0][2]
                return best, (f"token match {sorted(want)} -> OSM "
                              f"{best['name']!r} (names differ between "
                              f"catalogues)")

    if lonlat and len(lonlat) == 2:
        lon0, lat0 = float(lonlat[0]), float(lonlat[1])
        best = min(dams, key=lambda d: (d["center_ll"][0] - lon0) ** 2
                   + (d["center_ll"][1] - lat0) ** 2)
        dx = (best["center_ll"][0] - lon0) * 111_320.0 * math.cos(
            math.radians(lat0))
        dy = (best["center_ll"][1] - lat0) * 110_540.0
        dist_km = math.hypot(dx, dy) / 1000.0
        # Beyond a few km it is a different structure, not a spelling variant.
        if dist_km <= 5.0:
            return best, (f"nearest mapped dam to the catalogue coordinate: "
                          f"OSM {best['name']!r} at {dist_km:.2f} km "
                          f"(name {name!r} did not match)")
        return None, (f"nearest mapped dam {best['name']!r} is {dist_km:.1f} km "
                      f"from the catalogue coordinate - too far to assume "
                      f"they are the same structure")

    return None, f"no name or coordinate match for {name!r}"

def _dam_cells(dem: D.DEM, dam: dict) -> Tuple[Tuple[int, int], np.ndarray]:
    """Map the OSM dam way onto grid cells; return (centre_rc, line_cells)."""
    lons = [p[0] for p in dam["geometry_ll"]]
    lats = [p[1] for p in dam["geometry_ll"]]
    rows, cols, inside = EX.ll_to_rowcol(dem, lons, lats)
    rows, cols = rows[inside], cols[inside]
    if rows.size == 0:
        raise ValueError("Dam geometry falls outside the model domain")

    # densify along the crest so the barrier is a continuous line of cells
    pts = [(int(rows[0]), int(cols[0]))]
    for k in range(1, rows.size):
        r0, c0 = pts[-1]
        r1, c1 = int(rows[k]), int(cols[k])
        n = max(abs(r1 - r0), abs(c1 - c0), 1)
        for m in range(1, n + 1):
            pts.append((int(round(r0 + (r1 - r0) * m / n)),
                        int(round(c0 + (c1 - c0) * m / n))))
    line = np.array(sorted(set(pts)))
    centre = (int(np.median(line[:, 0])), int(np.median(line[:, 1])))
    return centre, line


@dataclass
class DamFrame:
    """Local coordinate frame of the dam axis.

    The river at a given dam can run in any direction, so upstream/downstream
    must be defined by the dam's own orientation rather than by grid rows or
    columns.  `signed` is the perpendicular distance of every cell from the
    crest line (in cells); `upstream_mask` is the impounded half-plane.
    """
    centre_rc: Tuple[int, int]
    along: np.ndarray          # unit vector along the crest, (drow, dcol)
    perp: np.ndarray           # unit vector normal to the crest
    upstream_sign: float       # +1 or -1: which side of `perp` impounds water
    signed: np.ndarray         # perpendicular distance field, cells
    upstream_mask: np.ndarray
    seed_rc: Tuple[int, int]   # deepest upstream cell -> flood-fill seed
    tailwater_rc: Tuple[int, int]   # deepest cell just downstream (the dam toe)
    downstream_dir: Tuple[float, float]   # (dx, dy) in map units, for momentum


def _dam_frame(dem: D.DEM, dam_line: np.ndarray, probe_cells: int = 14) -> DamFrame:
    pts = dam_line.astype(float)
    centre = pts.mean(axis=0)

    # principal axis of the crest line
    cov = np.cov((pts - centre).T)
    if np.ndim(cov) == 0 or np.any(~np.isfinite(cov)):
        along = np.array([1.0, 0.0])
    else:
        w, v = np.linalg.eigh(cov)
        along = v[:, int(np.argmax(w))]
    along = along / (np.linalg.norm(along) + 1e-12)
    perp = np.array([-along[1], along[0]])

    rr, cc = np.mgrid[0:dem.ny, 0:dem.nx]
    signed = (rr - centre[0]) * perp[0] + (cc - centre[1]) * perp[1]

    # a band either side of the crest, restricted to the crest's own extent
    span = (pts - centre) @ along
    alongdist = (rr - centre[0]) * along[0] + (cc - centre[1]) * along[1]
    near = (np.abs(alongdist) <= max(np.abs(span).max(), 3.0))

    pos = near & (signed > 3) & (signed < probe_cells)
    neg = near & (signed < -3) & (signed > -probe_cells)

    # Compare the THALWEG on each side, approximated by a low percentile, not
    # the median.  A band either side of the crest is mostly valley wall, and in
    # a gorge the walls easily outweigh the channel, so a median comparison
    # picks a side essentially at random.  The low percentile tracks the valley
    # floor, and a river always loses elevation downstream -- so the impounded
    # side is the one whose floor sits higher.
    z_pos = float(np.percentile(dem.z[pos], 5)) if pos.any() else -1e9
    z_neg = float(np.percentile(dem.z[neg], 5)) if neg.any() else -1e9
    upstream_sign = 1.0 if z_pos > z_neg else -1.0
    upstream_mask = (signed * upstream_sign) > 0

    # seed the flood fill at the deepest cell just upstream of the crest
    seed_band = near & upstream_mask & (np.abs(signed) > 1) & (np.abs(signed) < probe_cells)
    if seed_band.any():
        zz = np.where(seed_band, dem.z, np.inf)
        seed = np.unravel_index(int(np.argmin(zz)), zz.shape)
    else:
        seed = (int(centre[0]), int(centre[1]))

    # The dam toe: the LOWEST cell in a band just downstream.  A fixed offset
    # from the crest centre lands on the dam body or an abutment, which for a
    # 260 m structure is 200+ m above the river.  Anything seeded there -- the
    # breach source patch, the SPH downstream profile -- ends up perched on a
    # hillside instead of in the channel.
    dn_band = near & (~upstream_mask) & (np.abs(signed) > 1) & \
        (np.abs(signed) < probe_cells)
    if dn_band.any():
        zz = np.where(dn_band, dem.z, np.inf)
        toe = np.unravel_index(int(np.argmin(zz)), zz.shape)
    else:
        toe = (int(centre[0]), int(centre[1]))

    # downstream direction in map coordinates: +col is +x, +row is -y
    dperp = -upstream_sign * perp
    dx, dy = float(dperp[1]), float(-dperp[0])
    n = math.hypot(dx, dy) or 1.0

    return DamFrame(centre_rc=(int(centre[0]), int(centre[1])), along=along,
                    perp=perp, upstream_sign=upstream_sign, signed=signed,
                    upstream_mask=upstream_mask,
                    seed_rc=(int(seed[0]), int(seed[1])),
                    tailwater_rc=(int(toe[0]), int(toe[1])),
                    downstream_dir=(dx / n, dy / n))


def _pool_mask(reservoir: RES.Reservoir, dem: D.DEM) -> Optional[np.ndarray]:
    if reservoir.mask is None:
        return None
    m = reservoir.mask
    return m if m.shape == dem.shape else None


def _breach_cells(dem: D.DEM, frame: DamFrame,
                  geom: B.BreachGeometry) -> np.ndarray:
    """Cells that carry the breach discharge into the downstream valley.

    A PATCH, not a line.  Injecting the full breach discharge into one or two
    cells raises the water surface there by tens of metres per second, which the
    solver then converts into an unphysical radial jet: at 180 m posting a 280 m
    breach spans barely two cells, so a 1e6 m3/s peak implies dh/dt ~ 20 m/s.
    Spreading the source over the breach width and a few cells downstream keeps
    the injected depth rate within what the receiving reach can convey, which is
    also what the physical jet does within a few hundred metres of the dam.
    """
    half = max(int(geom.b_top() / dem.dx / 2), 1)
    half = max(half, 2)                       # never narrower than 5 cells
    depth_cells = 3                           # extent downstream of the crest
    r0, c0 = frame.tailwater_rc               # the dam toe, not the crest
    dn = -frame.upstream_sign * frame.perp

    cells = []
    for k in range(-half, half + 1):
        for d in range(0, depth_cells):
            r = r0 + frame.along[0] * k + dn[0] * d
            c = c0 + frame.along[1] * k + dn[1] * d
            ri, ci = int(round(r)), int(round(c))
            if 0 <= ri < dem.ny and 0 <= ci < dem.nx:
                cells.append((ri, ci))
    if not cells:
        return np.array([[r0, c0]])

    cand = np.unique(np.array(cells), axis=0)
    # Keep the lower half of the band: the valley floor, not the flanks. A
    # patch that straddles the hillsides injects water where it cannot drain
    # and the solver piles it into a spurious mound hundreds of metres deep.
    zz = dem.z[cand[:, 0], cand[:, 1]]
    keep = max(len(cand) // 2, min(6, len(cand)))
    idx = np.argsort(zz)[:keep]
    return cand[np.sort(idx)]


def _blended_hydrograph(br: B.BreachResult, iface: C.TransferInterface):
    """SPH drives the first seconds; the weir closure carries the long tail."""
    weir = br.hydrograph()
    sph = iface.hydrograph()
    t_switch = float(iface.t[-1]) if iface.t.size else 0.0
    blend = max(t_switch * 0.25, 1.0)

    def q(t: float) -> float:
        if t <= t_switch - blend:
            return sph(t)
        if t >= t_switch:
            return weir(t)
        f = (t - (t_switch - blend)) / blend
        return (1 - f) * sph(t) + f * weir(t)
    return q


_CONVERT = __import__("re").compile(r"\{\{convert\|([0-9.]+)\|([a-zA-Z0-9]+)")


def _published_value(field: str) -> Optional[Tuple[float, str]]:
    """Pull a number+unit out of a Wikipedia {{convert|...}} infobox field."""
    if not field:
        return None
    m = _CONVERT.search(field)
    if m:
        return float(m.group(1)), m.group(2)
    m = __import__("re").search(r"([0-9][0-9,.]*)\s*([a-zA-Z0-9]+)", field)
    if m:
        try:
            return float(m.group(1).replace(",", "")), m.group(2)
        except ValueError:
            return None
    return None


def _published_reservoir(dossier: dict) -> Tuple[Optional[float], Optional[float]]:
    """(gross capacity MCM, dam height m) from the published record, if present."""
    wiki = (dossier.get("wikipedia") or {}).get("fields", {})
    cap = _published_value(wiki.get("res_capacity_total", ""))
    hgt = _published_value(wiki.get("dam_height", ""))
    cap_mcm = None
    if cap:
        val, unit = cap
        cap_mcm = val * 1000.0 if unit.lower() in ("km3", "km³") else val
    return cap_mcm, (hgt[0] if hgt else None)


def _reservoir_cross_check(res_summary: dict, dossier: dict) -> dict:
    """Compare DEM-derived storage against the published engineering record.

    A crucial caveat is recorded here rather than buried: Copernicus GLO-30 is
    a DIGITAL SURFACE MODEL from 2011-2015 radar.  Where a reservoir already
    existed at acquisition time, the DEM records the *water surface*, not the
    drowned valley floor.  DEM hypsometry therefore measures the storage
    between that water surface and the crest, which is a lower bound on gross
    storage -- it cannot see the bathymetry underneath.  Matching the published
    gross capacity is not expected, and a shortfall is the correct behaviour,
    not an error.  Pre-impoundment topography or a bathymetric survey is needed
    to close the gap.
    """
    out = {
        "dem_capacity_mcm": res_summary["capacity_mcm"],
        "dem_max_depth_m": res_summary["max_depth_m"],
        "dem_surface_km2": res_summary["area_at_crest_km2"],
        "messages": [],
        "pass": True,
        "dem_is_surface_model": True,
        "caveat": ("Copernicus GLO-30 is a DSM: for an existing reservoir it "
                   "records the water surface, so DEM-derived storage is a "
                   "LOWER BOUND on gross capacity and the surface area is "
                   "measured at crest level, not at FRL."),
    }
    wiki = (dossier.get("wikipedia") or {}).get("fields", {})
    out["published_fields"] = wiki
    if not wiki:
        out["messages"].append("No published record available for cross-check.")
        return out

    cap = _published_value(wiki.get("res_capacity_total", ""))
    if cap:
        val, unit = cap
        pub_mcm = val * 1000.0 if unit.lower() in ("km3", "km³") else val
        out["published_capacity_mcm"] = round(pub_mcm, 1)
        out["capacity_ratio_dem_over_published"] = round(
            res_summary["capacity_mcm"] / pub_mcm, 3) if pub_mcm else None
        if out["capacity_ratio_dem_over_published"] and \
                out["capacity_ratio_dem_over_published"] > 3.0:
            out["pass"] = False
            out["messages"].append(
                f"DEM storage ({res_summary['capacity_mcm']:.0f} MCM) exceeds "
                f"the published gross capacity ({pub_mcm:.0f} MCM) by more than "
                "3x - the pool is probably leaking past the dam axis. Check the "
                "upstream half-plane and the barrier mask.")

    surf = _published_value(wiki.get("res_surface", ""))
    if surf:
        out["published_surface_km2"] = surf[0]

    hgt = _published_value(wiki.get("dam_height", ""))
    if hgt:
        out["published_dam_height_m"] = hgt[0]
        if res_summary["max_depth_m"] > 1.6 * hgt[0]:
            out["pass"] = False
            out["messages"].append(
                f"Impounded depth ({res_summary['max_depth_m']:.0f} m) far "
                f"exceeds the published dam height ({hgt[0]:.0f} m).")
    return out


def _validate(scn: Scenario, dem: D.DEM, swe: SW.SWEResult,
              hz: HZ.HazardProduct, manifest: dict) -> VAL.ValidationReport:
    window = scn.sentinel1_window
    if window is None:
        # default: a recent dry-season pass, giving the permanent water baseline
        window = ("2024-01-01", "2024-03-31")
    items = ds.find_sentinel1(scn.bbox_ll, window[0], window[1])
    if not items:
        raise ds.SourceUnavailable(
            f"No Sentinel-1 GRD scene over {scn.bbox_ll} in {window}")
    item = items[0]
    db = ds.sentinel1_backscatter(item, dem, scn.bbox_ll)
    slope = VAL.terrain_slope_deg(dem.z, dem.dx, dem.dy)
    water, thr = ds.sar_water_mask(db, slope_deg=slope)

    manifest["data_sources"]["sentinel1"] = {
        "item_id": item["id"],
        "datetime": item["properties"]["datetime"],
        "platform": item["properties"].get("platform"),
        "collection": f"{ds.S1_COLLECTION} via Microsoft Planetary Computer",
        "otsu_threshold_db": round(thr, 2),
    }
    rep = VAL.validate_extent(
        swe.wet_mask(0.05), water, dem.cell_area, mode=scn.validation_mode,
        baseline_water=water if scn.validation_mode == "context" else None,
        scene=manifest["data_sources"]["sentinel1"],
        valid=np.isfinite(db))
    return rep


def _write_terrain_3d(swe: Optional[SW.SWEResult], dem: D.DEM,
                      primary: SW.SWEResult, hz: HZ.HazardProduct,
                      out: Path, dam_line: Optional[np.ndarray] = None,
                      rgb: Optional[np.ndarray] = None) -> dict:
    """Collect the depth frames and hand them to the 3D exporter.

    Falls back to the single max-depth field when no time frames were written,
    so the 3D view still shows the flood envelope rather than nothing.
    """
    frames, times = [], []
    if swe is not None and swe.frame_dir is not None and swe.frame_times:
        for i, t in enumerate(swe.frame_times):
            f = swe.frame_dir / f"frame_{i:04d}.npz"
            if f.exists():
                frames.append(np.load(f)["h"].astype("float32"))
                times.append(t)
    if not frames:
        frames = [primary.h_max]
        times = [float(primary.stats.get("t_end_s", 0.0))]

    return OUT.export_terrain_3d(
        dem, frames, times, out / "terrain3d.json",
        hazard=hz.hazard_class.astype("uint8"), dam_cells=dam_line, rgb=rgb)


def _write_frame_overlays(swe: Optional[SW.SWEResult], dem: D.DEM,
                          out: Path) -> dict:
    """Turn the saved depth frames into WGS84 PNGs for the time animation."""
    if swe is None or swe.frame_dir is None or not swe.frame_times:
        return {"count": 0, "times_s": [], "files": []}
    ov = out / "overlays" / "frames"
    ov.mkdir(parents=True, exist_ok=True)
    vmax = max(float(swe.h_max.max()) * 0.6, 1.0)
    files, bounds = [], None
    for i, t in enumerate(swe.frame_times):
        f = swe.frame_dir / f"frame_{i:04d}.npz"
        if not f.exists():
            continue
        h = np.load(f)["h"].astype("float32")
        meta = OUT.raster_to_png_4326(h, dem, ov / f"f{i:04d}.png", 0.0, vmax,
                                      "depth", mask_below=0.05, max_px=900)
        bounds = meta["bounds_ll"]
        files.append(f"frames/{meta['file']}")
    return {"count": len(files), "times_s": [round(t, 1) for t in swe.frame_times],
            "files": files, "bounds_ll": bounds, "vmax": vmax}
