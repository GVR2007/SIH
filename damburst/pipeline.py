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

from .core import adequacy as ADQ
from .core import breach as B
from .core import coupling as C
from .core import datasources as ds
from .core import dem as D
from .core import exposure as EX
from .core import export as OUT
from .core import failure_probability as FP
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
    # --- adaptive model selection ------------------------------------
    # The framework chooses the near-field physics itself from the terrain and
    # the release scale. Set auto_model_selection=False to force `run_sph`.
    auto_model_selection: bool = True
    bed_slope_deg_c: float = ADQ.BED_SLOPE_DEG_C
    curvature_ratio_c: float = ADQ.CURVATURE_RATIO_C
    # --- possibility of breach (see docs/MATHEMATICAL_FORMULATION.md Part B)
    # P(overtopping) is ROUTED when flood statistics are supplied; every other
    # mechanism falls back to published base rates and is labelled as such.
    # Left at 0.0 the whole block is skipped and the run makes no probability
    # claim at all, which is the correct default.
    mean_annual_flood_m3s: float = 0.0
    flood_cv: float = 0.6
    spillway_capacity_factor: float = 1.0
    spillway_crest_length_m: float = 0.0     # 0 = size the placeholder
    spillway_sill_m: float = 0.0             # 0 = derive from the crest
    tailwater: bool = True               # Villemonte submergence on the breach weir
    tailwater_slope: float = 0.0         # 0 = derive from the DEM long profile
    froude_max: float = SW.FROUDE_MAX    # 0 disables the steep-terrain limiter
    steep_slope_deg: float = SW.STEEP_SLOPE_DEG
    sentinel1_window: Optional[Tuple[str, str]] = None
    baseline_window: Optional[Tuple[str, str]] = None   # pre-event, benchmark mode
    validation_mode: str = "context"
    satellite_basemap: bool = True
    population_product: str = "1km"
    iso3: str = "IND"
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
    match = [d for d in dams if scn.dam_name.lower() in d["name"].lower()]
    if not match:
        raise ds.SourceUnavailable(
            f"No OSM dam matching {scn.dam_name!r} inside {scn.bbox_ll}. "
            f"Found: {[d['name'] for d in dams]}")
    dam = match[0]
    dossier = ds.dam_dossier(dam)
    manifest["dam"] = dossier
    manifest["data_sources"]["dam"] = dossier["sources"]

    _log(progress, "ingest", "Fetching OSM exposure + WorldPop", 16)
    exposure = ds.fetch_exposure_osm(scn.bbox_ll)
    roads_geom = ds.fetch_roads_geom(scn.bbox_ll)
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

    # Channel geometry node.  `channel_mask` used to be hardcoded to None, so
    # `burn_channel` could never run whatever `channel_burn_m` was set to.  The
    # mask now comes from the real OSM waterway centrelines, which is exactly
    # what burn_channel's docstring says it needs; if OSM has no waterway here
    # and a burn was requested, fall back to D8 flow accumulation on the DEM.
    channel_mask = None
    if scn.channel_burn_m > 0:
        channel_mask, ch_src = _channel_mask(dem, scn.bbox_ll)
        manifest["data_sources"]["channel_network"] = ch_src
    dem_c = D.condition(dem, channel_mask=channel_mask, burn=scn.channel_burn_m,
                        fill_sinks=True)
    dem_c.manning = dem.manning
    dem_c.landcover = dem.landcover
    manifest["qc"]["conditioning"] = dem_c.meta.get("conditioning")
    manifest["qc"]["landcover"] = D.landcover_summary(dem_c.landcover) \
        if dem_c.landcover is not None else {}

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

    # Tailwater rating, so the Villemonte submergence correction in
    # `weir_discharge` can actually engage.  It never did before: no tailwater
    # was passed, so the breach discharged freely for the whole event and the
    # peak was an upper bound.  The rating is Manning normal depth in the
    # receiving reach, with the reach slope and roughness read off the DEM
    # thalweg and the WorldCover map at the dam toe.
    tw_fn, tw_meta = _tailwater_rating(scn, dem_c, frame, geom, reservoir)
    manifest["qc"]["tailwater"] = tw_meta

    br = B.simulate_breach(reservoir, geom, h_init,
                           failure_mode=scn.failure_mode,
                           growth_law=scn.growth_law, inflow=inflow,
                           tailwater=tw_fn,
                           t_end=scn.breach_hours * 3600.0)
    results["breach"] = br.to_dict()
    manifest["qc"]["breach"] = br.checks

    OUT.export_hydrograph_csv(out / "tables" / "breach_hydrograph.csv", br.t,
                              {"Q_m3s": br.q, "reservoir_level_m": br.h_res,
                               "breach_top_width_m": br.b_top,
                               "breach_invert_m": br.invert,
                               "storage_m3": br.volume})

    # -- 4b. ADAPTIVE MODEL SELECTION -------------------------------------
    #
    # The framework decides for itself which physics the near field needs,
    # instead of running everything and inviting the reader to pick. The
    # criterion is the non-hydrostatic index along the real receiving reach,
    # evaluated at the breach-jet velocity: where the depth-averaged equations
    # are valid, the particle model is skipped; where they are not, it is run
    # and the handover is placed where the flow recovers a hydrostatic profile.
    _log(progress, "select", "Adaptive model selection", 38)
    s_dn_sel, z_dn_sel, path_sel = C.thalweg_profile(dem_c, frame.tailwater_rc,
                                                     length_m=1500.0)
    plan = ADQ.select_models(
        s_dn_sel, z_dn_sel,
        head_m=h_init - geom.invert,
        breach_width_m=max(geom.b_top(), dem_c.dx),
        peak_q_m3s=br.peak_q,
        dp_m=scn.sph_dp,
        downstream_m=900.0,
        force=None if scn.auto_model_selection else scn.run_sph,
        bed_slope_deg_c=scn.bed_slope_deg_c,
        curvature_ratio_c=scn.curvature_ratio_c,
        breach_invert_m=geom.invert,
        toe_bed_m=float(dem_c.z[frame.tailwater_rc]),
        corridor=ADQ.corridor_relief(dem_c.z, path_sel, dem_c.dx, dem_c.dy))
    results["model_selection"] = plan.to_dict()
    manifest["qc"]["model_selection"] = plan.to_dict()
    _log(progress, "select", plan.reason, 40)

    # -- 5. SPH near field -----------------------------------------------
    sph_res = None
    iface = None
    if plan.run_near_field:
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
                water_surface_m=(ws or {}).get("elevation_m"),
                transfer_x_m=plan.transfer_x_m)
            results["sph"] = sph_res.stats
            results["transfer_interface"] = iface.to_dict()
            results["sph_vs_weir"] = C.compare_hydrographs(
                br, iface, head_m=h_init - geom.invert,
                sph_stats=sph_res.stats)
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
        sph_window_s = float(iface.t[-1]) if iface.t.size else 0.0
        sim_s = scn.sim_hours * 3600.0
        configs.append((
            "sph_initialised",
            _blended_hydrograph(br, iface),
            # Renamed from "coupled".  SPH supplies the source hydrograph for
            # the first `sph_window_s` seconds only -- 12-20 s against a 3-hour
            # simulation, i.e. ~0.2% of it -- after which the weir closure
            # takes over completely.  Calling that "coupled" oversells it: the
            # particle model sets the INITIAL CONDITION of the far field, it
            # does not drive it.  The name now says what the configuration is.
            f"2D shallow-water initialised by the SPH transfer-section "
            f"hydrograph for the first {sph_window_s:.1f} s "
            f"({100.0 * sph_window_s / max(sim_s, 1.0):.2f}% of the "
            f"simulation), weir closure thereafter"))

    for k, (key, hyd, desc) in enumerate(configs):
        _log(progress, "swe", f"Running 2D model: {key}", 50 + 20 * k)
        t0 = time.time()
        model = SW.SWE2D(dem_c.z, dem_c.dx, dem_c.dy, dem_c.manning,
                         open_edges=True, order=2,
                         froude_max=scn.froude_max,
                         steep_slope_deg=scn.steep_slope_deg)
        model.set_still_water(h_init, mask=_pool_mask(reservoir, dem_c))
        src = SW.PointSource(rows=breach_cells[:, 0], cols=breach_cells[:, 1],
                             hydrograph=hyd, name="breach")
        # Frames are written for EVERY configuration, so the animation and the
        # 3D view can show whichever model the statistics were taken from.
        # Writing them only for grid_standalone meant the dashboard rendered a
        # different model from the one it reported numbers for.
        fdir = out / "frames" / key

        def _p(t, tend, info, key=key):
            _log(progress, "swe", f"{key}: t={t / 60:.1f}/{tend / 60:.0f} min",
                 None, **info)

        r = model.run(t_end=scn.sim_hours * 3600.0, sources=[src],
                      source_direction=flow_dir, n_frames=scn.n_frames,
                      frame_dir=fdir, progress=_p, cfl=0.40, dt_max=15.0)
        swe_products[key] = r
        model_runs[key] = {
            "description": desc,
            "peak_inflow_m3s": round(
                _hydrograph_peak(hyd, scn.sim_hours * 3600.0,
                                 extra_times=(iface.t if iface is not None else None)), 1),
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
        sph_done = bool(sph_res.stats.get("completed", True))
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
            # A truncated particle run must not sit in the table looking like
            # an ordinary result next to two complete ones.
            "status": "ok" if sph_done else "truncated",
            "caveat": None if sph_done else (
                f"Terminated at {sph_res.stats.get('simulated_s')} s of "
                f"{sph_res.stats.get('requested_s')} s requested: "
                f"{sph_res.stats.get('stop_reason')}"),
        }
        if not sph_done and "sph_initialised" in model_runs:
            model_runs["sph_initialised"]["status"] = "derived_from_truncated_sph"
            model_runs["sph_initialised"]["caveat"] = (
                "The SPH hydrograph that initialises this configuration was "
                "truncated; see the sph_nearfield row.")

    # PRIMARY MODEL.
    # `grid_standalone` is the primary result.  The SPH-initialised
    # configuration differs from it only over the first few seconds and, where
    # the particle run was truncated, inherits that truncation -- so it must
    # not be the run that hazard, exposure and loss are computed from.  It
    # stays in the comparison as a sensitivity, which is what it is.
    primary_key = "grid_standalone"
    primary = swe_products[primary_key]
    results["primary_model"] = primary_key
    results["primary_model_rationale"] = (
        "grid_standalone is the primary configuration: it is driven end to end "
        "by the empirical breach hydrograph with no dependence on the "
        "near-field particle run. sph_initialised is reported alongside as a "
        "sensitivity on the first seconds of the release.")
    model_runs[primary_key]["is_primary"] = True

    # Quantitative agreement between the far-field configurations, so the
    # comparison is more than a table of scalar maxima.
    if len(swe_products) > 1:
        results["model_agreement"] = [
            VAL.field_agreement(primary.h_max, r.h_max, dem_c.cell_area,
                                label=f"{primary_key} vs {k}")
            for k, r in swe_products.items() if k != primary_key]

    results["model_comparison"] = C.build_comparison_table(model_runs)

    # -- 7. Hazard --------------------------------------------------------
    _log(progress, "hazard", "Hazard rating and classification", 82)
    hz = HZ.build_hazard(primary.h_max, primary.v_max, dem_c.cell_area,
                         landcover=dem_c.landcover,
                         arrival_s=primary.arrival_s,
                         duration_s=primary.duration_s)
    results["hazard"] = hz.summary

    # -- 7b. Post-run model adequacy --------------------------------------
    # The same criterion that chose the models, now applied to the solution
    # that was actually produced: over how much of the inundated area were the
    # equations we used valid? A hazard raster cannot say this about itself.
    _log(progress, "adequacy", "Scoring model adequacy over the solution", 84)
    adq = ADQ.adequacy_field(dem_c.z, primary.h_max, primary.v_max,
                             dem_c.dx, dem_c.dy,
                             bed_slope_deg_c=scn.bed_slope_deg_c,
                             curvature_ratio_c=scn.curvature_ratio_c)
    results["model_adequacy"] = adq.summary
    manifest["qc"]["model_adequacy"] = adq.summary
    _log(progress, "adequacy", adq.summary["verdict"], 85)

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
                                   hz.hazard_class, hz.dv)
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
    losses = EX.estimate_losses(buildings, roads, lc_impact, scn.asset_values)

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

    # -- 8b. Possibility of breach ----------------------------------------
    # Everything above is CONDITIONAL on failure. This is the other factor.
    if scn.mean_annual_flood_m3s > 0:
        _log(progress, "failure", "Routing the design-flood family", 90)
        try:
            if scn.spillway_crest_length_m > 0:
                spill = FP.Spillway(
                    crest_length_m=scn.spillway_crest_length_m,
                    sill_elevation_m=(scn.spillway_sill_m or crest - 3.0),
                    source="USER-SUPPLIED spillway rating")
            else:
                spill = FP.default_spillway(
                    crest, reservoir,
                    mean_annual_flood_m3s=scn.mean_annual_flood_m3s,
                    flood_cv=scn.flood_cv,
                    capacity_factor=scn.spillway_capacity_factor)
            results["failure_probability"] = FP.failure_probability_report(
                reservoir, crest, spillway=spill,
                mean_annual_flood_m3s=scn.mean_annual_flood_m3s,
                flood_cv=scn.flood_cv, barrier_type=scn.barrier_type,
                # Start the flood routing from the spillway sill (the
                # normal operating level), NOT from `h_init`.
                #
                # `loading="FRL"` sets h_init to the CREST, because the crest
                # is what the dam-line elevations give. That is the right
                # antecedent condition for a breach scenario -- a full
                # reservoir is the worst case -- but it is a nonsense starting
                # point for flood routing: a reservoir already at crest level
                # has zero freeboard and overtops on any flood at all, which
                # would report P(overtopping) ~ 1 as if it were a finding.
                # `None` lets the routine use the sill.
                starting_level=None)
            manifest["data_sources"]["spillway"] = spill.to_dict()
        except Exception as exc:                       # noqa: BLE001
            results["failure_probability"] = {
                "error": f"{type(exc).__name__}: {exc}"}
    else:
        results["failure_probability"] = {
            "computed": False,
            "note": ("No flood statistics supplied, so no probability of "
                     "failure is claimed. Every hazard and exposure figure in "
                     "this run is CONDITIONAL on failure occurring. Supply "
                     "--mean-annual-flood to route the design-flood family."),
        }

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
    OUT.write_geotiff(rasters / "model_adequacy_nhi.tif", adq.nhi, dem_c,
                      description=("Non-hydrostatic index; >=1 means the "
                                   "depth-averaged shallow-water assumptions "
                                   "are violated at that cell"))

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

    overlays["adequacy"] = OUT.raster_to_png_4326(
        adq.nhi, dem_c, ov / "model_adequacy.png", 0.0, 3.0, "hazard",
        mask_below=0.02)

    # Shaded relief, so the dashboard has a basemap even with no internet.
    overlays["basemap"] = OUT.hillshade_png_4326(dem_c, ov / "basemap.png",
                                                 rgb=sat_rgb)

    # Terrain + time-varying water surface for the 3D viewer.
    #
    # BOTH the 3D payload and the 2D animation are built from `primary` -- the
    # same run every reported statistic comes from.  They used to be hardwired
    # to `grid_standalone` while the numbers came from the coupled run, so the
    # dashboard animated one model and tabulated another.
    _log(progress, "export", "Writing 3D terrain and water surface", 97)
    try:
        results["terrain3d"] = _write_terrain_3d(
            primary, dem_c, primary, hz, out, dam_line=dam_line, rgb=sat_rgb)
        results["terrain3d"]["source_model"] = primary_key
    except Exception as exc:                           # noqa: BLE001
        results["terrain3d"] = {"error": f"{type(exc).__name__}: {exc}"}
        _log(progress, "export", f"3D export failed: {exc}", 97)

    frame_meta = _write_frame_overlays(primary, dem_c, out)
    if isinstance(frame_meta, dict):
        frame_meta["source_model"] = primary_key

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
        # Scale-independent: critical flow through the breach's own section.
        # This is the gate that actually binds on large Indian dams.
        "breach_critical_flow": manifest["qc"].get("breach", {})
            .get("peak_discharge_critical_flow", {}).get("pass"),
        # May be None ("not applicable") when the reservoir is outside the
        # range the empirical regressions were fitted to.
        "breach_peak_envelope": manifest["qc"].get("breach", {})
            .get("peak_discharge_envelopes", {}).get("pass"),
        "breach_geometry": manifest["qc"].get("breach", {})
            .get("geometry", {}).get("pass"),
        "exposure": manifest["qc"].get("exposure", {}).get("pass"),
    }
    # The SPH balance gate only exists if the near field was actually modelled.
    # Listing it as "inconclusive" when the selector deliberately decided the
    # particle model was unnecessary would turn a correct decision into a
    # blemish on the QC panel.
    if plan.run_near_field:
        gates["sph_balance"] = manifest["qc"].get(
            "sph_balance", {}).get("balance_pass")
    else:
        manifest["qc"]["sph_balance"] = {
            "not_applicable": True,
            "reason": ("the near-field particle model was not required for "
                       "this scenario; see qc.model_selection"),
        }
    manifest["qc"]["gates"] = gates
    manifest["qc"]["failed_gates"] = [k for k, v in gates.items() if v is False]
    # A gate that returned None did not pass -- it could not be evaluated.
    # Counting None as a pass is how a 16x envelope exceedance used to end up
    # inside a green "overall_pass". It is now reported as its own category.
    manifest["qc"]["inconclusive_gates"] = [
        k for k, v in gates.items() if v is None]
    manifest["qc"]["overall_pass"] = not manifest["qc"]["failed_gates"]
    manifest["qc"]["gate_legend"] = {
        "true": "evaluated and passed",
        "false": "evaluated and failed - counted in failed_gates",
        "null": ("could not be evaluated for this scenario (e.g. an empirical "
                 "regression outside its calibration range). NOT a pass; "
                 "listed in inconclusive_gates."),
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _log(progress, "done", f"Run complete in {manifest['timings']['total_s']}s", 100)

    return {"run_id": run_id, "dir": str(out), "manifest": manifest,
            "results": results}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _hydrograph_peak(hyd: Callable[[float], float], t_end: float,
                     extra_times: Optional[np.ndarray] = None,
                     n: int = 2000) -> float:
    """Peak of a source hydrograph, sampled so short features cannot be missed.

    This used to be `max(hyd(t) for t in linspace(0, t_end, 400))`.  Over a
    3-hour simulation that is one sample every 27 s, while the SPH-driven
    portion of the blended hydrograph is 12-20 s long -- so the sampler stepped
    straight over it and the SPH-initialised configuration reported a peak
    IDENTICAL to the standalone run, making two different models look like they
    agreed exactly.

    The fix is to sample the union of a coarse global grid, a fine grid over
    the early transient, and the actual sample times of any supplied series.
    """
    ts = [np.linspace(0.0, t_end, n)]
    # Dense coverage of the first 10 minutes, where every fast feature lives.
    ts.append(np.linspace(0.0, min(600.0, t_end), 1200))
    if extra_times is not None and np.size(extra_times):
        ts.append(np.asarray(extra_times, dtype=float))
    grid = np.unique(np.concatenate(ts))
    grid = grid[(grid >= 0.0) & (grid <= t_end)]
    return float(max(hyd(float(tt)) for tt in grid))


def _channel_mask(dem: D.DEM, bbox_ll) -> Tuple[Optional[np.ndarray], str]:
    """Channel-network raster for stream burning, from OSM then from the DEM.

    Preference order matters: an OSM waterway centreline is surveyed, a D8
    accumulation threshold is inferred.  Use the real data when it exists.
    """
    try:
        ways = ds.fetch_waterways(bbox_ll)
    except Exception as exc:                           # noqa: BLE001
        ways = []
        osm_err = str(exc)
    else:
        osm_err = None

    if ways:
        mask = np.zeros(dem.shape, dtype=bool)
        hit = 0
        for w in ways:
            coords = w.get("coords_ll") or []
            if len(coords) < 2:
                continue
            rows, cols, inside = EX.ll_to_rowcol(
                dem, [c[0] for c in coords], [c[1] for c in coords])
            rr, cc = rows[inside], cols[inside]
            if rr.size:
                mask[rr, cc] = True
                hit += 1
        if mask.any():
            return mask, (f"OpenStreetMap waterway centrelines (ODbL), "
                          f"{hit} ways burned")

    # Fallback: the DEM's own drainage network.  O(N) over cells, so only run
    # when a burn was explicitly requested and OSM had nothing.
    acc = D.d8_flow_accumulation(dem.z, dem.dx)
    thresh = float(np.percentile(acc, 99.5))
    mask = acc >= thresh
    return mask, (f"D8 flow accumulation on the conditioned DEM, "
                  f"threshold {thresh:.0f} cells "
                  f"(OSM waterways unavailable{': ' + osm_err if osm_err else ''})")


def _tailwater_rating(scn: "Scenario", dem: D.DEM, frame: "DamFrame",
                      geom: B.BreachGeometry, reservoir: RES.Reservoir):
    """Build the downstream stage rating that drives Villemonte submergence.

    Returns (callable or None, provenance dict).  The reach slope comes from
    the DEM thalweg below the dam toe and the roughness from the WorldCover
    Manning map there, so nothing here is a typed-in number.
    """
    if not scn.tailwater:
        return None, {"enabled": False,
                      "note": ("Tailwater disabled: the breach discharges "
                               "freely and peak outflow is an upper bound.")}
    try:
        s_dn, z_dn, _path = C.thalweg_profile(dem, frame.tailwater_rc,
                                              length_m=3000.0)
        if z_dn.size < 3:
            raise ValueError("thalweg too short")
        slope = scn.tailwater_slope or max(
            float((z_dn[0] - z_dn[-1]) / max(s_dn[-1] - s_dn[0], 1.0)), 1e-4)
        r0, c0 = frame.tailwater_rc
        n_bed = float(dem.manning[r0, c0]) if dem.manning is not None else 0.035
        width = max(geom.b_top(), dem.dx)
        invert = float(dem.z[r0, c0])
    except Exception as exc:                           # noqa: BLE001
        return None, {"enabled": False,
                      "error": f"{type(exc).__name__}: {exc}",
                      "note": "Tailwater rating unavailable; free discharge."}

    fn = B.normal_depth_tailwater(invert, width, slope, n_bed)
    return fn, {
        "enabled": True,
        "method": "Manning normal depth in a wide rectangular reach",
        "reach_slope": round(slope, 5),
        "reach_slope_source": ("DEM thalweg over 3 km below the dam toe"
                               if not scn.tailwater_slope else "user-supplied"),
        "manning_n": round(n_bed, 4),
        "manning_n_source": "ESA WorldCover class at the dam toe",
        "conveyance_width_m": round(width, 1),
        "invert_m": round(invert, 2),
        "note": ("Rating curve, not a backwater solution: the reach is assumed "
                 "to convey the breach discharge at normal depth. Conservative "
                 "in a steep gorge, where the true tailwater is lower."),
    }


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
    """Compare the storage the DEM ACTUALLY MEASURED against the published record.

    Copernicus GLO-30 is a DIGITAL SURFACE MODEL from 2011-2015 radar.  Where a
    reservoir already existed at acquisition time, the DEM records the *water
    surface*, not the drowned valley floor.  DEM hypsometry therefore measures
    only the storage between that water surface and the crest -- a lower bound
    on gross storage.

    THE CIRCULARITY THIS AVOIDS.  When the hybrid reconstruction runs, gross
    capacity is set BY the published figure (`a = V_pub / d_crest**b`), so
    comparing the reconstructed capacity against the published capacity is
    comparing a number against the number it was built from: the ratio is ~1 by
    construction and the gate cannot fail.  This check therefore uses
    `dem_only_capacity_mcm` -- what the terrain independently contributed --
    and reports the reconstructed figure separately, labelled as reconstructed.
    """
    deriv = res_summary.get("derivation", {}) or {}
    reconstructed = "hybrid" in str(deriv.get("method", ""))
    dem_only = deriv.get("dem_only_capacity_mcm")

    out = {
        "reported_capacity_mcm": res_summary["capacity_mcm"],
        "capacity_is_reconstructed": bool(reconstructed),
        "dem_only_capacity_mcm": dem_only,
        "dem_max_depth_m": res_summary["max_depth_m"],
        "dem_surface_km2": res_summary["area_at_crest_km2"],
        "messages": [],
        "pass": True,
        "dem_is_surface_model": True,
        "caveat": ("Copernicus GLO-30 is a DSM: for an existing reservoir it "
                   "records the water surface, so DEM-only storage is a LOWER "
                   "BOUND on gross capacity. Where the hybrid reconstruction "
                   "ran, the reported capacity is SET BY the published figure "
                   "and must not be presented as a DEM measurement."),
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

        if reconstructed and dem_only is not None:
            # Independent comparison: terrain-measured storage vs published.
            out["capacity_ratio_dem_only_over_published"] = round(
                dem_only / pub_mcm, 4) if pub_mcm else None
            out["published_capacity_share_of_reported"] = round(
                1.0 - min(dem_only / max(res_summary["capacity_mcm"], 1e-9), 1.0), 4)
            out["independence_note"] = (
                f"The reported {res_summary['capacity_mcm']:.0f} MCM is a "
                f"RECONSTRUCTION anchored on the published {pub_mcm:.0f} MCM; "
                f"comparing the two would be circular. The DEM independently "
                f"measured {dem_only:.1f} MCM above the DSM water plate, i.e. "
                f"{100.0 * dem_only / max(res_summary['capacity_mcm'], 1e-9):.1f}% "
                "of the storage used. The rest comes from the published record.")
            out["messages"].append(out["independence_note"])
            # Sanity in the other direction: the sliver the DEM sees must not
            # exceed the published gross capacity.
            if dem_only > pub_mcm:
                out["pass"] = False
                out["messages"].append(
                    f"DEM-only storage ({dem_only:.0f} MCM) exceeds the "
                    f"published gross capacity ({pub_mcm:.0f} MCM) - the pool "
                    "is leaking past the dam axis. Check the upstream "
                    "half-plane and the barrier mask.")
        else:
            out["capacity_ratio_dem_over_published"] = round(
                res_summary["capacity_mcm"] / pub_mcm, 3) if pub_mcm else None
            if out["capacity_ratio_dem_over_published"] and \
                    out["capacity_ratio_dem_over_published"] > 3.0:
                out["pass"] = False
                out["messages"].append(
                    f"DEM storage ({res_summary['capacity_mcm']:.0f} MCM) "
                    f"exceeds the published gross capacity ({pub_mcm:.0f} MCM) "
                    "by more than 3x - the pool is probably leaking past the "
                    "dam axis. Check the upstream half-plane and the barrier "
                    "mask.")

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
    # The baseline is passed in BOTH modes.
    #
    # In `context` it supplies `permanent_water_km2` for the dashboard; in
    # `benchmark` it is what gets subtracted from the model and the observation
    # so the score measures the flood signal rather than the river and the
    # reservoir being permanently wet.  Passing it only in `context` -- as this
    # used to -- meant the subtraction never ran and a benchmark score would
    # have been inflated by every permanently wet cell in the domain.
    #
    # NOTE on what `water` is in each mode.  In `context` the scene is a
    # dry-season pass, so `water` IS the permanent baseline and observed==
    # baseline by construction.  In `benchmark` the caller must supply
    # `sentinel1_window` bracketing the real flood, and a separate pre-event
    # scene is fetched for the baseline.
    baseline = water
    if scn.validation_mode == "benchmark" and scn.baseline_window:
        b_items = ds.find_sentinel1(scn.bbox_ll, *scn.baseline_window)
        if b_items:
            b_db = ds.sentinel1_backscatter(b_items[0], dem, scn.bbox_ll)
            baseline, b_thr = ds.sar_water_mask(b_db, slope_deg=slope)
            manifest["data_sources"]["sentinel1_baseline"] = {
                "item_id": b_items[0]["id"],
                "datetime": b_items[0]["properties"]["datetime"],
                "otsu_threshold_db": round(b_thr, 2),
                "role": "pre-event permanent-water baseline",
            }

    rep = VAL.validate_extent(
        swe.wet_mask(0.05), water, dem.cell_area, mode=scn.validation_mode,
        baseline_water=baseline,
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
