# Technical-approach diagram &rarr; code mapping

Every node of the SIH technical-approach flowchart, and where it is implemented.

## Left column — inputs, barrier, breach

| Diagram node | Implementation |
|---|---|
| DEM / Hydrology / Dam / Satellite | `core/datasources.py` — Copernicus GLO-30, OSM, WorldPop, WorldCover, Sentinel-1 |
| Input Data QC & Harmonization | `core/dem.py :: qc_report`, `core/datasources.py :: make_grid` (UTM reprojection) |
| Spatial Database | `data/cache/` — content-hashed cache of every fetched layer |
| DEM conditioning | `core/dem.py :: condition` (void fill &rarr; channel burn &rarr; priority-flood sink fill) |
| Channel geometry & bathymetry | `core/dem.py :: burn_channel`, `d8_flow_accumulation`, `steepest_descent_path` |
| Computational domain & mesh | `core/datasources.py :: make_grid`, `Grid` |
| Roughness map | `core/dem.py :: roughness_from_worldcover` (ESA WorldCover &rarr; Manning *n*) |
| Dam record type, height, crest, spillway | `core/datasources.py :: dam_dossier` (OSM + Wikidata + Wikipedia) |
| Reservoir H–V–A curve + Loading Level | `core/reservoir.py :: hva_from_dem`, `hybrid_hva`, `detect_water_surface` |
| Inflow Q_in(t) + tailwater rating | `core/reservoir.py :: inflow_hydrograph`; tailwater in `breach.weir_discharge` |
| Barrier type &rarr; Engineered / Natural | `core/breach.py :: seed_breach` (`barrier_type`) |
| Empirical Seed | `froehlich_2008`, `von_thun_gillette`, `macdonald_langridge`, `costa_schuster_peak` |
| Scenario: Dam Break / River Blockage / Water Release | `pipeline.py :: Scenario`, `scenarios.py` |
| Failure Mode: Partial / Progressive Erosion / Piping / Overtopping / Instantaneous | `core/breach.py :: FAILURE_MODES`, `breach_section` |
| Adaptive &Delta;t | `core/breach.py :: simulate_breach` (storage- and formation-limited step) |
| Geometry growth law | `core/breach.py :: growth_fraction` (linear / sine / erosion) |
| head H &rarr; Q_breach | `core/breach.py :: weir_discharge` (trapezoidal weir + Villemonte submergence) |
| Volume Balance | `core/reservoir.py :: volume_balance`, `breach.sanity_checks` |
| invert H–V–A | `core/reservoir.py :: Reservoir.level` |
| Reservoir drained? | `simulate_breach` termination on `stop_fraction` |
| Sanity checks: breach opening geometry, Q(t), t_peak, volume, reservoir level(t) | `core/breach.py :: sanity_checks` |
| Go / solver verification gate | `tests/test_swe_benchmarks.py`, `cli.py verify` |

## Centre column — SPH, Delft3D, coupling

| Diagram node | Implementation |
|---|---|
| Domain decomposition, Near field & Far field, Overlap Zone | `core/coupling.py :: TransferInterface` (`overlap_start_m`, `overlap_end_m`) |
| SPH | `core/sph.py` — weakly-compressible SPH, cubic spline kernel, Tait EOS |
| SPH configuration | `core/sph.py :: SPHConfig` |
| Particle initialisation | `core/sph.py :: build_slice` (from the real DEM long profile) |
| Run SPH near field | `core/sph.py :: run_sph` |
| SPH–Delft3D Transfer Interface | `core/coupling.py :: run_near_field` &rarr; `TransferInterface` |
| Balance | `core/coupling.py :: compare_hydrographs` (peak / volume / NSE gate) |
| Delft3D setup | `core/swe2d.py :: SWE2D.__init__`, `set_still_water` |
| Delft3D mesh attribution | roughness + bathymetry assignment in `pipeline.run_scenario` |
| Delft3D-FM | `core/swe2d.py` — HLL + MUSCL + Audusse well-balanced + SSP-RK2 |
| Delft3D standalone | model configuration `grid_standalone` |
| Delft3D coupled | model configuration `coupled` (`pipeline._blended_hydrograph`) |
| Stability watch | adaptive CFL + divergence guard in `SWE2D.run`; balance gate in `compare_hydrographs` |
| Initial + Boundary | `set_still_water`, `PointSource`, transmissive edges in `_finalise` |
| Model comparison | `core/coupling.py :: build_comparison_table` &rarr; dashboard **Models** tab |

## Output extraction and hazard

| Diagram node | Implementation |
|---|---|
| Depth Grids | `SWEResult.h_max` &rarr; `rasters/depth_max.tif` |
| Velocity grids | `SWEResult.v_max` &rarr; `rasters/velocity_max.tif` |
| Arrival time + Duration | `SWEResult.arrival_s`, `duration_s` &rarr; GeoTIFFs |
| Model Comparison Table | `results.json :: model_comparison` |
| Derive Hazard Variables | `core/hazard.py :: build_hazard` |
| Hazard rating | `hazard_rating` — Defra FD2321 HR = d(v+0.5) + DF |
| Low / Moderate / Significant / Extreme | `HAZARD_CLASSES`, `classify` |
| Hazard raster + area/class | `class_areas` &rarr; `rasters/hazard_class.tif` |
| Depth-damage curves &rarr; damage class &rarr; loss estimate | `damage_fraction` (JRC Huizinga 2017), `damage_class`, `exposure.estimate_losses` |

## Right column — exposure, QC, dashboard

| Diagram node | Implementation |
|---|---|
| Exposure Data | `core/datasources.py :: fetch_exposure_osm`, `fetch_roads_geom`, `fetch_population` |
| Zonal Intersection | `core/exposure.py` |
| Critical facilities | `facility_impact` |
| Crop raster | `landcover_impact` (WorldCover cropland) |
| Road lines | `road_impact` (`polyline_length_in_mask`) |
| BLDG polygon | `building_impact` |
| POP raster | `population_impact` (WorldPop) |
| QC checks (Yes/No gate) | `exposure.qc_checks`, `manifest.json :: qc` |
| Hazard map | `overlays/hazard.png`, `rasters/hazard_*.tif` |
| Evacuation timeline table | `exposure.evacuation_timeline` &rarr; **Evacuation** tab |
| District breakdown | `exposure.district_breakdown` (geoBoundaries ADM2) |
| Impact summary | `results.json :: impact` &rarr; **Impact** tab |

## Validation and near-real-time branch

| Diagram node | Implementation |
|---|---|
| Sentinel-1 / Sentinel-2 | `core/datasources.py :: find_sentinel1`, `sentinel1_backscatter` |
| Google Earth Engine | Replaced by the **Microsoft Planetary Computer STAC API**, which serves the same Sentinel-1/2 archive with a free token and no Earth Engine account. See note below. |
| OBSERVED FLOOD EXTENT | `core/datasources.py :: sar_water_mask` (Otsu + slope masking) |
| VALIDATION (IoU / CSI / NSE) | `core/validation.py :: extent_metrics`, `nash_sutcliffe`, `kling_gupta` |
| CALIBRATION + LIVE FLOOD LAYER | `validation.validate_extent` modes `benchmark` / `context` |

**On Google Earth Engine.** The problem statement names GEE for the near-real-time
branch. GEE requires a registered, approved account and its Python client is
awkward to ship in a judged demo. The Planetary Computer exposes the identical
Sentinel-1 GRD and Sentinel-2 L2A archives as STAC + COGs with an anonymous
token, so the framework uses that instead. The interface is deliberately thin —
`find_sentinel1` / `sentinel1_backscatter` — so a GEE backend can be dropped in
behind the same two functions if the evaluators require GEE specifically.

## Dashboard and export

| Diagram node | Implementation |
|---|---|
| Dashboard 2D/3D flood map time animation | `web/index.html` — 2D: Leaflet + per-frame WGS84 PNG overlays over a DEM hillshade. 3D: three.js WebGL terrain mesh with a shader-discarded water surface, fed by `core/export.py :: export_terrain_3d`. One time slider drives whichever view is active. The 3D payload is rebuilt on demand by `core/export.py :: build_terrain3d_from_run` when absent, so it is derived from run artefacts rather than stored. |
| Scenario compare | run selector + **Models** tab |
| Large-volume tiling | windowed COG reads, compressed frame store, downsampled overlays |
| Export &rarr; .shp / .kml | `core/export.py :: export_flood_extent_shp`, `export_kml`, `export_geojson` |
| Tech stack (Docker, React, REST API, PostgreSQL, Python, GEE…) | REST API `api/main.py` (FastAPI); front end is dependency-free vanilla JS + vendored Leaflet rather than React, so it runs offline with no build step |

## Deviations from the diagram, and why

1. **GEE &rarr; Planetary Computer** — same data, no account gate. See above.
2. **PostgreSQL/PostGIS &rarr; file-based run store** — each run is a
   self-describing directory (`manifest.json`, GeoTIFF, SHP, KMZ, CSV). Nothing
   in the pipeline needs transactional storage, and a directory is far easier to
   hand to evaluators. A PostGIS loader would sit behind `core/export.py`.
3. **React &rarr; vanilla JS** — no build step, no CDN at demo time.
4. **Delft3D &rarr; an equivalent solver** — Delft3D-FM is not redistributable and
   cannot be scripted in a self-contained prototype. `core/swe2d.py` implements
   the same governing equations and numerical class (well-balanced Godunov
   finite volume with wetting/drying), and is verified against analytical
   solutions. It is described as "Delft3D-class", never as Delft3D itself.
5. **SPH is a 2D vertical slice**, not 3D — it resolves the breach jet along the
   centreline, which is what the coupling needs, at a cost that fits a demo.
