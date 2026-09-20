# Technical-approach diagram &rarr; code mapping

Every node of the SIH technical-approach flowchart, and where it is implemented.

**How to read the Status column.** A mapping table is worthless if it points at
code the pipeline never calls, so every row carries one of:

| Status | Meaning |
|---|---|
| **live** | Executed on every run with default settings. |
| **opt-in** | Executed when the relevant scenario flag is set; reachable from the CLI and the REST API. |
| **available** | Implemented and tested, but not wired into the default pipeline. Not evidence of a working node on a default run. |

Rows previously marked as implementing a node while being unreachable have been
either wired in or demoted to `available`; see `docs/RESULTS_AUDIT.md` §10.

## Left column — inputs, barrier, breach

| Diagram node | Implementation | Status |
|---|---|---|
| DEM / Hydrology / Dam / Satellite | `core/datasources.py` — Copernicus GLO-30, OSM, WorldPop, WorldCover, Sentinel-1 | **live** |
| Input Data QC & Harmonization | `core/dem.py :: qc_report`, `core/datasources.py :: make_grid` (UTM reprojection) | **live** |
| Spatial Database | `data/cache/` — content-hashed cache of every fetched layer | **live** |
| DEM conditioning | `core/dem.py :: condition` (void fill &rarr; channel burn &rarr; priority-flood sink fill) | **live** (burn is opt-in) |
| Channel geometry & bathymetry | `core/dem.py :: burn_channel`, `d8_flow_accumulation`, `steepest_descent_path` | **opt-in** — `--channel-burn-m`. `pipeline._channel_mask` builds the mask from OSM waterway centrelines, falling back to `d8_flow_accumulation`. `steepest_descent_path` is **live** (SPH long profile). |
| Computational domain & mesh | `core/datasources.py :: make_grid`, `Grid` | **live** |
| Roughness map | `core/dem.py :: roughness_from_worldcover` (ESA WorldCover &rarr; Manning *n*) | **live** |
| Dam record type, height, crest, spillway | `core/datasources.py :: dam_dossier` (OSM + Wikidata + Wikipedia) | **live** |
| Reservoir H–V–A curve + Loading Level | `core/reservoir.py :: hva_from_dem`, `hybrid_hva`, `detect_water_surface` | **live** — note `hybrid_hva` anchors gross storage on the PUBLISHED capacity; see README §3.1 |
| Inflow Q_in(t) + tailwater rating | `core/reservoir.py :: inflow_hydrograph`; tailwater in `breach.weir_discharge` | inflow **opt-in** (`--inflow-m3s`); tailwater **live** via `breach.TailwaterRating` (Manning normal depth), disable with `--no-tailwater` |
| Barrier type &rarr; Engineered / Natural | `core/breach.py :: seed_breach` (`barrier_type`) | **live** |
| Empirical Seed | `froehlich_2008`, `von_thun_gillette`, `macdonald_langridge`, `costa_schuster_peak` | Froehlich/Costa–Schuster **live** via `--seed-method auto`; all four selectable with `--seed-method` |
| Scenario: Dam Break / River Blockage / Water Release | `pipeline.py :: Scenario`, `scenarios.py` | **live** |
| Failure Mode: Partial / Progressive Erosion / Piping / Overtopping / Instantaneous | `core/breach.py :: FAILURE_MODES`, `breach_section` | **live** |
| Adaptive &Delta;t | `core/breach.py :: simulate_breach` (storage- and formation-limited step) | **live** |
| Geometry growth law | `core/breach.py :: growth_fraction` (linear / sine / erosion) | **live** |
| head H &rarr; Q_breach | `core/breach.py :: weir_discharge` (trapezoidal weir + Villemonte submergence) | **live** — Villemonte now engages, because a tailwater rating is supplied; `orifice_discharge` is **live** in `piping` mode before roof collapse |
| Volume Balance | `core/reservoir.py :: volume_balance`, `breach.sanity_checks` | **live** — `sanity_checks` delegates to `volume_balance`, so there is one implementation |
| invert H–V–A | `core/reservoir.py :: Reservoir.level` | **live** |
| Reservoir drained? | `simulate_breach` termination on `stop_fraction` | **live** |
| Sanity checks: breach opening geometry, Q(t), t_peak, volume, reservoir level(t) | `core/breach.py :: sanity_checks` | **live** — adds a scale-independent critical-flow ceiling |
| Go / solver verification gate | `tests/test_swe_benchmarks.py`, `cli.py verify` | **live** |

## Centre column — SPH, Delft3D, coupling

| Diagram node | Implementation | Status |
|---|---|---|
| Domain decomposition, Near field & Far field, Overlap Zone | `core/coupling.py :: TransferInterface` (`overlap_start_m`, `overlap_end_m`) | **live** — transfer section now placed at ~1.5x the breach head, not at a multiple of the particle spacing |
| SPH | `core/sph.py` — weakly-compressible SPH, cubic spline kernel, Tait EOS | **live** |
| SPH configuration | `core/sph.py :: SPHConfig` | **live** |
| Particle initialisation | `core/sph.py :: build_slice` (from the real DEM long profile) | **live** |
| Run SPH near field | `core/sph.py :: run_sph` | **live** — reports `completed` / `stop_reason`; a truncated run is flagged in the Models tab |
| SPH–Delft3D Transfer Interface | `core/coupling.py :: run_near_field` &rarr; `TransferInterface` | **live** |
| Balance | `core/coupling.py :: compare_hydrographs` (peak / volume / NSE gate) | **live** — the gate requires BOTH the critical-flow ratio and `completed` |
| Delft3D setup | `core/swe2d.py :: SWE2D.__init__`, `set_still_water` | **live** |
| Delft3D mesh attribution | roughness + bathymetry assignment in `pipeline.run_scenario` | **live** |
| Delft3D-FM | `core/swe2d.py` — HLL + MUSCL + Audusse well-balanced + SSP-RK2 | **live** |
| Delft3D standalone | model configuration `grid_standalone` | **live** — this is the PRIMARY configuration |
| Delft3D coupled | model configuration `coupled` (`pipeline._blended_hydrograph`) | **live**, renamed `sph_initialised`: SPH supplies the source for the first 12&ndash;20 s only (~0.2% of the run), so it is a sensitivity, not a driver |
| Stability watch | adaptive CFL + divergence guard in `SWE2D.run`; balance gate in `compare_hydrographs` | **live** — plus a steep-terrain Froude limiter with per-run statistics |
| Initial + Boundary | `set_still_water`, `PointSource`, transmissive edges in `_finalise` | **live** |
| Model comparison | `core/coupling.py :: build_comparison_table` &rarr; dashboard **Models** tab | **live** — plus `validation.field_agreement` (IoU / NSE / KGE / RMSE between configurations) |

## Output extraction and hazard

| Diagram node | Implementation | Status |
|---|---|---|
| Depth Grids | `SWEResult.h_max` &rarr; `rasters/depth_max.tif` | **live** |
| Velocity grids | `SWEResult.v_max` &rarr; `rasters/velocity_max.tif` | **live** — percentiles reported alongside the max |
| Arrival time + Duration | `SWEResult.arrival_s`, `duration_s` &rarr; GeoTIFFs | **live** |
| Model Comparison Table | `results.json :: model_comparison` | **live** |
| Derive Hazard Variables | `core/hazard.py :: build_hazard` | **live** |
| Hazard rating | `hazard_rating` — Defra FD2321 HR = d(v+0.5) + DF | **live** |
| Low / Moderate / Significant / Extreme | `HAZARD_CLASSES`, `classify` | **live** |
| Hazard raster + area/class | `class_areas` &rarr; `rasters/hazard_class.tif` | **live** |
| Depth-damage curves &rarr; damage class &rarr; loss estimate | `damage_fraction` (JRC Huizinga 2017), `damage_class`, `exposure.estimate_losses` | **live** — curve chosen per building from its OSM `building=*` tag |

## Right column — exposure, QC, dashboard

| Diagram node | Implementation | Status |
|---|---|---|
| Exposure Data | `core/datasources.py :: fetch_exposure_osm`, `fetch_roads_geom`, `fetch_population` | **live** |
| Zonal Intersection | `core/exposure.py` | **live** |
| Critical facilities | `facility_impact` | **live** |
| Crop raster | `landcover_impact` (WorldCover cropland) | **live** |
| Road lines | `road_impact` (`polyline_length_in_mask`) | **live** |
| BLDG polygon | `building_impact` | **live** — POINT SAMPLE at the OSM way centre, not a polygon intersection; stated in the output |
| POP raster | `population_impact` (WorldPop) | **live** |
| QC checks (Yes/No gate) | `exposure.qc_checks`, `manifest.json :: qc` | **live** — gates are now pass / fail / inconclusive |
| Hazard map | `overlays/hazard.png`, `rasters/hazard_*.tif` | **live** |
| Evacuation timeline table | `exposure.evacuation_timeline` &rarr; **Evacuation** tab | **live** |
| District breakdown | `exposure.district_breakdown` (geoBoundaries ADM2) | **live** |
| Impact summary | `results.json :: impact` &rarr; **Impact** tab | **live** |

## Validation and near-real-time branch

| Diagram node | Implementation | Status |
|---|---|---|
| Sentinel-1 / Sentinel-2 | `core/datasources.py :: find_sentinel1`, `sentinel1_backscatter` | **live** |
| Google Earth Engine | Replaced by the **Microsoft Planetary Computer STAC API**, which serves the same Sentinel-1/2 archive with a free token and no Earth Engine account. See note below. | **live** (substitution) |
| OBSERVED FLOOD EXTENT | `core/datasources.py :: sar_water_mask` (Otsu + slope masking) | **live** |
| VALIDATION (IoU / CSI / NSE) | `core/validation.py :: extent_metrics`, `nash_sutcliffe`, `kling_gupta` | `extent_metrics` **opt-in** (`--validation-mode benchmark`); `nash_sutcliffe`/`kling_gupta`/`rmse` **live** via `field_agreement` in the model comparison |
| CALIBRATION + LIVE FLOOD LAYER | `validation.validate_extent` modes `benchmark` / `context` | `context` **live**; `benchmark` **opt-in** and requires `--baseline-window` for an honest score |

**On Google Earth Engine.** The problem statement names GEE for the near-real-time
branch. GEE requires a registered, approved account and its Python client is
awkward to ship in a judged demo. The Planetary Computer exposes the identical
Sentinel-1 GRD and Sentinel-2 L2A archives as STAC + COGs with an anonymous
token, so the framework uses that instead. The interface is deliberately thin —
`find_sentinel1` / `sentinel1_backscatter` — so a GEE backend can be dropped in
behind the same two functions if the evaluators require GEE specifically.

## Dashboard and export

| Diagram node | Implementation | Status |
|---|---|---|
| Dashboard 2D/3D flood map time animation | `web/index.html` — 2D: Leaflet + per-frame WGS84 PNG overlays over a DEM hillshade. 3D: three.js WebGL terrain mesh with a shader-discarded water surface, fed by `core/export.py :: export_terrain_3d`. One time slider drives whichever view is active. The 3D payload is rebuilt on demand by `core/export.py :: build_terrain3d_from_run` when absent, so it is derived from run artefacts rather than stored. | **live** |
| Scenario compare | run selector + **Models** tab | **live** |
| Large-volume tiling | windowed COG reads, compressed frame store, downsampled overlays | **live** |
| Export &rarr; .shp / .kml | `core/export.py :: export_flood_extent_shp`, `export_kml`, `export_geojson` | **live** |
| Tech stack (Docker, React, REST API, PostgreSQL, Python, GEE…) | REST API `api/main.py` (FastAPI); front end is dependency-free vanilla JS + vendored Leaflet rather than React, so it runs offline with no build step | **live** |

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



---

## Beyond the diagram: adaptive model selection

The technical-approach diagram treats the choice of near-field model as given:
SPH runs, its output crosses a transfer interface, the grid model takes over.
`core/adequacy.py` adds the decision the diagram assumes away.

| Node (new) | Implementation | Status |
|---|---|---|
| Non-hydrostatic index NHI(x) | `adequacy.profile_nhi`, `adequacy.adequacy_field` | **live** |
| Decide whether the near field needs a non-hydrostatic model | `adequacy.select_models` | **live**, override with `--no-auto-model` / `--no-sph` |
| Plunging-jet test at the breach | `select_models`, plunge / critical-depth ratio | **live** |
| Cross-valley relief in the flood corridor | `adequacy.corridor_relief` | **live** |
| Locate the transfer section from the physics | `adequacy.locate_transfer_section` | **live**, replaces the old `12 x dp` rule |
| Post-run adequacy of the solution produced | `adequacy.adequacy_field` -> `rasters/model_adequacy_nhi.tif`, **Model choice** tab | **live** |

## Beyond the diagram: possibility of breach

The diagram, and the framework as originally written, computes consequences
CONDITIONAL on failure. `core/failure_probability.py` supplies the other factor.

| Node (new) | Implementation | Status |
|---|---|---|
| Spillway rating | `failure_probability.Spillway`, `default_spillway` | **opt-in** (`--spillway-crest-length`, or a flagged placeholder) |
| Design-flood family Q_T | `gumbel_quantile`, `design_hydrograph` | **opt-in** (`--mean-annual-flood`) |
| Route the flood -> P(overtopping) | `route_flood`, `overtopping_probability` | **opt-in**, reuses `reservoir.route_step` |
| Event tree over failure mechanisms | `failure_probability_report` | **opt-in** |
| Fragility / reliability index | `lognormal_fragility`, `reliability_index` | **available** - formulation implemented, no site-specific parameters exist for these dams |
| Base rates where nothing can be computed | `BASE_RATES_PER_YEAR` (ICOLD / Foster et al. 2000) | **live** when the block runs, and labelled as a population average |

Formulation: [`MATHEMATICAL_FORMULATION.md`](MATHEMATICAL_FORMULATION.md) Part B.

---

## Functions NOT wired into the default pipeline

Listed explicitly so this document cannot be read as claiming more than the code
does. These are implemented and importable, but a default run does not call
them, so they are not evidence for a working node:

| Function | Why it exists | Why it is not called |
|---|---|---|
| `breach.scenario_matrix` | Runs the failure-mode ensemble (overtopping / piping / instantaneous) in one call | The dashboard compares MODEL configurations, not failure modes. Run the CLI three times with `--failure-mode` to build the ensemble. Its docstring no longer claims the dashboard uses it. |
| `reservoir.analytic_reservoir` | Power-law H–V–A from gazetteer capacity + dam height alone | Only needed when there is no DEM at all; the hybrid path covers every preset. |
| `reservoir.route_step` | One explicit continuity step | `simulate_breach` inlines the same two lines inside its adaptive-dt loop. |
| `sph.ritter_reference` | Ritter analytical solution for checking the particle model | The SWE Ritter benchmark in `tests/` is the one that runs; the SPH check is manual. |
| `swe2d.add_baseflow` | Pre-wet the channel with a low-flow depth | Dam-break runs start from a dry downstream bed by design. Useful for a river-blockage scenario with antecedent flow. |

Everything else in `core/` is reached by a default run or by a documented flag.
