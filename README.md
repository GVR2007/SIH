# DamBurst

**Dam Break Inundation Modelling Using Hydrodynamic Modelling of any River**
SIH 2026 &middot; Problem Statement **26161** &middot; National Technical Research Organisation (NTRO)

A working modelling framework that simulates dam-break / river-blockage failure,
routes the resulting flood wave with **both** a Smooth Particle Hydrodynamics
near-field model and a Delft3D-class 2D hydrodynamic far-field model, compares
the two, and turns the result into hazard, exposure and loss products with a
dashboard and `.shp` / `.kml` export.

Everything runs on **real, live, open data**. There is no synthetic terrain, no
invented dam, no placeholder flood extent.

---

## 1. Data sources (all public, no registration)

| Layer | Source | Licence |
|---|---|---|
| Elevation | **Copernicus GLO-30 DSM** COGs, AWS Open Data `copernicus-dem-30m` | Free/open |
| Land cover &rarr; Manning *n* | **ESA WorldCover v200** (10 m) via Microsoft Planetary Computer | CC-BY 4.0 |
| Population | **WorldPop 2020** (India) | CC-BY 4.0 |
| Dam geometry, buildings, roads, settlements, critical facilities | **OpenStreetMap** via Overpass API | ODbL |
| Dam engineering record (cross-check only) | **Wikidata** + **Wikipedia** infobox | CC0 / CC-BY-SA |
| Districts | **geoBoundaries** gbOpen ADM2 | CC-BY 4.0 |
| Observed water extent | **Sentinel-1 RTC** via Microsoft Planetary Computer (free SAS token) | Free/open |
| Basemap / 3D terrain texture | **Sentinel-2 L2A** true colour via Microsoft Planetary Computer | Free/open |

Rasters are read as **windowed COG range requests**, so a run downloads only the
few MB it actually needs. Everything is cached under `data/cache/`, so after the
first run the demo works offline.

If a source is unreachable the pipeline **raises** — it never silently
substitutes a guess.

---

## 2. Install and run

```bash
cd damburst
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

List the built-in study areas:

```bash
.venv/bin/python -m damburst.cli list
```

Run a scenario end to end (Tehri Dam, Bhagirathi, Uttarakhand):

```bash
.venv/bin/python -m damburst.cli run tehri --quality fast
```

Start the dashboard:

```bash
.venv/bin/python -m damburst.cli serve
```

Verify the solver against analytical solutions:

```bash
.venv/bin/python -m damburst.cli verify
```

Any OpenStreetMap-mapped dam works, not just the presets:

```bash
.venv/bin/python -m damburst.cli run --custom mydam --bbox 76.30 31.20 76.80 31.60 --dam "Bhakra"
```

---

## 3. What the framework actually computes

### 3.1 Reservoir H–V–A from the DEM

The stage–area–volume curve is **derived from the terrain**, by connected
flood-fill upstream of the OSM dam axis at 40 successive levels — not assumed
from a power law and not typed in from a gazetteer.

The impounded side is determined from the **dam's own orientation** (principal
axis of the crest line, then the higher-bed side of its normal). Splitting the
domain by grid row or column fails silently whenever the river does not run
along that axis, and the fill then escapes downstream.

> **Known limitation, reported in every run.** GLO-30 is a *surface* model from
> 2011–2015 radar. Where a reservoir already existed, the DEM records the water
> surface, not the drowned valley floor. DEM hypsometry therefore gives a
> **lower bound** on gross storage. The QC panel compares against the published
> capacity and states this explicitly. Closing the gap needs pre-impoundment
> topography or a bathymetric survey.

### 3.2 Breach formation

Empirical seed regressions, selected by barrier type:

* **Froehlich (2008)** — average breach width and formation time (default for engineered dams)
* **Von Thun & Gillette (1990)**
* **MacDonald & Langridge-Monopolis (1984)**
* **Costa & Schuster (1988)** — landslide-dam peak-discharge envelope (default for natural blockages)

Failure modes: overtopping, piping (with an orifice phase before roof collapse),
progressive erosion, instantaneous. Growth laws: linear, sine (HEC-RAS), erosion
(t^0.5). Outflow is a trapezoidal broad-crested weir with Villemonte submergence
correction, coupled to level-pool depletion on an adaptive timestep.

**Sanity gates** (all reported): mass-balance closure, routed peak vs the
Froehlich-1995 and Costa–Schuster-1988 published envelopes, and breach
width-to-height ratio.

### 3.3 2D shallow-water far field (Delft3D-FM class)

Godunov finite volume on the DEM mesh:

* HLL approximate Riemann solver
* **second-order MUSCL** reconstruction of the water-surface elevation with a
  minmod limiter, advanced by SSP-RK2 (first order available as a fallback)
* **Audusse et al. (2004)** hydrostatic reconstruction — well balanced
* Kurganov–Petrova depth desingularisation for wet/dry fronts
* semi-implicit Manning friction from the WorldCover roughness map
* adaptive CFL timestep, Numba-JIT parallel kernels

### 3.4 SPH near field

Weakly-compressible SPH on a **vertical slice taken from the real DEM long
profile**: cubic spline kernel, Tait EOS, Monaghan artificial viscosity, XSPH,
dynamic boundary particles, Verlet integration with Shepard density
re-initialisation. It resolves what a depth-averaged model structurally cannot —
the plunging breach jet, vertical accelerations, the non-hydrostatic surge front.

### 3.5 Coupling and model comparison

Three configurations are produced and compared side by side:

| Model | What it is |
|---|---|
| `grid_standalone` | 2D SWE driven by the empirical breach weir hydrograph |
| `sph_nearfield` | particle model of the breach jet |
| `coupled` | 2D SWE driven by the hydrograph SPH actually delivers across the transfer section |

The transfer interface samples depth and depth-averaged velocity at a section
downstream of the breach; the overlap zone is the reach between. A balance gate
compares SPH and weir peak, volume and NSE.

### 3.6 Hazard, exposure, loss

* Hazard rating **HR = d(v + 0.5) + DF** (Defra FD2320/FD2321), four classes
* Depth–damage curves from **JRC Huizinga et al. (2017)**, Asia
* Structural thresholds from **Clausen & Clark (1990)** on d&middot;v
* Zonal intersection with OSM buildings/roads/facilities, WorldPop population,
  WorldCover cropland, geoBoundaries districts
* Evacuation timeline banded by modelled first-arrival time

> Monetary loss needs asset unit values, which are an **economic input, not
> something derivable from a DEM**. They are explicit, overridable parameters and
> every currency figure is tagged with the rates that produced it. Physical
> exposure counts are the defensible output; currency is derived.

### 3.7 Validation against Sentinel-1

Otsu-thresholded SAR backscatter with terrain-slope masking gives the observed
water extent. Two modes, always recorded in the manifest:

* **`benchmark`** — run a real historical flood and score IoU / CSI / POD / FAR / F1
  against the SAR extent. This is the only mode that measures predictive skill.
* **`context`** — show the pre-event water body so the dashboard can separate
  permanent water from newly inundated land. **Not a skill score.**

A dam-break scenario is hypothetical; there is no satellite image of a flood
that has not happened, and the framework does not pretend otherwise.

---

## 4. Solver verification

`python -m damburst.cli verify` runs the analytical benchmarks:

| Test | Result |
|---|---|
| Lake at rest, irregular wet/dry bathymetry | max &#124;&eta;&minus;level&#124; ~1e-15 m, max &#124;u&#124; ~1e-13 m/s |
| Closed-basin mass conservation | relative error 0.0 |
| Ritter dry-bed dam break (dx = 0.5 m, t = 15 s) | L1(h) = 0.005 m = **0.05 %** of h&#8320; |
| Ritter grid convergence | rate ~1.0 |
| Ritter front position | **&minus;7.4 %** at dx = 0.5 m, &minus;5.6 % at dx = 0.25 m |

The front-position bias is a known property of shock-capturing schemes at a dry
front and it is **negative — the model predicts arrival slightly late**. That
matters for evacuation timing, so it is stated here rather than hidden; second
order more than halves it relative to first order (&minus;11.2 % &rarr; &minus;7.4 %).

---

## 5. Outputs

```
runs/<run_id>/
  manifest.json      provenance, QC gates, every parameter used
  results.json       hydrographs, hazard, impact, model comparison
  rasters/           depth_max, velocity_max, hazard_rating, hazard_class,
                     arrival_time_s, duration_s, dem_conditioned  (GeoTIFF)
  vectors/           flood_extent.shp + .prj, flood_extent.kmz, .geojson
  overlays/          WGS84 PNGs + per-frame animation tiles, plus basemap.png
                     (Sentinel-2 true colour, relief-shaded; DEM hillshade
                      fills granule gaps so the map never goes blank)
  terrain3d.json     elevation grid + per-frame water depths for the 3D view
  tables/            breach hydrograph, settlements, evacuation timeline,
                     districts, critical facilities  (CSV)
```

---

## 6. Deliverable coverage

| PS deliverable | Where |
|---|---|
| (i) Generalised dam-break / river-blockage framework with SPH **and** a Delft3D-class model, loss and damage | `core/breach.py`, `core/sph.py`, `core/swe2d.py`, `core/coupling.py`, `core/hazard.py`, `core/exposure.py` |
| (ii) Customised tool, scenarios from different input datasets | `pipeline.py`, `scenarios.py`, `cli.py`, any OSM dam + bbox |
| (iii) Dashboard GUI (2D map **and** 3D terrain), large data, `.shp` / `.kml` export | `api/main.py`, `web/index.html`, `core/export.py` |
| (iv) Near-real-time flood analysis from open satellite data | `core/datasources.py` (Sentinel-1 via Planetary Computer), `core/validation.py` |
| (v) Real Indian river and dam, open source | Tehri / Koteshwar / Bhakra / Srisailam presets, live OSM + Copernicus + WorldPop |

See [`docs/APPROACH_MAPPING.md`](docs/APPROACH_MAPPING.md) for the node-by-node
mapping from the technical-approach diagram to the code.

---

## 6b. Imagery basemap

The 2D map and the 3D terrain are both draped in **real Sentinel-2 L2A true
colour** (B04/B03/B02), mosaicked from the least-cloudy granules in the search
window and relief-shaded with the DEM hillshade so terrain still reads.

Three things this has to get right, and does:

* **Mosaic, don't pick.** One S2 granule is ~110 km square and a basin-sized
  bbox straddles several, so a single scene covers a fraction of the grid. The
  Tehri domain takes 4 granules, all under 1% cloud.
* **Gaps fall back to hillshade**, never to black. Coverage is 87.6% on the
  Tehri domain; the remainder is synthetic relief, and the seam is not obvious.
* **Coarse read, then upsample.** This is a texture, not an analysis layer.
  Reading 10 m bands at model resolution over a basin took ~9 minutes; reading
  decimated and upsampling takes ~170 s, and the mosaic is **cached** so every
  later run over the same area is instant (measured: 169 s &rarr; 0.0 s).

## 6a. 3D view

The dashboard has a **2D map / 3D terrain** toggle. The 3D view is rendered in
WebGL (three.js, vendored locally &mdash; no CDN) from `terrain3d.json`, which
ships the actual numbers rather than a picture of them:

* elevation as Float32, exact, downsampled to ~224 cells on the long axis;
* each depth frame as UInt8 quantised against a shared cap &mdash; 30 frames of
  Float32 at model resolution would be tens of megabytes.

The terrain is a lit vertex-coloured mesh; the water is a separate mesh whose
vertices sit at `bed + depth`, drawn with a shader that **discards** dry
fragments. A plain transparent sheet would film the whole valley and misread as
total inundation.

The payload is **derived, never stored as a fixture**. `terrain3d.json` is
written during a run, but if it is absent the API rebuilds it on first request
from the artefacts every finished run already holds &mdash; the conditioned DEM,
the hazard raster, the solver's depth frames and the dam geometry in
`results.json`. Runs produced before the 3D view existed therefore gain it
without re-simulating: measured rebuild time is **under 0.1 s**. Satellite
colour is attached only if the Sentinel-2 mosaic for that area is already
cached, because mosaicking granules inside an HTTP request would block the
client for minutes; otherwise the terrain falls back to elevation-tinted relief.

Controls: orbit / zoom / pan, a vertical-exaggeration slider (1&ndash;6&times;,
since real relief is nearly flat at valley scale), and a depth / hazard-class
surface switch. The time slider drives whichever view is on screen.

Sanity check on the Tehri run &mdash; water volume in the 3D surface grows to
**3,180 MCM** against **3,154 MCM** released by the breach model, and the final
wetted area is **80.0 km&sup2;** against **80.6 km&sup2;** from the 2D hazard
raster. The 3D view is the same solver output, not a separate illustration.

---

## 7. Honest limitations

1. **Reservoir bathymetry** is invisible to a DSM (§3.1). DEM storage is a lower bound.
2. **Front arrival is biased ~5–7 % late** at practical resolutions (§4).
3. **SPH is a 2D vertical slice**, not full 3D — it captures the jet structure
   along the breach centreline, not lateral spreading inside the breach.
4. **Validation `context` mode is not a skill score.** Only `benchmark` mode,
   run against a real observed flood, measures accuracy.
5. **Monetary loss rests on assumed unit rates** (§3.6).
6. **OSM completeness varies.** Building counts in rural Uttarakhand are a lower
   bound; the framework reports what is mapped, and says so.
7. **Manning roughness** is a land-cover lookup (Chow 1959; Arcement & Schneider),
   not calibrated to gauged events at these sites.
