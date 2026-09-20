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

Every physical assumption is a flag, and every flag is also a field on the REST
API's `POST /api/run` body — nothing in the model is reachable from only one of
the two interfaces:

```bash
# pick a different empirical breach regression
.venv/bin/python -m damburst.cli run tehri --seed-method von_thun_gillette

# piping failure with an upstream flood inflow, channel burned from OSM waterways
.venv/bin/python -m damburst.cli run tehri --failure-mode piping     --inflow-m3s 1200 --channel-burn-m 3

# substitute audited unit rates instead of the placeholder ones
.venv/bin/python -m damburst.cli run tehri     --value-residential 1800000 --road-damage-factor 0.2

# score against a real observed flood (needs BOTH windows)
.venv/bin/python -m damburst.cli run --custom kosi --bbox ... --dam "..."     --validation-mode benchmark     --sentinel1-window 2023-08-10 2023-08-20     --baseline-window 2023-05-01 2023-05-31

# free-discharging breach, and no steep-terrain Froude cap (raw SWE)
.venv/bin/python -m damburst.cli run tehri --no-tailwater --froude-max 0
```

Run `.venv/bin/python -m damburst.cli run --help` for the full list.

---

## 3. What the framework actually computes

### 3.1 Reservoir H–V–A

Two things build this curve, and it matters which is which.

**Above the reservoir water surface**, storage is measured from the terrain:
connected flood-fill upstream of the OSM dam axis at 40 successive levels, i.e.
true DEM hypsometry, not a power-law fit.

**Below the water surface, storage is reconstructed, not measured.** GLO-30 is
a *surface* model from 2011–2015 radar: where a reservoir already existed it
records the water plate, and the drowned valley underneath is invisible. For
Tehri the terrain therefore accounts for about **4 MCM out of roughly
3,400 MCM** — nothing like enough to drive a breach. Below the plate the
framework fits the standard two-parameter power law `V = a·d^b`, with the two
parameters pinned by two external constraints:

* the **published gross capacity** (CWC / Wikidata / Wikipedia infobox), and
* the **satellite-observed surface area** at the DSM water level (ESA WorldCover).

So `a = V_published / d_crest^b`, and the curve becomes

```
V(crest) = V_published * (d_ws / d_crest)^b   +   DEM hypsometry above the plate
```

— i.e. **the published figure sets the scale of the storage** (about 95% of it
at Tehri, 3,362 of 3,540 MCM), and the terrain contributes only the small
second term. Every run records this in `results.json` under
`reservoir.derivation`: `dem_only_capacity_mcm`, `published_capacity_mcm`,
`shape_exponent_b` and whether that exponent was *solved* from the two
constraints or *defaulted* to 2.4 because they did not bracket a root.

For the avoidance of doubt: **on every preset in this repository, the reservoir
volume that drives the simulation comes overwhelmingly from the published
engineering record, not from the DEM.** What the DEM independently contributes
is the hypsometry above the water plate, the surface area that constrains the
exponent, the reservoir's planform extent and the downstream terrain the flood
is actually routed over. Closing the remaining gap needs pre-impoundment
topography or a bathymetric survey.

The impounded side is determined from the **dam's own orientation** (principal
axis of the crest line, then the higher-bed side of its normal). Splitting the
domain by grid row or column fails silently whenever the river does not run
along that axis, and the fill then escapes downstream.

> **The QC gate here is deliberately not a comparison against the published
> capacity.** That figure is an *input* to the reconstruction, so comparing the
> result against it would be circular and could never fail. The gate instead
> checks the independently-measured `dem_only_capacity_mcm` against the
> published value and reports what fraction of the storage came from each.

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

All four seed regressions are selectable with `--seed-method`; `auto` picks
Froehlich for engineered dams and the Costa–Schuster branch for natural
blockages. The natural-blockage branch applies three **judgement factors**
(wider, slower, partial scour) on top of the Froehlich seed — they are named
constants in `core/breach.py`, recorded in every run, and are calibration, not
regression.

**Sanity gates** (all reported):

* **mass-balance closure** — delegated to `reservoir.volume_balance`;
* **critical flow through the breach's own section**,
  `Q ≤ (2/3)^1.5·√g·B·H^1.5`. This is a hard physical ceiling for free
  discharge and, unlike the empirical envelopes, it is **valid at any scale**,
  so it is the gate that actually binds on a 260 m dam;
* **routed peak vs the Froehlich-1995 and Costa–Schuster-1988 envelopes.**
  Those regressions were fitted to structures under ~100 m and ~1 km³. Every
  preset here is outside that range, so the gate reports **`null` — not
  applicable — rather than a pass**, and `null` is counted as inconclusive,
  never as green. A gate that returns "pass" on a 16× exceedance is not a gate;
* **breach width-to-height ratio.**

### 3.3 2D shallow-water far field (Delft3D-FM class)

Godunov finite volume on the DEM mesh:

* HLL approximate Riemann solver
* **second-order MUSCL** reconstruction of the water-surface elevation with a
  minmod limiter, advanced by SSP-RK2 (first order available as a fallback)
* **Audusse et al. (2004)** hydrostatic reconstruction — well balanced
* Kurganov–Petrova depth desingularisation for wet/dry fronts
* semi-implicit Manning friction from the WorldCover roughness map
* adaptive CFL timestep, Numba-JIT parallel kernels
* a **steep-terrain Froude limiter**: the shallow-water equations assume a
  hydrostatic pressure distribution, which a Himalayan gorge violates. Left
  alone they integrate water into free fall and report 120–130 m/s, because
  √(2g·400 m) really is ~88 m/s — those are the equations speaking well outside
  their validity. On cells whose own bed slope exceeds 12° the velocity is
  capped at `Fr ≤ 4`, and every run reports how many cell-steps were limited.
  Flat-bed cells never enter that branch, so the analytical benchmarks in §4
  are bit-for-bit unaffected — checked, not assumed. Disable with
  `--froude-max 0`.

Velocity is reported as a distribution (median / p90 / p99 / p99.9) alongside
the single-cell maximum, with the depth, bed slope and Froude number at the
cell that set that maximum. The maximum of a dam-break velocity field is one
cell on the worst terrain in the domain; the percentiles are what an
operational reader should quote.

### 3.3b Automated adaptive model selection  — the technical novelty

A framework that owns two solvers has to answer a question most of them dodge:
**which one is valid where?** The usual answer is to run both and print the
numbers side by side, which tells a reader the models disagree but not which to
believe — and it leaves the SPH→SWE handover a free parameter. In this codebase
it was previously `min(12 × dp, …)`: the *particle spacing*, a numerical setting
dressed up as a physical overlap zone.

`core/adequacy.py` computes the answer from the flow instead. The shallow-water
equations rest on one substantive assumption — hydrostatic pressure, which needs
vertical acceleration ≪ g. Three things break it, and each is tracked
separately rather than collapsed into one opaque number:

```
NHI = max(  |grad z| / tan(theta_c)  ,  u^2*|kappa| / (g*c)  ,  |grad eta| / s_c  )
             bed slope                   curvature -> a_z        surface steepness
```

`NHI < 1` means depth-averaging is defensible; `NHI >= 1` means it is not. A
fourth test covers the plunging jet: a breach whose invert sits above the toe
free-falls that drop, and if the fall exceeds the critical depth the near field
is unambiguously non-hydrostatic.

**Three things this buys:**

1. **The particle model runs only when it earns its cost.** Tehri with a
   full-height breach: NHI peaks at 0.03 along the receiving reach and the
   breach scours to within 12 m of the riverbed (0.23× critical depth), so
   there is no plunging jet — **SPH is skipped**. The same dam as a *landslide
   blockage* leaves a 56 m residual barrier → a plunging jet at **1.3× critical
   depth** → **SPH is selected**. Same terrain, same tool, opposite decision,
   and the reason is printed.
2. **The handover locates itself**, at the first point downstream where NHI
   recovers below 1. Changing the particle spacing no longer moves it.
3. **The run scores its own trustworthiness.** After solving, the same
   criterion is applied to the solution. On the Tehri run, **73% of the flood
   volume** (54% of the wetted area) sits where the depth-averaged equations do
   not hold — the flood fills a gorge with 38° walls.

The area and volume fractions are both reported because they diverge sharply in
steep terrain and it matters which is quoted: a flood filling a gorge wets the
valley sides, so many wet cells are steep while holding little of the water.
The verdict is decided on volume.

**The two questions are different, and the framework says so.** The near-field
decision (does the breach jet need SPH?) and the far-field verdict (are the
equations valid where the flood went?) can legitimately disagree, as they do at
Tehri. A particle model at the breach would not fix a far-field validity
problem; that needs a non-hydrostatic far-field solver or a finer DEM, neither
of which is in this prototype. The run reports the gap rather than absorbing it.

> `--no-auto-model` restores manual control; the automatic verdict is still
> recorded so the override is visible.

### 3.4 SPH near field

Weakly-compressible SPH on a **vertical slice taken from the real DEM long
profile**: cubic spline kernel, Tait EOS, Monaghan artificial viscosity, XSPH,
dynamic boundary particles, Verlet integration with Shepard density
re-initialisation. It targets what a depth-averaged model structurally cannot
represent — the plunging breach jet, vertical accelerations, the non-hydrostatic
surge front.

**What it currently achieves, stated plainly.** At demo settings the slice
carries ~4,000 fluid particles at `dp = 4 m`, and on the presets it does not
reach its requested end time: the timestep collapses after 12–20 s on a
pressure instability. Every run records `completed`, `simulated_s`,
`requested_s` and `stop_reason` in `results.json`, the Models tab renders them,
and the balance gate fails on an incomplete solve. A truncated particle run is
a diagnostic of the near field's character, not a converged result, and nothing
in the reported hazard depends on it — which is why `grid_standalone` is the
primary model (§3.5).

### 3.5 Coupling and model comparison

Three configurations are produced and compared side by side:

| Model | What it is | Role |
|---|---|---|
| `grid_standalone` | 2D SWE driven by the empirical breach weir hydrograph | **primary** — hazard, exposure and loss are computed from this |
| `sph_nearfield` | particle model of the breach jet | diagnostic |
| `sph_initialised` | 2D SWE whose source is the SPH transfer hydrograph for the first seconds, weir closure thereafter | sensitivity |

**Be clear about what the third row is.** SPH runs for 12–20 s against a
3-hour simulation — around **0.2%** of it. For the remaining 99.8% the
configuration is identical to `grid_standalone`. The particle model therefore
sets the *initial condition* of the far field; it does not drive it. Calling
that "coupled" oversells it, so it is named `sph_initialised`, it is reported as
a sensitivity, and `grid_standalone` is the primary result. The one place the
difference shows is the first seconds, where SPH starts from an already-open
breach while the weir needs its formation time — at Bhakra that raised peak
depth from 27 m to 62 m, which is a reason to report it, not to build on it.

The transfer interface samples depth and depth-averaged velocity at a section
**1.5 breach-heads downstream** — a physical length scale for the non-hydrostatic
near field, not a multiple of the particle spacing. The balance gate requires
BOTH that SPH's unit discharge sits near the critical-flow ceiling AND that the
particle run reached its requested end time; a truncated solve cannot pass.

Beyond the scalar table, `validation.field_agreement` scores the far-field
configurations against each other spatially (IoU on the wet mask; NSE, KGE,
RMSE and bias on depth), because two runs can share a peak depth and still
inundate different valleys.

### 3.6 Hazard, exposure, loss

* Hazard rating **HR = d(v + 0.5) + DF** (Defra FD2320/FD2321), four classes
* Depth–damage curves from **JRC Huizinga et al. (2017)**, Asia
* Structural thresholds from **Clausen & Clark (1990)** on d&middot;v
* Zonal intersection with OSM buildings/roads/facilities, WorldPop population,
  WorldCover cropland, geoBoundaries districts
* Evacuation timeline banded by modelled first-arrival time

> Monetary loss needs asset unit values, which are an **economic input, not
> something derivable from a DEM**. They are explicit parameters, overridable
> from both the CLI (`--value-residential`, `--road-damage-factor`, …) and the
> REST API (`asset_values`), and every currency figure is tagged with the rates
> that produced it — including `road_partial_damage_factor`, which typically
> drives **over 90% of the total** and therefore must not be a literal buried in
> the accounting code. Each run prints the percentage split so the dominant
> assumption is visible before anyone quotes a rupee figure. Physical exposure
> counts are the defensible output; currency is derived.

Buildings are classified per feature from their OSM `building=*` tag onto the
matching JRC curve and unit rate; the fraction that carried a specific tag
(rather than falling back to residential) is reported. Building exposure is a
**point sample** of the depth raster at the OSM way centre, not a polygon
intersection — at 90–150 m cells the two agree for all but the largest
structures.

### 3.7 Validation against Sentinel-1

Otsu-thresholded SAR backscatter with terrain-slope masking gives the observed
water extent. Two modes, always recorded in the manifest:

* **`benchmark`** — run a real historical flood and score IoU / CSI / POD / FAR / F1
  against the SAR extent. This is the only mode that measures predictive skill.
  It needs `--sentinel1-window` bracketing the event **and** `--baseline-window`
  for a pre-event scene; without the baseline, permanent water counts as a hit
  in both layers, and the report says so instead of returning a flattering
  number. **No benchmark case has been run for this submission**, so no
  predictive skill has been measured — see §7.
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

The 3D payload, the 2D animation frames and every reported statistic are all
built from the **same** run — whichever configuration is `primary_model`, whose
name is written into `terrain3d.source_model`. They used to be wired
separately, with the animation showing `grid_standalone` while the tables
reported the coupled run, so a "the 3D view is the same solver output" claim
was comparing two different models that happened to agree closely.

Sanity check on a Tehri run: water volume in the 3D surface tracks the volume
the breach model released, and the final wetted area tracks the 2D hazard
raster, both to well under 1%. Those are now genuinely the same field sampled
two ways rather than two models agreeing.

---

## 6c. Possibility of breach

Everything above is **conditional on failure**. Risk is
`P(failure) × consequence | failure`, and quoting the consequence with no
probability attached overstates what the framework knows.

`core/failure_probability.py` supplies the missing factor where it can:
**P(overtopping)** is computed by routing a Gumbel design-flood family through
the same level-pool equation the breach model uses, against a spillway rating.
Everything else — internal erosion, seismic, structural — falls back to
ICOLD / Foster et al. (2000) population base rates and is labelled as a base
rate, not an estimate.

The honest result is the sensitivity, not the number. Sweeping the one input
that is not in any open dataset:

| Spillway capacity | First overtopping | Annual P |
|---|---|---|
| 2.0 × design | > 10,000 y | < 1e-4 |
| 1.0 × design | > 10,000 y | < 1e-4 |
| 0.5 × design | 500 y | 2.0e-3 |
| 0.25 × design | 100 y | 1.0e-2 |

Two orders of magnitude, driven entirely by data the framework does not have:
spillway geometry lives in the CWC National Register of Large Dams, not in
OpenStreetMap. Starting the reservoir at FRL rather than the spillway sill
moves the 0.5× case from a 500-year to a 100-year event.

Note also that P(overtopping) is **not** P(failure): an embankment erodes when
overtopped, a concrete gravity dam may not. The conditional term is explicit.

Supply `--mean-annual-flood` to enable this; without it the run makes no
probability claim at all, which is the correct default. Full formulation,
including the event tree and the fragility/reliability-index treatment, is in
[`docs/MATHEMATICAL_FORMULATION.md`](docs/MATHEMATICAL_FORMULATION.md) Part B.

---

## 7. Honest limitations

1. **Reservoir storage below the water plate is reconstructed, not measured**
   (§3.1). On every preset, gross capacity is set by the published engineering
   record; the DEM contributes ~0.1% of Tehri's volume. This is the single
   largest external dependency in the framework.
2. **Routed peak discharge is not independently corroborated at this scale.**
   It sits below the critical-flow ceiling through its own breach section
   (the only scale-valid check available) but 3–16× above the Froehlich-1995
   and Costa–Schuster-1988 envelopes, which cannot arbitrate for a 260 m,
   3.4 km³ structure. Reported as inconclusive, not as a pass.
3. **Front arrival is biased ~5–7% late** at practical resolutions (§4). The
   bias is negative, so the model predicts arrival slightly *late* — the wrong
   direction for evacuation planning, which is why it is stated here.
4. **The shallow-water assumption is violated on the steepest terrain.** The
   Froude limiter (§3.3) bounds the consequence; it does not make the equations
   valid there. Treat velocities on >12° slopes as indicative.
5. **SPH does not converge on the presets.** It truncates on a pressure
   instability after 12–20 s (§3.4), it is a 2D vertical slice rather than full
   3D, and no reported hazard figure depends on it.
6. **No predictive skill has been measured.** Validation runs in `context` mode
   only; `benchmark` mode is implemented and reachable but has not been
   exercised against a real gauged flood (§3.7).
7. **Monetary loss rests on assumed unit rates** (§3.6), and one of them — the
   road partial-damage factor — typically drives over 90% of the total.
8. **OSM completeness varies.** Building counts in rural Uttarakhand are a
   lower bound; the QC panel flags the people-per-mapped-building ratio when it
   becomes absurd, and it does.
9. **Manning roughness** is a land-cover lookup (Chow 1959; Arcement &
   Schneider), not calibrated to gauged events at these sites.
10. **Tailwater submergence barely matters at these heads.** The Villemonte
    correction is applied and engages on a few hundred routing steps, but with
    170 m of head the submergence ratio rarely reaches the modular limit near
    the peak, so it changes peak outflow by well under 1%.

For a line-by-line audit of which reported numbers are measured, assumed or
reconstructed, see [`docs/RESULTS_AUDIT.md`](docs/RESULTS_AUDIT.md).
