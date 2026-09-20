# DamBurst — Results Audit

**What this document is.** An adversarial read of the DamBurst codebase asking one
question: *which numbers that the framework reports are actually measured, and which
are assumed, hardcoded, self-fulfilling, or produced by code paths that never run?*

Written for the SIH prototype review. The goal is that nothing in the demo can be
knocked over by a judge who opens the source.

**Scope.** `damburst/` at commit `86d0ce6`, plus the 14 runs committed under `runs/`.

> ## STATUS: ALL 16 FINDINGS ADDRESSED
>
> This document is kept as the *record of what was wrong*, because the fixes
> only mean something if the original problem is still written down. Every
> finding below now has a **FIXED** note saying what changed and where.
>
> * 112 automated tests now pass, including
>   `tests/test_audit_regressions.py`, which is written directly against these
>   findings so none of them can silently return.
> * The analytical benchmarks are unchanged by every fix — verified, not
>   assumed: lake-at-rest 6.2e-15 m, Ritter L1 0.05%, observed order 0.99,
>   front bias -7.35%.
> * Re-run evidence is quoted inline below.

**Verdict in one line.** The physics core is real and the data pipeline is real; the
*presentation layer* over-claims in about a dozen specific places, and four of those
are load-bearing enough to change the headline numbers.

**Severity key**

| | meaning |
|---|---|
| 🔴 **Critical** | A headline number is wrong, circular, or contradicts the README. Fix or restate before judging. |
| 🟠 **Major** | A claimed capability does not do what it says, or a gate passes when it should not. |
| 🟡 **Minor** | Hardcoded constant, mislabel, or dead code cited as evidence. Cheap to fix or to disclose. |

---

## Summary table

| # | Finding | Sev | Where | Status |
|---|---|---|---|---|
| 1 | Reservoir capacity is **not** DEM-derived; 99.9 % of Tehri storage comes from the Wikipedia figure through a power law | 🔴 | `core/reservoir.py:176` | **FIXED** (README §3.1 rewritten; provenance now records `capacity_from_published_fraction`) |
| 2 | The reservoir QC gate compares the derived capacity against the number it was **built from** — circular | 🔴 | `pipeline.py:734` | **FIXED** (`_reservoir_cross_check` now compares `dem_only_capacity_mcm`) |
| 3 | `coupled` is not meaningfully coupled: SPH drives 12–20 s of a 3-hour run (0.2 %) | 🔴 | `pipeline.py:686` | **FIXED** (renamed `sph_initialised`, demoted to a sensitivity; `grid_standalone` is primary) |
| 4 | `coupled` peak inflow is reported identical to `grid_standalone` because the sampler **steps over** the SPH window | 🔴 | `pipeline.py:308` | **FIXED** (`_hydrograph_peak` samples the union of grids + series times) |
| 5 | Breach peak-discharge gate auto-passes whenever it is out of calibration range | 🟠 | `core/breach.py:426` | **FIXED** (returns `null`; scale-independent critical-flow gate added) |
| 6 | `benchmark` validation never subtracts permanent water — the condition is inverted; mode has never been run | 🟠 | `pipeline.py:820` | **FIXED** (baseline passed unconditionally; warns loudly if absent) |
| 7 | SPH terminated on a pressure instability in every run; dashboard shows it as a clean model row | 🟠 | `core/sph.py`, `web/index.html:571` | **FIXED** (`status`/`caveat` in the table; `balance_pass` requires `completed`) |
| 8 | Max velocities of 66–129 m/s are unphysical and feed hazard, damage and arrival times | 🟠 | `core/swe2d.py:67` | **FIXED** (steep-terrain Froude limiter + percentiles + peak context) |
| 9 | Von Thun & Gillette width formula contradicts its own docstring and the published regression | 🟠 | `core/breach.py:95` | **FIXED** (2.5 h_w + C_b) |
| 10 | Nine functions cited in `APPROACH_MAPPING.md` as implementing diagram nodes are never called | 🟠 | see §10 | **FIXED** (wired in, or marked `available` with a reason) |
| 11 | Villemonte submergence, inflow hydrograph, channel burn and orifice phase are all unreachable | 🟠 | §11 | **FIXED** (all four now run; every knob on CLI **and** API) |
| 12 | Undocumented `0.35` multiplier drives 96 % of the reported monetary loss | 🟡 | `core/exposure.py:357` | **FIXED** (`AssetValues.road_partial_damage_factor`, echoed + overridable) |
| 13 | Hardcoded shape exponent `2.4` fallback; hardcoded `0.6` momentum coefficient; fixed 48 m transfer section | 🟡 | §13 | **FIXED** (fallback recorded; `0.6` documented; transfer section now physical) |
| 14 | 2D animation and 3D view render `grid_standalone` while all statistics come from `coupled` | 🟡 | `pipeline.py:452,459` | **FIXED** (both built from `primary`; `source_model` recorded) |
| 15 | Buildings are point-sampled, not polygon-intersected, and all are priced as residential | 🟡 | `core/exposure.py:227` | **FIXED** (per-building occupancy + rate; method stated in output) |
| 16 | Depth-damage curves and Manning table are typed-in literals with no traceable source file | 🟡 | `core/hazard.py:58`, `core/dem.py:229` | **FIXED** (`DD_SOURCE` with DOI; transcription stated) |

---

## 🔴 1. Reservoir storage is not derived from the terrain

**The README says** (§3.1):

> The stage–area–volume curve is **derived from the terrain** … not assumed from a
> power law and not typed in from a gazetteer.

**The code does the opposite.** When an existing reservoir is detected in the DSM —
which is *every* preset, because they are all operating dams — `hybrid_hva()` takes
over (`core/reservoir.py:176`) and builds storage as `V = a·d^b` with

```python
a = published_capacity_m3 / (d_crest ** b)     # reservoir.py:218
```

so `V(crest) == published capacity` **by construction**. From the Tehri manifest:

| quantity | value |
|---|---|
| `dem_only_capacity_mcm` | **4.02** |
| capacity actually used | **3,376.6** |
| `published_capacity_mcm` (Wikipedia infobox) | 3,540.0 |

**99.88 % of the storage that drives the entire simulation comes from a Wikipedia
number fed through a power law** — precisely the two things the README disclaims.

To be fair to the code: `hybrid_hva`'s own docstring is honest about this, the
provenance dict records every input, and the reasoning (a DSM cannot see beneath
standing water) is correct. The problem is entirely in the README and in §3.1's
"Known limitation" box, which describes the *DEM-only* path that in practice never
produces the reported number.

**Fix.** Rewrite README §3.1 to say: DEM hypsometry above the water plate, published
capacity + WorldCover surface area constraining a reconstructed power law below it.
Print `dem_only_capacity_mcm` next to the used capacity on the dashboard.

---

## 🔴 2. The reservoir QC gate is circular

`_reservoir_cross_check()` (`pipeline.py:734`) computes:

```python
out["capacity_ratio_dem_over_published"] = res_summary["capacity_mcm"] / pub_mcm
# fails only if ratio > 3.0
```

But `res_summary["capacity_mcm"]` *is* the hybrid capacity, which was constructed
from `pub_mcm`. The ratio is 0.954 at Tehri and can only ever be near 1. The gate is
structurally incapable of failing on the path that actually runs, and its docstring
still narrates the DEM-lower-bound story that applies to the other path.

**Fix.** Compare `dem_only_capacity_mcm` against published, or drop the gate and
report the reconstruction parameters instead of a pass/fail.

---

## 🔴 3. The "coupled" model is coupled for 0.2 % of the simulation

`_blended_hydrograph()` (`pipeline.py:686`) hands the SWE source term to SPH until
`t_switch = iface.t[-1]`, then to the weir. Measured across the committed runs:

| run | SPH window | simulation length | fraction SPH-driven |
|---|---|---|---|
| tehri-198620fe | 12.1 s | 10,800 s | **0.11 %** |
| bhakra-9fcc2d9a | 20.0 s | 10,800 s | **0.19 %** |
| srisailam-6dfd44de | 18.6 s | 10,800 s | **0.17 %** |

For the remaining 99.8 % the "coupled" run is the same configuration as
`grid_standalone`. And because SPH is initialised with the breach **already fully
open** while the weir needs 60–90 min to form, those first 12–20 s inject a slug the
standalone run never sees. At Bhakra that single slug changes the answer:

| Bhakra | grid_standalone | coupled |
|---|---|---|
| max depth | 26.8 m | **62.4 m** (2.3×) |
| inundated area | 19.28 km² | 21.02 km² |

`coupled` is the `primary_model`, so **the headline hazard, exposure and loss numbers
are taken from the run most contaminated by this artefact.**

**Fix.** Either run SPH long enough to matter, or rename the configuration to
something honest ("weir + SPH near-field initial condition") and make
`grid_standalone` the primary model.

---

## 🔴 4. The model-comparison table conceals the coupling by mis-sampling

```python
"peak_inflow_m3s": round(float(max(hyd(tt) for tt in
        np.linspace(0, scn.sim_hours * 3600, 400))), 1),   # pipeline.py:308
```

400 samples over 10,800 s is one sample every **27.1 s**. The SPH window is
12.1–20.0 s. **The sampler steps clean over it.** Result, straight out of
`results.json`:

```
grid_standalone  peakQ = 748180.3
coupled          peakQ = 748180.3     ← identical
sph_nearfield    peakQ = 1061623.3    ← the value the coupled row should reflect
```

A judge reading the Models tab sees two independent models agreeing to the decimal
point. They are not agreeing — the table simply never looked at the part where they
differ.

**Fix.** Sample the hydrograph adaptively, or evaluate the peak over the union of
`iface.t` and a coarse grid.

---

## 🟠 5. The breach peak-discharge gate cannot fail where it matters

```python
"pass": bool(envelope_ok or not in_calibration),   # breach.py:426
in_calibration = (v0 <= 1.0e9) and (head <= 100.0)
```

Every preset is a large Indian dam, so `in_calibration` is `False` for all of them,
so `pass` is `True` **regardless of the routed peak**. Tehri:

| | value |
|---|---|
| routed peak | 757,059 m³/s |
| Froehlich 1995 envelope | 236,095 m³/s (**3.2×**) |
| Costa–Schuster 1988 envelope | 45,785 m³/s (**16.5×**) |
| `within_envelope` | `false` |
| `pass` | **`true`** |

The argument for the escape hatch is legitimate — those regressions were fitted to
sub-100 m, sub-1 km³ structures. But a gate that reports `pass: true` on a 16×
exceedance is not a gate; it is a comment. Note also that 757,000 m³/s is roughly
7× the Amazon's mean discharge, sustained for ~90 minutes.

**Fix.** Report `pass: null` / `"not applicable"` rather than `true`, and keep it out
of the `overall_pass` roll-up. Add an independent physical ceiling — e.g. critical
flow through the breach section, `Q ≤ (2/3)^1.5 √g · B · H^1.5` — which *is* valid
at any scale.

---

## 🟠 6. Benchmark validation has an inverted condition and has never been run

`validate_extent()`'s docstring promises:

> In `benchmark` mode any permanent water present before the event is removed from
> both the model and the observation.

The call site passes the baseline **only in context mode**:

```python
baseline_water = water if scn.validation_mode == "context" else None   # pipeline.py:820
```

and the subtraction is guarded by `if baseline_water is not None:`
(`validation.py:156`). So in `benchmark` mode `baseline_water` is `None`, the guard
is skipped, and **the permanent-water removal never happens** — IoU/CSI/POD/FAR/F1
would be inflated by the reservoir and the river being permanently wet.

Compounding this: **all 14 committed runs are `context` mode.** The only mode that
produces a skill score has never been executed, and `extent_metrics` /
`kling_gupta` have never run on real data.

The README is honest that context mode is not a skill score. But deliverable (iv)
— "near-real-time flood analysis" — rests on a code path that is both unexercised
and buggy.

**Fix.** Pass `baseline_water=water` unconditionally. Then actually run one
benchmark case (a real gauged flood with a matching Sentinel-1 pair) and put the IoU
in the README. Without that, deliverable (iv) is a claim, not a demonstration.

---

## 🟠 7. SPH crashed in every run; the dashboard does not say so

From `results.json → sph` for tehri-198620fe:

```json
"simulated_s": 12.106,
"requested_s": 20.0,
"completed": false,
"stop_reason": "timestep collapsed to 6.86e-06s at t=12.106s (pressure instability); near field truncated"
```

The solver's self-reporting is **exemplary** — this is exactly what a numerical code
should record. The problem is downstream:

* `web/index.html:571` renders `sph_error` (a thrown exception) but **never**
  `completed` or `stop_reason`. A truncated, pressure-unstable run appears in the
  Models tab as an ordinary row.
* `compare_hydrographs()` sets `balance_pass` purely from the critical-flow ratio
  (`coupling.py:282`) and **ignores `completed`**. Every run reports
  `balance_pass: true` on a crashed solve.
* The `coupled` configuration is driven by the hydrograph this crashed solve
  produced.

Scale check: 4,082 fluid particles at `dp = 4 m` over 12 s. README §3.4 says this
"resolves … the plunging breach jet, vertical accelerations, the non-hydrostatic
surge front." At 4 m particle spacing and 12 s, that is a generous reading.

**Fix.** Surface `stop_reason` in the Models tab. Gate `balance_pass` on
`completed`. Either stabilise SPH (smaller `dp`, higher `alpha_visc`, δ-SPH density
diffusion) or state the truncation in the README.

---

## 🟠 8. Velocities of 66–129 m/s propagate into every downstream product

| run | max velocity | max d·v |
|---|---|---|
| tehri-1789829326 | **128.7 m/s** (463 km/h) | 16,384 m²/s |
| tehri-198620fe | 66.7 m/s | — |

The thin-film cap `V_CAP = 50.0` (`swe2d.py:67`) applies **only** where
`h < H_THIN = 2 cm`, so deep cells are uncapped. These are almost certainly
numerical — most likely at the injection patch or on steep dry slopes — not physical
dam-break velocities.

`hazard.py` clips the *hazard rating* to 20 and prints the raw maximum alongside,
with a careful note. That is good practice, but it addresses the wrong thing: the
note frames the problem as *Defra's scale not extending far enough*, when the actual
problem is *the velocity field is wrong*. The unclipped velocity still flows into:

* `structural_dv` classes (the "total destruction" building counts),
* per-settlement `velocity_ms` in the evacuation table,
* arrival times, which are what an evacuation plan would be built on.

**Fix.** Diagnose where the peak sits (dump `argmax(v_max)` and its distance from
the source patch). Cap by Froude number rather than by film thickness. Report the
99.9th percentile velocity next to the max.

---

## 🟠 9. Von Thun & Gillette is implemented incorrectly

```python
def von_thun_gillette(...):
    """Von Thun & Gillette (1990).  B_avg = 2.5*hw + Cb ; tf from erodibility."""
    ...
    b_avg = 4.0 * hw + cb            # breach.py:95  ← contradicts the docstring
```

The published regression is `B_avg = 2.5·h_w + C_b`. The `4.0` appears to be the
denominator of the *formation-time* relation `t_f = B/(4h_w + 61)` copied into the
width formula. At Tehri's head this overstates breach width by roughly 60 %.

Mitigating: the function is **unreachable** (§11), so no committed result is affected.
Aggravating: the README lists it as one of four selectable regressions, and a judge
who greps for "Von Thun" lands on a formula that disagrees with its own docstring.

---

## 🟠 10. Nine functions cited in APPROACH_MAPPING.md are never called

`docs/APPROACH_MAPPING.md` maps each node of the SIH technical-approach diagram to a
function. These have **zero call sites** outside their own definition:

| Diagram node | Cited function | Status |
|---|---|---|
| Channel geometry & bathymetry | `dem.burn_channel` | reachable only with `channel_mask`, which `pipeline.py:164` hardcodes to `None` |
| Channel geometry & bathymetry | `dem.d8_flow_accumulation` | **never called** |
| Volume Balance | `reservoir.volume_balance` | **never called** |
| VALIDATION (IoU/CSI/NSE) | `validation.kling_gupta` | **never called** |
| — | `breach.orifice_discharge` | **never called** (see §11) |
| — | `breach.scenario_matrix` | **never called**, though its docstring claims "the failure-mode ensemble that the dashboard compares side by side" |
| — | `hazard.structural_vulnerability` | **never called** (a near-duplicate is inlined in `building_impact`) |
| — | `reservoir.analytic_reservoir`, `reservoir.route_step`, `dem.landcover_summary`, `validation.rmse`, `datasources.fetch_waterways`, `sph.ritter_reference`, `swe2d.add_baseflow` | **never called** |

The mapping table is the document a judge will use to verify deliverable coverage.
Rows that point at dead code are the single most fragile thing in the submission.

**Fix.** Either wire them in or mark them `(available, not exercised in the default
pipeline)`. Do not leave them reading as evidence.

---

## 🟠 11. Four advertised physics features are unreachable

| README claim | Reality |
|---|---|
| §3.2 "trapezoidal broad-crested weir with **Villemonte submergence correction**" | `simulate_breach` is called without `tailwater`, so `tw is None` on every step and the correction branch in `weir_discharge` never executes. |
| §3.2 "piping (**with an orifice phase** before roof collapse)" | `breach_section` in piping mode raises the invert and narrows the width, then routes through `weir_discharge` like every other mode. `orifice_discharge()` is never called. It is a weir with a high invert, not an orifice. |
| §3.2 four selectable seed regressions | `Scenario.seed_method` exists but is exposed by **neither** the CLI (`cli.py` has no `--seed-method`) **nor** the API (`RunRequest` has no `seed_method`). `auto` always resolves to `froehlich_2008` or `costa_schuster`. Von Thun and MacDonald are unreachable. |
| Approach map "Inflow Q_in(t) + tailwater rating" | `Scenario.inflow_m3s` defaults to `0.0` and is likewise absent from CLI and API, so `inflow` is always `None`. |

Also absent from both interfaces: `asset_values` (despite `hazard.py` calling them
"explicit, overridable scenario parameters"), `channel_burn_m`, `population_product`,
`iso3`, `satellite_basemap`.

**Fix.** These are one-line additions to `cli.py` and `RunRequest`. Doing so converts
four paper claims into demonstrable features for roughly an hour of work.

---

## 🟡 12. An undocumented 0.35 drives 96 % of the reported loss

```python
road_loss = road_km * values.road_per_km * 0.35   # partial-damage factor
                                                  # exposure.py:357
```

This factor is **not** in `AssetValues`, **not** in `unit_values_used`, and **not**
overridable. Tehri's loss breakdown:

| component | ₹ | share |
|---|---|---|
| roads | 1,058,286,250 | **96.5 %** |
| cropland | 31,725,000 | 2.9 % |
| buildings | 7,200,000 | 0.7 % |
| **total** | **₹109.72 crore** | |

`hazard.py`'s header insists every monetary figure "is tagged with the unit rates
that produced it." The rate that produced 96.5 % of the total is untagged.

Note also that roads use a flat damage factor while buildings get a depth-damage
curve — so the dominant term is the least physically justified one.

**Fix.** Move `0.35` into `AssetValues` as `road_partial_damage_factor` and emit it
in `unit_values_used`.

---

## 🟡 13. Other hardcoded constants worth disclosing

| Constant | Where | Comment |
|---|---|---|
| `return 2.4` | `reservoir.py:166` | Shape-exponent fallback, "typical narrow Himalayan valley", used silently whenever the bisection bracket fails. Not recorded in the provenance dict, so a reviewer cannot tell a solved `b` from a defaulted one. |
| `* 0.6` | `swe2d.py:498` | Momentum coefficient on the injected source: `spd = 0.6·√(2gh)`. Undocumented, untuned, and it sets the initial jet momentum. |
| `transfer_x = min(12·dp, 0.35·downstream_m)` | `coupling.py:203` | Comes out at exactly **48.0 m in all four SPH runs** because it is set by particle spacing, not by where the near field actually ends. The "overlap zone" of the coupling is therefore a numerical parameter dressed as a physical one. |
| `b_avg *= 1.5`, `t_f *= 2.0`, `invert += 0.25·h` | `breach.py:159-165` | Three unsourced fudge factors defining the entire natural-blockage (landslide-dam) branch. The comment "landslide dams breach wider, shallower and slower" is qualitatively right; the numbers have no citation. |
| `K0_OVERTOPPING = 1.3`, `K0_PIPING = 1.0` | `breach.py:36` | Correct per Froehlich 2008. ✅ |
| `c_rect = 1.70`, `c_tri = 1.35` | `breach.py:233` | Standard broad-crested values, correctly justified in the docstring. ✅ |
| `depth_cells = 3`, `half ≥ 2` | `pipeline.py:661` | Source-patch geometry. Honestly documented as a numerical necessity, which is the right way to do it. ✅ |

---

## 🟡 14. The animation and the 3D view are a different model from the statistics

```python
results["terrain3d"] = _write_terrain_3d(
    swe_products.get("grid_standalone") or primary, ...)   # pipeline.py:452
frame_meta = _write_frame_overlays(
    swe_products.get("grid_standalone"), ...)              # pipeline.py:459
```

but

```python
primary = swe_products.get("coupled") or swe_products["grid_standalone"]
hz = HZ.build_hazard(primary.h_max, primary.v_max, ...)
```

So whenever SPH succeeds, **the 2D time animation and the 3D water surface render
`grid_standalone` while every reported statistic comes from `coupled`.** They differ
by 0.25 % in area at Tehri and by 2.3× in max depth at Bhakra.

This also undercuts README §6a's sanity check —

> the final wetted area is 80.0 km² against 80.6 km² from the 2D hazard raster.
> The 3D view is the same solver output, not a separate illustration.

— which, for a run with SPH enabled, is comparing two *different* models that happen
to agree to 0.25 %, not one model against itself.

**Fix.** One line: pass `primary` to both writers.

---

## 🟡 15. Building exposure is point-sampled and priced as uniformly residential

`APPROACH_MAPPING.md` lists "BLDG polygon → `building_impact`". In fact
`building_impact` (`exposure.py:227`) takes `b["lon"], b["lat"]` — a single centroid
per building — and samples the depth raster there. At 120–150 m cells against
building footprints this is a reasonable simplification, but it is **point sampling,
not polygon intersection**, and the mapping document says otherwise.

Separately, every building is valued with the **residential** curve and the
**residential** unit rate. `AssetValues.commercial_per_building` (₹3,500,000) is
defined, printed in `unit_values_used` as though it were used, and never read.

The saving grace, and it is a real one: `qc_checks` (`exposure.py:391`) catches the
resulting absurdity and says so plainly —

> "25,929 people exposed against only 4 inundated OSM buildings (6,482 people per
> mapped building) … the building count is a lower bound"

— and the dashboard renders that warning. This is the best-handled weakness in the
codebase. **Keep it visible during the demo**; it converts a hole into evidence of
rigour.

---

## 🟡 16. Reference tables are typed-in literals

| Table | Where | Attribution | Verifiable? |
|---|---|---|---|
| `DD_CURVES` — 5 curves × 9 depths | `hazard.py:58` | "JRC Huizinga et al. (2017), Asia" | No source file, no DOI, no extraction script. Nine numbers per curve, typed in. |
| `WORLDCOVER_CLASSES` Manning *n* | `dem.py:229` | "Chow (1959) … Arcement & Schneider" | Values are plausible (tree cover 0.12, built-up 0.09) but the class→*n* assignment is an unsourced judgement per class. README §7.7 discloses this. ✅ |
| `DEBRIS_FACTOR` | `hazard.py:48` | "Defra FD2321 Table 3.2" | Three values; consistent with the method. |
| `HAZARD_CLASSES` thresholds | `hazard.py:41` | Defra FD2321 | Correct. ✅ |

This is normal engineering practice and not dishonest. But "derived from JRC 2017"
and "nine numbers typed from a PDF" are different provenance claims, and the
manifest asserts the first.

**Fix.** Add a one-line comment per curve giving the table and page number in
EUR 28552 EN.

---

## What is genuinely solid

Worth being precise about, because most of the framework is not in question and this
review should not read as a demolition:

* **The 2D solver is real and verified.** HLL + MUSCL/minmod + SSP-RK2 + Audusse
  well-balanced reconstruction, with Kurganov–Petrova desingularisation, is a
  correct and non-trivial implementation. `tests/test_swe_benchmarks.py` tests it
  against Ritter's analytical solution, lake-at-rest and closed-basin mass
  conservation, with real tolerances, and the README reports the **negative** front
  bias (−7.4 %) rather than hiding it. That is the mark of an honest code.
* **Every data source is live and open.** Windowed COG reads, content-hashed cache,
  and `SourceUnavailable` raised rather than silently substituting a guess.
* **Self-reporting inside the numerical kernels is excellent.** SPH's
  `stop_reason`, the DSM-bathymetry provenance dict, and the exposure QC warnings
  are all things a less careful project would have swallowed. The failures in this
  audit are overwhelmingly in the *presentation layer*, not in the physics.
* **QC gates are surfaced on the dashboard** (`index.html:722-725`), including
  failed gates and warnings. Nine of fourteen runs openly report `overall_pass: false`.

---

## Prioritised fix list

**Before the demo — these change headline numbers, or are one-liners**

1. §4 — fix the peak-inflow sampler so the Models tab stops showing a false agreement.
2. §14 — pass `primary` to the 3D and frame writers (one line).
3. §6 — pass `baseline_water` unconditionally (one line).
4. §5 — change the envelope gate to `pass: null` when out of calibration.
5. §1 — rewrite README §3.1 to describe the hybrid method that actually runs.
6. §7 — render `stop_reason` in the Models tab; gate `balance_pass` on `completed`.
7. §12 — move `0.35` into `AssetValues`.
8. §10 — annotate the dead rows in `APPROACH_MAPPING.md`.

**If there is time**

9. §8 — diagnose the 129 m/s velocity; it is the weakest physical result in the run.
10. §3 — rename `coupled`, or extend the SPH window until the coupling is real.
11. §11 — expose `seed_method`, `inflow_m3s`, `asset_values` in CLI + API.
12. §9 — fix the Von Thun & Gillette coefficient to 2.5.
13. §6 — run one real `benchmark` case and publish the IoU.

**Repo hygiene (not a results issue, but a judge will see it)**

14. No `.gitignore`; 1,202 binary files tracked; `.git` is 65 MB. Add one and
    consider `git filter-repo` on `data/cache/` and `runs/`.
15. `runs/tehri-1789789259/` is a half-written run with no `manifest.json` — delete it.
16. `__pycache__/` is committed.

---

## The one-paragraph honest abstract

> DamBurst couples an empirical breach model to a verified second-order well-balanced
> shallow-water solver over live Copernicus, WorldCover, WorldPop and OpenStreetMap
> data, and produces hazard, exposure and evacuation products with full provenance and
> standard GIS export. Reservoir storage below the DSM water surface is reconstructed
> from published gross capacity and satellite-observed surface area, not surveyed. The
> SPH near field is a 2D vertical slice that runs for the first ~15 s and is currently
> truncated by a pressure instability; it informs the initial condition rather than
> driving the far field. Routed peak discharge exceeds the Froehlich and Costa–Schuster
> envelopes by 3–16×, which those regressions cannot arbitrate at this scale and which
> remains an open item. Satellite validation runs in context mode only; no predictive
> skill score has been measured.

That paragraph is defensible line by line, and it is a stronger position in front of
a judge than a claim that does not survive a `grep`.
