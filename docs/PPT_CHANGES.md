# PPT change list

What to change in the deck, given (a) Varshith's review comments and (b) the
results audit in [`RESULTS_AUDIT.md`](RESULTS_AUDIT.md).

I have not seen the current deck, so this is organised by **claim**, not by
slide number. Search the deck for each claim on the left and apply the change
on the right. Anything marked 🔴 will not survive a judge who opens the repo.

---

## 1. The three review comments, and what each means for the deck

> **"Automated scenario comparison does not look feasible."**

He is right, and the audit found the same thing independently from the numbers.
The old three-way comparison was `grid_standalone` vs `sph_nearfield` vs
`coupled`, and on a real Tehri run the "coupled" model came out at
**IoU 0.9998, NSE 1.000, RMSE 0.096 m** against the standalone run — because
SPH drove only 12 s of a 3-hour simulation, i.e. 0.11% of it. It was not a
comparison of two models; it was the same model twice.

**Deck action:** delete the "automated scenario comparison" framing entirely.
Do not present a three-model comparison table as a headline capability.

> **"Focus on one technical novelty — adaptive model selection / adaptive
> spatial resolution / uncertainty-aware scenario generation."**

Chosen: **automated adaptive model selection.** Implemented in
`damburst/core/adequacy.py`. See §3 below for the slide.

> **"Mathematical formulation on the flow dynamics in case of breach and the
> possibility of breach to be worked out."**

Both written: [`MATHEMATICAL_FORMULATION.md`](MATHEMATICAL_FORMULATION.md),
Part A (flow dynamics) and Part B (possibility of breach). Part B is now
partly implemented too — `core/failure_probability.py` routes a design-flood
family to get P(overtopping). See §4 and §5.

---

## 2. 🔴 Claims to delete or rewrite

| Current claim (search for this) | Why it fails | Replace with |
|---|---|---|
| 🔴 "H–V–A curve **derived from the DEM**" / "storage derived from terrain" | On every preset, gross storage is set by the **published capacity** through a power law. The DEM contributes **0.12 MCM of 3,375 MCM** at Tehri — 0.004%. A judge who opens `results.json` sees `dem_only_capacity_mcm: 0.12`. | "Storage above the reservoir water plate is measured from the DEM; below it, bathymetry is **reconstructed** from published capacity + satellite-observed surface area. The DSM cannot see through water." |
| 🔴 "Coupled SPH + 2D model" as a headline | SPH drives 0.11–0.19% of the run and terminated on a pressure instability in every run tried. | "SPH resolves the **near field**; the framework decides **whether it is needed** (§3). The far field is 2D SWE." |
| 🔴 Three-way model comparison table | The two 2D configurations agree to IoU 0.9998 — see above. | The **model-selection** slide (§3). |
| 🔴 "Validated against Sentinel-1" | All 14 runs were `context` mode, which is **not a skill score**. No IoU/CSI/POD was ever computed. The `benchmark` path also had an inverted condition that skipped the permanent-water subtraction (now fixed). | "Sentinel-1 provides the pre-event water baseline so the dashboard separates permanent water from newly inundated land. **No predictive skill score has been measured** — that needs a real gauged flood." |
| 🔴 Any single "₹ X crore loss" headline | 96.5% of that number came from an **undocumented `0.35`** hardcoded in the loss function, on top of placeholder unit rates. | Lead with **physical exposure** (people, buildings, km of road, ha of cropland). If currency appears at all, show the rate table next to it. |
| 🟠 "Max velocity 128.7 m/s" or any max-only figure | 463 km/h. It is one cell on terrain too steep for the equations. | Median / p90 / p99 / p99.9 **and** the max, labelled. At Tehri: median 6.4, p99.9 45.1, max 65.5 m/s. |
| 🟠 "All QC gates pass" | 9 of 14 runs reported `overall_pass: false`, and the peak-discharge gate auto-passed a **16× exceedance** because it excused itself when out of calibration range. | "QC gates are pass / fail / **inconclusive**, and inconclusive is not a pass." Show a run with a real failed gate — it reads as rigour, not weakness. |
| 🟠 Approach-diagram → code mapping slide | Nine cited functions had **zero call sites**. | Use the updated `APPROACH_MAPPING.md`, which now marks every row **live / opt-in / available**. |

---

## 3. NEW SLIDE — the technical novelty

Title: **Automated adaptive model selection — the framework decides which
physics applies, and where**

**The problem with what everyone else does.** A framework that owns two
solvers usually runs both and shows the numbers side by side. That tells a
reader the models disagree, not which to believe. And it makes the SPH→SWE
handover a free parameter — in our own code it was previously
`min(12 × dp, …)`, i.e. **the particle spacing**, a numerical setting dressed
up as a physical overlap zone.

**What we do instead.** Compute how badly the shallow-water assumptions are
violated, and let that decide. The SWE assume hydrostatic pressure, which
needs vertical acceleration ≪ g. Three things break it:

```
NHI = max(  |∇z| / tan θc  ,  u²·|κ| / (g·c)  ,  |∇η| / sc  )
         bed slope        curvature→a_z        surface steepness
```

`NHI < 1` → depth-averaging is defensible. `NHI ≥ 1` → it is not.

**Three things this buys, none of which a side-by-side table gives:**

1. **Run the particle model only when it is needed.** Tehri, full-height
   breach: NHI peaks at 0.03 and the breach scours to within 12 m of the
   riverbed (0.23× critical depth) → **no plunging jet, SPH skipped**, ~5
   minutes saved with no physics lost. Same dam as a **landslide blockage**:
   a 56 m residual barrier → plunging jet at **1.3× critical depth** →
   **SPH selected**. Same terrain, same tool, opposite decision, stated reason.
2. **The handover locates itself** — at the first point downstream where NHI
   recovers below 1. Change the particle spacing and the section does not move
   (there is a test for this).
3. **The run scores its own trustworthiness.** After solving, the same
   criterion is applied to the solution: at Tehri, **73% of the flood volume**
   sits where the depth-averaged equations do not hold (54% of the wetted
   area), because the flood fills a gorge with 38° walls. No other dam-break
   framework tells you that about its own output.

**Suggested visual:** two panels of the same map — the depth raster, and the
NHI raster beside it with the ≥1 region highlighted; plus the one-line decision
string the framework emitted.

**Honesty line for the slide (keep it):** the near-field decision and the
far-field verdict answer different questions and can disagree. A particle model
at the breach would not fix a far-field validity problem; that needs a
non-hydrostatic far-field solver or a finer DEM, neither of which is in this
prototype.

---

## 4. NEW SLIDE — mathematical formulation

Varshith asked for this explicitly. One slide, equations only, pointing at
`docs/MATHEMATICAL_FORMULATION.md` for the rest.

**Reservoir depletion**

$$dV/dt = Q_{in}(t) - Q_b(h,t), \qquad h = \mathcal{H}^{-1}(V)$$

**Breach outflow** — trapezoidal broad-crested weir, with the two regimes the
code actually implements:

$$Q_b = C_r b H^{3/2} + C_t z_s H^{5/2} \quad\text{(free surface)}$$
$$Q_b = C_d A_o \sqrt{2g(h-z_c)} \quad\text{(pressurised, piping before roof collapse)}$$
$$Q_b \leftarrow Q_b (1-S^{3/2})^{0.385}, \quad S = H_t/H > 0.67 \quad\text{(Villemonte submergence)}$$

**Breach growth** — Froehlich (2008) seed, sine growth law:

$$\bar B = 0.27 K_0 V_w^{0.32} h_b^{0.04}, \quad t_f = 63.2\sqrt{V_w/(g h_b^2)}, \quad \phi(t) = \tfrac12(1-\cos(\pi t/t_f))$$

**Far field** — 2D shallow water, conservative form:

$$\partial_t \mathbf{U} + \partial_x \mathbf{F} + \partial_y \mathbf{G} = \mathbf{S}(\mathbf{U}, \nabla z) - \boldsymbol{\tau}$$

solved with HLL + MUSCL/minmod + SSP-RK2 + Audusse well-balanced
reconstruction, Kurganov–Petrova wet/dry desingularisation.

**Verification** (put these numbers on the slide — they are the strongest thing
in the project and they reproduce exactly):

| Test | Result |
|---|---|
| Lake at rest, irregular wet/dry bed | max\|Δη\| = 6.2 × 10⁻¹⁵ m |
| Closed-basin mass conservation | relative error 0.0 |
| Ritter dry-bed dam break | L₁(h) = 0.0052 m = **0.05%** of h₀ |
| Grid convergence | observed order **0.99** |
| Ritter front position | **−7.35%** (arrival predicted *late* — the unsafe direction, stated) |

---

## 5. NEW SLIDE — possibility of breach

The deck currently answers "what happens if it fails". Varshith asked for "how
likely is it to fail". Make the distinction explicit — it is the single easiest
way to look rigorous:

> Every hazard, exposure and evacuation figure in this framework is
> **conditional on failure**.
> Risk = P(failure) × consequence | failure.
> We compute the right-hand factor. Here is what we can and cannot say about
> the left one.

**Event tree** — overtopping / internal erosion / seismic / structural /
cascade (Tehri → Koteshwar; the presets include the cascade pair and the
conditional probability there is ≈ 1, so cascade risk is *not* the product of
independent probabilities).

**What is computed:** P(overtopping), by routing a Gumbel design-flood family
through the same level-pool equation the breach model uses, against a spillway
rating.

**The honest result** — and this is the slide's real point. Sweeping the one
number we do not have (spillway capacity):

| Spillway capacity | First overtopping | Annual P |
|---|---|---|
| 2.0 × design | > 10,000 y | < 1e-4 |
| 1.0 × design | > 10,000 y | < 1e-4 |
| 0.5 × design | 500 y | 2.0e-3 |
| 0.25 × design | 100 y | 1.0e-2 |

Starting the reservoir at FRL instead of the spillway sill moves the 0.5× case
from a 500-year to a 100-year event.

**Two orders of magnitude, driven entirely by data we do not have.** The
framework is honest about which: spillway geometry is in the CWC National
Register of Large Dams, not in OpenStreetMap. Fragility for internal erosion
and seismic failure falls back to ICOLD/Foster population base rates
(~10⁻⁴–10⁻⁵ per dam-year) and is labelled as a base rate, not an estimate.

**Also say:** P(overtopping) ≠ P(failure). An embankment erodes when
overtopped; a concrete gravity dam may pass substantial overtopping without
breaching.

---

## 6. Slides to strengthen (not replace)

| Slide | Change |
|---|---|
| Data sources | Keep as is — it is genuinely strong. Add that a run downloads only a few MB via windowed COG range requests and is then fully offline. |
| Solver verification | Promote it. Analytical benchmarks against Ritter with an observed convergence order of 0.99 are more convincing than any screenshot. |
| Dashboard | Add the two new tabs: **Model choice** (the decision trace) and **Breach risk**. |
| Limitations | Expand from a footnote to a full slide. Counter-intuitively this is the strongest slide you can have in front of a technical panel: the QC panel flags "25,929 people exposed against 4 mapped OSM buildings" by itself. Showing that you caught it reads as rigour. |
| Tech stack | Add: 112 automated tests, including a regression suite tied to a written results audit. |

---

## 7. Two things worth saying out loud in the presentation

1. **"We audited our own results and found four load-bearing problems."**
   Then name one — the peak-discharge gate that reported `pass: true` on a 16×
   exceedance because it excused itself. Fixed: it now returns `null`
   ("inconclusive"), and a scale-independent critical-flow ceiling was added
   as the gate that actually binds. This is a better story than "everything
   passed".

2. **"The framework tells you where not to trust it."** 73% of the Tehri flood
   volume is in terrain where the shallow-water equations do not strictly hold,
   and the run says so, in the manifest and on the dashboard. That is the
   novelty doing something a hazard map cannot do for itself.

---

## 8. Do not put these in the deck

* Any single monetary loss figure without the rate table beside it.
* Any max-only velocity or depth.
* "Validated" — until a `benchmark`-mode run against a real gauged flood exists.
* "Real-time" — a fast run is ~4 minutes at 120 m for 3 simulated hours.
* Delft3D by name. The solver is *Delft3D-class* (same governing equations and
  numerical class), and the repo says so. Keep that wording.
