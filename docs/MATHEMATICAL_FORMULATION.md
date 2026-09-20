# Mathematical formulation — breach flow dynamics and breach likelihood

SIH 2026 · PS 26161 · DamBurst

Written for the technical review. Two things are set out here:

* **Part A — flow dynamics of a breach**: the governing equations, closures and
  discretisation actually implemented, with the code symbol next to each term
  so a reviewer can go from the equation to the line that evaluates it.
* **Part B — possibility of breach**: how the *likelihood* of failure is
  formulated, which is a separate question from what happens once it occurs,
  and which the framework currently does not answer. §B.5 says what would have
  to be built.

Notation is consistent throughout: $h$ depth, $\eta = z + h$ water-surface
elevation, $z$ bed, $\mathbf{u}=(u,v)$ depth-averaged velocity, $n$ Manning
roughness, $g = 9.80665\ \mathrm{m\,s^{-2}}$.

---

# Part A — Flow dynamics

## A.1 Reservoir depletion (level-pool routing)

The impoundment is treated as a horizontal free surface, so its state reduces
to a single scalar, the stored volume $V(t)$:

$$\frac{dV}{dt} = Q_{\text{in}}(t) - Q_b\big(h_{\text{res}},\,t\big)$$

with the stage recovered by inverting the hypsometric curve,
$h_{\text{res}} = \mathcal{H}^{-1}(V)$.

**Hypsometry.** $\mathcal{H}$ is built two ways and the run records which:

$$
A(h) = \!\!\sum_{(i,j)\in\mathcal{P}(h)}\!\! \Delta x\,\Delta y ,
\qquad
V(h) = \!\!\sum_{(i,j)\in\mathcal{P}(h)}\!\! \big(h - z_{ij}\big)\,\Delta x\,\Delta y
$$

where $\mathcal{P}(h)$ is the 4-connected set of cells below level $h$ reachable
from a seed on the impounded side of the dam axis and not crossing the barrier
mask. This is exact DEM hypsometry — no power law.

Where the DSM already contains a filled reservoir, the terrain below the water
plate at $z_{ws}$ is invisible, and storage there is **reconstructed** as

$$V(h) = a\,(h - z_{\text{base}})^{\,b}, \qquad A(h) = \frac{dV}{dh} = a\,b\,(h-z_{\text{base}})^{\,b-1}$$

with $a,b$ fixed by two external constraints — the observed surface area at the
plate and the published gross capacity:

$$
\frac{A(z_{ws})}{V_{\text{pub}}} = \frac{b}{d_{ws}}\left(\frac{d_{ws}}{d_{\text{crest}}}\right)^{b},
\qquad
a = \frac{V_{\text{pub}}}{d_{\text{crest}}^{\,b}},
\qquad d_\bullet = \bullet - z_{\text{base}}
$$

The first is solved for $b$ by bisection on $[1.05, 6]$. **This makes gross
storage an input from the published record, not a DEM measurement** — see
README §3.1.

> `reservoir.hva_from_dem`, `reservoir.hybrid_hva`, `reservoir._solve_shape_exponent`

## A.2 Breach geometry evolution

The opening is a trapezoid of bottom width $b$, side slope $z_s$ (H:V) and
invert $\zeta$. Its final state $(b_f, z_s, \zeta_f, h_b)$ comes from an
empirical regression; **Froehlich (2008)** is the default for engineered fills:

$$\bar{B} = 0.27\,K_0\,V_w^{0.32}\,h_b^{0.04},
\qquad
t_f = 63.2\,\sqrt{\frac{V_w}{g\,h_b^{2}}}$$

$K_0 = 1.3$ overtopping, $1.0$ piping. Alternatives, selectable with
`--seed-method`:

| Regression | Width | Formation time |
|---|---|---|
| Von Thun & Gillette (1990) | $\bar B = 2.5\,h_w + C_b$ | $t_f = \bar B/(4h_w + 61)$ h |
| MacDonald & Langridge-Monopolis (1984) | from $V_{er} = 0.0261\,(V_w h_b)^{0.769}$ | $t_f = 0.0179\,V_{er}^{0.364}$ h |
| Costa & Schuster (1988), natural | Froehlich × 1.5 | Froehlich × 2.0 |

**Growth law.** A dimensionless growth fraction $\phi(t) \in [0,1]$ interpolates
from closed to final:

$$
\phi(t)=
\begin{cases}
t/t_f & \text{linear}\\[2pt]
\tfrac12\big(1-\cos(\pi t/t_f)\big) & \text{sine (HEC-RAS)}\\[2pt]
\sqrt{t/t_f} & \text{erosion, headcut-dominated}
\end{cases}
$$

so that $b(t) = \phi b_f$, $z_s(t) = \phi z_{s,f}$,
$\zeta(t) = \zeta_f + h_b\,(1-\phi)$.

> `breach.growth_fraction`, `breach.breach_section`

## A.3 Breach outflow $Q_b$

**Free-surface (weir) regime.** Broad-crested trapezoidal weir, rectangular
plus triangular contributions, with head $H = h_{\text{res}} - \zeta$:

$$Q_b = C_r\,b\,H^{3/2} \;+\; C_t\,z_s\,H^{5/2},
\qquad C_r = 1.70,\; C_t = 1.35$$

$C_r = 1.70$ corresponds to $C_d = 0.55$ in $Q = C_d b\sqrt{2g}H^{3/2}$, the
standard broad-crested value (NWS DAMBRK / HEC-RAS).

**Submergence (Villemonte 1947).** With tailwater head $H_t$ above the invert
and submergence ratio $S = H_t/H$:

$$Q_b \leftarrow Q_b\,\big(1 - S^{3/2}\big)^{0.385}, \qquad S > S_{\text{mod}} = 0.67$$

The tailwater elevation is supplied by a Manning normal-depth rating in the
receiving reach, evaluated with a one-step explicit lag:

$$z_{tw} = \zeta_{\text{toe}} + \left(\frac{q\,n}{\sqrt{S_0}}\right)^{3/5},
\qquad q = \frac{Q_b^{\,(k-1)}}{W}$$

with $S_0$ from the DEM thalweg over 3 km below the toe and $n$ from the
WorldCover class at the toe.

**Pressurised (orifice) regime.** In a piping failure before roof collapse
($\phi < 0.5$) the opening is submerged on all sides, so discharge scales with
$\sqrt{H}$ to the centroid, not $H^{3/2}$:

$$Q_b = C_d\,A_o\,\sqrt{2g\,(h_{\text{res}} - z_c)},
\qquad C_d = 0.6,\; z_c = \tfrac12(\zeta + \zeta_{\text{soffit}})$$

> `breach.weir_discharge`, `breach.orifice_discharge`, `breach.TailwaterRating`

**Adaptive time step.** The routing step is limited so no step drains more than
0.5% of live storage or advances the breach by more than 2% of $t_f$:

$$\Delta t = \min\!\left(\Delta t_{\max},\; \frac{0.005\,V}{Q_b},\; 0.02\,t_f\right)$$

## A.4 Far field — 2D shallow-water equations

Conservative form with bed slope and friction source terms:

$$
\frac{\partial}{\partial t}
\begin{pmatrix} h \\ hu \\ hv \end{pmatrix}
+\frac{\partial}{\partial x}
\begin{pmatrix} hu \\ hu^2 + \tfrac12 g h^2 \\ huv \end{pmatrix}
+\frac{\partial}{\partial y}
\begin{pmatrix} hv \\ huv \\ hv^2 + \tfrac12 g h^2 \end{pmatrix}
=
\begin{pmatrix} q_{\text{src}} \\ -gh\,\partial_x z - \tau_x \\ -gh\,\partial_y z - \tau_y \end{pmatrix}
$$

with Manning friction

$$\tau_x = g\,n^2\,\frac{u\,\lVert\mathbf{u}\rVert}{h^{1/3}},\qquad
\tau_y = g\,n^2\,\frac{v\,\lVert\mathbf{u}\rVert}{h^{1/3}}$$

### Discretisation

**HLL Riemann flux** at each face, with wave speeds
$s_L = \min(u_L - c_L,\ u^* - c^*)$, $s_R = \max(u_R + c_R,\ u^* + c^*)$,
$c=\sqrt{gh}$:

$$
\mathbf{F}^{\text{HLL}} =
\begin{cases}
\mathbf{F}_L, & s_L \ge 0\\[2pt]
\dfrac{s_R\mathbf{F}_L - s_L\mathbf{F}_R + s_L s_R(\mathbf{U}_R-\mathbf{U}_L)}{s_R-s_L}, & s_L<0<s_R\\[2pt]
\mathbf{F}_R, & s_R \le 0
\end{cases}
$$

**MUSCL reconstruction** of $\eta$, $u$, $v$ with a minmod limiter, giving
second order in space:

$$\sigma_i = \operatorname{minmod}\!\left(\frac{q_i - q_{i-1}}{\Delta x},\ \frac{q_{i+1}-q_i}{\Delta x}\right),
\qquad
\operatorname{minmod}(a,b)=\begin{cases}a & |a|<|b|,\ ab>0\\ b & |b|\le|a|,\ ab>0\\ 0 & ab\le 0\end{cases}$$

**Audusse et al. (2004) hydrostatic reconstruction** — this is what makes the
scheme *well balanced*, i.e. exactly preserves $\eta = \text{const}$, $\mathbf{u}=0$
over arbitrary bathymetry:

$$z^* = \max(z_L, z_R),\qquad
h_L^* = \max\big(0,\ \eta_L - z^*\big),\qquad
h_R^* = \max\big(0,\ \eta_R - z^*\big)$$

with the bed-slope source discretised consistently so the residual vanishes at
rest. Verified to $\max|\eta - \eta_0| \sim 10^{-15}$ m (§A.7).

**Kurganov–Petrova desingularisation** for wet/dry fronts, avoiding division by
a vanishing depth:

$$u = \frac{\sqrt{2}\,h\,(hu)}{\sqrt{h^4 + \max(h^4,\ \varepsilon^4)}}$$

**SSP-RK2 time integration**:

$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t\,\mathcal{L}(\mathbf{U}^n),
\qquad
\mathbf{U}^{n+1} = \tfrac12\mathbf{U}^n + \tfrac12\left[\mathbf{U}^{(1)} + \Delta t\,\mathcal{L}(\mathbf{U}^{(1)})\right]$$

**CFL condition**:

$$\Delta t = \mathrm{CFL}\cdot\frac{\min(\Delta x,\Delta y)}{\max_{ij}\big(\lVert\mathbf{u}\rVert + \sqrt{gh}\big)},
\qquad \mathrm{CFL} = 0.40$$

**Friction** is integrated semi-implicitly so it cannot reverse the flow at
small depth:

$$(hu)^{n+1} = \frac{(hu)^*}{1 + \Delta t\,g n^2 \lVert\mathbf{u}\rVert h^{-4/3}}$$

> `swe2d._flux_x/_flux_y`, `_slopes_x/_slopes_y`, `_residual`, `_stage`, `_finalise`

### Validity limit on steep terrain

The hydrostatic assumption behind (A.4) requires $\partial z/\partial x \ll 1$.
In a Himalayan gorge it is not small, and the equations then integrate water
into effectively free fall. On cells with bed slope
$\theta > \theta_c = 12°$ the velocity is limited to

$$\lVert\mathbf{u}\rVert \le \mathrm{Fr}_{\max}\sqrt{gh},\qquad \mathrm{Fr}_{\max}=4$$

Cells with $\theta \le \theta_c$ — every cell in the analytical benchmarks —
are untouched, so §A.7 is unaffected. This limiter is a **model-adequacy
indicator as much as a numerical fix**: the field $\theta_{ij} > \theta_c$ is
precisely the region where a depth-averaged model should not be trusted.

## A.5 Near field — weakly-compressible SPH

A 2D vertical slice along the breach centreline, taken from the real DEM long
profile. For particle $i$ with smoothing length $h_s = 1.3\,dp$:

**Cubic spline kernel** ($q = r/h_s$):

$$W(q) = \alpha_d\begin{cases}1 - \tfrac32 q^2 + \tfrac34 q^3 & 0\le q<1\\ \tfrac14(2-q)^3 & 1\le q<2\\ 0 & q\ge2\end{cases},
\qquad \alpha_d = \frac{10}{7\pi h_s^2}$$

**Continuity and momentum** (Monaghan form):

$$\frac{D\rho_i}{Dt} = \sum_j m_j\,\mathbf{v}_{ij}\cdot\nabla_i W_{ij}$$

$$\frac{D\mathbf{v}_i}{Dt} = -\sum_j m_j\left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} + \Pi_{ij}\right)\nabla_i W_{ij} + \mathbf{g}$$

**Tait equation of state**, $\gamma = 7$, $c_0 = 12\sqrt{g\,H_{\max}}$:

$$p = \frac{\rho_0 c_0^2}{\gamma}\left[\left(\frac{\rho}{\rho_0}\right)^{\gamma} - 1\right]$$

**Monaghan artificial viscosity** ($\alpha = 0.05$, active in compression only):

$$\Pi_{ij} = \begin{cases}\dfrac{-\alpha\,\bar c_{ij}\,\mu_{ij}}{\bar\rho_{ij}} & \mathbf{v}_{ij}\cdot\mathbf{r}_{ij}<0\\ 0 & \text{otherwise}\end{cases},
\qquad \mu_{ij} = \frac{h_s\,\mathbf{v}_{ij}\cdot\mathbf{r}_{ij}}{\lVert\mathbf{r}_{ij}\rVert^2 + 0.01h_s^2}$$

**XSPH** velocity correction ($\epsilon = 0.5$) and periodic **Shepard density
re-initialisation** every 30 steps:

$$\rho_i \leftarrow \frac{\sum_j m_j W_{ij}}{\sum_j (m_j/\rho_j) W_{ij}}$$

**Time step**: $\Delta t = \mathrm{CFL}\cdot\min\!\big(h_s/c_{\max},\ \sqrt{h_s/|\mathbf{a}|_{\max}}\big)$, CFL $= 0.20$.

> `sph.run_sph`, `sph._interactions`, `sph._eos`, `sph._shepard`

## A.6 Coupling

The transfer section sits at $x_T = 1.5\,H$ downstream of the toe — a physical
length scale for the non-hydrostatic near field. There, depth and
depth-averaged velocity are Shepard-interpolated from the particles:

$$d(t) = \max_i z_i\big|_{x_i \approx x_T} - z_{\text{bed}},
\qquad
\bar u(t) = \frac{\sum_i m_i u_i W_i}{\sum_i m_i W_i},
\qquad
q(t) = d\,\bar u$$

and extruded by the breach width, $Q_{\text{SPH}} = q\,W_b$. The far-field
source is then blended over $[t_T - \Delta, t_T]$:

$$Q_{\text{src}}(t) = \big(1-f\big)\,Q_{\text{SPH}}(t) + f\,Q_{\text{weir}}(t),
\qquad f = \frac{t-(t_T-\Delta)}{\Delta}$$

**Stated plainly**: $t_T \approx 12$–$20$ s against a 3-hour simulation, so this
sets the *initial condition* of the far field rather than driving it.

## A.7 Verification

| Test | Analytical result | Measured |
|---|---|---|
| Lake at rest, irregular wet/dry bed | $\eta \equiv \text{const}$, $\mathbf{u}\equiv 0$ | $\max\lvert\Delta\eta\rvert = 6.2\times10^{-15}$ m, $\max\lvert u\rvert = 8.4\times10^{-14}$ m/s |
| Closed-basin mass conservation | $\Delta V = 0$ | relative error $0.0$ |
| Ritter (1892) dry-bed dam break | $h(x,t) = \frac{1}{9g}\!\left(2c_0 - \frac{x}{t}\right)^2$, $u = \frac{2}{3}\!\left(\frac{x}{t}+c_0\right)$ | $L_1(h) = 0.0052$ m $= 0.05\%$ of $h_0$ |
| Ritter grid convergence | order 2 | observed rate $0.94\to0.99$ |
| Ritter front position, $x_f = 2c_0t$ | — | $-7.35\%$ at $\Delta x = 0.5$ m; $-5.62\%$ at $0.25$ m |

The front bias is **negative** — the model predicts arrival slightly *late*,
which is the unsafe direction for evacuation timing, and is therefore reported
rather than buried. Second order more than halves it ($-11.2\% \to -7.35\%$).

> `tests/test_swe_benchmarks.py`, `python -m damburst.cli verify`

---

# Part B — Possibility of breach

Part A answers *"what happens if the dam fails?"*. It says nothing about
*"how likely is that?"*. These are different questions and the framework
currently answers only the first. This part sets out the formulation for the
second and is explicit about what is not yet implemented.

## B.1 Why the distinction matters

The hazard products in Part A are **conditional on failure**:

$$\text{Hazard map} = \Pr\big(\text{depth} > d \mid \text{failure}\big)$$

Risk, which is what an emergency planner actually needs, requires the
unconditional form:

$$\text{Risk} = \Pr(\text{failure}) \times \text{Consequence} \mid \text{failure}$$

Reporting only the left factor's consequence, with no probability attached,
overstates what the framework knows. A scenario is not a forecast.

## B.2 Event-tree decomposition

Total annual failure probability decomposes over initiating mechanisms, which
are the same modes §A.2 already implements:

$$P_f = \sum_{m} P(\text{IE}_m)\cdot P(\text{failure} \mid \text{IE}_m)$$

| $m$ | Initiating event | Typical dominant term |
|---|---|---|
| Overtopping | inflow flood exceeding spillway capacity | hydrological loading |
| Piping / internal erosion | seepage gradient exceeding critical | ageing, filter performance |
| Seismic | ground motion exceeding design | site seismicity |
| Structural / foundation | — | inspection record |
| Upstream cascade | failure of a dam above (e.g. Tehri → Koteshwar) | conditional on upstream $P_f$ |

The last row matters for this problem statement: the presets already include a
cascade pair, and $P(\text{Koteshwar fails} \mid \text{Tehri fails}) \approx 1$
for a wave of this size, so cascade risk is not the product of independent
probabilities.

## B.3 Load–resistance (fragility) formulation

For each mechanism, failure is the event that load $S$ exceeds resistance $R$:

$$P_f^{(m)} = \Pr(R - S \le 0) = \int_{-\infty}^{\infty} F_R(x)\,f_S(x)\,dx$$

With both lognormal — the usual assumption — this closes in the familiar form

$$P_f^{(m)} = \Phi\!\left(-\beta\right),
\qquad
\beta = \frac{\ln(\mu_R/\mu_S)}{\sqrt{\beta_R^2 + \beta_S^2}}$$

where $\beta$ is the reliability index. A **fragility curve** is the same
expression read as a function of an intensity measure $IM$ (flood peak, PGA):

$$P\big(\text{failure}\mid IM\big) = \Phi\!\left(\frac{\ln(IM/\theta)}{\zeta}\right)$$

$\theta$ = median capacity, $\zeta$ = log-standard deviation.

**Overtopping, worked through.** Failure requires the reservoir level to exceed
the crest, so the load is the routed flood peak and the resistance is the
freeboard plus spillway capacity:

$$P(\text{IE}_{\text{OT}}) = \Pr\Big(\max_t h_{\text{res}}(t) > z_{\text{crest}}\Big)$$

which is obtained by routing the design-flood family through the **same**
level-pool equation as §A.1:

$$\frac{dV}{dt} = Q_{\text{in}}^{(T)}(t) - Q_{\text{spillway}}(h) ,
\qquad
Q_{\text{spillway}} = C_s L_s (h - z_{\text{sill}})^{3/2}$$

for return periods $T$ with exceedance $1/T$. The framework already has
`reservoir.inflow_hydrograph` (gamma-shaped flood) and the routing loop, so
this is a small extension rather than a new model.

## B.4 Base rates — the honest anchor

Where site-specific fragility data does not exist, historical base rates bound
the answer. ICOLD / Foster et al. (2000) statistics on ~11,000 embankment dams
give an order-of-magnitude annual failure probability of

$$P_f \sim 10^{-4}\ \text{to}\ 10^{-5}\ \text{per dam-year}$$

with roughly half of all failures by overtopping and a third by internal
erosion. For a specific dam this must be adjusted by age, inspection record,
spillway adequacy and seismic setting — which is exactly the information
DamBurst does **not** pull from OSM or Wikidata.

## B.5 What is and is not implemented

**Implemented.** Everything in Part A: the conditional consequence of an
assumed failure, with the failure mode, growth law and loading level as
explicit user-chosen scenario parameters.

**Not implemented.** Any $P_f$. The framework does not estimate the probability
of failure and must not be presented as doing so. Concretely, adding it needs:

1. a spillway rating $(C_s, L_s, z_{\text{sill}})$ per dam — not in OSM, needs
   the CWC National Register of Large Dams;
2. a design-flood family $Q_{\text{in}}^{(T)}(t)$ — from CWC gauge records or a
   regional flood-frequency relation;
3. fragility parameters $(\theta, \zeta)$ per mechanism, or the base rates of
   §B.4 as an explicit placeholder;
4. a seismic hazard curve for the site (PGA vs return period).

Items 1–2 are the shortest path: they reuse the existing routing loop and would
turn `loading` from a user-chosen fraction into a return-period-indexed
distribution, which is also the natural entry point for an uncertainty-aware
scenario generator.

## B.6 Propagating uncertainty into the hazard map

Once $P_f$ and parameter uncertainty exist, the deterministic hazard raster
becomes a probabilistic one. With breach parameters
$\boldsymbol{\theta} = (\bar B, t_f, z_s, \phi\text{-law}, n, \dots)$ drawn from
their distributions:

$$\Pr\big(d_{ij} > d^*\big) = \int \mathbb{1}\!\left[d_{ij}(\boldsymbol{\theta}) > d^*\right] p(\boldsymbol{\theta})\,d\boldsymbol{\theta}
\;\approx\; \frac{1}{N}\sum_{k=1}^{N}\mathbb{1}\!\left[d_{ij}(\boldsymbol{\theta}_k) > d^*\right]$$

The cost structure makes this tractable in a specific way: the **breach model
is milliseconds** (so $N \sim 10^4$ hydrographs is free), while the **2D solver
is 60–250 s** (so $N \sim 10$–$30$ at most). The practical scheme is therefore
a large Monte Carlo on §A.1–A.3 to get the hydrograph distribution, and a small
stratified sample of that distribution — by hydrograph peak and volume
quantile — pushed through the 2D solver. Reported as depth exceedance
probability and an arrival-time confidence band rather than a single line.

---

## References

* Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R., Perthame, B. (2004). A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows. *SIAM J. Sci. Comput.* 25(6), 2050–2065.
* Costa, J.E., Schuster, R.L. (1988). The formation and failure of natural dams. *GSA Bulletin* 100(7), 1054–1068.
* Foster, M., Fell, R., Spannagle, M. (2000). The statistics of embankment dam failures and accidents. *Can. Geotech. J.* 37(5), 1000–1024.
* Froehlich, D.C. (1995). Peak outflow from breached embankment dam. *J. Water Resour. Plan. Manage.* 121(1), 90–97.
* Froehlich, D.C. (2008). Embankment dam breach parameters and their uncertainties. *J. Hydraul. Eng.* 134(12), 1708–1721.
* Harten, A., Lax, P.D., van Leer, B. (1983). On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. *SIAM Review* 25(1), 35–61.
* Huizinga, J., de Moel, H., Szewczyk, W. (2017). *Global flood depth-damage functions.* JRC Technical Report EUR 28552 EN.
* Kurganov, A., Petrova, G. (2007). A second-order well-balanced positivity preserving central-upwind scheme for the Saint-Venant system. *Commun. Math. Sci.* 5(1), 133–160.
* MacDonald, T.C., Langridge-Monopolis, J. (1984). Breaching characteristics of dam failures. *J. Hydraul. Eng.* 110(5), 567–586.
* Monaghan, J.J. (1994). Simulating free surface flows with SPH. *J. Comput. Phys.* 110(2), 399–406.
* Ritter, A. (1892). Die Fortpflanzung der Wasserwellen. *Z. Verein. Deutsch. Ing.* 36(33), 947–954.
* Villemonte, J.R. (1947). Submerged weir discharge studies. *Engineering News-Record* 139, 866–869.
* Von Thun, J.L., Gillette, D.R. (1990). *Guidance on breach parameters.* USBR internal memorandum.
* Defra / Environment Agency (2006). *Flood risks to people, Phase 2.* FD2321/TR2.
