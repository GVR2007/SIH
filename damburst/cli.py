"""Command-line entry point.

    python -m damburst.cli list
    python -m damburst.cli run tehri --quality fast --failure-mode piping
    python -m damburst.cli serve --port 8000
    python -m damburst.cli verify
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .pipeline import RUNS, Scenario, run_scenario
from .scenarios import FAST, FULL, PRESETS, preset_scenario


def cmd_list(args):
    print(f"{'key':<12} {'river':<14} {'state':<20} title")
    print("-" * 92)
    for k, p in PRESETS.items():
        print(f"{k:<12} {p['river']:<14} {p['state']:<20} {p['title']}")
    print("\nAny other OpenStreetMap-mapped dam works too:")
    print("  python -m damburst.cli run --custom NAME --bbox W S E N --dam 'Dam Name'")


def cmd_run(args):
    if args.custom:
        if not (args.bbox and args.dam):
            sys.exit("--custom needs --bbox W S E N and --dam 'Name'")
        scn = Scenario(name=args.custom, bbox_ll=tuple(args.bbox), dam_name=args.dam)
        scn = replace(scn, **(FULL if args.quality == "full" else FAST))
    else:
        scn = preset_scenario(args.preset, args.quality)

    over = {}
    for f in ("barrier_type", "failure_mode", "growth_law", "loading",
              "res_m", "sim_hours", "breach_hours", "sph_dp", "sph_seconds",
              "validation_mode", "seed_method", "inflow_m3s", "channel_burn_m",
              "manning_scale", "n_frames", "population_product", "iso3",
              "froude_max", "steep_slope_deg", "tailwater_slope",
              "bed_slope_deg_c", "curvature_ratio_c",
              "mean_annual_flood_m3s", "flood_cv",
              "spillway_capacity_factor", "spillway_crest_length_m",
              "spillway_sill_m"):
        v = getattr(args, f, None)
        if v is not None:
            over[f] = v
    if args.no_sph:
        # An explicit --no-sph IS an override, so it must also switch off the
        # automatic selector -- otherwise the framework would cheerfully
        # decide to run SPH anyway and the flag would silently do nothing.
        over["run_sph"] = False
        over["auto_model_selection"] = False
    if args.no_tailwater:
        over["tailwater"] = False
    if args.no_auto_model:
        over["auto_model_selection"] = False
    if args.no_satellite:
        over["satellite_basemap"] = False
    if args.sentinel1_window:
        over["sentinel1_window"] = tuple(args.sentinel1_window)
    if args.baseline_window:
        over["baseline_window"] = tuple(args.baseline_window)

    # Asset unit values are an economic input, so they have to be overridable
    # from the command line -- `hazard.py` calls them "explicit, overridable
    # scenario parameters" and until now neither interface exposed them.
    av = {}
    for f in ("currency", "residential_per_building", "commercial_per_building",
              "road_per_km", "cropland_per_hectare"):
        v = getattr(args, f, None)
        if v is not None:
            av[f] = v
    if av:
        over["asset_values"] = replace(scn.asset_values, **av)

    scn = replace(scn, **over)

    print(f"Scenario: {scn.name}  dam={scn.dam_name}  bbox={scn.bbox_ll}")
    sph_mode = ("auto (the framework decides from the non-hydrostatic index)"
                if scn.auto_model_selection
                else ("forced on" if scn.run_sph else "forced off"))
    print(f"  {scn.failure_mode} / {scn.barrier_type}, grid {scn.res_m} m, "
          f"{scn.sim_hours} h, near-field model: {sph_mode}")
    out = run_scenario(scn)

    r = out["results"]
    print("\n" + "=" * 70)
    print(f"RUN {out['run_id']}  ->  {out['dir']}")
    print("=" * 70)
    res, br, hz = r["reservoir"], r["breach"], r["hazard"]
    print(f"Reservoir   capacity {res['capacity_mcm']:.1f} MCM, "
          f"max depth {res['max_depth_m']:.1f} m, "
          f"surface {res['area_at_crest_km2']:.2f} km2")
    print(f"Breach      peak {br['peak_q_m3s']:,.0f} m3/s at "
          f"{br['t_peak_min']:.1f} min, released {br['released_mcm']:.1f} MCM, "
          f"final width {br['final_top_width_m']:.0f} m")
    print(f"Inundation  {hz['inundated_area_km2']:.2f} km2, "
          f"max depth {hz['max_depth_m']:.1f} m, "
          f"max velocity {hz['max_velocity_ms']:.1f} m/s")
    pop = r["impact"]["population"]
    print(f"Exposure    {pop['total_exposed']:,.0f} people, "
          f"{r['impact']['buildings']['exposed']:,} buildings, "
          f"{r['impact']['roads']['total_inundated_km']:.1f} km road/rail, "
          f"{len(r['impact']['settlements'])} settlements")
    ms = r.get("model_selection")
    if ms:
        print("\nModel selection (the framework's own decision):")
        print("  " + _wrap(ms["reason"], 68))
    adq = r.get("model_adequacy")
    if adq:
        print("\nAdequacy of the solution produced:")
        print("  " + _wrap(adq["verdict"], 68))
    fp = r.get("failure_probability") or {}
    if fp.get("annual_probability_of_failure"):
        print(f"\nPossibility of breach   P(failure) ~ "
              f"{fp['annual_probability_of_failure']:.2e}/y "
              f"(1 in {fp['return_period_y']:,.0f} y)")
        ot = fp["mechanisms"]["overtopping"]
        if ot.get("computed"):
            first = next((x["return_period_y"] for x in ot["by_return_period"]
                          if x["overtopped"]), None)
            print(f"  overtopping routed; first overtops at T = "
                  f"{first if first else '>10000'} y")
        print("  Every hazard/exposure figure above is CONDITIONAL on failure.")
    elif fp.get("computed") is False:
        print("\nPossibility of breach   not computed "
              "(no flood statistics supplied)")
        print("  Every hazard/exposure figure above is CONDITIONAL on failure.")

    print("\nModel comparison:")
    for m in r["model_comparison"]:
        print(f"  {m['model']:<18} peakQ={_f(m['peak_inflow_m3s'])} "
              f"area={_f(m['inundated_km2'])} km2  maxd={_f(m['max_depth_m'])} m  "
              f"cpu={_f(m['wallclock_s'])} s")
    print("\nExports:", ", ".join(
        [r["exports"]["shapefile"], r["exports"]["kmz"], r["exports"]["geojson"]]
        + r["exports"]["rasters"][:3] + ["..."]))
    print(f"\nOpen the dashboard:  python -m damburst.cli serve")


def _f(v, d=1):
    return "-" if v is None else f"{v:,.{d}f}"


def _wrap(text, width=68, indent="  "):
    import textwrap
    return ("\n" + indent).join(textwrap.wrap(str(text), width))


def cmd_serve(args):
    import uvicorn
    print(f"Dashboard -> http://{args.host}:{args.port}/")
    uvicorn.run("damburst.api.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


def cmd_verify(args):
    """Run the analytical solver benchmarks."""
    here = Path(__file__).resolve().parents[1] / "tests" / "test_swe_benchmarks.py"
    import runpy
    sys.argv = [str(here)]
    runpy.run_path(str(here), run_name="__main__")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="damburst",
        description="Dam-break inundation modelling: SPH + 2D hydrodynamics "
                    "on real open geospatial data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list built-in study areas").set_defaults(fn=cmd_list)

    r = sub.add_parser("run", help="run a scenario end to end")
    r.add_argument("preset", nargs="?", default="tehri", choices=list(PRESETS))
    r.add_argument("--custom", help="custom run name (with --bbox and --dam)")
    r.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    r.add_argument("--dam", help="dam name as mapped in OpenStreetMap")
    r.add_argument("--quality", choices=["fast", "full"], default="fast")
    r.add_argument("--barrier-type", dest="barrier_type",
                   choices=["engineered", "natural"])
    r.add_argument("--failure-mode", dest="failure_mode",
                   choices=["overtopping", "piping", "progressive_erosion",
                            "instantaneous"])
    r.add_argument("--growth-law", dest="growth_law",
                   choices=["sine", "linear", "erosion"])
    r.add_argument("--seed-method", dest="seed_method",
                   choices=["auto", "froehlich_2008", "von_thun_gillette",
                            "macdonald", "costa_schuster"],
                   help="empirical breach-geometry regression (default: auto)")
    r.add_argument("--loading", help="'FRL' or a depth fraction like 0.85")
    r.add_argument("--inflow-m3s", dest="inflow_m3s", type=float,
                   help="constant upstream inflow Q_in during the breach")
    r.add_argument("--no-tailwater", action="store_true",
                   help="disable the downstream rating; the breach then "
                        "discharges freely and peak Q is an upper bound")
    r.add_argument("--tailwater-slope", dest="tailwater_slope", type=float,
                   help="override the DEM-derived receiving-reach slope")
    r.add_argument("--res-m", dest="res_m", type=float)
    r.add_argument("--sim-hours", dest="sim_hours", type=float)
    r.add_argument("--breach-hours", dest="breach_hours", type=float)
    r.add_argument("--n-frames", dest="n_frames", type=int)
    r.add_argument("--channel-burn-m", dest="channel_burn_m", type=float,
                   help="burn the OSM waterway centreline this deep into the "
                        "DEM to recover sub-pixel channel conveyance")
    r.add_argument("--manning-scale", dest="manning_scale", type=float)
    r.add_argument("--froude-max", dest="froude_max", type=float,
                   help="Froude cap on steep cells; 0 disables the limiter")
    r.add_argument("--steep-slope-deg", dest="steep_slope_deg", type=float,
                   help="bed slope above which the Froude cap applies")
    r.add_argument("--sph-dp", dest="sph_dp", type=float)
    r.add_argument("--sph-seconds", dest="sph_seconds", type=float)
    r.add_argument("--validation-mode", dest="validation_mode",
                   choices=["context", "benchmark"])
    r.add_argument("--sentinel1-window", dest="sentinel1_window", nargs=2,
                   metavar=("START", "END"),
                   help="event window, e.g. 2023-08-10 2023-08-20")
    r.add_argument("--baseline-window", dest="baseline_window", nargs=2,
                   metavar=("START", "END"),
                   help="pre-event window for the permanent-water baseline "
                        "(benchmark mode; without it the score is inflated)")
    r.add_argument("--population-product", dest="population_product",
                   choices=["1km", "100m"])
    r.add_argument("--iso3", dest="iso3")
    r.add_argument("--no-satellite", action="store_true",
                   help="skip the Sentinel-2 basemap mosaic")
    # -- adaptive model selection -----------------------------------------
    r.add_argument("--no-auto-model", action="store_true",
                   help="disable adaptive model selection and honour --no-sph "
                        "literally instead of letting the framework decide")
    r.add_argument("--bed-slope-deg-c", dest="bed_slope_deg_c", type=float,
                   help="bed slope at which depth-averaging is judged invalid")
    r.add_argument("--curvature-ratio-c", dest="curvature_ratio_c", type=float,
                   help="vertical-acceleration/g ratio threshold")
    # -- possibility of breach ---------------------------------------------
    r.add_argument("--mean-annual-flood", dest="mean_annual_flood_m3s",
                   type=float,
                   help="mean annual flood at the site (m3/s). Supplying it "
                        "enables routed P(overtopping); without it the run "
                        "makes no probability claim.")
    r.add_argument("--flood-cv", dest="flood_cv", type=float,
                   help="coefficient of variation of the annual flood series")
    r.add_argument("--spillway-capacity-factor",
                   dest="spillway_capacity_factor", type=float,
                   help="scale the placeholder spillway; sweep it to see how "
                        "much P(overtopping) depends on an unknown")
    r.add_argument("--spillway-crest-length", dest="spillway_crest_length_m",
                   type=float, help="real spillway crest length (m), CWC register")
    r.add_argument("--spillway-sill", dest="spillway_sill_m", type=float,
                   help="real spillway sill elevation (m MSL)")
    r.add_argument("--no-sph", action="store_true")
    # -- asset unit values (economic inputs, not measurements) -------------
    r.add_argument("--currency", dest="currency")
    r.add_argument("--value-residential", dest="residential_per_building",
                   type=float, help="replacement value per residential building")
    r.add_argument("--value-commercial", dest="commercial_per_building",
                   type=float)
    r.add_argument("--value-road-per-km", dest="road_per_km", type=float)
    r.add_argument("--value-cropland-per-ha", dest="cropland_per_hectare",
                   type=float)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="start the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    sub.add_parser("verify", help="run solver analytical benchmarks"
                   ).set_defaults(fn=cmd_verify)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
