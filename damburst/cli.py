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
              "validation_mode"):
        v = getattr(args, f, None)
        if v is not None:
            over[f] = v
    if args.no_sph:
        over["run_sph"] = False
    scn = replace(scn, **over)

    print(f"Scenario: {scn.name}  dam={scn.dam_name}  bbox={scn.bbox_ll}")
    print(f"  {scn.failure_mode} / {scn.barrier_type}, grid {scn.res_m} m, "
          f"{scn.sim_hours} h, SPH={'on' if scn.run_sph else 'off'}")
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
    r.add_argument("--loading", help="'FRL' or a depth fraction like 0.85")
    r.add_argument("--res-m", dest="res_m", type=float)
    r.add_argument("--sim-hours", dest="sim_hours", type=float)
    r.add_argument("--breach-hours", dest="breach_hours", type=float)
    r.add_argument("--sph-dp", dest="sph_dp", type=float)
    r.add_argument("--sph-seconds", dest="sph_seconds", type=float)
    r.add_argument("--validation-mode", dest="validation_mode",
                   choices=["context", "benchmark"])
    r.add_argument("--no-sph", action="store_true")
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
