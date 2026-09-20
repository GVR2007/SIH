"""FastAPI backend for the dam-break modelling dashboard.

Provides scenario submission, live progress, result retrieval and download of
the .shp / .kmz / GeoTIFF products.  Runs execute in a worker thread so the
dashboard can stream progress while the solver is running.
"""

from __future__ import annotations

import io
import json
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..pipeline import RUNS, ROOT, Scenario, run_scenario
from ..scenarios import PRESETS, preset_scenario

app = FastAPI(title="DamBurst - Dam Break Inundation Modelling Framework",
              version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

WEB = ROOT / "web"
_jobs: Dict[str, dict] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    preset: Optional[str] = Field(None, description="Preset key, e.g. 'tehri'")
    name: Optional[str] = None
    bbox_ll: Optional[List[float]] = None
    dam_name: Optional[str] = None
    barrier_type: Optional[str] = None
    failure_mode: Optional[str] = None
    growth_law: Optional[str] = None
    loading: Optional[str] = None
    res_m: Optional[float] = None
    sim_hours: Optional[float] = None
    breach_hours: Optional[float] = None
    n_frames: Optional[int] = None
    run_sph: Optional[bool] = None
    sph_dp: Optional[float] = None
    sph_seconds: Optional[float] = None
    manning_scale: Optional[float] = None
    validation_mode: Optional[str] = None
    sentinel1_window: Optional[List[str]] = None
    # --- previously unreachable from BOTH the CLI and this API -----------
    seed_method: Optional[str] = Field(
        None, description="auto | froehlich_2008 | von_thun_gillette | "
                          "macdonald | costa_schuster")
    inflow_m3s: Optional[float] = None
    channel_burn_m: Optional[float] = None
    tailwater: Optional[bool] = None
    tailwater_slope: Optional[float] = None
    froude_max: Optional[float] = None
    steep_slope_deg: Optional[float] = None
    baseline_window: Optional[List[str]] = Field(
        None, description="pre-event window for the permanent-water baseline; "
                          "required for an honest benchmark score")
    population_product: Optional[str] = None
    iso3: Optional[str] = None
    satellite_basemap: Optional[bool] = None
    # --- adaptive model selection (the technical novelty) ----------------
    auto_model_selection: Optional[bool] = Field(
        None, description="let the framework choose the near-field physics "
                          "from the non-hydrostatic index (default true)")
    bed_slope_deg_c: Optional[float] = None
    curvature_ratio_c: Optional[float] = None
    # --- possibility of breach -------------------------------------------
    mean_annual_flood_m3s: Optional[float] = Field(
        None, description="enables routed P(overtopping); omit to make no "
                          "probability claim")
    flood_cv: Optional[float] = None
    spillway_capacity_factor: Optional[float] = None
    spillway_crest_length_m: Optional[float] = None
    spillway_sill_m: Optional[float] = None
    # Asset unit values are an economic input, not a measurement, so they have
    # to be overridable per study area.
    asset_values: Optional[Dict[str, Any]] = Field(
        None, description="any of residential_per_building, "
                          "commercial_per_building, road_per_km, "
                          "cropland_per_hectare, road_partial_damage_factor")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    from ..core.swe2d import HAVE_NUMBA
    return {"ok": True, "numba": HAVE_NUMBA, "runs_dir": str(RUNS)}


@app.get("/api/presets")
def presets():
    return {"presets": [{"key": k, **v} for k, v in PRESETS.items()]}


@app.post("/api/run")
def start_run(req: RunRequest):
    try:
        scn = _build_scenario(req)
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(400, f"Invalid scenario: {exc}")

    run_id = f"{scn.name}-{uuid.uuid4().hex[:8]}"
    with _lock:
        _jobs[run_id] = {"run_id": run_id, "state": "queued", "pct": 0,
                         "stage": "queued", "message": "waiting for worker",
                         "log": [], "started": time.time(),
                         "scenario": scn.to_dict()}

    def _progress(p: dict):
        with _lock:
            j = _jobs[run_id]
            if p.get("pct") is not None:
                j["pct"] = p["pct"]
            j["stage"] = p.get("stage", j["stage"])
            j["message"] = p.get("message", "")
            j["log"].append({"t": round(time.time() - j["started"], 1),
                             "stage": p.get("stage"), "message": p.get("message")})
            del j["log"][:-200]

    def _worker():
        with _lock:
            _jobs[run_id]["state"] = "running"
        try:
            out = run_scenario(scn, run_id=run_id, progress=_progress)
            with _lock:
                _jobs[run_id].update(state="done", pct=100, stage="done",
                                     message="complete", dir=out["dir"])
        except Exception as exc:                        # noqa: BLE001
            with _lock:
                _jobs[run_id].update(state="error", stage="error",
                                     message=f"{type(exc).__name__}: {exc}",
                                     traceback=traceback.format_exc())

    threading.Thread(target=_worker, daemon=True).start()
    return {"run_id": run_id, "state": "queued"}


@app.get("/api/runs")
def list_runs():
    """Newest first.

    The dashboard auto-loads the first finished entry, so ordering is not
    cosmetic: listing in-memory jobs ahead of disk runs made it open whichever
    run happened to be registered first this session rather than the one the
    user just produced.
    """
    out = []
    with _lock:
        for rid, j in _jobs.items():
            row = {k: j[k] for k in ("run_id", "state", "pct", "stage",
                                     "message") if k in j}
            row["_t"] = j.get("started", 0.0)
            out.append(row)
    known = {o["run_id"] for o in out}
    for d in RUNS.glob("*"):
        if d.is_dir() and d.name not in known and (d / "results.json").exists():
            out.append({"run_id": d.name, "state": "done", "pct": 100,
                        "stage": "done", "message": "from disk",
                        "_t": d.stat().st_mtime})
    out.sort(key=lambda r: -r.get("_t", 0.0))
    for r in out:
        r.pop("_t", None)
    return {"runs": out}


@app.get("/api/runs/{run_id}/status")
def status(run_id: str):
    with _lock:
        j = _jobs.get(run_id)
        if j:
            return {k: v for k, v in j.items() if k != "traceback"}
    if (RUNS / run_id / "results.json").exists():
        return {"run_id": run_id, "state": "done", "pct": 100, "stage": "done"}
    raise HTTPException(404, "unknown run")


@app.get("/api/runs/{run_id}/results")
def results(run_id: str):
    p = RUNS / run_id / "results.json"
    if not p.exists():
        raise HTTPException(404, "results not ready")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/runs/{run_id}/manifest")
def manifest(run_id: str):
    p = RUNS / run_id / "manifest.json"
    if not p.exists():
        raise HTTPException(404, "manifest not ready")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/runs/{run_id}/sph")
def sph_snapshots(run_id: str):
    p = RUNS / run_id / "tables" / "sph_snapshots.json"
    if not p.exists():
        raise HTTPException(404, "no SPH output for this run")
    return JSONResponse(json.loads(p.read_text()))


_t3d_locks: Dict[str, threading.Lock] = {}


@app.get("/api/runs/{run_id}/file/{path:path}")
def run_file(run_id: str, path: str):
    base = (RUNS / run_id).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(404, "not found")

    # The 3D payload is DERIVED, not stored: any finished run already holds the
    # conditioned DEM, the hazard raster and the solver's depth frames, so build
    # it on first request instead of leaving every run made before the 3D view
    # existed permanently flat. Cached to disk afterwards.
    if path == "terrain3d.json" and not target.exists():
        if not (base / "results.json").exists():
            raise HTTPException(404, "run not finished")
        lock = _t3d_locks.setdefault(run_id, threading.Lock())
        with lock:                                   # concurrent tabs ask at once
            if not target.exists():
                try:
                    from ..core.export import build_terrain3d_from_run
                    build_terrain3d_from_run(base)
                except Exception as exc:             # noqa: BLE001
                    raise HTTPException(
                        422, f"could not build 3D data: {type(exc).__name__}: {exc}")

    if not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.get("/api/runs/{run_id}/download/{kind}")
def download(run_id: str, kind: str):
    base = RUNS / run_id
    if kind == "kmz":
        f = base / "vectors" / "flood_extent.kmz"
        if not f.exists():
            raise HTTPException(404, "kmz not found")
        return FileResponse(f, filename=f"{run_id}_flood_extent.kmz",
                            media_type="application/vnd.google-earth.kmz")
    if kind == "geojson":
        f = base / "vectors" / "flood_extent.geojson"
        if not f.exists():
            raise HTTPException(404, "geojson not found")
        return FileResponse(f, filename=f"{run_id}_flood_extent.geojson")

    groups = {
        "shp": list((base / "vectors").glob("flood_extent.*")),
        "rasters": list((base / "rasters").glob("*.tif")),
        "tables": list((base / "tables").glob("*.csv")),
        "all": (list((base / "vectors").glob("*")) +
                list((base / "rasters").glob("*.tif")) +
                list((base / "tables").glob("*")) +
                [base / "manifest.json", base / "results.json"]),
    }
    files = groups.get(kind)
    if not files:
        raise HTTPException(404, f"unknown download kind {kind!r}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists() and f.is_file():
                zf.write(f, arcname=f"{run_id}/{f.relative_to(base)}")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_{kind}.zip"'})


@app.get("/", response_class=HTMLResponse)
def index():
    f = WEB / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>DamBurst</h1><p>web/index.html missing</p>")
    # The dashboard is a single hand-edited file with no content hash in its
    # URL, so a browser will happily keep serving a stale copy after an edit --
    # which looks exactly like the fix not working. Never cache the shell.
    return HTMLResponse(f.read_text(), headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    })


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


# ---------------------------------------------------------------------------

def _build_scenario(req: RunRequest) -> Scenario:
    if req.preset:
        scn = preset_scenario(req.preset)
    else:
        if not (req.bbox_ll and req.dam_name and req.name):
            raise ValueError("custom runs need name, bbox_ll and dam_name")
        scn = Scenario(name=req.name, bbox_ll=tuple(req.bbox_ll),
                       dam_name=req.dam_name)

    overrides = {}
    for f in ("barrier_type", "failure_mode", "growth_law", "loading", "res_m",
              "sim_hours", "breach_hours", "n_frames", "run_sph", "sph_dp",
              "sph_seconds", "manning_scale", "validation_mode",
              "seed_method", "inflow_m3s", "channel_burn_m", "tailwater",
              "tailwater_slope", "froude_max", "steep_slope_deg",
              "population_product", "iso3", "satellite_basemap",
              "auto_model_selection", "bed_slope_deg_c",
              "curvature_ratio_c", "mean_annual_flood_m3s",
              "flood_cv", "spillway_capacity_factor",
              "spillway_crest_length_m", "spillway_sill_m"):
        v = getattr(req, f)
        if v is not None:
            overrides[f] = v
    # Setting run_sph explicitly is an override; it must disable the automatic
    # selector, or the flag would be accepted and then ignored.
    if req.run_sph is not None and req.auto_model_selection is None:
        overrides["auto_model_selection"] = False
    if req.name:
        overrides["name"] = req.name
    if req.sentinel1_window:
        overrides["sentinel1_window"] = tuple(req.sentinel1_window)
    if req.baseline_window:
        overrides["baseline_window"] = tuple(req.baseline_window)
    if req.bbox_ll:
        overrides["bbox_ll"] = tuple(req.bbox_ll)
    if req.asset_values:
        allowed = {"currency", "residential_per_building",
                   "commercial_per_building", "road_per_km",
                   "cropland_per_hectare", "road_partial_damage_factor"}
        bad = set(req.asset_values) - allowed
        if bad:
            raise ValueError(f"unknown asset_values keys: {sorted(bad)}; "
                             f"allowed: {sorted(allowed)}")
        overrides["asset_values"] = replace(scn.asset_values,
                                            **req.asset_values)
    return replace(scn, **overrides)
