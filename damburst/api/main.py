"""FastAPI backend for the dam-break modelling dashboard.

Provides scenario submission, live progress, result retrieval and download of
the .shp / .kmz / GeoTIFF products.  Runs execute in a worker thread so the
dashboard can stream progress while the solver is running.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core import datasources as ds
from ..core.hazard import AssetValues
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
    country: Optional[str] = None


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


@app.get("/api/dams")
def dam_index(country: str = "IND", q: str = "", limit: int = 400,
              min_height_m: float = 0.0):
    """Every dam in a country, from the Wikidata national index.

    The presets are five worked examples, not the extent of what the framework
    can model: the DEM, land cover, population and imagery are all fetched on
    demand for whatever domain a dam implies, so any dam in this list is
    runnable.
    """
    try:
        idx = ds.fetch_dam_index(country)
    except Exception as exc:                           # noqa: BLE001
        raise HTTPException(503, f"dam index unavailable: {exc}")
    if q:
        needle = q.lower()
        idx = [d for d in idx
               if needle in (d["name"] or "").lower()
               or needle in (d.get("admin") or "").lower()
               or needle in (d.get("river") or "").lower()]
    if min_height_m:
        idx = [d for d in idx if (d.get("height_m") or 0) >= min_height_m]
    return {"country": country.upper(), "total": len(idx),
            "dams": idx[:max(1, limit)],
            "source": "Wikidata Query Service (CC0)"}


@app.get("/api/dams/near")
def dams_near(lat: float, lon: float, radius_deg: float = 0.25,
              country: str = "IND"):
    """Dams around a clicked point, nearest first, Wikidata plus OSM."""
    try:
        found = ds.dams_near(lat, lon, radius_deg, country)
    except Exception as exc:                           # noqa: BLE001
        raise HTTPException(503, f"dam lookup failed: {exc}")
    for d in found:
        d["suggested_bbox_ll"], d["bbox_note"] = ds.auto_bbox(
            d["lat"], d["lon"], d.get("height_m"))
    return {"count": len(found), "dams": found}


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

def _slug(name: str) -> str:
    """Run-id-safe short name from a dam name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "dam"


def _build_scenario(req: RunRequest) -> Scenario:
    if req.preset:
        scn = preset_scenario(req.preset)
    elif req.bbox_ll and req.dam_name and req.name:
        scn = Scenario(name=req.name, bbox_ll=tuple(req.bbox_ll),
                       dam_name=req.dam_name)
    elif req.dam_name:
        # Dam name alone: look it up in the national index and derive a
        # screening domain from its published height. This is what makes every
        # dam in the country runnable without the caller hand-picking a box.
        match = None
        for d in ds.fetch_dam_index(req.country or "IND"):
            if d["name"].lower() == req.dam_name.strip().lower():
                match = d
                break
        if match is None:
            raise ValueError(
                f"{req.dam_name!r} is not in the {req.country or 'IND'} dam "
                f"index. Supply bbox_ll explicitly, or query /api/dams?q= to "
                f"find the exact name.")
        bbox, note = ds.auto_bbox(match["lat"], match["lon"],
                                  match.get("height_m"))
        name = req.name or _slug(match["name"])
        scn = Scenario(name=name, bbox_ll=bbox, dam_name=match["name"],
                       domain_note=note,
                       dam_lonlat=(match["lon"], match["lat"]))
    else:
        raise ValueError("a run needs a preset, a dam_name, or "
                         "name + bbox_ll + dam_name")

    overrides = {}
    for f in ("barrier_type", "failure_mode", "growth_law", "loading", "res_m",
              "sim_hours", "breach_hours", "n_frames", "run_sph", "sph_dp",
              "sph_seconds", "manning_scale", "validation_mode"):
        v = getattr(req, f)
        if v is not None:
            overrides[f] = v
    if req.name:
        overrides["name"] = req.name
    if req.sentinel1_window:
        overrides["sentinel1_window"] = tuple(req.sentinel1_window)
    if req.bbox_ll:
        overrides["bbox_ll"] = tuple(req.bbox_ll)
    return replace(scn, **overrides)
