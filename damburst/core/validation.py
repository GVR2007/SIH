"""Validation against observed satellite flood extent.

Implements the "Sentinel-1 / Sentinel-2 -> OBSERVED FLOOD EXTENT ->
VALIDATION (IoU / CSI / NSE) -> CALIBRATION + LIVE FLOOD LAYER" branch, and the
"framework for near real time flood analysis through open source data"
deliverable.

Observed extent comes from real Sentinel-1 GRD scenes served by the Microsoft
Planetary Computer (free SAS token, no account).  SAR is used rather than
optical because it sees through the cloud that accompanies a flood.

Honest-comparison note
----------------------
A dam-break simulation is a *hypothetical* event; there is no satellite image of
a flood that has not happened.  So this module is used in two distinct ways and
the run manifest always records which:

  `benchmark`  the model is run for a real, observed historical flood event and
               scored against the SAR extent for that date.  This is the only
               mode that measures predictive skill.
  `context`    a SAR scene is shown as the pre-event baseline water extent
               (reservoir + river), so the dashboard can distinguish permanent
               water from newly inundated land.  This is NOT a skill score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Categorical extent metrics
# ---------------------------------------------------------------------------

def contingency(model: np.ndarray, observed: np.ndarray,
                valid: Optional[np.ndarray] = None) -> Dict[str, int]:
    m = model.astype(bool)
    o = observed.astype(bool)
    if valid is not None:
        m = m & valid
        o = o & valid
        dom = valid
    else:
        dom = np.ones_like(m, dtype=bool)
    return {
        "hits": int((m & o).sum()),
        "false_alarms": int((m & ~o & dom).sum()),
        "misses": int((~m & o & dom).sum()),
        "correct_negatives": int((~m & ~o & dom).sum()),
    }


def extent_metrics(model: np.ndarray, observed: np.ndarray,
                   valid: Optional[np.ndarray] = None) -> Dict[str, object]:
    """IoU / CSI, probability of detection, false-alarm ratio, bias."""
    c = contingency(model, observed, valid)
    h, f, m_, cn = (c["hits"], c["false_alarms"], c["misses"],
                    c["correct_negatives"])
    union = h + f + m_
    return {
        **c,
        "iou": round(h / union, 4) if union else None,
        "csi": round(h / union, 4) if union else None,     # CSI == IoU here
        "pod": round(h / (h + m_), 4) if (h + m_) else None,
        "far": round(f / (h + f), 4) if (h + f) else None,
        "bias": round((h + f) / (h + m_), 4) if (h + m_) else None,
        "accuracy": round((h + cn) / max(h + f + m_ + cn, 1), 4),
        "f1": round(2 * h / (2 * h + f + m_), 4) if (2 * h + f + m_) else None,
    }


# ---------------------------------------------------------------------------
# Continuous metrics
# ---------------------------------------------------------------------------

def nash_sutcliffe(obs: Sequence[float], sim: Sequence[float]) -> float:
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    if obs.size < 2:
        return float("nan")
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def kling_gupta(obs: Sequence[float], sim: Sequence[float]) -> float:
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    if obs.size < 2 or obs.std() == 0 or sim.std() == 0:
        return float("nan")
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean()) if obs.mean() != 0 else float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(obs: Sequence[float], sim: Sequence[float]) -> float:
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((obs[ok] - sim[ok]) ** 2)))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    mode: str                              # "benchmark" | "context"
    scene: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, object] = field(default_factory=dict)
    permanent_water_km2: Optional[float] = None
    model_extent_km2: Optional[float] = None
    observed_extent_km2: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "scene": self.scene,
            "metrics": self.metrics,
            "permanent_water_km2": self.permanent_water_km2,
            "model_extent_km2": self.model_extent_km2,
            "observed_extent_km2": self.observed_extent_km2,
            "notes": self.notes,
        }


def validate_extent(model_wet: np.ndarray, observed_water: np.ndarray,
                    cell_area: float, mode: str = "context",
                    baseline_water: Optional[np.ndarray] = None,
                    scene: Optional[dict] = None,
                    valid: Optional[np.ndarray] = None) -> ValidationReport:
    """Score (benchmark) or contextualise (context) the modelled extent.

    In `benchmark` mode any permanent water present before the event is removed
    from both the model and the observation, so the score reflects the flood
    signal and is not inflated by the river and reservoir always being wet.
    """
    rep = ValidationReport(mode=mode, scene=scene or {})

    obs = observed_water.astype(bool)
    mod = model_wet.astype(bool)

    if baseline_water is not None:
        base = baseline_water.astype(bool)
        rep.permanent_water_km2 = round(float(base.sum()) * cell_area / 1e6, 4)
        if mode == "benchmark":
            obs = obs & ~base
            mod = mod & ~base

    rep.model_extent_km2 = round(float(mod.sum()) * cell_area / 1e6, 4)
    rep.observed_extent_km2 = round(float(obs.sum()) * cell_area / 1e6, 4)

    if mode == "benchmark":
        rep.metrics = extent_metrics(mod, obs, valid)
        rep.notes.append(
            "Scored against an observed Sentinel-1 flood extent for a real "
            "event; permanent water removed from both layers.")
    else:
        rep.metrics = {
            "overlap_with_observed_water_km2":
                round(float((mod & obs).sum()) * cell_area / 1e6, 4),
            "modelled_beyond_observed_water_km2":
                round(float((mod & ~obs).sum()) * cell_area / 1e6, 4),
        }
        rep.notes.append(
            "CONTEXT ONLY - not a skill score. The dam-break scenario is "
            "hypothetical, so no satellite image of it exists. The SAR scene "
            "shows the pre-event water body; the difference is the land the "
            "scenario would newly inundate.")
    return rep


def terrain_slope_deg(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    gy, gx = np.gradient(z, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))
