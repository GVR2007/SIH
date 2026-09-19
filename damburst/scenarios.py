"""Built-in study areas.

Every preset points at a real, OpenStreetMap-mapped dam on a real Indian river.
Nothing about the dam is hard-coded beyond its name and a bounding box: crest
elevation, storage and reservoir extent are all derived from the Copernicus
GLO-30 DEM at run time, and the published engineering record is pulled from
Wikidata/Wikipedia purely as a cross-check.

Any other OSM-mapped dam works too -- supply `name`, `bbox_ll` and `dam_name`
to the API instead of a preset.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from .pipeline import Scenario

PRESETS: Dict[str, dict] = {
    "tehri": {
        "title": "Tehri Dam - Bhagirathi (Ganga), Uttarakhand",
        "dam_name": "Tehri",
        "river": "Bhagirathi",
        "state": "Uttarakhand",
        "barrier_type": "engineered",
        "bbox_ll": [78.02, 29.82, 78.98, 30.78],
        "note": ("India's tallest dam (260.5 m, earth and rock-fill). The "
                 "downstream reach runs to Koteshwar and on toward Devprayag."),
        "downstream_towns": ["New Tehri", "Chamba", "Koteshwar", "Devprayag",
                             "Rishikesh"],
    },
    "koteshwar": {
        "title": "Koteshwar Dam - Bhagirathi, Uttarakhand",
        "dam_name": "Koteshwar",
        "river": "Bhagirathi",
        "state": "Uttarakhand",
        "barrier_type": "engineered",
        "bbox_ll": [78.25, 30.00, 78.72, 30.32],
        "note": ("Concrete gravity dam immediately downstream of Tehri; the "
                 "cascade case for a Tehri-triggered failure."),
        "downstream_towns": ["Devprayag"],
    },
    "bhakra": {
        "title": "Bhakra Dam - Sutlej, Himachal Pradesh / Punjab",
        "dam_name": "Bhakra",
        "river": "Sutlej",
        "state": "Himachal Pradesh",
        "barrier_type": "engineered",
        "bbox_ll": [76.30, 31.20, 76.80, 31.60],
        "note": "Concrete gravity dam, Gobind Sagar reservoir.",
        "downstream_towns": ["Nangal"],
    },
    "srisailam": {
        "title": "Srisailam Dam - Krishna, Andhra Pradesh / Telangana",
        "dam_name": "Srisailam",
        "river": "Krishna",
        "state": "Andhra Pradesh",
        "barrier_type": "engineered",
        "bbox_ll": [78.70, 16.00, 79.20, 16.40],
        "note": "Masonry gravity dam in a deep gorge on the Krishna.",
        "downstream_towns": ["Nagarjuna Sagar"],
    },
}

# Defaults tuned so a demo run finishes in a couple of minutes on a laptop.
FAST = dict(res_m=120.0, sim_hours=3.0, breach_hours=3.0, n_frames=36,
            run_sph=True, sph_dp=4.0, sph_seconds=20.0)

# Higher fidelity for the final presentation run.
FULL = dict(res_m=60.0, sim_hours=8.0, breach_hours=6.0, n_frames=72,
            run_sph=True, sph_dp=2.5, sph_seconds=45.0)


def preset_scenario(key: str, quality: str = "fast") -> Scenario:
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; available: {sorted(PRESETS)}")
    p = PRESETS[key]
    base = Scenario(
        name=key,
        bbox_ll=tuple(p["bbox_ll"]),
        dam_name=p["dam_name"],
        barrier_type=p["barrier_type"],
    )
    return replace(base, **(FULL if quality == "full" else FAST))
