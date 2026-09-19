"""Depth-damage functions and asset values from the published international record.

Nothing in this module invents a damage curve or a unit rate.  Both come from

    Huizinga, J., De Moel, H., Szewczyk, W. (2017).  Global flood depth-damage
    functions: Methodology and the database with guidelines.  EUR 28552 EN,
    Publications Office of the European Union.  doi:10.2760/16510, JRC105688.

which is the reference set used by the JRC's own global flood risk work, by the
World Bank / GFDRR CCDR screening tools and by Deltares' FIAT.  The extracted
tables live in `data/reference/jrc_flood_damage.json` with their citation, so a
reviewer can check every number against the source PDF.

Two things the JRC database gives us that a hand-written table cannot:

  * a damage curve *per continent* rather than one global curve, and
  * a maximum damage value *per country*, from international construction-cost
    surveys (EC Harris 2010, Turner & Townsend 2013) for buildings and from
    World Bank WDI agricultural value added for cropland.

Price-level and currency conversion is done live against the World Bank
indicator API rather than with a frozen exchange rate -- see `convert_value`.
The JRC tables are EUR at 2010 price level; every figure this module returns
records the conversion chain that produced it.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REFERENCE = (Path(__file__).resolve().parents[2]
             / "data" / "reference" / "jrc_flood_damage.json")

# Asset classes the JRC database distinguishes.
ASSET_CLASSES = ("residential", "commercial", "industrial",
                 "transport", "infrastructure", "agriculture")

# Continent of a country, for curve selection.  Only the countries present in
# the JRC cost tables need an entry; `continent_for` falls back to the generic
# global curve when a country is not listed, which is exactly what the JRC
# guidelines prescribe for unlisted countries.
_CONTINENT = {
    "IND": "Asia", "BGD": "Asia", "PAK": "Asia", "NPL": "Asia", "LKA": "Asia",
    "CHN": "Asia", "JPN": "Asia", "KOR": "Asia", "THA": "Asia", "VNM": "Asia",
    "IDN": "Asia", "MYS": "Asia", "PHL": "Asia", "KHM": "Asia", "LAO": "Asia",
    "MMR": "Asia", "BTN": "Asia", "AFG": "Asia", "IRN": "Asia", "TUR": "Asia",
    "USA": "North America", "CAN": "North America", "MEX": "North America",
    "BRA": "South America", "ARG": "South America", "CHL": "South America",
    "COL": "South America", "PER": "South America",
    "AUS": "Oceania", "NZL": "Oceania",
    "ZAF": "Africa", "EGY": "Africa", "NGA": "Africa", "KEN": "Africa",
    "GHA": "Africa", "TZA": "Africa", "UGA": "Africa", "MOZ": "Africa",
}

# JRC cost tables are keyed by country name, not ISO3.
_ISO3_TO_JRC_NAME = {
    "IND": "India", "CHN": "China", "JPN": "Japan", "KOR": "South Korea",
    "THA": "Thailand", "VNM": "Vietnam", "IDN": "Indonesia", "MYS": "Malaysia",
    "LKA": "Sri Lanka", "TUR": "Turkey", "USA": "USA", "CAN": "Canada",
    "BRA": "Brazil", "AUS": "Australia", "NZL": "New Zealand",
    "ZAF": "South Africa", "EGY": "Egypt", "GHA": "Ghana", "UGA": "Uganda",
    "GBR": "United Kingdom", "DEU": "Germany", "FRA": "France",
    "NLD": "Netherlands", "ITA": "Italy", "ESP": "Spain", "RUS": "Russia",
    "SGP": "Singapore", "ARE": "United Arab Emirates", "QAT": "Qatar",
    "OMN": "Oman", "POL": "Poland", "IRL": "Ireland", "TWN": "Taiwan",
    "HKG": "Hong Kong", "CHE": "Switzerland", "SWE": "Sweden",
    "DNK": "Denmark", "FIN": "Finland", "AUT": "Austria", "BEL": "Belgium",
    "PRT": "Portugal", "GRC": "Greece", "ROU": "Romania", "UKR": "Ukraine",
    "SAU": "Saudi Arabia", "BHR": "Bahrain", "HUN": "Hungary",
    "CZE": "Czech Republic", "SVK": "Slovakia", "HRV": "Croatia",
    "SRB": "Serbia", "BGR": "Bulgaria", "LVA": "Latvia", "CMR": "Cameroon",
}

# WDI agriculture table is keyed by the World Bank country name.
_ISO3_TO_WDI_NAME = dict(_ISO3_TO_JRC_NAME, USA="United States",
                         RUS="Russian Federation", KOR="Korea, Rep.",
                         HKG="Hong Kong SAR, China", EGY="Egypt, Arab Rep.",
                         VNM="Vietnam", IRN="Iran, Islamic Rep.")


class ReferenceUnavailable(RuntimeError):
    """Raised when a published value or a live conversion factor is missing."""


# ---------------------------------------------------------------------------
# Reference database
# ---------------------------------------------------------------------------

_DB: Optional[dict] = None


def db() -> dict:
    global _DB
    if _DB is None:
        if not REFERENCE.exists():
            raise ReferenceUnavailable(
                f"JRC reference database missing at {REFERENCE}. "
                "It is extracted from the published JRC105688 PDF; see the "
                "module docstring.")
        _DB = json.loads(REFERENCE.read_text())
    return _DB


def citation() -> str:
    return db()["_citation"]


def continent_for(iso3: str) -> Optional[str]:
    return _CONTINENT.get(iso3.upper())


# ---------------------------------------------------------------------------
# Damage curves
# ---------------------------------------------------------------------------

@dataclass
class DamageCurve:
    """A published depth-damage function.  `factor(depth)` is dimensionless."""
    asset_class: str
    depth_m: np.ndarray
    damage_factor: np.ndarray
    source: str

    def factor(self, depth) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float64)
        return np.interp(d, self.depth_m, self.damage_factor,
                         left=0.0, right=float(self.damage_factor[-1]))

    def to_dict(self) -> dict:
        return {"asset_class": self.asset_class,
                "depth_m": self.depth_m.tolist(),
                "damage_factor": self.damage_factor.tolist(),
                "source": self.source}


def curve(asset_class: str, iso3: str = "IND") -> DamageCurve:
    """The published curve for this asset class in this country's continent.

    Falls back to the JRC generic global function (Table 5-4) where the
    continent has no published curve for the class -- which is what the JRC
    guidelines prescribe, not an invention of ours.
    """
    if asset_class not in ASSET_CLASSES:
        raise ValueError(f"unknown asset class {asset_class!r}; "
                         f"available: {ASSET_CLASSES}")
    d = db()
    cont = continent_for(iso3)
    node = (d["curves"].get(cont) or {}).get(asset_class)
    if node is not None:
        src = (f"JRC Huizinga et al. (2017), average continental damage "
               f"function for {cont} - {asset_class}")
    else:
        node = d["global_generic"].get(asset_class)
        if node is None:
            # No continental curve and no generic one: infrastructure and
            # agriculture are the only classes with generic curves, so for the
            # building classes use the nearest published continental set.
            for fallback in ("Asia", "Europe", "North America"):
                node = (d["curves"].get(fallback) or {}).get(asset_class)
                if node is not None:
                    src = (f"JRC Huizinga et al. (2017), {fallback} curve used "
                           f"for {asset_class} (no curve published for {cont})")
                    break
            if node is None:
                raise ReferenceUnavailable(
                    f"No published curve for {asset_class} in {cont or iso3}")
        else:
            src = (f"JRC Huizinga et al. (2017) Table 5-4 generic global "
                   f"function for {asset_class} (no {cont} curve published)")
    return DamageCurve(asset_class=asset_class,
                       depth_m=np.asarray(node["depth_m"], dtype=np.float64),
                       damage_factor=np.asarray(node["damage_factor"],
                                                dtype=np.float64),
                       source=src)


def all_curves(iso3: str = "IND") -> Dict[str, DamageCurve]:
    return {c: curve(c, iso3) for c in ASSET_CLASSES}


# ---------------------------------------------------------------------------
# Currency and price level -- live, never a frozen rate
# ---------------------------------------------------------------------------

WB_API = "https://api.worldbank.org/v2"
# LCU per US$, period average.  EMU gives EUR per US$.
WB_FX = "PA.NUS.FCRF"
# GDP deflator, index (2015 = 100 in the current WDI vintage).
WB_DEFLATOR = "NY.GDP.DEFL.ZS"

_wb_cache: Dict[Tuple[str, str], Dict[int, float]] = {}


def _wb_series(iso3: str, indicator: str) -> Dict[int, float]:
    """Fetch a World Bank indicator series.  Free, no key, no registration."""
    key = (iso3.upper(), indicator)
    if key in _wb_cache:
        return _wb_cache[key]
    from .datasources import http_json, SourceUnavailable
    url = (f"{WB_API}/country/{iso3}/indicator/{indicator}"
           f"?format=json&per_page=200")
    try:
        j = http_json(url)
    except SourceUnavailable as exc:
        raise ReferenceUnavailable(
            f"World Bank {indicator} for {iso3} unavailable: {exc}") from exc
    if not isinstance(j, list) or len(j) < 2 or not j[1]:
        raise ReferenceUnavailable(
            f"World Bank returned no {indicator} series for {iso3}")
    series = {int(r["date"]): float(r["value"])
              for r in j[1] if r.get("value") is not None}
    if not series:
        raise ReferenceUnavailable(
            f"World Bank {indicator} series for {iso3} is empty")
    _wb_cache[key] = series
    return series


def _nearest(series: Dict[int, float], year: int) -> Tuple[int, float]:
    if year in series:
        return year, series[year]
    y = min(series, key=lambda k: (abs(k - year), -k))
    return y, series[y]


@dataclass
class Conversion:
    """A fully documented EUR(2010) -> local-currency(target year) factor."""
    factor: float
    currency: str
    price_year: int
    chain: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"factor": round(self.factor, 4), "currency": self.currency,
                "price_year": self.price_year, "chain": self.chain}


def eur2010_to_local(iso3: str = "IND", target_year: Optional[int] = None
                     ) -> Conversion:
    """Convert a JRC EUR-2010 unit value into present-day local currency.

    Chain, every step from the World Bank indicator API at run time:

        EUR(2010) -> USD(2010)          EMU official exchange rate, 2010
        USD(2010) -> LCU(2010)          country official exchange rate, 2010
        LCU(2010) -> LCU(target year)   country GDP deflator ratio

    Nothing here is a stored constant, so the rupee figures move with the
    published exchange rate and deflator instead of ageing silently.
    """
    iso3 = iso3.upper()
    target_year = target_year or datetime.date.today().year
    chain: List[str] = []

    emu = _wb_series("EMU", WB_FX)
    y_emu, eur_per_usd = _nearest(emu, 2010)
    chain.append(f"EUR->USD: World Bank {WB_FX} EMU {y_emu} = "
                 f"{eur_per_usd:.4f} EUR/USD")

    lcu = _wb_series(iso3, WB_FX)
    y_lcu, lcu_per_usd = _nearest(lcu, 2010)
    chain.append(f"USD->LCU: World Bank {WB_FX} {iso3} {y_lcu} = "
                 f"{lcu_per_usd:.4f} LCU/USD")

    defl = _wb_series(iso3, WB_DEFLATOR)
    y0, d0 = _nearest(defl, 2010)
    y1, d1 = _nearest(defl, target_year)
    chain.append(f"price level: World Bank {WB_DEFLATOR} {iso3} "
                 f"{y0}={d0:.2f} -> {y1}={d1:.2f}")

    factor = (1.0 / eur_per_usd) * lcu_per_usd * (d1 / d0)
    return Conversion(factor=factor, currency=_currency_of(iso3),
                      price_year=y1, chain=chain)


_CURRENCY = {"IND": "INR", "USA": "USD", "GBR": "GBP", "JPN": "JPY",
             "CHN": "CNY", "BRA": "BRL", "ZAF": "ZAR", "AUS": "AUD",
             "CAN": "CAD", "IDN": "IDR", "VNM": "VND", "BGD": "BDT",
             "PAK": "PKR", "NPL": "NPR", "LKA": "LKR", "THA": "THB"}


def _currency_of(iso3: str) -> str:
    return _CURRENCY.get(iso3.upper(), "LCU")


# ---------------------------------------------------------------------------
# Maximum damage values
# ---------------------------------------------------------------------------

@dataclass
class MaxDamage:
    """Published maximum damage for one asset class, in local currency."""
    asset_class: str
    value: float
    unit: str                    # "per_m2", "per_m", "per_ha"
    currency: str
    price_year: int
    source: str
    conversion: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"asset_class": self.asset_class, "value": round(self.value, 2),
                "unit": self.unit, "currency": self.currency,
                "price_year": self.price_year, "source": self.source,
                "conversion": self.conversion}


def _building_cost_eur_m2(iso3: str, key: str) -> Tuple[float, str]:
    """Construction cost from JRC Appendix B, preferring the newer survey."""
    name = _ISO3_TO_JRC_NAME.get(iso3.upper())
    table = db()["construction_costs_eur_m2_2010"]
    rec = table.get(name) if name else None
    if not rec:
        raise ReferenceUnavailable(
            f"JRC Appendix B has no construction cost for {iso3} "
            f"({name or 'unmapped'}). Supply an explicit override.")
    for survey, label in (("turner_townsend_2013", "Turner & Townsend (2013)"),
                          ("ec_harris_2010", "EC Harris (2010)")):
        if survey in rec and key in rec[survey]:
            return float(rec[survey][key]), (
                f"JRC Huizinga et al. (2017) Appendix B Table B-2, {label}, "
                f"{name}: {rec[survey][key]} EUR/m2 (2010 price level)")
    raise ReferenceUnavailable(
        f"No {key} construction cost for {name} in JRC Appendix B")


def max_damage(asset_class: str, iso3: str = "IND",
               target_year: Optional[int] = None,
               conversion: Optional[Conversion] = None) -> MaxDamage:
    """Published maximum damage for an asset class, in local currency.

    Buildings come from the international construction-cost surveys compiled in
    JRC Appendix B (EUR/m2).  Roads come from the continental infrastructure
    value (EUR/m).  Cropland comes from World Bank agricultural value added per
    hectare (US$/ha), which is the JRC's own basis for agriculture.
    """
    iso3 = iso3.upper()
    conv = conversion or eur2010_to_local(iso3, target_year)
    d = db()

    if asset_class in ("residential", "commercial", "industrial"):
        key = {"residential": "res", "commercial": "com",
               "industrial": "ind"}[asset_class]
        eur, src = _building_cost_eur_m2(iso3, key)
        return MaxDamage(asset_class, eur * conv.factor, "per_m2",
                         conv.currency, conv.price_year, src, conv.to_dict())

    if asset_class in ("infrastructure", "transport"):
        cont = continent_for(iso3) or "Asia"
        per_m = d["infrastructure_eur_per_m_2010"].get(cont)
        if per_m is None:
            raise ReferenceUnavailable(
                f"No published infrastructure value for {cont}")
        return MaxDamage(
            asset_class, float(per_m) * conv.factor, "per_m",
            conv.currency, conv.price_year,
            f"JRC Huizinga et al. (2017) section 3.2.5, {cont} infrastructure: "
            f"{per_m} EUR/m (2010 price level)", conv.to_dict())

    if asset_class == "agriculture":
        name = _ISO3_TO_WDI_NAME.get(iso3) or _ISO3_TO_JRC_NAME.get(iso3)
        va = d["agriculture_va_usd_ha"].get(name) if name else None
        if va is None:
            raise ReferenceUnavailable(
                f"JRC Appendix A has no agricultural value added for {iso3}")
        # Appendix A is US$; route it through the same published FX series.
        lcu = _wb_series(iso3, WB_FX)
        y, lcu_per_usd = _nearest(lcu, 2010)
        defl = _wb_series(iso3, WB_DEFLATOR)
        y0, d0 = _nearest(defl, 2010)
        y1, d1 = _nearest(defl, conv.price_year)
        value = float(va) * lcu_per_usd * (d1 / d0)
        return MaxDamage(
            asset_class, value, "per_ha", conv.currency, y1,
            f"JRC Huizinga et al. (2017) Appendix A, {name}: {va} US$/ha "
            f"agricultural value added (World Bank WDI)",
            {"factor": round(lcu_per_usd * (d1 / d0), 4),
             "currency": conv.currency, "price_year": y1,
             "chain": [f"USD->LCU: World Bank {WB_FX} {iso3} {y} = "
                       f"{lcu_per_usd:.4f} LCU/USD",
                       f"price level: {WB_DEFLATOR} {y0}={d0:.2f} -> "
                       f"{y1}={d1:.2f}"]})

    raise ValueError(f"unknown asset class {asset_class!r}")


def max_damage_set(iso3: str = "IND", target_year: Optional[int] = None
                   ) -> Dict[str, MaxDamage]:
    """Every asset class at once, sharing one conversion chain."""
    conv = eur2010_to_local(iso3, target_year)
    out = {}
    for c in ASSET_CLASSES:
        try:
            out[c] = max_damage(c, iso3, conversion=conv)
        except ReferenceUnavailable:
            continue
    return out


# ---------------------------------------------------------------------------
# OSM building -> asset class
# ---------------------------------------------------------------------------

# OpenStreetMap `building=*` values, grouped onto the JRC asset classes.
# https://wiki.openstreetmap.org/wiki/Key:building
_BUILDING_CLASS = {
    "residential": {
        "yes", "house", "residential", "apartments", "detached", "terrace",
        "semidetached_house", "bungalow", "dormitory", "hut", "cabin",
        "farm", "static_caravan", "houseboat", "ger", "annexe",
    },
    "commercial": {
        "commercial", "retail", "shop", "supermarket", "kiosk", "office",
        "hotel", "restaurant", "mall", "marketplace",
    },
    "industrial": {
        "industrial", "warehouse", "factory", "manufacture", "works",
        "hangar", "storage_tank", "silo", "digester", "brewery",
    },
    "infrastructure": {
        "hospital", "school", "university", "college", "public",
        "civic", "government", "train_station", "transportation",
        "fire_station", "police", "toilets", "service", "bridge",
    },
    "agriculture": {
        "barn", "cowshed", "farm_auxiliary", "greenhouse", "stable",
        "sty", "livestock", "slurry_tank",
    },
}
_LOOKUP = {v: cls for cls, vals in _BUILDING_CLASS.items() for v in vals}


def classify_building(tags: Optional[dict]) -> Tuple[str, str]:
    """Map an OSM building's tags onto a JRC asset class.

    Returns (asset_class, basis).  `building=yes` carries no information, so a
    more specific tag is consulted before falling back to residential -- which
    is the dominant class in the OSM building stock and the JRC default.
    """
    t = tags or {}
    b = (t.get("building") or "").lower()
    if b and b != "yes":
        cls = _LOOKUP.get(b)
        if cls:
            return cls, f"building={b}"
    for key in ("amenity", "shop", "office", "tourism", "industrial",
                "healthcare", "craft"):
        v = (t.get(key) or "").lower()
        if not v:
            continue
        cls = _LOOKUP.get(v)
        if cls:
            return cls, f"{key}={v}"
        if key in ("shop", "office", "tourism", "craft"):
            return "commercial", f"{key}={v}"
        if key in ("amenity", "healthcare"):
            if v in ("hospital", "clinic", "school", "college", "university",
                     "police", "fire_station", "townhall"):
                return "infrastructure", f"{key}={v}"
            return "commercial", f"{key}={v}"
        if key == "industrial":
            return "industrial", f"{key}={v}"
    if b == "yes":
        return "residential", "building=yes (no more specific tag)"
    return "residential", "untagged, JRC default class"


def floor_area_m2(footprint_m2: float, tags: Optional[dict]) -> float:
    """Gross floor area = footprint x storeys.

    JRC maximum damage values are per square metre of *floor* area, so a
    multi-storey building is worth more than its footprint.  `building:levels`
    is a real OSM tag; where it is absent the footprint is used unchanged
    (one storey), which understates rather than inflates the loss.
    """
    t = tags or {}
    levels = t.get("building:levels")
    try:
        n = float(levels)
        if not (1.0 <= n <= 200.0):
            n = 1.0
    except (TypeError, ValueError):
        n = 1.0
    return float(footprint_m2) * n


# ---------------------------------------------------------------------------
# Damage classes (presentation only)
# ---------------------------------------------------------------------------

DAMAGE_CLASS_NAMES = {0: "None", 1: "Minor", 2: "Moderate",
                      3: "Major", 4: "Destroyed"}


def damage_class(frac: np.ndarray) -> np.ndarray:
    """0 none, 1 minor (<25%), 2 moderate (<50%), 3 major (<75%), 4 destroyed."""
    cls = np.zeros(np.shape(frac), dtype=np.int8)
    f = np.asarray(frac)
    cls[f > 0.001] = 1
    cls[f >= 0.25] = 2
    cls[f >= 0.50] = 3
    cls[f >= 0.75] = 4
    return cls


def provenance(iso3: str = "IND", target_year: Optional[int] = None) -> dict:
    """Everything a reviewer needs to audit the monetary figures."""
    curves = {c: curve(c, iso3).to_dict() for c in ASSET_CLASSES}
    rates = {k: v.to_dict() for k, v in max_damage_set(iso3, target_year).items()}
    return {
        "citation": citation(),
        "source_pdf": db()["_source_pdf"],
        "country": iso3.upper(),
        "continent": continent_for(iso3),
        "curves": curves,
        "max_damage": rates,
        "note": ("Damage curves and unit values are published international "
                 "reference data, not local assumptions. Where an audited "
                 "state schedule-of-rates exists for the study area it should "
                 "replace the construction-cost survey values."),
    }
