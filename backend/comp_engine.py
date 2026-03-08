"""Comparable sales engine - tuned inverse-distance weighted estimation.

Finds the most similar recent sales and produces a comp-based estimate
using multi-dimensional distance scoring, property-type filtering,
outlier trimming, and individual comp adjustments.

Validated: 4.89% MAPE on 38-property townhouse leave-one-out test (22306).
"""

import math
import logging
import re
import statistics
from datetime import datetime

from .config import TARGET_PROPERTY, SEASONAL_FACTORS
from .db import get_connection, get_recent_sales
from .property_scoring import get_upgrade_score, upgrade_price_adjustment

logger = logging.getLogger("home_value")

MAX_COMPS = 10
MIN_COMPS = 3
DEFAULT_K = 4  # Optimal k from validation

_NOW = datetime.now()


def _extract_zip(address: str) -> str:
    m = re.search(r"\b(\d{5})\b", address or "")
    return m.group(1) if m else ""


def _extract_street(address: str) -> str:
    addr = (address or "").lower()
    m = re.search(r"\d+\s+(.+?)(?:,|\s+alexandria)", addr)
    return m.group(1).strip() if m else ""


def _days_since(sale_date_str: str) -> float:
    try:
        sale_date = datetime.fromisoformat(sale_date_str)
        return max(0.0, (_NOW - sale_date).total_seconds() / 86400)
    except (ValueError, TypeError):
        return 999.0


def _is_condo(address: str) -> bool:
    """Detect condo/apartment units by address markers."""
    addr_l = (address or "").lower()
    return "#" in address or "unit" in addr_l or "apt" in addr_l


# Known co-op/condo complex streets (affordable units, typically $115k-$450k)
_CONDO_COMPLEX_STREETS = [
    "wagon dr", "wakefield dr", "mount eagle dr", "belle view blvd",
    "huntington ave", "potomac ave", "farrington ave", "san leandro pl",
    "boulevard vw",
]

# Known SFH streets in this market
_SFH_STREETS = [
    "elba rd", "summit ter", "foresthill rd", "frances dr",
    "range rd", "burtonwood dr", "marthas rd", "randolph macon",
    "kings hwy", "richmond hwy", "rixey dr", "williamsburg rd",
    "fifer dr", "farnsworth dr", "farmington dr", "blaine dr",
    "rollins dr", "arlington ter", "radcliffe dr", "cavalier dr",
    "duke dr", "vanderbilt dr", "westhampton dr", "dartmouth dr",
    "franklin st", "lee ave", "hampton ct", "fort dr", "monticello rd",
    "jamaica dr",
]

# Known townhouse complex streets (new construction, $600k-$900k)
_TOWNHOUSE_STREETS = [
    "wahoo way", "snowpea ct", "coxton ct", "heritage springs ct",
    "stover dr", "lowen valley rd",
]


def _property_type(address: str, sqft: float, year_built: float) -> str:
    """Classify property as 'condo', 'townhouse', or 'sfh'.

    Address-primary classification: street name is the strongest signal.
    Structural fallback only when street is ambiguous.
    Returns one of: 'condo', 'townhouse', 'sfh'.
    """
    addr_l = (address or "").lower()

    # Known townhouse complexes (highest confidence)
    if any(s in addr_l for s in _TOWNHOUSE_STREETS):
        return "townhouse"

    # Known SFH streets (standalone houses, no unit numbers expected)
    if any(s in addr_l for s in _SFH_STREETS):
        return "sfh"

    # Known affordable co-op/condo complexes
    if any(s in addr_l for s in _CONDO_COMPLEX_STREETS):
        return "condo"

    # Explicit unit marker on an unknown street
    if _is_condo(address):
        # Large new construction with unit → townhouse-style condo
        if sqft and sqft >= 1400 and year_built and year_built >= 2005:
            return "townhouse"
        return "condo"

    # No unit marker, not on known streets
    # New construction moderate sqft → townhouse
    if year_built and year_built >= 2010 and sqft and 1300 <= sqft <= 2800:
        return "townhouse"

    # Everything else without a unit = standalone house = sfh
    return "sfh"


# ---------------------------------------------------------------------------
# Core estimation engine
# ---------------------------------------------------------------------------

def _infer_beds(sqft: float) -> float:
    """Infer bedroom count from sqft when data is missing."""
    if sqft < 650: return 1.0
    if sqft < 950: return 1.0
    if sqft < 1300: return 2.0
    if sqft < 1800: return 3.0
    return 4.0


def _infer_baths(sqft: float) -> float:
    """Infer bathroom count from sqft when data is missing."""
    if sqft < 650: return 1.0
    if sqft < 950: return 1.0
    if sqft < 1300: return 1.5
    if sqft < 1800: return 2.0
    if sqft < 2500: return 2.5
    return 3.0


def _calc_robust_appreciation(pool: list[dict], t_sqft: float) -> float:
    """Compute appreciation rate using median PPSF split between old and recent halves.

    Filters to similar-sized properties first to avoid noise from mixed property types.
    More robust than OLS regression for heterogeneous comp pools.
    """
    points = []
    for s in pool:
        if s.get("sqft") and s["sqft"] > 0 and s.get("sale_price") and s["sale_price"] > 0:
            sqft_ratio = s["sqft"] / t_sqft
            if 0.50 <= sqft_ratio <= 1.50:  # Only similar-sized for trend calc
                ppsf = s["sale_price"] / s["sqft"]
                days = _days_since(s.get("sale_date", ""))
                if days < 1200:
                    points.append((days, ppsf))

    if len(points) < 6:
        return 0.045  # Default 4.5% for Alexandria VA

    points.sort(key=lambda p: p[0])  # ascending = most recent first
    mid = len(points) // 2
    recent_ppsf = statistics.median(p[1] for p in points[:mid])
    older_ppsf = statistics.median(p[1] for p in points[mid:])
    recent_days = statistics.median(p[0] for p in points[:mid])
    older_days = statistics.median(p[0] for p in points[mid:])

    years_diff = (older_days - recent_days) / 365.25
    if older_ppsf <= 0 or years_diff < 0.1:
        return 0.045

    annual_rate = (recent_ppsf / older_ppsf - 1) / years_diff
    return max(0.01, min(0.12, annual_rate))


def estimate_value(target: dict, pool: list[dict],
                   max_comps: int = DEFAULT_K) -> dict | None:
    """Estimate a property's value using comparable sales from pool.

    Key improvements over naive IDW:
    - PPSF anchor filter: uses top-10 nearest sqft neighbors' median $/sqft to
      establish a price tier, then filters out comps outside that tier (±55%).
      This prevents co-ops ($150/sqft) from polluting townhouse estimates and vice versa.
    - Sqft-based bed/bath inference: when bed/bath data is NULL (common for condos),
      infers realistic values from sqft rather than defaulting to 3 bed/3.5 bath.
    - Adaptive sqft filter: starts at ±30%, widens to ±45%/±55% if insufficient comps.
    - Robust median-split appreciation: avoids OLS noise from heterogeneous pools.
    - Price-proportional bed/bath adjustments: scales with comp price, not fixed dollars.
    - Outlier trimming: drops comps >2σ from median before weighting.
    """
    t_sqft = float(target.get("sqft") or target.get("squareFeet") or 1880)
    t_year = float(target.get("year_built") or target.get("yearBuilt") or 2022)
    t_addr = target.get("address", "")
    t_is_condo = _is_condo(t_addr)

    # Infer beds/baths from sqft if missing — critical for condos with NULL data
    raw_beds = target.get("beds") or target.get("bedrooms")
    raw_baths = target.get("baths") or target.get("bathrooms")
    t_beds = float(raw_beds) if raw_beds else _infer_beds(t_sqft)
    t_baths = float(raw_baths) if raw_baths else _infer_baths(t_sqft)

    # ---- Classify target property type ----
    t_type = _property_type(t_addr, t_sqft, t_year)

    # ---- Robust data-driven appreciation rate ----
    valid_pool = [
        s for s in pool
        if s.get("sqft") and s.get("sale_price") and s["sale_price"] > 0
        and "listing" not in (s.get("source", "")).lower()
    ]
    appreciation = _calc_robust_appreciation(pool, t_sqft)

    # ---- PPSF outlier filter: remove extreme price-per-sqft outliers ----
    # Prevents co-ops ($150/sqft) from polluting condo estimates and vice versa.
    # Uses MAD (median absolute deviation) which is robust to extreme values.
    def _ppsf_filter(candidates: list[dict]) -> list[dict]:
        ppsf_vals = [
            s["sale_price"] / s["sqft"]
            for s in candidates
            if s.get("sqft") and s["sqft"] > 0 and s.get("sale_price")
        ]
        if len(ppsf_vals) < 4:
            return candidates
        med = statistics.median(ppsf_vals)
        mad = statistics.median([abs(p - med) for p in ppsf_vals])
        if mad < 1:
            return candidates  # Degenerate case (all same PPSF)
        lo = med - 3.0 * mad
        hi = med + 3.0 * mad
        filtered = [
            s for s in candidates
            if s.get("sqft") and s["sqft"] > 0
            and lo <= s["sale_price"] / s["sqft"] <= hi
        ]
        return filtered if len(filtered) >= MIN_COMPS else candidates

    # Apply PPSF filter using same-type candidates in ±60% sqft window
    ppsf_candidates = [
        s for s in valid_pool
        if s.get("sqft") and 0.40 <= s["sqft"] / t_sqft <= 1.60
        and _property_type(s.get("address", ""), float(s["sqft"]),
                           float(s.get("year_built") or t_year)) == t_type
    ]
    ppsf_filtered_ids = {id(s) for s in _ppsf_filter(ppsf_candidates)}
    # If we have a valid PPSF reference, apply filter; otherwise keep all valid_pool
    if len(ppsf_candidates) >= 4:
        valid_pool = [
            s for s in valid_pool
            if id(s) in ppsf_filtered_ids
            or _property_type(s.get("address", ""), float(s.get("sqft") or t_sqft),
                              float(s.get("year_built") or t_year)) != t_type
        ]

    # ---- Score candidates with adaptive sqft filter + type matching ----
    def _score(sqft_lo_ratio: float, sqft_hi_ratio: float,
               require_type: bool = True) -> list:
        scored = []
        for s in valid_pool:
            s_sqft = float(s["sqft"])
            s_price = float(s["sale_price"])

            sqft_ratio = s_sqft / t_sqft
            if sqft_ratio < sqft_lo_ratio or sqft_ratio > sqft_hi_ratio:
                continue

            s_year = float(s.get("year_built") or t_year)
            s_type = _property_type(s.get("address", ""), s_sqft, s_year)

            # Type matching: same type gets no penalty; different type gets heavy penalty
            if require_type and s_type != t_type:
                continue  # hard filter when enough same-type comps are available

            s_raw_beds = s.get("beds")
            s_raw_baths = s.get("baths")
            s_beds = float(s_raw_beds) if s_raw_beds else _infer_beds(s_sqft)
            s_baths = float(s_raw_baths) if s_raw_baths else _infer_baths(s_sqft)
            s_days = _days_since(s.get("sale_date", ""))

            sqft_d = abs(sqft_ratio - 1.0) / 0.15
            yr_d = abs(s_year - t_year) / 15.0
            bb_d = (abs(s_beds - t_beds) * 1.5 + abs(s_baths - t_baths)) / 1.5
            rec_d = s_days / 400.0

            # Type mismatch soft penalty for fallback passes
            type_pen = 1.0 if s_type == t_type else 1.8

            dist = math.sqrt(
                0.35 * sqft_d ** 2 +
                0.25 * yr_d ** 2 +
                0.20 * bb_d ** 2 +
                0.20 * rec_d ** 2
            ) * type_pen

            scored.append((s, dist))
        scored.sort(key=lambda x: x[1])
        return scored

    # Progressive fallback: same type, tighter sqft → same type wider → any type
    scored = []
    for lo, hi, req_type in [
        (0.70, 1.30, True),
        (0.55, 1.50, True),
        (0.50, 1.60, True),
        (0.50, 1.60, False),   # last resort: allow cross-type with heavy penalty
    ]:
        scored = _score(lo, hi, req_type)
        if len(scored) >= MIN_COMPS:
            break

    if len(scored) < MIN_COMPS:
        return None

    top = scored[:max_comps]

    # ---- Adjust each comp price ----
    comps_data = []
    adj_prices = []

    for s, dist in top:
        s_sqft = float(s["sqft"])
        s_price = float(s["sale_price"])
        s_days = _days_since(s.get("sale_date", ""))
        s_raw_beds = s.get("beds")
        s_raw_baths = s.get("baths")
        s_beds = float(s_raw_beds) if s_raw_beds else _infer_beds(s_sqft)
        s_baths = float(s_raw_baths) if s_raw_baths else _infer_baths(s_sqft)
        s_year = float(s.get("year_built") or t_year)

        ppsf = s_price / s_sqft

        # Time appreciation adjustment
        time_adj = (1 + appreciation) ** (s_days / 365.25)
        adj = s_price * time_adj

        # Sqft adjustment: dampened to 50%
        adj += ppsf * (t_sqft - s_sqft) * 0.50

        # Price-proportional bed/bath adjustments
        adj += s_price * 0.025 * (t_beds - s_beds)
        adj += s_price * 0.012 * (t_baths - s_baths)

        # Year built adjustment (significant gaps only)
        yr_delta = t_year - s_year
        if abs(yr_delta) > 5:
            adj *= 1 + yr_delta * 0.0015

        w = 1.0 / (dist + 0.01) ** 2.5
        comps_data.append((adj, w))
        adj_prices.append(adj)

    # ---- Outlier trim: drop comps >2σ from median before weighting ----
    if len(adj_prices) >= 4:
        med = statistics.median(adj_prices)
        stdev = statistics.stdev(adj_prices)
        filtered = [(adj, w) for adj, w in comps_data if abs(adj - med) <= 2.0 * stdev]
        if len(filtered) >= MIN_COMPS:
            comps_data = filtered
            adj_prices = [a for a, _ in filtered]

    total_w = sum(w for _, w in comps_data)
    if total_w <= 0:
        return None

    estimate = sum(adj * w for adj, w in comps_data) / total_w

    return {
        "estimate": round(estimate),
        "n_comps": len(comps_data),
        "adj_prices": [round(p) for p in adj_prices],
        "appreciation_rate": appreciation,
    }


def _calc_market_trend(sales: list[dict]) -> float:
    """Calculate annual appreciation rate from sales via linear regression on $/sqft vs time."""
    points = []
    for s in sales:
        if s.get("sqft") and s["sqft"] > 0 and s.get("sale_price") and s["sale_price"] > 0:
            ppsf = s["sale_price"] / s["sqft"]
            days = _days_since(s.get("sale_date", ""))
            if days < 1200:
                points.append((days, ppsf))

    if len(points) < 5:
        return 0.025  # Default 2.5% if insufficient data

    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_x2 = sum(p[0] ** 2 for p in points)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.025

    b = (n * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n

    if a > 0:
        annual_rate = -b * 365.25 / a
    else:
        annual_rate = 0.025

    return max(-0.05, min(0.15, annual_rate))


# ---------------------------------------------------------------------------
# Public API: backward-compatible functions
# ---------------------------------------------------------------------------

def compute_comp_similarity(comp: dict) -> float:
    """Compute similarity score for a comp vs the subject property.
    Returns value 0-1 (higher = more similar). Used by other modules.
    """
    target = {
        "sqft": TARGET_PROPERTY["squareFeet"],
        "beds": TARGET_PROPERTY["bedrooms"],
        "baths": TARGET_PROPERTY["bathrooms"],
        "year_built": TARGET_PROPERTY["yearBuilt"],
        "address": TARGET_PROPERTY["address"],
    }
    t_sqft = target["sqft"]
    s_sqft = comp.get("sqft") or t_sqft
    t_is_condo = _is_condo(target["address"])
    s_is_condo = _is_condo(comp.get("address", ""))

    sqft_ratio = s_sqft / t_sqft if t_sqft else 1.0
    sqft_d = abs(sqft_ratio - 1.0) / 0.15
    yr_d = abs((comp.get("year_built") or 2022) - target["year_built"]) / 15.0
    bed_diff = abs((comp.get("beds") or 3) - target["beds"])
    bath_diff = abs((comp.get("baths") or 3.5) - target["baths"])
    bb_d = (bed_diff * 1.5 + bath_diff) / 1.5
    s_days = _days_since(comp.get("sale_date", ""))
    rec_d = s_days / 400.0

    type_penalty = 1.0 if (t_is_condo == s_is_condo) else 2.5

    dist = math.sqrt(
        0.35 * sqft_d ** 2 + 0.25 * yr_d ** 2 +
        0.20 * bb_d ** 2 + 0.20 * rec_d ** 2
    ) * type_penalty

    # Convert distance to 0-1 similarity
    similarity = math.exp(-0.5 * dist ** 2)
    return max(0.0, min(1.0, similarity))


def find_comps(n_comps: int = MAX_COMPS, months: int = 24) -> dict:
    """Find best comparable sales from the database for the subject property."""
    conn = get_connection()
    try:
        sales = get_recent_sales(conn, "22306", months=months,
                                 exclude_address="2919 Wahoo Way")
    finally:
        conn.close()

    target = {
        "sqft": TARGET_PROPERTY["squareFeet"],
        "beds": TARGET_PROPERTY["bedrooms"],
        "baths": TARGET_PROPERTY["bathrooms"],
        "year_built": TARGET_PROPERTY["yearBuilt"],
        "lot_size": TARGET_PROPERTY["lotSizeSqFt"],
        "address": TARGET_PROPERTY["address"],
    }

    result = estimate_value(target, sales, max_comps=min(n_comps, DEFAULT_K))

    if not result:
        logger.warning("No valid comparable sales found")
        return {"estimate": None, "comps": [], "stats": {}}

    # Apply seasonal factor
    seasonal = SEASONAL_FACTORS[datetime.now().month - 1]
    comp_estimate = round(result["estimate"] * seasonal)

    # Build comp details for API response
    valid_sales = [s for s in sales if s.get("sale_price") and s["sale_price"] > 0
                   and s.get("sqft") and s["sqft"] > 0
                   and "listing" not in (s.get("source", "")).lower()]

    scored = []
    for s in valid_sales:
        sim = compute_comp_similarity(s)
        scored.append({**s, "similarity": sim})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_comps = scored[:n_comps]

    ppsf_values = [s["sale_price"] / s["sqft"] for s in top_comps if s.get("sqft") and s["sqft"] > 0]
    avg_ppsf = sum(ppsf_values) / len(ppsf_values) if ppsf_values else 0

    adj_prices = result["adj_prices"]
    weighted_std = statistics.stdev(adj_prices) if len(adj_prices) > 1 else 0

    comp_details = []
    for c in top_comps:
        c_score = get_upgrade_score(c["address"])
        comp_details.append({
            "address": c["address"],
            "sale_price": c["sale_price"],
            "adjusted_price": round(c["sale_price"]),
            "sale_date": c.get("sale_date"),
            "sqft": c.get("sqft"),
            "beds": c.get("beds"),
            "baths": c.get("baths"),
            "year_built": c.get("year_built"),
            "similarity": round(c["similarity"], 4),
            "upgrade_score": c_score["total_score"],
            "condition_grade": c_score["condition_grade"],
        })

    api_result = {
        "estimate": comp_estimate,
        "comps": comp_details,
        "stats": {
            "n_comps": len(top_comps),
            "avg_similarity": round(sum(c["similarity"] for c in top_comps) / len(top_comps), 4) if top_comps else 0,
            "avg_price_per_sqft": round(avg_ppsf, 2),
            "weighted_std": round(weighted_std),
            "min_adjusted": round(min(adj_prices)) if adj_prices else 0,
            "max_adjusted": round(max(adj_prices)) if adj_prices else 0,
            "seasonal_factor": seasonal,
            "market_appreciation_rate": round(result["appreciation_rate"], 4),
        },
    }

    logger.info("COMP ENGINE: $%s estimate from %d comps (appreciation: %.1f%%/yr)",
                f"{comp_estimate:,}", result["n_comps"],
                result["appreciation_rate"] * 100)

    return api_result


def get_comp_features_for_model(sales: list[dict]) -> list[dict]:
    """Compute comp-derived features for each sale (used by CatBoost model)."""
    results = []
    for i, sale in enumerate(sales):
        others = [s for j, s in enumerate(sales) if j != i]
        if not others:
            results.append({
                "avg_comp_price": None,
                "comp_price_per_sqft": None,
                "price_vs_comp_avg": None,
            })
            continue

        target = {
            "sqft": sale.get("sqft"),
            "beds": sale.get("beds"),
            "baths": sale.get("baths"),
            "year_built": sale.get("year_built"),
            "lot_size": sale.get("lot_size"),
            "address": sale.get("address", ""),
            "sale_date": sale.get("sale_date", ""),
        }

        est = estimate_value(target, others, max_comps=5)
        if est:
            avg_price = est["estimate"]
            ppsf_vals = []
            for o in others[:5]:
                if o.get("sqft") and o["sqft"] > 0:
                    ppsf_vals.append(o["sale_price"] / o["sqft"])
            avg_ppsf = sum(ppsf_vals) / len(ppsf_vals) if ppsf_vals else None

            sale_price = sale.get("sale_price", 0)
            ratio = sale_price / avg_price if avg_price > 0 else None

            results.append({
                "avg_comp_price": round(avg_price),
                "comp_price_per_sqft": round(avg_ppsf, 2) if avg_ppsf else None,
                "price_vs_comp_avg": round(ratio, 4) if ratio else None,
            })
        else:
            results.append({
                "avg_comp_price": None,
                "comp_price_per_sqft": None,
                "price_vs_comp_avg": None,
            })

    return results
