"""ML valuation engine - Gaussian kernel weighted regression with uncertainty quantification."""

import json
import math
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

from .config import (
    TARGET_PROPERTY, FEATURE_WEIGHTS, LISTING_DISCOUNT,
    MARKET_TREND_ANNUAL_PCT, KERNEL_BANDWIDTH, SEASONAL_FACTORS,
    T_CRITICAL, T_CRITICAL_INF, SOURCES, FALLBACK_COMPARABLES,
    CACHE_FILE, CACHE_TTL_SECONDS, DATA_DIR,
)

logger = logging.getLogger("home_value")

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def gaussian_kernel(distance: float, bandwidth: float) -> float:
    return math.exp(-0.5 * (distance / bandwidth) ** 2)


def get_t_critical(df: int, level: float) -> float:
    keys = sorted(T_CRITICAL.keys())
    for k in keys:
        if df <= k:
            return T_CRITICAL[k][level]
    return T_CRITICAL_INF[level]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_recency_score(sale_date_str: str | None) -> float:
    if not sale_date_str:
        return 0.0
    try:
        sale_date = datetime.fromisoformat(sale_date_str)
    except (ValueError, TypeError):
        return 0.0
    days_since = (datetime.now() - sale_date).total_seconds() / 86400
    return math.exp(-0.693 * days_since / 180)


def compute_location_proximity(comp_address: str | None) -> float:
    if not comp_address:
        return 0.3
    addr = comp_address.lower()
    if "wahoo" in addr:
        return 1.0
    if "22306" in addr:
        return 0.7
    if "alexandria" in addr:
        return 0.5
    return 0.3


def compute_property_age_score(comp_year_built: int | None) -> float:
    if not comp_year_built:
        return 0.5
    age_diff = abs(TARGET_PROPERTY["yearBuilt"] - comp_year_built)
    return gaussian_kernel(age_diff, 10)


def compute_bed_bath_score(comp_beds: int | None, comp_baths: float | None) -> float:
    bed_diff = abs((comp_beds or 3) - TARGET_PROPERTY["bedrooms"])
    bath_diff = abs((comp_baths or 2) - TARGET_PROPERTY["bathrooms"])
    return gaussian_kernel(bed_diff + bath_diff, 2)


def compute_sqft_similarity(comp_sqft: int | None) -> float:
    if not comp_sqft:
        return 0.5
    ratio = comp_sqft / TARGET_PROPERTY["squareFeet"]
    return gaussian_kernel(abs(1 - ratio), KERNEL_BANDWIDTH)


def compute_lot_size_similarity(comp_lot_sqft: int | None) -> float:
    if not comp_lot_sqft:
        return 0.5
    ratio = comp_lot_sqft / TARGET_PROPERTY["lotSizeSqFt"]
    return gaussian_kernel(abs(1 - ratio), KERNEL_BANDWIDTH)


# ---------------------------------------------------------------------------
# Composite similarity
# ---------------------------------------------------------------------------

def compute_similarity_score(comp: dict) -> float:
    features = {
        "recency": compute_recency_score(comp.get("closedDate") or comp.get("listedDate")),
        "locationProximity": compute_location_proximity(comp.get("address")),
        "propertyAge": compute_property_age_score(comp.get("yearBuilt")),
        "bedBathMatch": compute_bed_bath_score(comp.get("bedrooms"), comp.get("bathrooms")),
        "sqftSimilarity": compute_sqft_similarity(comp.get("squareFeet")),
        "lotSizeSimilarity": compute_lot_size_similarity(comp.get("lotSizeSqFt")),
    }
    score = sum(FEATURE_WEIGHTS[k] * features.get(k, 0) for k in FEATURE_WEIGHTS)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Price adjustments
# ---------------------------------------------------------------------------

def adjust_price(comp: dict) -> int | None:
    price = comp.get("price")
    if not price or price <= 0:
        return None

    price = float(price)

    # Sqft delta adjustment
    comp_sqft = comp.get("squareFeet")
    if comp_sqft and comp_sqft > 0:
        price_per_sqft = price / comp_sqft
        sqft_delta = TARGET_PROPERTY["squareFeet"] - comp_sqft
        price += price_per_sqft * sqft_delta * 0.5

    # Listing discount
    is_listing = not comp.get("closedDate")
    if is_listing:
        price *= (1 - LISTING_DISCOUNT)

    # Market trend time adjustment
    date_str = comp.get("closedDate") or comp.get("listedDate")
    if date_str:
        try:
            sale_date = datetime.fromisoformat(date_str)
            years_since = (datetime.now() - sale_date).total_seconds() / (365.25 * 86400)
            price *= (1 + MARKET_TREND_ANNUAL_PCT / 100) ** years_since
        except (ValueError, TypeError):
            pass

    return round(price)


def get_seasonal_factor() -> float:
    return SEASONAL_FACTORS[datetime.now().month - 1]


# ---------------------------------------------------------------------------
# Uncertainty quantification (t-distribution)
# ---------------------------------------------------------------------------

def compute_uncertainty(adjusted_prices: list[int], weights: list[float]) -> dict:
    n = len(adjusted_prices)
    if n < 2:
        val = adjusted_prices[0] if adjusted_prices else 0
        return {
            "weightedMean": val,
            "weightedStdDev": 0,
            "effectiveSampleSize": n,
            "coeffOfVariation": 0,
            "intervals": {
                "prediction90": {"low": val, "high": val},
                "confidence80": {"low": val, "high": val},
            },
        }

    total_weight = sum(weights)
    weighted_mean = sum(p * w for p, w in zip(adjusted_prices, weights)) / total_weight

    sum_wt2 = sum(w * w for w in weights)
    effective_n = (total_weight ** 2) / sum_wt2
    bessel_factor = total_weight / (total_weight - sum_wt2 / total_weight)

    weighted_variance = bessel_factor * sum(
        w * (p - weighted_mean) ** 2 for p, w in zip(adjusted_prices, weights)
    ) / total_weight
    weighted_std = math.sqrt(max(0, weighted_variance))

    cv = weighted_std / weighted_mean if weighted_mean > 0 else 0

    df = max(1, round(effective_n - 1))
    se_mean = weighted_std / math.sqrt(effective_n)

    t90 = get_t_critical(df, 0.90)
    prediction_spread = t90 * weighted_std * math.sqrt(1 + 1 / effective_n)

    t80 = get_t_critical(df, 0.80)
    confidence_spread = t80 * se_mean

    return {
        "weightedMean": round(weighted_mean),
        "weightedStdDev": round(weighted_std),
        "effectiveSampleSize": effective_n,
        "coeffOfVariation": cv,
        "intervals": {
            "prediction90": {
                "low": round(weighted_mean - prediction_spread),
                "high": round(weighted_mean + prediction_spread),
            },
            "confidence80": {
                "low": round(weighted_mean - confidence_spread),
                "high": round(weighted_mean + confidence_spread),
            },
        },
    }


# ---------------------------------------------------------------------------
# ML valuation engine
# ---------------------------------------------------------------------------

def ml_valuation(comparables: dict) -> dict:
    all_comps = []
    for c in (comparables.get("recentSales") or []):
        all_comps.append({**c, "_type": "sale"})
    for c in (comparables.get("currentListings") or []):
        all_comps.append({**c, "_type": "listing"})
    all_comps = [c for c in all_comps if c.get("price") and c["price"] > 0]

    if not all_comps:
        return {"estimate": None, "confidence": 0, "mlDetails": None}

    scored = []
    for comp in all_comps:
        sim = compute_similarity_score(comp)
        adj = adjust_price(comp)
        if adj and adj > 0:
            scored.append({
                **comp,
                "similarityScore": sim,
                "rawPrice": comp["price"],
                "adjustedPrice": adj,
                "isListing": comp["_type"] == "listing",
            })

    if not scored:
        return {"estimate": None, "confidence": 0, "mlDetails": None}

    scored.sort(key=lambda c: c["similarityScore"], reverse=True)

    weights = [gaussian_kernel(1 - c["similarityScore"], KERNEL_BANDWIDTH) for c in scored]
    adjusted_prices = [c["adjustedPrice"] for c in scored]
    uncertainty = compute_uncertainty(adjusted_prices, weights)

    seasonal = get_seasonal_factor()
    seasonal_estimate = round(uncertainty["weightedMean"] * seasonal)

    avg_sim = sum(c["similarityScore"] for c in scored) / len(scored)
    comp_count_factor = min(1.0, len(scored) / 8)
    cv_penalty = max(0.0, 1 - uncertainty["coeffOfVariation"] * 5)
    confidence_pct = round(avg_sim * comp_count_factor * cv_penalty * 100)

    intervals = {
        "prediction90": {
            "low": round(uncertainty["intervals"]["prediction90"]["low"] * seasonal),
            "high": round(uncertainty["intervals"]["prediction90"]["high"] * seasonal),
        },
        "confidence80": {
            "low": round(uncertainty["intervals"]["confidence80"]["low"] * seasonal),
            "high": round(uncertainty["intervals"]["confidence80"]["high"] * seasonal),
        },
    }

    return {
        "estimate": seasonal_estimate,
        "confidence": confidence_pct,
        "mlDetails": {
            "intervals": intervals,
            "stats": {
                "weightedStdDev": uncertainty["weightedStdDev"],
                "effectiveSampleSize": uncertainty["effectiveSampleSize"],
                "coeffOfVariation": uncertainty["coeffOfVariation"],
                "seasonalFactor": seasonal,
            },
            "compScores": [
                {
                    "address": c.get("address"),
                    "rawPrice": c["rawPrice"],
                    "adjustedPrice": c["adjustedPrice"],
                    "similarityScore": c["similarityScore"],
                    "isListing": c["isListing"],
                }
                for c in scored
            ],
        },
    }


# ---------------------------------------------------------------------------
# Data source scraping
# ---------------------------------------------------------------------------

def extract_prices_from_text(text: str) -> list[int]:
    matches = []
    for m in re.finditer(r"\$\s*([\d,]+(?:\.\d{2})?)", text):
        val = int(m.group(1).replace(",", "").split(".")[0])
        if 200_000 <= val <= 2_000_000:
            matches.append(val)
    return matches


def extract_comps_from_text(text: str) -> list[dict]:
    comps = []
    for line in text.split("\n"):
        price_m = re.search(r"\$\s*([\d,]+)", line)
        addr_m = re.search(
            r"(\d+\s+[\w\s]+(?:Way|St|Rd|Dr|Ct|Ln|Ave|Blvd|Pl|Cir))", line, re.IGNORECASE
        )
        if price_m and addr_m:
            price = int(price_m.group(1).replace(",", ""))
            if 200_000 <= price <= 2_000_000:
                comps.append({"address": addr_m.group(1).strip(), "price": price})
    return comps


async def fetch_source(source: dict, timeout: int = 15) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(source["url"], headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                text = await resp.text()
                prices = extract_prices_from_text(text)
                comps = extract_comps_from_text(text)
                return {
                    "label": source["label"],
                    "ok": True,
                    "extractedValue": prices[0] if prices else None,
                    "signalCount": len(prices) + len(comps),
                    "prices": prices,
                    "comps": comps,
                }
    except Exception as e:
        return {
            "label": source["label"],
            "ok": False,
            "error": str(e),
            "extractedValue": None,
            "signalCount": 0,
            "prices": [],
            "comps": [],
        }


# ---------------------------------------------------------------------------
# Service: cache, merge, build payload
# ---------------------------------------------------------------------------

class HomeValueService:
    def __init__(self):
        self._cached_data: dict | None = None
        self._load_cache()

    def _load_cache(self):
        try:
            if CACHE_FILE.exists():
                self._cached_data = json.loads(CACHE_FILE.read_text())
                logger.info("Cache loaded from disk")
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)

    def _save_cache(self, data: dict):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)

    def _is_cache_fresh(self) -> bool:
        if not self._cached_data or "meta" not in self._cached_data:
            return False
        try:
            gen_at = datetime.fromisoformat(self._cached_data["meta"]["generatedAt"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - gen_at).total_seconds()
            return age < CACHE_TTL_SECONDS
        except Exception:
            return False

    @staticmethod
    def _merge_source_comps(source_results: list[dict], fallback: dict) -> dict:
        sales = list(fallback.get("recentSales", []))
        listings = list(fallback.get("currentListings", []))

        for src in source_results:
            if src.get("ok") and isinstance(src.get("comps"), list):
                for comp in src["comps"]:
                    is_dup = any(
                        s.get("address", "").lower() == comp.get("address", "").lower()
                        for s in sales
                    )
                    if not is_dup and comp.get("price"):
                        sales.append({
                            "address": comp["address"],
                            "price": comp["price"],
                            "closedDate": datetime.now().strftime("%Y-%m-%d"),
                            "source": src["label"],
                        })

        return {"recentSales": sales, "currentListings": listings}

    @staticmethod
    def _build_valuation(comparables: dict, source_results: list[dict]) -> dict:
        ml = ml_valuation(comparables)

        source_values = [s["extractedValue"] for s in source_results if s.get("ok") and s.get("extractedValue")]

        estimate = ml["estimate"]
        methodology = "ML-Weighted Kernel Regression"

        if not estimate and source_values:
            source_values.sort()
            estimate = source_values[len(source_values) // 2]
            methodology = "Source Median (ML fallback)"

        if not estimate:
            all_prices = [c["price"] for c in (comparables.get("recentSales") or []) + (comparables.get("currentListings") or []) if c.get("price", 0) > 0]
            if all_prices:
                estimate = round(sum(all_prices) / len(all_prices))
                methodology = "Simple Average (fallback)"

        confidence_pct = ml["confidence"] or 50
        conf_range = (
            ml["mlDetails"]["intervals"]["confidence80"]
            if ml.get("mlDetails")
            else {"low": round((estimate or 0) * 0.95), "high": round((estimate or 0) * 1.05)}
        )

        return {
            "estimate": estimate,
            "confidenceRange": conf_range,
            "confidencePct": confidence_pct,
            "methodology": methodology,
            "lastRefreshedAt": datetime.now(timezone.utc).isoformat(),
            "mlDetails": ml.get("mlDetails"),
        }

    @staticmethod
    def _build_financials(estimate: int | None) -> dict:
        balance = TARGET_PROPERTY["currentBalance"]
        ltv_ratio = (balance / estimate) * 100 if estimate else None
        pmi_threshold = round(balance / 0.8)
        delta = pmi_threshold - estimate if estimate else None

        return {
            "ltv": {
                "currentBalance": balance,
                "ratioPct": round(ltv_ratio, 2) if ltv_ratio is not None else None,
            },
            "pmi": {
                "thresholdHomeValue": pmi_threshold,
                "deltaToThreshold": delta,
            },
        }

    async def _generate_data(self) -> dict:
        logger.info("Generating fresh data...")

        import asyncio
        source_results = await asyncio.gather(
            *(fetch_source(s) for s in SOURCES),
            return_exceptions=True,
        )
        source_results = [
            r if isinstance(r, dict) else {"label": "unknown", "ok": False, "error": str(r), "prices": [], "comps": [], "extractedValue": None, "signalCount": 0}
            for r in source_results
        ]

        fallback = FALLBACK_COMPARABLES
        comparables = self._merge_source_comps(source_results, fallback)
        valuation = self._build_valuation(comparables, source_results)
        financials = self._build_financials(valuation["estimate"])

        all_sale_prices = sorted(
            [c["price"] for c in (comparables.get("recentSales") or []) if c.get("price", 0) > 0]
        )
        median_price = all_sale_prices[len(all_sale_prices) // 2] if all_sale_prices else None

        data = {
            "valuation": valuation,
            "financials": financials,
            "property": {
                "address": TARGET_PROPERTY["address"],
                "squareFeet": TARGET_PROPERTY["squareFeet"],
                "yearBuilt": TARGET_PROPERTY["yearBuilt"],
            },
            "market": {
                "medianPrice": median_price,
                "yoyChangePct": MARKET_TREND_ANNUAL_PCT,
            },
            "comparables": comparables,
            "sources": [
                {
                    "label": s["label"],
                    "ok": s.get("ok", False),
                    "extractedValue": s.get("extractedValue"),
                    "signalCount": s.get("signalCount", 0),
                    "error": s.get("error"),
                }
                for s in source_results
            ],
            "meta": {
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "targetProperty": TARGET_PROPERTY["address"],
            },
        }

        self._cached_data = data
        self._save_cache(data)
        logger.info("Data generated and cached")
        return data

    async def get_data(self, force_refresh: bool = False) -> dict:
        if not force_refresh and self._is_cache_fresh() and self._cached_data:
            return self._cached_data
        try:
            return await self._generate_data()
        except Exception as e:
            logger.error("Generate failed: %s", e)
            if self._cached_data:
                return self._cached_data
            raise

    async def force_refresh(self) -> dict:
        return await self._generate_data()
