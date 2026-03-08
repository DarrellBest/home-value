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
    """XGBoost-based valuation using comparable sales data.

    Delegates to training.xgb_valuation for the actual model training/prediction.
    """
    all_comps = []
    for c in (comparables.get("recentSales") or []):
        all_comps.append({**c, "_type": "sale"})
    for c in (comparables.get("currentListings") or []):
        all_comps.append({**c, "_type": "listing"})
    all_comps = [c for c in all_comps if c.get("price") and c["price"] > 0]

    if not all_comps:
        return {"estimate": None, "confidence": 0, "mlDetails": None}

    # Convert comps to sales-like format for XGBoost pipeline
    sales = []
    for c in all_comps:
        sales.append({
            "address": c.get("address", ""),
            "sale_price": c["price"],
            "sale_date": c.get("closedDate") or c.get("listedDate") or "",
            "sqft": c.get("squareFeet"),
            "year_built": c.get("yearBuilt"),
            "beds": c.get("bedrooms"),
            "baths": c.get("bathrooms"),
            "lot_size": c.get("lotSizeSqFt"),
            "source": c.get("source", "unknown"),
        })

    from .training import xgb_valuation
    return xgb_valuation(sales)


# ---------------------------------------------------------------------------
# Data source scraping
# ---------------------------------------------------------------------------

def _is_valid_price(val: int | float) -> bool:
    return 200_000 <= val <= 2_000_000


def extract_json_ld_prices(soup: BeautifulSoup) -> list[int]:
    """Extract prices from JSON-LD structured data (schema.org Residence/RealEstateListing)."""
    prices = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                _extract_jsonld_item_prices(item, prices)
        except (json.JSONDecodeError, TypeError):
            continue
    logger.debug("JSON-LD extracted %d prices: %s", len(prices), prices)
    return prices


def _extract_jsonld_item_prices(item: dict, prices: list[int]):
    """Recursively extract prices from a JSON-LD item."""
    if not isinstance(item, dict):
        return
    # Check @type for relevant schema types
    item_type = item.get("@type", "")
    if isinstance(item_type, list):
        item_type = " ".join(item_type)
    # Direct price field
    for key in ("price", "lowPrice", "highPrice", "value"):
        raw = item.get(key)
        if raw is not None:
            try:
                val = int(float(str(raw).replace(",", "").replace("$", "")))
                if _is_valid_price(val):
                    prices.append(val)
            except (ValueError, TypeError):
                pass
    # Nested offers / priceSpecification
    for nested_key in ("offers", "priceSpecification", "estimatedValue", "mainEntity"):
        nested = item.get(nested_key)
        if isinstance(nested, dict):
            _extract_jsonld_item_prices(nested, prices)
        elif isinstance(nested, list):
            for sub in nested:
                _extract_jsonld_item_prices(sub, prices)
    # Check @graph
    graph = item.get("@graph")
    if isinstance(graph, list):
        for sub in graph:
            _extract_jsonld_item_prices(sub, prices)


def extract_meta_prices(soup: BeautifulSoup) -> list[int]:
    """Extract prices from meta tags (og:price, twitter:data1, etc.)."""
    prices = []
    meta_attrs = [
        ("property", "og:price:amount"),
        ("property", "og:price"),
        ("name", "twitter:data1"),
        ("name", "price"),
        ("itemprop", "price"),
    ]
    for attr_name, attr_value in meta_attrs:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            try:
                val = int(float(tag["content"].replace(",", "").replace("$", "").strip()))
                if _is_valid_price(val):
                    prices.append(val)
                    logger.debug("Meta tag %s=%s -> %d", attr_name, attr_value, val)
            except (ValueError, TypeError):
                pass
    return prices


def extract_inline_json_prices(soup: BeautifulSoup) -> list[int]:
    """Extract prices from inline JavaScript JSON blobs (__NEXT_DATA__, __INITIAL_STATE__, etc.)."""
    prices = []
    patterns = [
        # Zillow Zestimate / __NEXT_DATA__
        (r'"zestimate"\s*:\s*(\d+)', "zestimate"),
        (r'"price"\s*:\s*(\d+)', "price"),
        (r'"listPrice"\s*:\s*(\d+)', "listPrice"),
        (r'"estimatedValue"\s*:\s*(\d+)', "estimatedValue"),
        (r'"homeValue"\s*:\s*(\d+)', "homeValue"),
        (r'"taxAssessedValue"\s*:\s*(\d+)', "taxAssessedValue"),
        # Redfin
        (r'"estimateInfo".*?"amount"\s*:\s*(\d+)', "redfin_estimate"),
        (r'"sectionPreviewText"\s*:\s*"\$\s*([\d,]+)"', "redfin_preview"),
        (r'"predictedValue"\s*:\s*(\d+)', "predictedValue"),
        # Realtor.com
        (r'"estimate".*?"price"\s*:\s*(\d+)', "realtor_estimate"),
        (r'"list_price"\s*:\s*(\d+)', "list_price"),
        # Homes.com
        (r'"currentValue"\s*:\s*(\d+)', "currentValue"),
        (r'"propertyValue"\s*:\s*(\d+)', "propertyValue"),
    ]
    for script in soup.find_all("script"):
        script_text = script.string or ""
        if not script_text or len(script_text) < 50:
            continue
        # Try to find __NEXT_DATA__ or window.__INITIAL_STATE__ JSON blobs
        for blob_pattern in [
            r'__NEXT_DATA__\s*=\s*(\{.*?\})\s*;',
            r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;',
            r'window\.propertyData\s*=\s*(\{.*?\})\s*;',
        ]:
            blob_match = re.search(blob_pattern, script_text, re.DOTALL)
            if blob_match:
                try:
                    blob = json.loads(blob_match.group(1))
                    _deep_extract_prices(blob, prices)
                except (json.JSONDecodeError, TypeError):
                    pass
        # Regex fallback on script content
        for pattern, label in patterns:
            for m in re.finditer(pattern, script_text):
                try:
                    val = int(m.group(1).replace(",", ""))
                    if _is_valid_price(val):
                        prices.append(val)
                        logger.debug("Inline JSON [%s] -> %d", label, val)
                except (ValueError, TypeError):
                    pass
    return prices


def _deep_extract_prices(data, prices: list[int], depth: int = 0):
    """Recursively extract price-like values from parsed JSON blobs."""
    if depth > 8:
        return
    price_keys = {
        "zestimate", "price", "listPrice", "estimatedValue", "homeValue",
        "taxAssessedValue", "predictedValue", "currentValue", "propertyValue",
        "list_price", "amount", "value", "estimate",
    }
    if isinstance(data, dict):
        for key, val in data.items():
            if key.lower().rstrip("_") in {k.lower() for k in price_keys}:
                if isinstance(val, (int, float)) and _is_valid_price(int(val)):
                    prices.append(int(val))
                    logger.debug("Deep JSON key=%s -> %d", key, int(val))
            elif isinstance(val, (dict, list)):
                _deep_extract_prices(val, prices, depth + 1)
    elif isinstance(data, list):
        for item in data[:50]:  # limit iteration
            _deep_extract_prices(item, prices, depth + 1)


def extract_prices_from_text(text: str) -> list[int]:
    """Fallback regex extraction of dollar amounts from plain text."""
    matches = []
    for m in re.finditer(r"\$\s*([\d,]+(?:\.\d{2})?)", text):
        val = int(m.group(1).replace(",", "").split(".")[0])
        if _is_valid_price(val):
            matches.append(val)
    return matches


def extract_comps_from_text(text: str) -> list[dict]:
    comps = []
    for line in text.split("\n"):
        price_m = re.search(r"\$\s*([\d,]+)", line)
        addr_m = re.search(
            r"(\d{2,5}\s+[\w\s]+(?:Way|St|Rd|Dr|Ct|Ln|Ave|Blvd|Pl|Cir))\b", line, re.IGNORECASE
        )
        if price_m and addr_m:
            addr = addr_m.group(1).strip()
            # Filter garbage: must start with a real house number and have reasonable length
            if len(addr) < 10 or not re.match(r"\d{2,5}\s+[A-Za-z]", addr):
                continue
            price = int(price_m.group(1).replace(",", ""))
            if _is_valid_price(price):
                comps.append({"address": addr, "price": price})
    return comps


def extract_jsonld_comps(soup: BeautifulSoup) -> list[dict]:
    """Extract comparable properties from JSON-LD (SingleFamilyResidence, Product with offers)."""
    comps = []
    seen_addresses = set()
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                comp = _parse_jsonld_comp(item)
                if comp and comp["address"].lower() not in seen_addresses:
                    seen_addresses.add(comp["address"].lower())
                    comps.append(comp)
        except (json.JSONDecodeError, TypeError):
            continue
    logger.debug("JSON-LD extracted %d comps", len(comps))
    return comps


def _parse_jsonld_comp(item: dict) -> dict | None:
    """Parse a single JSON-LD item into a comp dict if it has address and price."""
    item_type = item.get("@type", "")
    if isinstance(item_type, list):
        item_type = " ".join(item_type)

    relevant_types = ("SingleFamilyResidence", "Residence", "House", "Product",
                      "RealEstateListing", "Apartment", "Accommodation")
    if not any(t in item_type for t in relevant_types):
        return None

    # Extract address
    address = None
    addr_obj = item.get("address")
    if isinstance(addr_obj, dict):
        parts = [addr_obj.get("streetAddress", ""), addr_obj.get("addressLocality", ""),
                 addr_obj.get("addressRegion", ""), addr_obj.get("postalCode", "")]
        address = ", ".join(p for p in parts if p)
    elif isinstance(addr_obj, str):
        address = addr_obj
    if not address:
        address = item.get("name", "")
    if not address:
        return None

    # Extract price from offers or direct price
    price = None
    offers = item.get("offers")
    if isinstance(offers, dict):
        raw = offers.get("price")
        if raw is not None:
            try:
                price = int(float(str(raw).replace(",", "").replace("$", "")))
            except (ValueError, TypeError):
                pass
    if price is None:
        for key in ("price", "value"):
            raw = item.get(key)
            if raw is not None:
                try:
                    price = int(float(str(raw).replace(",", "").replace("$", "")))
                except (ValueError, TypeError):
                    pass
                if price:
                    break

    if not price or not _is_valid_price(price):
        return None

    return {"address": address, "price": price}


def _detect_blocked(html: str, status_code: int) -> str | None:
    """Detect if response is a blocked/captcha page. Returns reason or None."""
    html_lower = html.lower()
    if status_code == 403:
        if "captcha" in html_lower or "perimeterx" in html_lower:
            return "captcha (PerimeterX)"
        if "access denied" in html_lower:
            return "access denied (WAF)"
        return "403 Forbidden"
    if status_code == 429:
        return "rate limited (429)"
    if status_code >= 400:
        return f"HTTP {status_code}"
    if "captcha" in html_lower and len(html) < 20000:
        return "captcha detected in response"
    return None


def extract_all_signals(html: str, label: str) -> dict:
    """Run full extraction pipeline on HTML, returning prices and comps."""
    soup = BeautifulSoup(html, "html.parser")
    all_prices = []
    extraction_log = []

    # 1. JSON-LD structured data
    jsonld_prices = extract_json_ld_prices(soup)
    if jsonld_prices:
        all_prices.extend(jsonld_prices)
        extraction_log.append(f"JSON-LD: {len(jsonld_prices)} prices")

    # 2. Meta tags
    meta_prices = extract_meta_prices(soup)
    if meta_prices:
        all_prices.extend(meta_prices)
        extraction_log.append(f"Meta tags: {len(meta_prices)} prices")

    # 3. Inline JavaScript JSON
    inline_prices = extract_inline_json_prices(soup)
    if inline_prices:
        all_prices.extend(inline_prices)
        extraction_log.append(f"Inline JSON: {len(inline_prices)} prices")

    # 4. CSS selector extraction for specific sites
    selector_prices = _extract_by_selectors(soup, label)
    if selector_prices:
        all_prices.extend(selector_prices)
        extraction_log.append(f"CSS selectors: {len(selector_prices)} prices")

    # 5. Fallback: plain text regex
    if not all_prices:
        text_prices = extract_prices_from_text(html)
        if text_prices:
            all_prices.extend(text_prices)
            extraction_log.append(f"Text regex fallback: {len(text_prices)} prices")

    # Deduplicate while preserving order
    seen = set()
    unique_prices = []
    for p in all_prices:
        if p not in seen:
            seen.add(p)
            unique_prices.append(p)

    # Extract comps from both text and JSON-LD structured data
    text_comps = extract_comps_from_text(html)
    jsonld_comps = extract_jsonld_comps(soup)

    # Merge comps, preferring JSON-LD (more structured)
    seen_addrs = set()
    comps = []
    for comp in jsonld_comps + text_comps:
        key = comp["address"].lower().strip()
        if key not in seen_addrs:
            seen_addrs.add(key)
            comps.append(comp)

    if jsonld_comps:
        extraction_log.append(f"JSON-LD comps: {len(jsonld_comps)}")

    logger.info("[%s] Extraction: %s | %d unique prices, %d comps",
                label, "; ".join(extraction_log) if extraction_log else "no signals",
                len(unique_prices), len(comps))

    return {
        "prices": unique_prices,
        "comps": comps,
        "extractionLog": extraction_log,
    }


def _extract_by_selectors(soup: BeautifulSoup, label: str) -> list[int]:
    """Site-specific CSS selector extraction."""
    prices = []
    selectors = []

    label_lower = label.lower()
    if "zillow" in label_lower:
        selectors = [
            '[data-testid="zestimate-text"]',
            ".zestimate",
            'span[data-testid="price"]',
            ".ds-summary-row .ds-body",
        ]
    elif "redfin" in label_lower:
        selectors = [
            '[data-rf-test-id="avmLdpPrice"]',
            '[data-rf-test-id="avm-price"]',
            ".statsValue",
            ".RedfinEstimateValueHeader",
            ".estimate",
        ]
    elif "realtor" in label_lower:
        selectors = [
            '[data-testid="list-price"]',
            ".property-price",
            ".listing-price",
            ".price-section .price",
        ]
    elif "homes" in label_lower:
        selectors = [
            ".property-info-price",
            ".price-info .price",
            '[data-testid="property-price"]',
            ".home-value-estimate",
            ".estimated-value",
        ]

    for sel in selectors:
        for el in soup.select(sel):
            text = el.get_text(strip=True)
            m = re.search(r"\$?\s*([\d,]+)", text)
            if m:
                try:
                    val = int(m.group(1).replace(",", ""))
                    if _is_valid_price(val):
                        prices.append(val)
                        logger.debug("CSS selector [%s] %s -> %d", label, sel, val)
                except (ValueError, TypeError):
                    pass
    return prices


async def fetch_source(source: dict, timeout: int = 20) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.google.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Cache-Control": "no-cache",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(source["url"], headers=headers, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                html = await resp.text()
                logger.info("[%s] Fetched %d bytes, status %d, url=%s",
                            source["label"], len(html), resp.status, resp.url)

                # Check for blocked/captcha responses
                blocked_reason = _detect_blocked(html, resp.status)
                if blocked_reason:
                    logger.warning("[%s] BLOCKED: %s", source["label"], blocked_reason)
                    return {
                        "label": source["label"],
                        "ok": False,
                        "error": f"blocked: {blocked_reason}",
                        "extractedValue": None,
                        "signalCount": 0,
                        "prices": [],
                        "comps": [],
                        "extractionLog": [f"Blocked: {blocked_reason}"],
                    }

                signals = extract_all_signals(html, source["label"])
                prices = signals["prices"]
                comps = signals["comps"]
                signal_count = len(prices) + len(comps)

                # Compute median of extracted prices as the representative value
                extracted_value = None
                if prices:
                    sorted_prices = sorted(prices)
                    extracted_value = sorted_prices[len(sorted_prices) // 2]

                logger.info("[%s] OK: %d prices, %d comps, extractedValue=%s",
                            source["label"], len(prices), len(comps), extracted_value)

                return {
                    "label": source["label"],
                    "ok": True,
                    "extractedValue": extracted_value,
                    "signalCount": signal_count,
                    "prices": prices,
                    "comps": comps,
                    "extractionLog": signals["extractionLog"],
                }
    except Exception as e:
        logger.error("[%s] Fetch failed: %s", source["label"], e)
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
    def _build_training_stats(comparables: dict) -> dict:
        """Compute ML training stats using the full training pipeline."""
        from .training import compute_training_stats

        all_comps = list(comparables.get("recentSales") or [])
        all_comps += list(comparables.get("currentListings") or [])
        valid_comps = [c for c in all_comps if c.get("price") and c["price"] > 0]

        # Build sales-like dicts for compute_training_stats
        sales = []
        for c in valid_comps:
            sales.append({
                "address": c.get("address", ""),
                "sale_price": c["price"],
                "sale_date": c.get("closedDate") or c.get("listedDate") or "",
                "sqft": c.get("squareFeet"),
                "year_built": c.get("yearBuilt"),
                "beds": c.get("bedrooms"),
                "baths": c.get("bathrooms"),
                "lot_size": c.get("lotSizeSqFt"),
                "source": c.get("source", "unknown"),
            })

        return compute_training_stats(sales, valid_comps)

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

        # Try to use DB sales data, fall back to hardcoded comps
        try:
            from .db import get_connection, get_recent_sales, get_target_property_id, insert_valuation
            conn = get_connection()
            db_sales = get_recent_sales(conn, "22306", months=48, exclude_address="2919 Wahoo Way")
            conn.close()
        except Exception:
            db_sales = []

        if db_sales:
            db_comps = []
            for s in db_sales:
                db_comps.append({
                    "address": s["address"],
                    "price": s["sale_price"],
                    "closedDate": s["sale_date"],
                    "squareFeet": s.get("sqft"),
                    "yearBuilt": s.get("year_built"),
                    "bedrooms": s.get("beds"),
                    "bathrooms": s.get("baths"),
                    "lotSizeSqFt": s.get("lot_size"),
                    "source": s.get("source", "db"),
                })
            fallback_with_db = {
                "recentSales": db_comps,
                "currentListings": FALLBACK_COMPARABLES.get("currentListings", []),
            }
            comparables = self._merge_source_comps(source_results, fallback_with_db)
        else:
            fallback = FALLBACK_COMPARABLES
            comparables = self._merge_source_comps(source_results, fallback)

        valuation = self._build_valuation(comparables, source_results)
        financials = self._build_financials(valuation["estimate"])
        training_stats = self._build_training_stats(comparables)

        # Store valuation in DB
        try:
            from .db import get_connection, get_target_property_id, insert_valuation
            conn = get_connection()
            prop_id = get_target_property_id(conn)
            if valuation.get("estimate"):
                conf_range = valuation.get("confidenceRange", {})
                insert_valuation(
                    conn, prop_id,
                    estimate=valuation["estimate"],
                    confidence=valuation.get("confidencePct", 0),
                    confidence_low=conf_range.get("low", 0),
                    confidence_high=conf_range.get("high", 0),
                    ml_details=valuation.get("mlDetails"),
                    training_samples=len(db_sales) if db_sales else len(FALLBACK_COMPARABLES.get("recentSales", [])),
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to store valuation in DB: %s", e)

        all_sale_prices = sorted(
            [c["price"] for c in (comparables.get("recentSales") or []) if c.get("price", 0) > 0]
        )
        median_price = all_sale_prices[len(all_sale_prices) // 2] if all_sale_prices else None

        data = {
            "valuation": valuation,
            "financials": financials,
            "trainingStats": training_stats,
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
