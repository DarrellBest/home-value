"""Data fetcher module - Redfin GIS-CSV sold listings + Zillow ZHVI market data
+ Zillow Research sale price CSVs + Alexandria City Assessor scraper."""

import csv
import io
import re
import logging
import asyncio
from datetime import datetime
from urllib.parse import urlencode, quote

import aiohttp
from bs4 import BeautifulSoup

from .db import get_connection, upsert_property, insert_sale, insert_market_stat, get_sale_count

logger = logging.getLogger("home_value")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# 22306 bounding box (Alexandria, VA) - includes nearby zips for more data
# South: 38.745, North: 38.795, West: -77.11, East: -77.05
REDFIN_POLYGON = "-77.11 38.745,-77.05 38.745,-77.05 38.795,-77.11 38.795,-77.11 38.745"
REDFIN_GIS_CSV_URL = (
    "https://www.redfin.com/stingray/api/gis-csv?al=1&market=dc"
    "&min_price=100000&max_price=2000000"
    "&poly={poly}"
    "&sold_within_days=365&num_homes=350&v=8"
)

# Target zip codes - 22306 is primary, neighbors provide additional training data
TARGET_ZIPS = {"22306", "22303", "22307", "22309"}

# Zillow Home Value Index - monthly median values by ZIP
ZILLOW_ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]


def _parse_price(text: str | None) -> int | None:
    if not text:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(text))
    if not cleaned:
        return None
    try:
        val = int(float(cleaned))
        return val if 50_000 <= val <= 5_000_000 else None
    except (ValueError, TypeError):
        return None


def _parse_date(text: str | None) -> str | None:
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%dT%H:%M:%S",
                "%B-%d-%Y"):  # Redfin uses "August-8-2025"
        try:
            return datetime.strptime(text.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try "Month-Day-Year" with variations
    m = re.match(r"(\w+)-(\d+)-(\d{4})", text.strip())
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    try:
        return int(float(re.sub(r"[^\d.]", "", str(text))))
    except (ValueError, TypeError):
        return None


async def _fetch_with_retry(session: aiohttp.ClientSession, url: str,
                            timeout: int = 30) -> str | None:
    """Fetch URL with retry logic and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(
                url, headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=True,
            ) as resp:
                if resp.status == 200:
                    return await resp.text()
                elif resp.status == 403:
                    logger.warning("HTTP 403 from %s (attempt %d) - may be rate limited",
                                   url[:80], attempt + 1)
                elif resp.status >= 500:
                    logger.warning("HTTP %d from %s (attempt %d)", resp.status, url[:80], attempt + 1)
                else:
                    logger.warning("HTTP %d from %s", resp.status, url[:80])
                    return None  # Don't retry 4xx (except 403)
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching %s (attempt %d/%d)", url[:80], attempt + 1, MAX_RETRIES)
        except aiohttp.ClientError as e:
            logger.warning("Client error fetching %s (attempt %d): %s", url[:80], attempt + 1, e)

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF[attempt]
            logger.info("Retrying in %ds...", wait)
            await asyncio.sleep(wait)

    logger.error("All %d attempts failed for %s", MAX_RETRIES, url[:80])
    return None


async def fetch_redfin_sold_listings() -> dict:
    """Fetch recently sold listings from Redfin's GIS-CSV endpoint.

    Uses a polygon bounding box around 22306 Alexandria, VA.
    Returns dict with counts of new properties and sales inserted.
    """
    new_properties = 0
    new_sales = 0
    skipped = 0
    errors = []

    poly_encoded = REDFIN_POLYGON.replace(" ", "%20").replace(",", "%2C")
    url = REDFIN_GIS_CSV_URL.format(poly=poly_encoded)

    try:
        async with aiohttp.ClientSession() as session:
            text = await _fetch_with_retry(session, url, timeout=45)
            if not text:
                errors.append("Failed to fetch Redfin GIS-CSV after retries")
                return {"new_properties": 0, "new_sales": 0, "errors": errors}

            # Parse CSV - skip disclaimer line
            lines = text.strip().split("\n")
            clean_lines = [l for l in lines if not l.startswith('"In accordance')]
            reader = csv.DictReader(io.StringIO("\n".join(clean_lines)))

            conn = get_connection()
            try:
                for row in reader:
                    try:
                        zip_code = row.get("ZIP OR POSTAL CODE", "").strip()
                        if zip_code not in TARGET_ZIPS:
                            continue

                        address_parts = [
                            row.get("ADDRESS", "").strip(),
                            row.get("CITY", "").strip(),
                            row.get("STATE OR PROVINCE", "").strip(),
                            zip_code,
                        ]
                        address = ", ".join(p for p in address_parts if p)
                        if not address:
                            skipped += 1
                            continue

                        price = _parse_price(row.get("PRICE"))
                        sold_date = _parse_date(row.get("SOLD DATE"))

                        if not price:
                            skipped += 1
                            continue

                        sqft = _parse_int(row.get("SQUARE FEET"))
                        beds = _parse_int(row.get("BEDS"))
                        baths_str = row.get("BATHS", "")
                        baths = float(baths_str) if baths_str else None
                        year_built = _parse_int(row.get("YEAR BUILT"))
                        lot_size = _parse_int(row.get("LOT SIZE"))

                        prop_id = upsert_property(
                            conn, address,
                            sqft=sqft, beds=beds, baths=baths,
                            year_built=year_built, lot_size=lot_size,
                        )
                        new_properties += 1

                        if sold_date:
                            if insert_sale(conn, prop_id, price, sold_date, "redfin"):
                                new_sales += 1
                        else:
                            # No sold date - use price as listing reference
                            if insert_sale(conn, prop_id, price,
                                           datetime.now().strftime("%Y-%m-%d"), "redfin-listing"):
                                new_sales += 1

                    except Exception as e:
                        logger.debug("Error processing Redfin row: %s", e)
                        skipped += 1

                conn.commit()
            finally:
                conn.close()

    except Exception as e:
        logger.error("Redfin sold listings fetch failed: %s", e)
        errors.append(str(e))

    logger.info("Redfin sold listings: %d properties, %d sales, %d skipped",
                new_properties, new_sales, skipped)
    result = {"new_properties": new_properties, "new_sales": new_sales, "skipped": skipped}
    if errors:
        result["errors"] = errors
    return result


async def fetch_zillow_market_data() -> dict:
    """Fetch Zillow Home Value Index data for 22306.

    Downloads the ZHVI CSV and extracts monthly median values for the target ZIP.
    Returns dict with count of new market stat records.
    """
    new_stats = 0
    errors = []

    try:
        async with aiohttp.ClientSession() as session:
            text = await _fetch_with_retry(session, ZILLOW_ZHVI_URL, timeout=60)
            if not text:
                errors.append("Failed to fetch Zillow ZHVI CSV after retries")
                return {"new_stats": 0, "errors": errors}

            lines = text.strip().split("\n")
            header = lines[0].split(",")

            # Find the date columns (they start after the metadata columns)
            date_col_start = None
            for i, col in enumerate(header):
                if re.match(r"\d{4}-\d{2}-\d{2}", col):
                    date_col_start = i
                    break

            if date_col_start is None:
                errors.append("Could not find date columns in Zillow ZHVI CSV")
                return {"new_stats": 0, "errors": errors}

            # Find 22306 row
            target_row = None
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > 2 and parts[2] == "22306":
                    target_row = parts
                    break

            if not target_row:
                errors.append("ZIP 22306 not found in Zillow ZHVI data")
                return {"new_stats": 0, "errors": errors}

            conn = get_connection()
            try:
                # Import last 24 months of ZHVI data
                for i in range(max(date_col_start, len(target_row) - 24), len(target_row)):
                    if i >= len(header) or i >= len(target_row):
                        continue
                    date_str = header[i].strip()
                    value_str = target_row[i].strip()
                    if not date_str or not value_str:
                        continue

                    month = date_str[:7]  # "2025-01-31" -> "2025-01"
                    try:
                        median_price = int(float(value_str))
                    except (ValueError, TypeError):
                        continue

                    if 100_000 <= median_price <= 2_000_000:
                        insert_market_stat(conn, "22306", median_price, None, month, "zillow-zhvi")
                        new_stats += 1

                conn.commit()
            finally:
                conn.close()

    except Exception as e:
        logger.error("Zillow ZHVI fetch failed: %s", e)
        errors.append(str(e))

    logger.info("Zillow ZHVI: %d market stat records for 22306", new_stats)
    result = {"new_stats": new_stats}
    if errors:
        result["errors"] = errors
    return result


async def fetch_redfin_market_data() -> dict:
    """Fetch Redfin Data Center market stats for 22306.

    Uses the smaller city-level weekly housing data.
    Returns dict with count of new stat records.
    """
    new_stats = 0
    errors = []

    # Use the city-level data which is much smaller than the national file
    url = ("https://redfin-public-data.s3.us-west-2.amazonaws.com/"
           "redfin_market_tracker/city_market_tracker.tsv000.gz")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Redfin data center returned %d", resp.status)
                    errors.append(f"HTTP {resp.status}")
                    return {"new_stats": 0, "errors": errors}

                raw = await resp.read()

                import gzip
                try:
                    text = gzip.decompress(raw).decode("utf-8", errors="replace")
                except (gzip.BadGzipFile, OSError):
                    text = raw.decode("utf-8", errors="replace")

                conn = get_connection()
                try:
                    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
                    for row in reader:
                        region = row.get("region", "")
                        # Match Alexandria city
                        if "alexandria" not in region.lower():
                            continue
                        state = row.get("state", "") or row.get("state_code", "")
                        if state and "va" not in state.lower() and "virginia" not in state.lower():
                            continue

                        period = row.get("period_begin") or row.get("month_date_yyyymm")
                        month = _parse_date(period)
                        if not month:
                            month = period[:7] if period and len(period) >= 7 else None
                        if not month:
                            continue

                        median = _parse_price(
                            row.get("median_sale_price") or
                            row.get("median_sale_price_adjusted")
                        )
                        yoy_str = row.get("median_sale_price_yoy") or row.get("median_ppsf_yoy")
                        yoy = None
                        if yoy_str:
                            try:
                                yoy = round(float(yoy_str) * 100, 2)
                            except (ValueError, TypeError):
                                pass

                        if median or yoy is not None:
                            insert_market_stat(conn, "22306", median, yoy, month, "redfin")
                            new_stats += 1

                    conn.commit()
                finally:
                    conn.close()

    except asyncio.TimeoutError:
        logger.error("Redfin data center download timed out (file too large)")
        errors.append("Timeout downloading Redfin data center TSV")
    except Exception as e:
        logger.error("Redfin data center fetch failed: %s", e)
        errors.append(str(e))

    logger.info("Redfin market data: %d stat records", new_stats)
    result = {"new_stats": new_stats}
    if errors:
        result["errors"] = errors
    return result


###############################################################################
# Zillow Research Data - Median Sale Price by ZIP
###############################################################################

# Zillow publishes median sale price (not ZHVI) and sale counts per ZIP
ZILLOW_MEDIAN_SALE_URL = (
    "https://files.zillowstatic.com/research/public_csvs/median_sale_price/"
    "Metro_median_sale_price_uc_sfrcondo_month.csv"
)
ZILLOW_SALE_COUNT_URL = (
    "https://files.zillowstatic.com/research/public_csvs/sales_count_now/"
    "Metro_sales_count_now_uc_sfrcondo_month.csv"
)
# ZIP-level sale price data (more granular)
ZILLOW_ZIP_MEDIAN_SALE_URL = (
    "https://files.zillowstatic.com/research/public_csvs/median_sale_price/"
    "Zip_median_sale_price_uc_sfrcondo_sm_sa_month.csv"
)


async def fetch_zillow_research_sales() -> dict:
    """Fetch Zillow Research median sale price CSV by ZIP code.

    Downloads the ZIP-level median sale price CSV and extracts monthly data
    for target ZIP codes. Inserts as market stats (these are aggregate, not
    property-level, but supplement training data with price signals).
    """
    new_stats = 0
    errors = []

    urls_to_try = [
        ("Zip_median_sale_price", ZILLOW_ZIP_MEDIAN_SALE_URL),
        ("Metro_median_sale_price", ZILLOW_MEDIAN_SALE_URL),
    ]

    try:
        async with aiohttp.ClientSession() as session:
            text = None
            source_name = None
            for name, url in urls_to_try:
                text = await _fetch_with_retry(session, url, timeout=90)
                if text:
                    source_name = name
                    logger.info("Zillow Research: fetched %s (%d bytes)", name, len(text))
                    break
                logger.warning("Zillow Research: %s not available, trying next", name)

            if not text:
                errors.append("Failed to fetch any Zillow Research sale price CSV")
                return {"new_stats": 0, "errors": errors}

            reader = csv.DictReader(io.StringIO(text))
            fieldnames = reader.fieldnames or []

            # Find date columns (YYYY-MM-DD format)
            date_cols = [c for c in fieldnames if re.match(r"\d{4}-\d{2}-\d{2}", c)]
            if not date_cols:
                errors.append(f"No date columns found in {source_name}")
                return {"new_stats": 0, "errors": errors}

            # Only keep last 36 months of date columns
            date_cols = date_cols[-36:]

            conn = get_connection()
            try:
                for row in reader:
                    # Match by ZIP code or by region name containing Alexandria
                    zip_val = row.get("RegionName", "").strip()
                    region_type = row.get("RegionType", "").strip().lower()

                    is_target_zip = zip_val in TARGET_ZIPS
                    is_alexandria_metro = (
                        "alexandria" in zip_val.lower()
                        or ("metro" in region_type and "washington" in zip_val.lower())
                    )

                    if not is_target_zip and not is_alexandria_metro:
                        continue

                    zip_label = zip_val if is_target_zip else "22306"

                    for date_col in date_cols:
                        val_str = row.get(date_col, "").strip()
                        if not val_str:
                            continue
                        try:
                            median_price = int(float(val_str))
                        except (ValueError, TypeError):
                            continue
                        if 50_000 <= median_price <= 2_000_000:
                            month = date_col[:7]
                            insert_market_stat(conn, zip_label, median_price, None,
                                               month, "zillow-research")
                            new_stats += 1

                conn.commit()
            finally:
                conn.close()

    except Exception as e:
        logger.error("Zillow Research sales fetch failed: %s", e)
        errors.append(str(e))

    logger.info("Zillow Research: %d market stat records imported (source: %s)",
                new_stats, source_name or "none")
    result = {"new_stats": new_stats, "source": source_name}
    if errors:
        result["errors"] = errors
    return result


###############################################################################
# Alexandria City Assessor Scraper
###############################################################################

ALEX_ASSESSOR_BASE = "https://realestate.alexandriava.gov"

# Politeness delay between requests (seconds)
ASSESSOR_DELAY = 1.5

# Streets in/near 22306 to search (format: ^PREFIX^STREET^SUFFIX for the select value)
ASSESSOR_STREETS_22306 = [
    "^RICHMOND^HWY", "^RICHMOND^LN", "^BELLE VIEW^BLVD",
    "^FORT HUNT^RD", "^COLLINGWOOD^RD", "^HUNTINGTON^AVE",
    "^MOUNT EAGLE^DR", "^MOUNT EAGLE^PL", "^BEACON HILL^RD",
    "^PAUL SPRING^RD", "^EDGEHILL^DR", "^BATTERSEA^LN",
    "^SHENANDOAH^RD", "^STRATFORD^LN", "^WOODLAWN^TER",
    "^ELKIN^ST", "^RUSSELL^RD", "^JANNEYS^LN",
    "^TANEY^AVE", "^MEMORIAL^ST", "^POPKINS^LN",
    "^MIDDAY^LN", "^BISCAYNE^DR", "^CREEKSIDE^DR",
    "^GROVETON^ST", "^LUKENS^LN", "^WINDING^WAY",
]


async def fetch_alexandria_assessor() -> dict:
    """Scrape sales from the Alexandria City real estate assessor site.

    Searches by street name, collects account numbers from results,
    then fetches each property's detail page for sales history and attributes.
    """
    new_properties = 0
    new_sales = 0
    skipped = 0
    errors = []
    seen_accounts = set()

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Collect account numbers from street searches
            for street_val in ASSESSOR_STREETS_22306:
                await asyncio.sleep(ASSESSOR_DELAY)
                search_url = (
                    f"{ALEX_ASSESSOR_BASE}/index.php?"
                    f"action=address&StreetNumber=&StreetName={quote(street_val)}"
                    f"&UnitNo=&Search=Search"
                )
                try:
                    page = await _fetch_with_retry(session, search_url, timeout=20)
                    if not page:
                        continue
                    # Extract account numbers from detail.php links
                    accounts = set(re.findall(r'detail\.php\?accountno=(\d+)', page))
                    new_accounts = accounts - seen_accounts
                    seen_accounts.update(new_accounts)
                    logger.debug("Assessor street %s: %d accounts (%d new)",
                                 street_val, len(accounts), len(new_accounts))
                except Exception as e:
                    logger.debug("Assessor search failed for %s: %s", street_val, e)

            logger.info("Alexandria assessor: found %d unique accounts to fetch", len(seen_accounts))

            # Step 2: Fetch detail page for each account
            conn = get_connection()
            try:
                for acct in seen_accounts:
                    await asyncio.sleep(ASSESSOR_DELAY)
                    try:
                        p, s, sk = await _fetch_assessor_detail(session, conn, acct)
                        new_properties += p
                        new_sales += s
                        skipped += sk
                    except Exception as e:
                        logger.debug("Assessor detail failed for %s: %s", acct, e)
                        skipped += 1

                conn.commit()
            finally:
                conn.close()

    except Exception as e:
        logger.error("Alexandria assessor fetch failed: %s", e)
        errors.append(str(e))

    logger.info("Alexandria assessor: %d properties, %d sales, %d skipped",
                new_properties, new_sales, skipped)
    result = {"new_properties": new_properties, "new_sales": new_sales, "skipped": skipped}
    if errors:
        result["errors"] = errors
    return result


async def _fetch_assessor_detail(session: aiohttp.ClientSession,
                                  conn, acct: str) -> tuple[int, int, int]:
    """Fetch and parse a single property detail page from the Alexandria assessor.

    Extracts: address, year built, sqft, baths, and full sale history.
    Returns (new_properties, new_sales, skipped).
    """
    url = f"{ALEX_ASSESSOR_BASE}/detail.php?accountno={acct}"
    html = await _fetch_with_retry(session, url, timeout=20)
    if not html:
        return 0, 0, 1

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # Extract address from page title or content
    # Pattern: "NNNN STREET NAME" followed by city info
    addr_match = re.search(
        r'(\d+\s+[A-Z][A-Z\s]+(?:ST|AVE|DR|RD|LN|CT|PL|WAY|TER|BLVD|CIR|HWY|PKWY))\s*,?\s*ALEXANDRIA',
        text, re.IGNORECASE
    )
    if not addr_match:
        return 0, 0, 1

    address = f"{addr_match.group(1).strip()}, Alexandria, VA"

    # Extract property attributes
    sqft_match = re.search(r'Above Grade Living Area.*?(\d[\d,]+)', text)
    sqft = _parse_int(sqft_match.group(1)) if sqft_match else None

    year_match = re.search(r'Year Built:\s*(\d{4})', text)
    year_built = int(year_match.group(1)) if year_match else None

    full_baths_match = re.search(r'Full Baths:\s*(\d+)', text)
    half_baths_match = re.search(r'Half Baths:\s*(\d+)', text)
    baths = None
    if full_baths_match:
        baths = float(full_baths_match.group(1))
        if half_baths_match:
            baths += float(half_baths_match.group(1)) * 0.5

    # Upsert property
    prop_id = upsert_property(conn, address, sqft=sqft, baths=baths, year_built=year_built)
    new_properties = 1
    new_sales = 0

    # Extract sales from the sales table (last table with Sale Date / Sale Price headers)
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header_cells = [c.get_text(strip=True).lower() for c in rows[0].find_all(["th", "td"])]
        if "sale date" not in " ".join(header_cells) or "sale price" not in " ".join(header_cells):
            continue

        # Find column indices
        date_idx = price_idx = None
        for i, h in enumerate(header_cells):
            if "sale date" in h:
                date_idx = i
            elif "sale price" in h:
                price_idx = i
        if date_idx is None or price_idx is None:
            continue

        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) <= max(date_idx, price_idx):
                continue
            sale_date = _parse_date(cells[date_idx].get_text(strip=True))
            sale_price = _parse_price(cells[price_idx].get_text(strip=True))
            if sale_date and sale_price:
                if insert_sale(conn, prop_id, sale_price, sale_date, "alexandria-assessor"):
                    new_sales += 1

    return new_properties, new_sales, 0


def load_fallback_comps_to_db():
    """Load the hardcoded fallback comparables into the database as seed data."""
    from .config import FALLBACK_COMPARABLES

    conn = get_connection()
    new_sales = 0
    try:
        for comp in FALLBACK_COMPARABLES.get("recentSales", []):
            prop_id = upsert_property(
                conn,
                address=comp["address"],
                sqft=comp.get("squareFeet"),
                beds=comp.get("bedrooms"),
                baths=comp.get("bathrooms"),
                year_built=comp.get("yearBuilt"),
                lot_size=comp.get("lotSizeSqFt"),
            )
            if comp.get("price") and comp.get("closedDate"):
                if insert_sale(conn, prop_id, comp["price"], comp["closedDate"], comp.get("source", "fallback")):
                    new_sales += 1

        for comp in FALLBACK_COMPARABLES.get("currentListings", []):
            upsert_property(
                conn,
                address=comp["address"],
                sqft=comp.get("squareFeet"),
                beds=comp.get("bedrooms"),
                baths=comp.get("bathrooms"),
                year_built=comp.get("yearBuilt"),
                lot_size=comp.get("lotSizeSqFt"),
            )

        conn.commit()
        logger.info("Loaded %d fallback sales into database", new_sales)
    finally:
        conn.close()

    return new_sales


async def run_full_fetch() -> dict:
    """Run all data fetchers and return summary."""
    results = {}

    # Load fallback comps as seed data first
    seed_sales = load_fallback_comps_to_db()
    results["seed_sales"] = seed_sales

    # Run fetchers in parallel
    fetch_results = await asyncio.gather(
        fetch_redfin_sold_listings(),
        fetch_zillow_market_data(),
        fetch_redfin_market_data(),
        fetch_zillow_research_sales(),
        fetch_alexandria_assessor(),
        return_exceptions=True,
    )

    names = ["redfin_sold", "zillow_market", "redfin_market",
             "zillow_research", "alexandria_assessor"]
    for name, result in zip(names, fetch_results):
        if isinstance(result, dict):
            results[name] = result
        else:
            logger.error("%s fetch exception: %s", name, result)
            results[name] = {"error": str(result)}

    conn = get_connection()
    try:
        results["total_sales"] = get_sale_count(conn)
        # Also count by zip
        for z in sorted(TARGET_ZIPS):
            results[f"sales_{z}"] = get_sale_count(conn, z)
    finally:
        conn.close()

    logger.info("Full data fetch complete: %s", results)
    return results
