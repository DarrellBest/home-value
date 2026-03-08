"""SQLite database layer for home value data pipeline."""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR

logger = logging.getLogger("home_value")

DB_PATH = DATA_DIR / "home_value.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT NOT NULL UNIQUE,
    sqft INTEGER,
    beds INTEGER,
    baths REAL,
    year_built INTEGER,
    lot_size INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id INTEGER NOT NULL REFERENCES properties(id),
    sale_price INTEGER NOT NULL,
    sale_date TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(property_id, sale_date, source)
);

CREATE TABLE IF NOT EXISTS valuations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id INTEGER NOT NULL REFERENCES properties(id),
    estimate INTEGER NOT NULL,
    confidence REAL,
    confidence_low INTEGER,
    confidence_high INTEGER,
    ml_details TEXT,
    training_samples INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS market_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zip_code TEXT NOT NULL,
    median_price INTEGER,
    yoy_change REAL,
    month TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'redfin',
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(zip_code, month, source)
);

CREATE INDEX IF NOT EXISTS idx_sales_property ON sales(property_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_valuations_property ON valuations(property_id);
CREATE INDEX IF NOT EXISTS idx_valuations_created ON valuations(created_at);
CREATE INDEX IF NOT EXISTS idx_market_stats_zip ON market_stats(zip_code, month);
"""


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)


def upsert_property(conn: sqlite3.Connection, address: str, sqft: int | None = None,
                    beds: int | None = None, baths: float | None = None,
                    year_built: int | None = None, lot_size: int | None = None) -> int:
    cur = conn.execute("SELECT id FROM properties WHERE address = ?", (address,))
    row = cur.fetchone()
    if row:
        prop_id = row["id"]
        updates = []
        params = []
        for col, val in [("sqft", sqft), ("beds", beds), ("baths", baths),
                         ("year_built", year_built), ("lot_size", lot_size)]:
            if val is not None:
                updates.append(f"{col} = ?")
                params.append(val)
        if updates:
            params.append(prop_id)
            conn.execute(f"UPDATE properties SET {', '.join(updates)} WHERE id = ?", params)
        return prop_id
    else:
        cur = conn.execute(
            "INSERT INTO properties (address, sqft, beds, baths, year_built, lot_size) VALUES (?, ?, ?, ?, ?, ?)",
            (address, sqft, beds, baths, year_built, lot_size)
        )
        return cur.lastrowid


def insert_sale(conn: sqlite3.Connection, property_id: int, sale_price: int,
                sale_date: str, source: str = "manual") -> bool:
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sales (property_id, sale_price, sale_date, source) VALUES (?, ?, ?, ?)",
            (property_id, sale_price, sale_date, source)
        )
        return True
    except sqlite3.IntegrityError:
        return False


def insert_valuation(conn: sqlite3.Connection, property_id: int, estimate: int,
                     confidence: float, confidence_low: int, confidence_high: int,
                     ml_details: dict | None = None, training_samples: int = 0):
    conn.execute(
        "INSERT INTO valuations (property_id, estimate, confidence, confidence_low, confidence_high, ml_details, training_samples) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (property_id, estimate, confidence, confidence_low, confidence_high,
         json.dumps(ml_details) if ml_details else None, training_samples)
    )


def insert_market_stat(conn: sqlite3.Connection, zip_code: str, median_price: int | None,
                       yoy_change: float | None, month: str, source: str = "redfin"):
    conn.execute(
        "INSERT OR REPLACE INTO market_stats (zip_code, median_price, yoy_change, month, source) VALUES (?, ?, ?, ?, ?)",
        (zip_code, median_price, yoy_change, month, source)
    )


def get_recent_sales(conn: sqlite3.Connection, zip_code: str = "22306",
                     months: int = 12, exclude_address: str | None = None) -> list[dict]:
    """Get all nearby sales from the last N months.

    Includes adjacent zip codes (22306, 22307, 22309) for broader comparable pool.
    Optionally excludes a specific address (e.g. subject property) to prevent data leakage.
    """
    nearby_zips = ("22306", "22307", "22309", "22303")
    zip_clauses = " OR ".join("p.address LIKE ?" for _ in nearby_zips)
    zip_clauses += " OR p.address LIKE '%Alexandria, VA%'"
    params: list = [f"%{z}%" for z in nearby_zips]
    params.append(f"-{months} months")

    exclude_clause = ""
    if exclude_address:
        exclude_clause = "AND p.address NOT LIKE ?"
        params.append(f"%{exclude_address}%")

    query = f"""
        SELECT p.address, p.sqft, p.beds, p.baths, p.year_built, p.lot_size,
               s.sale_price, s.sale_date, s.source
        FROM sales s
        JOIN properties p ON s.property_id = p.id
        WHERE ({zip_clauses})
          AND s.sale_date >= date('now', ?)
          {exclude_clause}
        ORDER BY s.sale_date DESC
    """
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_valuation_history(conn: sqlite3.Connection, property_id: int,
                          weeks: int = 52) -> list[dict]:
    """Get valuation history for a property over the last N weeks."""
    query = """
        SELECT estimate, confidence, confidence_low, confidence_high,
               training_samples, created_at
        FROM valuations
        WHERE property_id = ?
          AND created_at >= datetime('now', ?)
        ORDER BY created_at ASC
    """
    rows = conn.execute(query, (property_id, f"-{weeks * 7} days")).fetchall()
    return [dict(r) for r in rows]


def get_target_property_id(conn: sqlite3.Connection) -> int:
    """Get or create the target property (2919 Wahoo Way)."""
    from .config import TARGET_PROPERTY
    return upsert_property(
        conn,
        address=TARGET_PROPERTY["address"],
        sqft=TARGET_PROPERTY["squareFeet"],
        beds=TARGET_PROPERTY["bedrooms"],
        baths=TARGET_PROPERTY["bathrooms"],
        year_built=TARGET_PROPERTY["yearBuilt"],
        lot_size=TARGET_PROPERTY["lotSizeSqFt"],
    )


def get_strict_filtered_sales(conn: sqlite3.Connection,
                              zip_code: str = "22306",
                              sqft_low: int = 1500, sqft_high: int = 2200,
                              price_low: int = 600000, price_high: int = 900000,
                              min_year_built: int = 2020,
                              months: int = 24,
                              exclude_address: str | None = None) -> list[dict]:
    """Get sales with ultra-strict filtering for high-quality comps.

    Filters: single ZIP, sqft range, price range, year built, recency.
    """
    params: list = [f"%{zip_code}%", sqft_low, sqft_high,
                    price_low, price_high, min_year_built, f"-{months} months"]

    exclude_clause = ""
    if exclude_address:
        exclude_clause = "AND p.address NOT LIKE ?"
        params.append(f"%{exclude_address}%")

    query = f"""
        SELECT p.address, p.sqft, p.beds, p.baths, p.year_built, p.lot_size,
               s.sale_price, s.sale_date, s.source
        FROM sales s
        JOIN properties p ON s.property_id = p.id
        WHERE p.address LIKE ?
          AND p.sqft BETWEEN ? AND ?
          AND s.sale_price BETWEEN ? AND ?
          AND p.year_built >= ?
          AND s.sale_date >= date('now', ?)
          AND s.source NOT LIKE '%listing%'
          {exclude_clause}
        ORDER BY s.sale_date DESC
    """
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_sale_count(conn: sqlite3.Connection, zip_code: str = "22306") -> int:
    nearby_zips = ("22306", "22307", "22309", "22303")
    zip_clauses = " OR ".join("p.address LIKE ?" for _ in nearby_zips)
    zip_clauses += " OR p.address LIKE '%Alexandria, VA%'"
    params = [f"%{z}%" for z in nearby_zips]
    cur = conn.execute(
        f"SELECT COUNT(*) as cnt FROM sales s JOIN properties p ON s.property_id = p.id WHERE ({zip_clauses})",
        params
    )
    return cur.fetchone()["cnt"]
