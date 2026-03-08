"""Market trend calibration - ZIP-level price trend calculation.

Computes local market momentum from recent sales and market statistics
to adjust valuations for micro-market dynamics.
"""

import logging
import math
from datetime import datetime

from .db import get_connection

logger = logging.getLogger("home_value")


def _compute_zip_trend_from_stats(conn, zip_code: str, months: int = 6) -> dict | None:
    """Compute trend from market_stats table (Zillow ZHVI / research data)."""
    rows = conn.execute("""
        SELECT median_price, month, source
        FROM market_stats
        WHERE zip_code = ?
          AND month >= date('now', ?)
        ORDER BY month ASC
    """, (zip_code, f"-{months} months")).fetchall()

    if len(rows) < 2:
        return None

    prices = [(r["month"], r["median_price"]) for r in rows if r["median_price"]]
    if len(prices) < 2:
        return None

    # Simple linear trend: fit price = m*month_index + b
    n = len(prices)
    x = list(range(n))
    y = [p[1] for p in prices]

    x_mean = sum(x) / n
    y_mean = sum(y) / n
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    ss_xx = sum((xi - x_mean) ** 2 for xi in x)

    if ss_xx == 0:
        return None

    slope = ss_xy / ss_xx  # $/month trend
    monthly_pct = (slope / y_mean) * 100 if y_mean > 0 else 0
    annualized_pct = monthly_pct * 12

    # Trend strength: R² of the linear fit
    y_pred = [y_mean + slope * (xi - x_mean) for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "source": "market_stats",
        "zip_code": zip_code,
        "months": n,
        "start_price": prices[0][1],
        "end_price": prices[-1][1],
        "monthly_change_pct": round(monthly_pct, 3),
        "annualized_pct": round(annualized_pct, 2),
        "slope_per_month": round(slope),
        "r_squared": round(max(0, r_squared), 4),
    }


def _compute_zip_trend_from_sales(conn, zip_code: str, months: int = 6) -> dict | None:
    """Compute trend from actual sales data (fallback if market_stats sparse)."""
    rows = conn.execute("""
        SELECT s.sale_price, s.sale_date, p.sqft
        FROM sales s
        JOIN properties p ON s.property_id = p.id
        WHERE p.address LIKE ?
          AND s.sale_date >= date('now', ?)
          AND s.source NOT LIKE '%listing%'
          AND s.sale_price > 0
          AND p.sqft > 0
        ORDER BY s.sale_date ASC
    """, (f"%{zip_code}%", f"-{months} months")).fetchall()

    if len(rows) < 5:
        return None

    # Use price per sqft to normalize across property sizes
    data = []
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["sale_date"])
            month_idx = (dt.year - 2020) * 12 + dt.month  # arbitrary epoch
            ppsf = r["sale_price"] / r["sqft"]
            data.append((month_idx, ppsf))
        except (ValueError, TypeError):
            continue

    if len(data) < 5:
        return None

    n = len(data)
    x = [d[0] for d in data]
    y = [d[1] for d in data]

    x_mean = sum(x) / n
    y_mean = sum(y) / n
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    ss_xx = sum((xi - x_mean) ** 2 for xi in x)

    if ss_xx == 0:
        return None

    slope = ss_xy / ss_xx  # $/sqft per month
    monthly_pct = (slope / y_mean) * 100 if y_mean > 0 else 0
    annualized_pct = monthly_pct * 12

    y_pred = [y_mean + slope * (xi - x_mean) for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "source": "sales_ppsf",
        "zip_code": zip_code,
        "n_sales": n,
        "avg_ppsf": round(y_mean, 2),
        "monthly_change_pct": round(monthly_pct, 3),
        "annualized_pct": round(annualized_pct, 2),
        "slope_ppsf_per_month": round(slope, 2),
        "r_squared": round(max(0, r_squared), 4),
    }


def compute_market_trends(zip_code: str = "22306", months: int = 6) -> dict:
    """Compute ZIP-level market trends from multiple data sources.

    Returns trend adjustment factor and detailed trend info.
    """
    conn = get_connection()
    try:
        # Try market_stats first (more reliable monthly data)
        stats_trend = _compute_zip_trend_from_stats(conn, zip_code, months)

        # Also compute from sales for comparison/fallback
        sales_trend = _compute_zip_trend_from_sales(conn, zip_code, months)

        # Also check adjacent ZIPs for broader market context
        adjacent_trends = {}
        for adj_zip in ("22307", "22309", "22303"):
            adj = _compute_zip_trend_from_stats(conn, adj_zip, months)
            if adj:
                adjacent_trends[adj_zip] = adj
            else:
                adj = _compute_zip_trend_from_sales(conn, adj_zip, months)
                if adj:
                    adjacent_trends[adj_zip] = adj
    finally:
        conn.close()

    # Determine primary trend and adjustment factor
    primary_trend = stats_trend or sales_trend

    if primary_trend:
        # Monthly change -> project forward to "now"
        monthly_pct = primary_trend["monthly_change_pct"]
        r_sq = primary_trend.get("r_squared", 0)

        # Dampen the adjustment by R² (don't trust noisy trends)
        # Also cap the adjustment to ±5% to avoid wild swings
        dampened_monthly = monthly_pct * min(r_sq, 1.0)
        # Apply 3 months of trend (project forward from midpoint of data)
        months_forward = 3
        raw_adjustment = dampened_monthly * months_forward / 100
        adjustment_factor = 1.0 + max(-0.05, min(0.05, raw_adjustment))
    else:
        adjustment_factor = 1.0
        monthly_pct = 0
        dampened_monthly = 0

    # Compute broader market context
    all_annualized = []
    if primary_trend:
        all_annualized.append(primary_trend["annualized_pct"])
    for adj in adjacent_trends.values():
        all_annualized.append(adj["annualized_pct"])

    market_consensus = (
        sum(all_annualized) / len(all_annualized) if all_annualized else 0
    )

    result = {
        "adjustment_factor": round(adjustment_factor, 4),
        "primary_trend": primary_trend,
        "sales_trend": sales_trend,
        "adjacent_trends": adjacent_trends,
        "market_consensus_annual_pct": round(market_consensus, 2),
        "zip_code": zip_code,
    }

    logger.info("MARKET TRENDS [%s]: adjustment=%.4f, annualized=%.1f%%, consensus=%.1f%%",
                zip_code, adjustment_factor,
                primary_trend["annualized_pct"] if primary_trend else 0,
                market_consensus)

    return result


def get_trend_feature(sale_date_str: str, zip_code: str = "22306") -> float:
    """Get the market trend value at a specific point in time.

    Used as a feature in the CatBoost model. Returns the trailing 6-month
    annualized price change at the time of sale.
    """
    conn = get_connection()
    try:
        # Get market_stats around the sale date
        rows = conn.execute("""
            SELECT median_price, month
            FROM market_stats
            WHERE zip_code = ?
              AND month <= ?
              AND month >= date(?, '-6 months')
            ORDER BY month ASC
        """, (zip_code, sale_date_str, sale_date_str)).fetchall()
    finally:
        conn.close()

    if len(rows) < 2:
        return 0.0

    prices = [r["median_price"] for r in rows if r["median_price"]]
    if len(prices) < 2:
        return 0.0

    # Simple annualized return over the period
    start, end = prices[0], prices[-1]
    if start <= 0:
        return 0.0

    total_return = (end - start) / start
    n_months = len(prices)
    annualized = total_return * (12 / max(n_months, 1))
    return round(annualized * 100, 2)  # as percentage
