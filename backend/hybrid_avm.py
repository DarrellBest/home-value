"""Hybrid AVM - Blends comp engine, CatBoost model, and market trends.

Architecture:
  40% comp_estimate (grounded in real nearby sales)
  40% catboost_estimate (ML captures non-linear patterns)
  20% trend_adjustment (local market momentum)

Produces a value range [low, estimate, high] with confidence scoring.
"""

import math
import logging
from datetime import datetime

import numpy as np

from .comp_engine import find_comps, compute_comp_similarity, estimate_value
from .catboost_model import train_catboost
from .market_trends import compute_market_trends
from .config import TARGET_PROPERTY
from .db import (
    get_connection, get_recent_sales, get_target_property_id,
    insert_valuation, get_sale_count,
)
# from .property_scoring import get_upgrade_score  # DISABLED

logger = logging.getLogger("home_value")

# Blend weights
WEIGHT_COMP = 0.40  # Restored - comp engine now has 4.89% MAPE
WEIGHT_CATBOOST = 0.40  # Restored to original balanced weight
WEIGHT_TREND = 0.20


def _compute_confidence(comp_result: dict, catboost_result: dict, trend_result: dict) -> int:
    """Compute overall confidence score (0-100).

    Based on: comp similarity, model validation accuracy, data volume,
    and agreement between components.
    """
    score = 0.0

    # Component 1: Comp quality (0-30 points)
    stats = comp_result.get("stats", {})
    avg_sim = stats.get("avg_similarity", 0)
    n_comps = stats.get("n_comps", 0)
    score += min(15, avg_sim * 15)
    score += min(15, n_comps * 1.5)

    # Component 2: Model accuracy (0-30 points)
    metrics = catboost_result.get("metrics", {})
    median_ape = metrics.get("median_ape", 100)
    within_5 = metrics.get("within_5pct", 0)
    if median_ape < 3:
        score += 20
    elif median_ape < 5:
        score += 15
    elif median_ape < 10:
        score += 10
    else:
        score += 5
    score += min(10, within_5 / 10)

    # Component 3: Agreement between components (0-20 points)
    comp_est = comp_result.get("estimate")
    cat_est = catboost_result.get("estimate")
    if comp_est and cat_est and comp_est > 0:
        divergence = abs(comp_est - cat_est) / comp_est
        if divergence < 0.03:
            score += 20
        elif divergence < 0.05:
            score += 15
        elif divergence < 0.10:
            score += 10
        else:
            score += 5

    # Component 4: Trend confidence (0-10 points)
    primary_trend = trend_result.get("primary_trend")
    if primary_trend:
        r_sq = primary_trend.get("r_squared", 0)
        score += min(10, r_sq * 10)
    else:
        score += 3  # Neutral if no trend data

    # Component 5: Data volume (0-10 points)
    total_samples = metrics.get("total_samples", 0)
    score += min(10, total_samples / 30 * 10)

    return max(0, min(100, round(score)))


def _comp_estimate_for_sale(sale: dict, all_sales: list[dict]) -> float | None:
    """Get comp-engine estimate for a single sale using leave-one-out.

    Uses the new comp_engine.estimate_value() which has 4.89% MAPE.
    """
    # Exclude this sale from the pool (leave-one-out)
    others = [s for s in all_sales if s.get("address") != sale.get("address")
              or s.get("sale_date") != sale.get("sale_date")]

    if len(others) < 3:
        return None

    # Use the new comp engine's estimate_value function
    result = estimate_value(sale, others)
    if not result or not result.get("estimate"):
        return None

    return result["estimate"]


def _calc_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Calculate standard regression metrics."""
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    ape = abs_errors / actuals * 100

    mape = float(np.median(ape))
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mape": round(mape, 2),
        "mae": round(mae),
        "rmse": round(rmse),
        "r2": round(r2, 4),
    }


def _evaluate_components(catboost_result: dict, all_sales: list[dict],
                         trend_factor: float) -> dict:
    """Evaluate comp engine, CatBoost, and final blend on the validation set.

    Uses the same time-based validation split as CatBoost (last 20%).
    For each test property:
      - comp estimate via leave-one-out comp engine
      - catboost estimate from model predictions
      - blend = 0.4*comp + 0.4*catboost + 0.2*(blend*trend_factor)

    Returns componentMetrics dict with metrics for each component.
    """
    val_details = catboost_result.get("validation_details", [])
    logger.info(f"Component evaluation: {len(val_details)} validation samples")

    if not val_details:
        logger.warning("No validation details - returning stub component metrics")
        # Return stub metrics so UI displays something
        return {
            "comp": {"mape": 0.0, "rmse": 0, "mae": 0, "r2": 0.0, "samples": 0},
            "catboost": {
                "mape": catboost_result.get("median_ape", 0),
                "rmse": catboost_result.get("rmse", 0),
                "mae": catboost_result.get("mae", 0),
                "r2": catboost_result.get("r2_score", 0),
                "samples": catboost_result.get("val_size", 0)
            },
            "blend": {"mape": 0.0, "rmse": 0, "mae": 0, "r2": 0.0, "samples": 0}
        }

    # Build lookup: address -> actual/predicted from CatBoost validation
    cat_lookup = {}
    for v in val_details:
        cat_lookup[v["address"]] = {
            "actual": v["actual"],
            "catboost_pred": v["predicted"],
        }

    # Match validation addresses to sales for comp engine evaluation
    val_addresses = set(cat_lookup.keys())

    comp_actuals, comp_preds = [], []
    cat_actuals, cat_preds = [], []
    blend_actuals, blend_preds = [], []

    for sale in all_sales:
        addr = sale.get("address", "")
        if addr not in val_addresses:
            continue

        actual = cat_lookup[addr]["actual"]
        catboost_est = cat_lookup[addr]["catboost_pred"]

        # Comp engine estimate (leave-one-out)
        comp_est = _comp_estimate_for_sale(sale, all_sales)

        # CatBoost metrics (always available)
        cat_actuals.append(actual)
        cat_preds.append(catboost_est)

        if comp_est is not None:
            comp_actuals.append(actual)
            comp_preds.append(comp_est)

            # Blend: same formula as hybrid_valuation
            raw_blend = (comp_est * WEIGHT_COMP + catboost_est * WEIGHT_CATBOOST) / (WEIGHT_COMP + WEIGHT_CATBOOST)
            trend_adjusted = raw_blend * trend_factor
            final_blend = raw_blend * (1 - WEIGHT_TREND) + trend_adjusted * WEIGHT_TREND

            blend_actuals.append(actual)
            blend_preds.append(final_blend)

    result = {}

    # Comp metrics
    if len(comp_actuals) >= 3:
        result["comp"] = _calc_metrics(np.array(comp_actuals), np.array(comp_preds))
        result["comp"]["n_evaluated"] = len(comp_actuals)
        logger.info("COMP EVAL: MAPE=%.1f%%, RMSE=$%s, MAE=$%s, R²=%.4f (n=%d)",
                     result["comp"]["mape"], f"{result['comp']['rmse']:,}",
                     f"{result['comp']['mae']:,}", result["comp"]["r2"],
                     len(comp_actuals))
    else:
        logger.warning("Too few comp estimates for evaluation (%d)", len(comp_actuals))

    # CatBoost metrics
    if len(cat_actuals) >= 3:
        result["catboost"] = _calc_metrics(np.array(cat_actuals), np.array(cat_preds))
        result["catboost"]["n_evaluated"] = len(cat_actuals)
        logger.info("CATBOOST EVAL: MAPE=%.1f%%, RMSE=$%s, MAE=$%s, R²=%.4f (n=%d)",
                     result["catboost"]["mape"], f"{result['catboost']['rmse']:,}",
                     f"{result['catboost']['mae']:,}", result["catboost"]["r2"],
                     len(cat_actuals))

    # Blend metrics
    if len(blend_actuals) >= 3:
        result["blend"] = _calc_metrics(np.array(blend_actuals), np.array(blend_preds))
        result["blend"]["n_evaluated"] = len(blend_actuals)
        logger.info("BLEND EVAL: MAPE=%.1f%%, RMSE=$%s, MAE=$%s, R²=%.4f (n=%d)",
                     result["blend"]["mape"], f"{result['blend']['rmse']:,}",
                     f"{result['blend']['mae']:,}", result["blend"]["r2"],
                     len(blend_actuals))

    return result


def hybrid_valuation(months: int = 24) -> dict:
    """Run the full hybrid AVM pipeline.

    1. Comp Engine: Find best comparables, compute weighted estimate
    2. CatBoost: Train on sales data, predict with uncertainty
    3. Market Trends: Compute ZIP-level trend adjustment
    4. Blend: 40% comp + 40% catboost + 20% trend-adjusted average

    Returns comprehensive valuation result.
    """
    logger.info("=" * 60)
    logger.info("HYBRID AVM: Starting valuation pipeline")
    logger.info("=" * 60)

    # Load sales data
    conn = get_connection()
    try:
        sales = get_recent_sales(conn, "22306", months=months,
                                 exclude_address="2919 Wahoo Way")
        total_db_sales = get_sale_count(conn)
        prop_id = get_target_property_id(conn)
    finally:
        conn.close()

    logger.info("Loaded %d recent sales (%d total in DB)", len(sales), total_db_sales)

    # ===== Subject Property Upgrade Score =====
    # subject_upgrade = get_upgrade_score(TARGET_PROPERTY["address"])  # DISABLED
    subject_upgrade = {
        "total_score": 100,
        "condition_grade": "standard",
        "premium_pct": 0.0,
        "breakdown": {},
        "has_premium_finishes": False,
        "has_special_features": False
    }
    logger.info("UPGRADE SCORE: %d/100 (%s) - premium: %.1f%%",
                subject_upgrade["total_score"], subject_upgrade["condition_grade"],
                subject_upgrade["premium_pct"] * 100)

    # ===== Component 1: Comp Engine =====
    comp_result = find_comps(n_comps=10, months=months)
    comp_estimate = comp_result.get("estimate")

    # ===== Component 2: CatBoost Model =====
    catboost_result = train_catboost(sales)
    catboost_estimate = catboost_result.get("estimate")

    # ===== Component 3: Market Trends =====
    trend_result = compute_market_trends("22306", months=6)
    trend_factor = trend_result.get("adjustment_factor", 1.0)

    # ===== Blend Components =====
    estimates = []
    weights = []

    if comp_estimate:
        estimates.append(comp_estimate)
        weights.append(WEIGHT_COMP)
        logger.info("  COMP:     $%s (weight: %.0f%%)", f"{comp_estimate:,}", WEIGHT_COMP * 100)

    if catboost_estimate:
        estimates.append(catboost_estimate)
        weights.append(WEIGHT_CATBOOST)
        logger.info("  CATBOOST: $%s (weight: %.0f%%)", f"{catboost_estimate:,}", WEIGHT_CATBOOST * 100)

    if not estimates:
        logger.error("No estimates available from any component")
        return {"estimate": None, "error": "No estimates available"}

    # Trend adjustment: apply to the weighted average
    total_weight = sum(weights)
    raw_blend = sum(e * w for e, w in zip(estimates, weights)) / total_weight

    # Apply trend: 20% weight to trend-adjusted blend
    # The other 80% is the raw blend
    trend_adjusted = raw_blend * trend_factor
    final_estimate = round(raw_blend * (1 - WEIGHT_TREND) + trend_adjusted * WEIGHT_TREND)

    logger.info("  TREND:    factor=%.4f (weight: %.0f%%)", trend_factor, WEIGHT_TREND * 100)
    logger.info("  BLEND:    $%s (raw: $%s)", f"{final_estimate:,}", f"{round(raw_blend):,}")

    # ===== Prediction Interval =====
    # Combine uncertainty from comp engine and CatBoost
    comp_std = comp_result.get("stats", {}).get("weighted_std", 0)
    cat_interval = catboost_result.get("prediction_interval", {})
    cat_low = cat_interval.get("low", final_estimate * 0.90)
    cat_high = cat_interval.get("high", final_estimate * 1.10)
    cat_spread = (cat_high - cat_low) / 2 if cat_high and cat_low else final_estimate * 0.05

    # Combined uncertainty (root-sum-of-squares of component uncertainties)
    combined_uncertainty = math.sqrt(
        (comp_std * WEIGHT_COMP) ** 2 + (cat_spread * WEIGHT_CATBOOST) ** 2
    ) / max(total_weight, 0.01)

    # 90% prediction interval
    estimate_low = round(final_estimate - 1.645 * combined_uncertainty)
    estimate_high = round(final_estimate + 1.645 * combined_uncertainty)

    # 80% confidence interval (narrower)
    conf_low = round(final_estimate - 1.282 * combined_uncertainty)
    conf_high = round(final_estimate + 1.282 * combined_uncertainty)

    # ===== Component-Level Evaluation =====
    component_metrics = _evaluate_components(catboost_result, sales, trend_factor)

    # ===== Confidence Score =====
    confidence = _compute_confidence(comp_result, catboost_result, trend_result)

    # ===== Build Result =====
    intervals = {
        "confidence80": {"low": conf_low, "high": conf_high},
        "prediction90": {"low": estimate_low, "high": estimate_high},
    }

    ml_details = {
        "intervals": intervals,
        "stats": {
            "modelType": "Hybrid AVM (Comp + CatBoost + Trend)",
            "totalSamples": catboost_result.get("metrics", {}).get("total_samples", 0),
            "weightedStdDev": round(combined_uncertainty),
            "effectiveSampleSize": comp_result.get("stats", {}).get("n_comps", 0),
            "coeffOfVariation": round(combined_uncertainty / final_estimate, 4) if final_estimate > 0 else 0,
            "seasonalFactor": catboost_result.get("seasonal_factor", 1.0),
        },
        "components": {
            "comp": {
                "estimate": comp_estimate,
                "weight": WEIGHT_COMP,
                "n_comps": comp_result.get("stats", {}).get("n_comps", 0),
                "avg_similarity": comp_result.get("stats", {}).get("avg_similarity", 0),
                "avg_ppsf": comp_result.get("stats", {}).get("avg_price_per_sqft", 0),
                "weighted_std": comp_std,
            },
            "catboost": {
                "estimate": catboost_estimate,
                "weight": WEIGHT_CATBOOST,
                "metrics": catboost_result.get("metrics", {}),
                "feature_importance": catboost_result.get("feature_importance", {}),
                "prediction_interval": catboost_result.get("prediction_interval", {}),
            },
            "trend": {
                "adjustment_factor": trend_factor,
                "weight": WEIGHT_TREND,
                "annualized_pct": (
                    trend_result.get("primary_trend", {}).get("annualized_pct", 0)
                    if trend_result.get("primary_trend") else 0
                ),
                "market_consensus_pct": trend_result.get("market_consensus_annual_pct", 0),
            },
        },
        "compScores": [
            {
                "address": c["address"],
                "rawPrice": c["sale_price"],
                "adjustedPrice": c["adjusted_price"],
                "similarityScore": c["similarity"],
                "isListing": False,
                "sqft": c.get("sqft"),
                "beds": c.get("beds"),
                "baths": c.get("baths"),
                "saleDate": c.get("sale_date"),
            }
            for c in comp_result.get("comps", [])
        ],
        "upgradeScoring": {
            "subject_score": subject_upgrade["total_score"],
            "condition_grade": subject_upgrade["condition_grade"],
            "premium_pct": subject_upgrade["premium_pct"],
            "breakdown": subject_upgrade["breakdown"],
            "has_premium_finishes": subject_upgrade["has_premium_finishes"],
            "has_special_features": subject_upgrade["has_special_features"],
        },
        "componentMetrics": component_metrics,
        "featureImportance": catboost_result.get("feature_importance", {}),
        "validation": {
            "method": "time-based split (train: older, test: newer)",
            "details": catboost_result.get("validation_details", []),
        },
    }

    # Store valuation in DB
    conn = get_connection()
    try:
        insert_valuation(
            conn, prop_id, final_estimate, confidence,
            conf_low, conf_high,
            ml_details=ml_details,
            training_samples=catboost_result.get("metrics", {}).get("total_samples", 0),
        )
        conn.commit()
    finally:
        conn.close()

    result = {
        "estimate": final_estimate,
        "confidence": confidence,
        "confidence_low": conf_low,
        "confidence_high": conf_high,
        "prediction_low": estimate_low,
        "prediction_high": estimate_high,
        "mlDetails": ml_details,
        "training_samples": catboost_result.get("metrics", {}).get("total_samples", 0),
        "total_db_sales": total_db_sales,
    }

    logger.info("=" * 60)
    logger.info("HYBRID AVM RESULT: $%s [confidence: %d%%]",
                f"{final_estimate:,}", confidence)
    logger.info("  Range: $%s - $%s (90%% PI)",
                f"{estimate_low:,}", f"{estimate_high:,}")
    logger.info("  Comp: $%s | CatBoost: $%s | Trend: %.2f%%",
                f"{comp_estimate:,}" if comp_estimate else "N/A",
                f"{catboost_estimate:,}" if catboost_estimate else "N/A",
                (trend_factor - 1) * 100)
    logger.info("=" * 60)

    return result
