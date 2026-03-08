"""Training pipeline - Hybrid AVM (Comp Engine + CatBoost + Market Trends).

Replaces the previous XGBoost-only approach with a 3-component hybrid:
  1. Comp Engine: Weighted comparable sales similarity search
  2. CatBoost: Gradient boosting on log(price) with comp-derived features
  3. Market Trends: ZIP-level trend calibration

Blend: 40% comp + 40% catboost + 20% trend adjustment.
"""

import logging
from datetime import datetime

from .config import TARGET_PROPERTY
from .db import get_connection, get_recent_sales, get_sale_count

logger = logging.getLogger("home_value")


# ---------------------------------------------------------------------------
# Public API: train_and_valuate (uses Hybrid AVM)
# ---------------------------------------------------------------------------

def train_and_valuate() -> dict:
    """Run Hybrid AVM pipeline: Comp Engine + CatBoost + Market Trends.

    Replaces the old XGBoost-only approach with a 3-component blend.
    """
    from .hybrid_avm import hybrid_valuation

    logger.info("Running Hybrid AVM pipeline (Comp + CatBoost + Trends)")
    result = hybrid_valuation(months=24)

    return {
        "estimate": result.get("estimate"),
        "confidence": result.get("confidence", 0),
        "confidence_low": result.get("confidence_low", 0),
        "confidence_high": result.get("confidence_high", 0),
        "training_samples": result.get("training_samples", 0),
        "total_db_sales": result.get("total_db_sales", 0),
        "weights_retrained": True,
        "trainingStats": _extract_training_stats(result),
    }


def _extract_training_stats(hybrid_result: dict) -> dict:
    """Extract training stats from hybrid AVM result for backward compatibility."""
    ml = hybrid_result.get("mlDetails", {})
    components = ml.get("components", {})
    cat_metrics = components.get("catboost", {}).get("metrics", {})

    kfold = cat_metrics.get("kFoldCV", {"nFolds": 0, "folds": [], "mean": {}, "std": {}})
    resid = cat_metrics.get("residual_analysis", {"bias": None, "residuals": [], "outliers": [], "outlierCount": 0})

    # Overfitting detection: compare CV mean RMSE vs test RMSE
    overfitting_warning = None
    if kfold.get("mean", {}).get("rmse") and cat_metrics.get("rmse"):
        cv_rmse = kfold["mean"]["rmse"]
        test_rmse = cat_metrics["rmse"]
        if cv_rmse > 0 and test_rmse / cv_rmse < 0.5:
            overfitting_warning = f"Test RMSE (${test_rmse:,}) is much lower than CV mean RMSE (${cv_rmse:,}), possible overfitting"

    return {
        "trainingDataCount": cat_metrics.get("total_samples", 0),
        "filteredSampleCount": cat_metrics.get("total_samples", 0),
        "testAccuracy": {
            "rmse": cat_metrics.get("rmse"),
            "mae": cat_metrics.get("mae"),
        },
        "r2Score": cat_metrics.get("r2_score"),
        "mape": cat_metrics.get("median_ape"),
        "medianAE": cat_metrics.get("mae"),
        "avgPriceError": cat_metrics.get("mae"),
        "dataFreshness": {"oldest": None, "newest": None},
        "dataSourceBreakdown": {},
        "lastTrainedAt": datetime.now().isoformat(),
        "cvFolds": kfold.get("nFolds", 0),
        "avgSimilarityScore": components.get("comp", {}).get("avg_similarity"),
        "featureImportance": ml.get("featureImportance", {}),
        "kFoldCV": kfold,
        "trainTestSplit": {
            "trainSize": cat_metrics.get("train_size", 0),
            "testSize": cat_metrics.get("val_size", 0),
        },
        "residualAnalysis": resid,
        "overfittingWarning": overfitting_warning,
        "hybridComponents": components,
        "componentMetrics": ml.get("componentMetrics", {}),
    }


# ---------------------------------------------------------------------------
# Backward-compatible API: xgb_valuation (now delegates to hybrid)
# ---------------------------------------------------------------------------

def xgb_valuation(sales: list[dict]) -> dict:
    """Hybrid AVM valuation using Comp Engine + CatBoost + Market Trends.

    Maintains the same return signature for backward compatibility.
    Called from valuation.py's ml_valuation().
    """
    from .hybrid_avm import hybrid_valuation

    result = hybrid_valuation(months=24)

    if not result.get("estimate"):
        return {"estimate": None, "confidence": 0, "mlDetails": None}

    return {
        "estimate": result["estimate"],
        "confidence": result.get("confidence", 0),
        "mlDetails": result.get("mlDetails"),
    }


# ---------------------------------------------------------------------------
# Backward-compatible API: compute_training_stats
# ---------------------------------------------------------------------------

def compute_training_stats(sales: list[dict], comps: list[dict]) -> dict:
    """Compute training statistics using the Hybrid AVM pipeline.

    This is called by valuation.py._build_training_stats() for the
    command-center dashboard. Delegates to the hybrid system.
    """
    now = datetime.now()

    # Data source breakdown
    source_counts: dict[str, int] = {}
    for s in sales:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    # Data freshness
    sale_dates = []
    for s in sales:
        try:
            sale_dates.append(datetime.fromisoformat(s.get("sale_date", "")))
        except (ValueError, TypeError):
            pass

    oldest_date = min(sale_dates).strftime("%Y-%m-%d") if sale_dates else None
    newest_date = max(sale_dates).strftime("%Y-%m-%d") if sale_dates else None

    # Run hybrid valuation for metrics
    try:
        from .hybrid_avm import hybrid_valuation
        result = hybrid_valuation(months=24)
        ml = result.get("mlDetails", {})
        components = ml.get("components", {})
        cat_metrics = components.get("catboost", {}).get("metrics", {})
        feat_imp = ml.get("featureImportance", {})

        kfold = cat_metrics.get("kFoldCV", {"nFolds": 0, "folds": [], "mean": {}, "std": {}})
        resid = cat_metrics.get("residual_analysis", {"bias": None, "residuals": [], "outliers": [], "outlierCount": 0})

        overfitting_warning = None
        if kfold.get("mean", {}).get("rmse") and cat_metrics.get("rmse"):
            cv_rmse = kfold["mean"]["rmse"]
            test_rmse = cat_metrics["rmse"]
            if cv_rmse > 0 and test_rmse / cv_rmse < 0.5:
                overfitting_warning = f"Test RMSE (${test_rmse:,}) is much lower than CV mean RMSE (${cv_rmse:,}), possible overfitting"

        return {
            "trainingDataCount": len(sales),
            "filteredSampleCount": cat_metrics.get("total_samples", len(sales)),
            "testAccuracy": {
                "rmse": cat_metrics.get("rmse"),
                "mae": cat_metrics.get("mae"),
            },
            "r2Score": cat_metrics.get("r2_score"),
            "mape": cat_metrics.get("median_ape"),
            "medianAE": cat_metrics.get("mae"),
            "avgPriceError": cat_metrics.get("mae"),
            "dataFreshness": {"oldest": oldest_date, "newest": newest_date},
            "dataSourceBreakdown": source_counts,
            "lastTrainedAt": now.isoformat(),
            "cvFolds": kfold.get("nFolds", 0),
            "avgSimilarityScore": components.get("comp", {}).get("avg_similarity"),
            "featureImportance": feat_imp,
            "kFoldCV": kfold,
            "trainTestSplit": {
                "trainSize": cat_metrics.get("train_size", 0),
                "testSize": cat_metrics.get("val_size", 0),
            },
            "residualAnalysis": resid,
            "overfittingWarning": overfitting_warning,
            "hybridComponents": components,
            "componentMetrics": ml.get("componentMetrics", {}),
        }
    except Exception as e:
        logger.error("Hybrid AVM failed in compute_training_stats: %s", e)
        return {
            "trainingDataCount": len(sales),
            "filteredSampleCount": 0,
            "testAccuracy": {"rmse": None, "mae": None},
            "r2Score": None, "mape": None, "medianAE": None,
            "avgPriceError": None,
            "dataFreshness": {"oldest": oldest_date, "newest": newest_date},
            "dataSourceBreakdown": source_counts,
            "lastTrainedAt": now.isoformat(),
            "cvFolds": 0, "avgSimilarityScore": None,
            "featureImportance": {},
            "kFoldCV": {"nFolds": 0, "folds": [], "mean": {}, "std": {}},
            "trainTestSplit": {},
            "residualAnalysis": {"residuals": [], "bias": None, "outliers": []},
            "overfittingWarning": None,
        }


def _db_sale_to_comp(sale: dict) -> dict:
    """Convert a DB sales row to a comp dict."""
    return {
        "address": sale["address"],
        "price": sale["sale_price"],
        "closedDate": sale["sale_date"],
        "squareFeet": sale.get("sqft"),
        "yearBuilt": sale.get("year_built"),
        "bedrooms": sale.get("beds"),
        "bathrooms": sale.get("baths"),
        "lotSizeSqFt": sale.get("lot_size"),
        "source": sale.get("source", "db"),
    }


async def run_weekly_update() -> dict:
    """Full weekly pipeline: fetch data, retrain, store valuation."""
    from .data_fetcher import run_full_fetch

    fetch_result = await run_full_fetch()
    valuation_result = train_and_valuate()

    new_sales = (
        fetch_result.get("redfin_sold", {}).get("new_sales", 0) +
        fetch_result.get("seed_sales", 0)
    )

    summary = {
        "fetch": fetch_result,
        "valuation": valuation_result,
        "message": f"Updated with {new_sales} new sales, {valuation_result['training_samples']} training samples (Hybrid AVM)",
    }

    logger.info(summary["message"])
    return summary
