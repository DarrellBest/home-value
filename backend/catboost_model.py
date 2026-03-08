"""CatBoost-based valuation model.

Predicts log(closing_price) using property, location, time/market,
and comp-derived features. Uses time-based train/test splits for
honest validation metrics.
"""

import math
import logging
import re
import numpy as np
from datetime import datetime

from catboost import CatBoostRegressor, Pool

from .config import TARGET_PROPERTY, SEASONAL_FACTORS
from .comp_engine import get_comp_features_for_model
from .market_trends import get_trend_feature
from .property_scoring import get_upgrade_score

logger = logging.getLogger("home_value")

CURRENT_YEAR = datetime.now().year

# Feature groups
FEATURE_NAMES = [
    # Property features
    "sqft", "beds", "baths", "year_built", "lot_size", "property_type_idx",
    # Location features
    "zip_idx", "location_score",
    # Time/market features
    "sale_year", "sale_month", "days_on_market_proxy", "market_trend_pct",
    # Comp-derived features
    "avg_comp_price", "comp_price_per_sqft", "price_vs_comp_avg",
    # Property condition/upgrade features
    "upgrade_score", "has_premium_finishes", "has_special_features",
]

# Condition grade encoding
CONDITION_GRADE_IDX = {"below_average": 0, "standard": 1, "upgraded": 2, "luxury": 3}

# ZIP code encoding
ZIP_ENCODING = {"22306": 0, "22307": 1, "22309": 2, "22303": 3}


def _extract_zip(address: str) -> str:
    m = re.search(r"\b(\d{5})\b", address or "")
    return m.group(1) if m else ""


def _location_score(address: str) -> float:
    """Numeric location score (0-1)."""
    addr = (address or "").lower()
    if "wahoo" in addr:
        return 1.0
    zip_code = _extract_zip(address)
    if zip_code == "22306":
        return 0.85
    if zip_code in ("22307", "22309", "22303"):
        return 0.60
    if "alexandria" in addr:
        return 0.40
    return 0.20


def _days_on_market_proxy(sale_date_str: str) -> float:
    """Proxy for days on market using sale date recency.

    Properties selling in hot months tend to sell faster.
    This is a proxy since we don't have actual DOM data.
    """
    try:
        dt = datetime.fromisoformat(sale_date_str)
        month = dt.month
        # Hot months (spring/summer) have lower DOM
        seasonal_dom = {
            1: 45, 2: 40, 3: 30, 4: 25, 5: 20, 6: 18,
            7: 20, 8: 22, 9: 28, 10: 35, 11: 40, 12: 45,
        }
        return float(seasonal_dom.get(month, 30))
    except (ValueError, TypeError):
        return 30.0


def build_feature_vector(sale: dict, comp_features: dict | None = None) -> list[float]:
    """Build feature vector for a single sale/property."""
    sqft = float(sale.get("sqft") or sale.get("squareFeet") or 0)
    beds = float(sale.get("beds") or sale.get("bedrooms") or 0)
    baths = float(sale.get("baths") or sale.get("bathrooms") or 0)
    year_built = float(sale.get("year_built") or sale.get("yearBuilt") or 0)
    lot_size = float(sale.get("lot_size") or sale.get("lotSizeSqFt") or 0)
    property_type_idx = 0  # All single-family for now

    # Location
    address = sale.get("address", "")
    zip_code = _extract_zip(address)
    zip_idx = ZIP_ENCODING.get(zip_code, len(ZIP_ENCODING))
    loc_score = _location_score(address)

    # Time/market
    sale_date_str = sale.get("sale_date") or sale.get("closedDate") or ""
    try:
        dt = datetime.fromisoformat(sale_date_str)
        sale_year = float(dt.year)
        sale_month = float(dt.month)
    except (ValueError, TypeError):
        sale_year = float(CURRENT_YEAR)
        sale_month = float(datetime.now().month)

    dom_proxy = _days_on_market_proxy(sale_date_str)

    # Market trend at time of sale
    trend_pct = get_trend_feature(sale_date_str, zip_code or "22306")

    # Comp-derived features
    cf = comp_features or {}
    avg_comp = float(cf.get("avg_comp_price") or 0)
    comp_ppsf = float(cf.get("comp_price_per_sqft") or 0)
    price_vs_comp = float(cf.get("price_vs_comp_avg") or 1.0)

    # Property condition/upgrade features
    score_data = get_upgrade_score(address)
    upgrade_score = float(score_data["total_score"])
    has_premium = 1.0 if score_data["has_premium_finishes"] else 0.0
    has_special = 1.0 if score_data["has_special_features"] else 0.0

    return [
        sqft, beds, baths, year_built, lot_size, float(property_type_idx),
        float(zip_idx), loc_score,
        sale_year, sale_month, dom_proxy, trend_pct,
        avg_comp, comp_ppsf, price_vs_comp,
        upgrade_score, has_premium, has_special,
    ]


def _build_dataset(sales: list[dict], comp_features_list: list[dict] | None = None):
    """Build feature matrix and log-price targets from sales data."""
    X, y, addresses = [], [], []

    for i, s in enumerate(sales):
        # Skip listings
        source = s.get("source", "")
        if "listing" in source.lower():
            continue

        price = s.get("sale_price") or s.get("price")
        if not price or price <= 0:
            continue

        sqft = s.get("sqft") or s.get("squareFeet")
        if not sqft or sqft <= 0:
            continue

        cf = comp_features_list[i] if comp_features_list and i < len(comp_features_list) else None
        features = build_feature_vector(s, cf)
        X.append(features)
        y.append(math.log(float(price)))  # Predict log(price)
        addresses.append(s.get("address", "Unknown"))

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), addresses


def _catboost_params():
    """CatBoost hyperparameters tuned for small real estate datasets."""
    return {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "verbose": 0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "od_type": "Iter",
        "od_wait": 50,
    }


def train_catboost(sales: list[dict]) -> dict:
    """Train CatBoost model with time-based validation.

    Uses time-based split: trains on older sales, validates on newer ones.
    Returns model, metrics, and prediction for subject property.
    """
    # Sort by date for time-based split
    dated_sales = []
    for s in sales:
        source = s.get("source", "")
        if "listing" in source.lower():
            continue
        price = s.get("sale_price") or s.get("price")
        sqft = s.get("sqft") or s.get("squareFeet")
        if not price or price <= 0 or not sqft or sqft <= 0:
            continue
        dated_sales.append(s)

    dated_sales.sort(key=lambda x: x.get("sale_date", "") or "")

    if len(dated_sales) < 10:
        logger.warning("Too few sales for CatBoost (%d). Need at least 10.", len(dated_sales))
        return {"estimate": None, "metrics": {}, "model": None}

    # Compute comp features for all sales
    comp_features = get_comp_features_for_model(dated_sales)

    # Build full dataset
    X, y, addresses = _build_dataset(dated_sales, comp_features)

    if len(X) < 10:
        return {"estimate": None, "metrics": {}, "model": None}

    # Time-based split: last 20% for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    addr_val = addresses[split_idx:]

    # Create CatBoost pools (all numeric features, no categoricals)
    train_pool = Pool(X_train, y_train, feature_names=FEATURE_NAMES)
    val_pool = Pool(X_val, y_val, feature_names=FEATURE_NAMES)

    # Train model
    params = _catboost_params()
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=0)

    # Validation metrics (in log space and real space)
    val_preds_log = model.predict(X_val)
    val_preds = np.exp(val_preds_log)
    val_actuals = np.exp(y_val)

    # Absolute percentage errors
    ape = np.abs((val_preds - val_actuals) / val_actuals) * 100
    median_ape = float(np.median(ape))
    mae = float(np.mean(np.abs(val_preds - val_actuals)))
    within_5pct = float(np.mean(ape <= 5.0) * 100)
    within_10pct = float(np.mean(ape <= 10.0) * 100)

    # RMSE on test set (real-space dollars)
    rmse = float(np.sqrt(np.mean((val_preds - val_actuals) ** 2)))

    # R² score on test set
    ss_res = float(np.sum((val_actuals - val_preds) ** 2))
    ss_tot = float(np.sum((val_actuals - np.mean(val_actuals)) ** 2))
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residual analysis
    residuals = val_preds - val_actuals  # signed residuals in dollars
    residual_bias = float(np.mean(residuals))  # positive = model overestimates
    residual_list = [round(float(r)) for r in residuals]
    # Outliers: residuals beyond 2 standard deviations
    res_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0
    outlier_threshold = 2.0 * res_std
    outlier_indices = np.where(np.abs(residuals) > outlier_threshold)[0]
    outliers = []
    for idx in outlier_indices:
        outliers.append({
            "address": addr_val[idx],
            "actual": round(float(val_actuals[idx])),
            "predicted": round(float(val_preds[idx])),
            "residual": round(float(residuals[idx])),
        })

    # K-Fold cross-validation (10-fold on full dataset)
    n_folds = min(10, len(X))  # cap at dataset size
    kfold_results = []
    if n_folds >= 3:
        fold_size = len(X) // n_folds
        indices = np.arange(len(X))
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else len(X)
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            if len(train_idx) < 5 or len(val_idx) < 1:
                continue

            fold_pool = Pool(X[train_idx], y[train_idx], feature_names=FEATURE_NAMES)
            fold_model = CatBoostRegressor(**{**params, "iterations": 300})
            fold_model.fit(fold_pool, verbose=0)

            fold_preds_log = fold_model.predict(X[val_idx])
            fold_preds = np.exp(fold_preds_log)
            fold_actuals = np.exp(y[val_idx])

            fold_ape = np.abs((fold_preds - fold_actuals) / fold_actuals) * 100
            fold_residuals = fold_preds - fold_actuals
            fold_ss_res = float(np.sum((fold_actuals - fold_preds) ** 2))
            fold_ss_tot = float(np.sum((fold_actuals - np.mean(fold_actuals)) ** 2))

            kfold_results.append({
                "fold": fold + 1,
                "rmse": round(float(np.sqrt(np.mean(fold_residuals ** 2)))),
                "mae": round(float(np.mean(np.abs(fold_residuals)))),
                "r2": round(1.0 - (fold_ss_res / fold_ss_tot) if fold_ss_tot > 0 else 0.0, 4),
                "median_ape": round(float(np.median(fold_ape)), 2),
                "samples": len(val_idx),
            })

    kfold_summary = {}
    if kfold_results:
        kfold_summary = {
            "nFolds": len(kfold_results),
            "folds": kfold_results,
            "mean": {
                "rmse": round(float(np.mean([f["rmse"] for f in kfold_results]))),
                "mae": round(float(np.mean([f["mae"] for f in kfold_results]))),
                "r2": round(float(np.mean([f["r2"] for f in kfold_results])), 4),
                "median_ape": round(float(np.mean([f["median_ape"] for f in kfold_results])), 2),
            },
            "std": {
                "rmse": round(float(np.std([f["rmse"] for f in kfold_results]))),
                "mae": round(float(np.std([f["mae"] for f in kfold_results]))),
                "r2": round(float(np.std([f["r2"] for f in kfold_results])), 4),
                "median_ape": round(float(np.std([f["median_ape"] for f in kfold_results])), 2),
            },
        }
    else:
        kfold_summary = {"nFolds": 0, "folds": [], "mean": {}, "std": {}}

    # Per-sample validation details
    val_details = []
    for i in range(len(val_preds)):
        val_details.append({
            "address": addr_val[i],
            "actual": round(float(val_actuals[i])),
            "predicted": round(float(val_preds[i])),
            "ape": round(float(ape[i]), 2),
        })

    # Feature importance
    raw_imp = model.get_feature_importance()
    total_imp = raw_imp.sum()
    feat_importance = {}
    for name, imp in zip(FEATURE_NAMES, raw_imp):
        feat_importance[name] = round(float(imp / total_imp * 100), 2) if total_imp > 0 else 0

    # Retrain on ALL data for final prediction
    full_pool = Pool(X, y, feature_names=FEATURE_NAMES)
    final_model = CatBoostRegressor(**params)
    final_model.fit(full_pool, verbose=0)

    # Predict subject property
    subject_comp_features = _get_subject_comp_features(dated_sales)
    tp = TARGET_PROPERTY
    subject_sale = {
        "address": tp["address"],
        "sqft": tp["squareFeet"],
        "beds": tp["bedrooms"],
        "baths": tp["bathrooms"],
        "year_built": tp["yearBuilt"],
        "lot_size": tp["lotSizeSqFt"],
        "sale_date": datetime.now().strftime("%Y-%m-%d"),
    }
    X_subject = np.array([build_feature_vector(subject_sale, subject_comp_features)],
                         dtype=np.float64)
    pred_log = final_model.predict(X_subject)[0]
    estimate_raw = math.exp(pred_log)

    # Apply upgrade premium adjustment post-prediction
    # CatBoost can't learn upgrade effects since training data lacks variation,
    # so we apply the subject's upgrade premium as a multiplier
    from .property_scoring import upgrade_price_adjustment
    upgrade_mult = 1.0  # DISABLED: was upgrade_price_adjustment(tp["address"])
    estimate = estimate_raw * upgrade_mult
    # Adjust pred_log for consistent interval calculation
    pred_log = math.log(estimate)

    logger.info("CATBOOST: raw=$%s, upgrade_mult=%.3f, adjusted=$%s",
                f"{round(estimate_raw):,}", upgrade_mult, f"{round(estimate):,}")

    # Uncertainty from validation residuals
    val_residuals_log = val_preds_log - y_val
    residual_std_log = float(np.std(val_residuals_log, ddof=1)) if len(val_residuals_log) > 1 else 0.1

    # Virtual ensemble: retrain on bootstrap samples for prediction interval
    n_boots = 5
    boot_preds = []
    for b in range(n_boots):
        rng = np.random.RandomState(42 + b)
        idx = rng.choice(len(X), size=len(X), replace=True)
        boot_pool = Pool(X[idx], y[idx], feature_names=FEATURE_NAMES)
        boot_model = CatBoostRegressor(**{**params, "random_seed": 42 + b})
        boot_model.fit(boot_pool, verbose=0)
        boot_preds.append(float(boot_model.predict(X_subject)[0]))

    pred_std_log = float(np.std(boot_preds, ddof=1)) if len(boot_preds) > 1 else residual_std_log
    combined_std_log = math.sqrt(residual_std_log ** 2 + pred_std_log ** 2)

    # Prediction interval in real space (via log-normal properties)
    estimate_low = math.exp(pred_log - 1.645 * combined_std_log)
    estimate_high = math.exp(pred_log + 1.645 * combined_std_log)

    # Apply seasonal factor
    seasonal = SEASONAL_FACTORS[datetime.now().month - 1]
    catboost_estimate = round(estimate * seasonal)

    metrics = {
        "median_ape": round(median_ape, 2),
        "mae": round(mae),
        "rmse": round(rmse),
        "r2_score": round(r2_score, 4),
        "within_5pct": round(within_5pct, 1),
        "within_10pct": round(within_10pct, 1),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "total_samples": len(X),
        "residual_std_log": round(residual_std_log, 4),
        "pred_std_log": round(pred_std_log, 4),
        "best_iteration": final_model.get_best_iteration() or params["iterations"],
        "residual_analysis": {
            "bias": round(residual_bias),
            "std": round(res_std),
            "residuals": residual_list,
            "outliers": outliers,
            "outlierCount": len(outliers),
        },
        "kFoldCV": kfold_summary,
    }

    result = {
        "estimate": catboost_estimate,
        "estimate_raw": round(estimate),
        "prediction_interval": {
            "low": round(estimate_low * seasonal),
            "high": round(estimate_high * seasonal),
        },
        "metrics": metrics,
        "feature_importance": feat_importance,
        "validation_details": val_details,
        "model": final_model,
        "seasonal_factor": seasonal,
        "boot_predictions_log": [round(math.exp(p), -2) for p in boot_preds],
    }

    logger.info("CATBOOST: $%s estimate (median APE: %.1f%%, MAE: $%s, %.0f%% within ±5%%)",
                f"{catboost_estimate:,}", median_ape, f"{mae:,.0f}", within_5pct)

    return result


def _get_subject_comp_features(sales: list[dict]) -> dict:
    """Get comp-derived features for the subject property."""
    from .comp_engine import find_comps

    comps = find_comps(n_comps=5)
    if comps.get("comps"):
        avg_price = sum(c["adjusted_price"] for c in comps["comps"]) / len(comps["comps"])
        ppsf_vals = [
            c["sale_price"] / c["sqft"]
            for c in comps["comps"]
            if c.get("sqft") and c["sqft"] > 0
        ]
        avg_ppsf = sum(ppsf_vals) / len(ppsf_vals) if ppsf_vals else 0

        return {
            "avg_comp_price": round(avg_price),
            "comp_price_per_sqft": round(avg_ppsf, 2),
            "price_vs_comp_avg": 1.0,  # Subject is the reference
        }

    return {
        "avg_comp_price": 0,
        "comp_price_per_sqft": 0,
        "price_vs_comp_avg": 1.0,
    }
