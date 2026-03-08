"""Property condition and upgrade scoring system.

Scores properties based on finishes, upgrades, and special features
to capture quality differences that basic features (sqft, beds, baths) miss.

Score of 100 = standard builder-grade finishes.
Score > 100 = premium upgrades.
Score < 100 = deferred maintenance or dated finishes.
"""

import logging

logger = logging.getLogger("home_value")

# --- Upgrade categories and point values ---

PREMIUM_FINISHES = {
    "hardwood_floors": 5,
    "quartz_countertops": 4,
    "stainless_appliances": 3,
    "granite_countertops": 3,
    "tile_floors": 2,
    "upgraded_cabinets": 3,
    "stone_backsplash": 2,
}

ARCHITECTURAL_UPGRADES = {
    "crown_molding": 3,
    "upgraded_stairs": 3,
    "wall_treatments": 2,
    "wainscoting": 2,
    "tray_ceilings": 2,
    "custom_built_ins": 3,
    "upgraded_lighting": 2,
}

SPECIAL_FEATURES = {
    "wet_bar": 5,
    "roof_deck": 10,
    "finished_basement": 8,
    "deck_patio": 4,
    "fireplace": 3,
    "smart_home": 3,
    "ev_charger": 2,
    "wine_cellar": 5,
    "outdoor_kitchen": 6,
}

CONDITION_ADJUSTMENTS = {
    "new_construction": 5,       # Built within 2 years
    "recently_renovated": 3,     # Major reno within 5 years
    "well_maintained": 0,        # Standard upkeep
    "dated_finishes": -5,        # Original 10+ year old finishes
    "deferred_maintenance": -10, # Visible wear/issues
}

# Condition grade thresholds
CONDITION_GRADES = {
    "luxury": 125,    # 125+
    "upgraded": 110,  # 110-124
    "standard": 100,  # 90-109
    "below_average": 0,  # <90
}


def compute_upgrade_score(upgrades: dict) -> dict:
    """Compute upgrade score from a dict of upgrade features.

    Args:
        upgrades: dict with keys:
            - premium_finishes: list of finish names
            - architectural: list of architectural upgrade names
            - special_features: list of special feature names
            - condition: str condition key (or None for default)

    Returns:
        dict with: total_score, breakdown, condition_grade,
                   has_premium_finishes, has_special_features
    """
    base_score = 100
    breakdown = {"base": base_score}

    # Premium finishes
    finishes = upgrades.get("premium_finishes", [])
    finish_pts = sum(PREMIUM_FINISHES.get(f, 0) for f in finishes)
    breakdown["premium_finishes"] = finish_pts

    # Architectural upgrades
    arch = upgrades.get("architectural", [])
    arch_pts = sum(ARCHITECTURAL_UPGRADES.get(a, 0) for a in arch)
    breakdown["architectural"] = arch_pts

    # Special features
    special = upgrades.get("special_features", [])
    special_pts = sum(SPECIAL_FEATURES.get(s, 0) for s in special)
    breakdown["special_features"] = special_pts

    # Condition adjustment
    condition = upgrades.get("condition", "well_maintained")
    condition_pts = CONDITION_ADJUSTMENTS.get(condition, 0)
    breakdown["condition"] = condition_pts

    total = base_score + finish_pts + arch_pts + special_pts + condition_pts

    # Determine grade
    if total >= CONDITION_GRADES["luxury"]:
        grade = "luxury"
    elif total >= CONDITION_GRADES["upgraded"]:
        grade = "upgraded"
    elif total >= CONDITION_GRADES["standard"]:
        grade = "standard"
    else:
        grade = "below_average"

    return {
        "total_score": total,
        "breakdown": breakdown,
        "condition_grade": grade,
        "has_premium_finishes": finish_pts > 0,
        "has_special_features": special_pts > 0,
        "premium_pct": round((total - 100) / 100, 4),  # e.g. 0.35 = 35% premium
    }


# --- Known property upgrade profiles ---

KNOWN_UPGRADES = {
    "2919 wahoo way": {
        "premium_finishes": [
            "hardwood_floors",
            "quartz_countertops",
            "stainless_appliances",
        ],
        "architectural": [
            "crown_molding",
            "upgraded_stairs",
            "wall_treatments",
        ],
        "special_features": [
            "wet_bar",
            "roof_deck",
        ],
        "condition": "new_construction",
    },
}


def get_upgrade_score(address: str) -> dict:
    """Get upgrade score for an address. Returns standard (100) if unknown."""
    addr_lower = (address or "").lower()

    for key, upgrades in KNOWN_UPGRADES.items():
        if key in addr_lower:
            result = compute_upgrade_score(upgrades)
            logger.debug("Upgrade score for %s: %d (%s)",
                         address, result["total_score"], result["condition_grade"])
            return result

    # Default: standard condition, no known upgrades
    return {
        "total_score": 100,
        "breakdown": {"base": 100},
        "condition_grade": "standard",
        "has_premium_finishes": False,
        "has_special_features": False,
        "premium_pct": 0.0,
    }


def upgrade_price_adjustment(address: str) -> float:
    """Return a price multiplier based on upgrade score.

    A score of 135 means the property is worth ~17.5% more than standard
    (half the raw premium - conservative to avoid overshooting).
    """
    score_data = get_upgrade_score(address)
    premium_pct = score_data["premium_pct"]
    # Apply 50% of the raw premium as price adjustment (conservative)
    return 1.0 + (premium_pct * 0.50)
