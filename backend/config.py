from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
CACHE_FILE = DATA_DIR / "home-value-cache.json"
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

TARGET_PROPERTY = {
    "address": "2919 Wahoo Way, Alexandria, VA 22306",
    "squareFeet": 1866,
    "yearBuilt": 2022,
    "lotSizeSqFt": 3200,
    "bedrooms": 4,
    "bathrooms": 3.5,
    "currentBalance": 590166,
}

FEATURE_WEIGHTS = {
    "recency": 0.25,
    "locationProximity": 0.25,
    "propertyAge": 0.10,
    "bedBathMatch": 0.15,
    "sqftSimilarity": 0.15,
    "lotSizeSimilarity": 0.10,
}

LISTING_DISCOUNT = 0.03
MARKET_TREND_ANNUAL_PCT = 3.5
KERNEL_BANDWIDTH = 0.3

SEASONAL_FACTORS = [
    0.97, 0.97, 0.99, 1.01, 1.03, 1.04,
    1.04, 1.03, 1.02, 1.00, 0.98, 0.96,
]

T_CRITICAL = {
    1:   {0.90: 6.314, 0.80: 3.078},
    2:   {0.90: 2.920, 0.80: 1.886},
    3:   {0.90: 2.353, 0.80: 1.638},
    4:   {0.90: 2.132, 0.80: 1.533},
    5:   {0.90: 2.015, 0.80: 1.476},
    6:   {0.90: 1.943, 0.80: 1.440},
    7:   {0.90: 1.895, 0.80: 1.415},
    8:   {0.90: 1.860, 0.80: 1.397},
    9:   {0.90: 1.833, 0.80: 1.383},
    10:  {0.90: 1.812, 0.80: 1.372},
    15:  {0.90: 1.753, 0.80: 1.341},
    20:  {0.90: 1.725, 0.80: 1.325},
    30:  {0.90: 1.697, 0.80: 1.310},
    60:  {0.90: 1.671, 0.80: 1.296},
    120: {0.90: 1.658, 0.80: 1.289},
}
T_CRITICAL_INF = {0.90: 1.645, 0.80: 1.282}

SOURCES = [
    {
        "label": "Zillow (zestimate proxy)",
        "url": "https://www.zillow.com/homedetails/2919-Wahoo-Way-Alexandria-VA-22306/",
    },
    {
        "label": "Redfin Estimate",
        "url": "https://www.redfin.com/VA/Alexandria/2919-Wahoo-Way-22306/",
    },
    {
        "label": "Realtor.com",
        "url": "https://www.realtor.com/realestateandhomes-detail/2919-Wahoo-Way_Alexandria_VA_22306/",
    },
]

FALLBACK_COMPARABLES = {
    "recentSales": [
        {"address": "2917 Wahoo Way, Alexandria, VA 22306", "price": 700000, "closedDate": "2025-11-15", "squareFeet": 1900, "yearBuilt": 2022, "bedrooms": 4, "bathrooms": 3.5, "lotSizeSqFt": 3100, "source": "fallback"},
        {"address": "2921 Wahoo Way, Alexandria, VA 22306", "price": 685000, "closedDate": "2025-09-20", "squareFeet": 1850, "yearBuilt": 2022, "bedrooms": 4, "bathrooms": 3.5, "lotSizeSqFt": 3250, "source": "fallback"},
        {"address": "2915 Wahoo Way, Alexandria, VA 22306", "price": 710000, "closedDate": "2025-08-05", "squareFeet": 1920, "yearBuilt": 2022, "bedrooms": 4, "bathrooms": 3.5, "lotSizeSqFt": 3300, "source": "fallback"},
        {"address": "6410 Richmond Hwy, Alexandria, VA 22306", "price": 650000, "closedDate": "2025-10-10", "squareFeet": 1750, "yearBuilt": 2020, "bedrooms": 3, "bathrooms": 3, "lotSizeSqFt": 4000, "source": "fallback"},
        {"address": "2800 Popkins Ln, Alexandria, VA 22306", "price": 725000, "closedDate": "2025-07-22", "squareFeet": 2100, "yearBuilt": 2021, "bedrooms": 4, "bathrooms": 4, "lotSizeSqFt": 3500, "source": "fallback"},
    ],
    "currentListings": [
        {"address": "2923 Wahoo Way, Alexandria, VA 22306", "price": 720000, "listedDate": "2026-02-01", "squareFeet": 1880, "yearBuilt": 2022, "bedrooms": 4, "bathrooms": 3.5, "lotSizeSqFt": 3150, "source": "fallback"},
        {"address": "6500 Tower Dr, Alexandria, VA 22306", "price": 695000, "listedDate": "2026-01-15", "squareFeet": 1800, "yearBuilt": 2021, "bedrooms": 4, "bathrooms": 3, "lotSizeSqFt": 3600, "source": "fallback"},
    ],
}
