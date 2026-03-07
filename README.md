# Home Value Predictor

ML-based home valuation with LCARS UI. Standalone FastAPI app for Mission Control integration.

## Features

- Gaussian kernel weighted regression with 6 features
- Uncertainty quantification with t-distribution confidence intervals
- Multi-source data scraping (Zillow, Redfin, Realtor.com)
- LTV/PMI financial analysis
- LCARS-themed frontend
- Disk cache with 6hr TTL and daily auto-refresh

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8551 --reload
```

## API

- `GET /health` - Health check
- `GET /api/valuation` - Full ML prediction payload
- `POST /api/refresh` - Force refresh data
- `GET /` - LCARS frontend
