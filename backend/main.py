"""Home Value Predictor - FastAPI application."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, CACHE_TTL_SECONDS
from .valuation import HomeValueService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("home_value")

service = HomeValueService()
_scheduler_task: asyncio.Task | None = None


async def _daily_refresh_loop():
    while True:
        await asyncio.sleep(CACHE_TTL_SECONDS)
        try:
            await service.force_refresh()
            logger.info("Scheduled refresh completed")
        except Exception as e:
            logger.warning("Scheduled refresh failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler_task
    # Startup: trigger initial fetch and start scheduler
    asyncio.create_task(service.get_data())
    _scheduler_task = asyncio.create_task(_daily_refresh_loop())
    logger.info("Home Value Predictor started")
    yield
    # Shutdown
    if _scheduler_task:
        _scheduler_task.cancel()


app = FastAPI(title="Home Value Predictor", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/valuation")
async def get_valuation(refresh: int = 0):
    data = await service.get_data(force_refresh=bool(refresh))
    return JSONResponse(content=data)


@app.post("/api/refresh")
async def force_refresh():
    data = await service.force_refresh()
    return JSONResponse(content=data)


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
