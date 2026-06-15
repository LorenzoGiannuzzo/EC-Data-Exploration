"""FastAPI entry point.

Phase 3: full analytical API.
    /health, /info           — meta
    /metadata/*              — ATECO catalogue, POD-set preview
    /clustering/*            — clustering pipeline
    /gse/*                   — GSE profile comparison
    /arera/*                 — ARERA profile comparison

CORS is wide-open in development so the Streamlit frontend (and `curl`,
Postman, …) can call the API freely. Tighten for production.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from data_explorer import __version__
from data_explorer.api.routers import arera, clustering, gse, metadata, overview
from data_explorer.config import settings
from data_explorer.db.queries import fetch_table_counts
from data_explorer.db.session import get_session


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(
    title="Data Explorer API",
    version=__version__,
    description="Backend service for electrical-load profile analysis.",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(metadata.router)
app.include_router(overview.router)
app.include_router(clustering.router)
app.include_router(gse.router)
app.include_router(arera.router)


# ── Meta endpoints (kept here for visibility at the API root) ────────────────
@app.get("/health", tags=["meta"])
def health(db: Session = Depends(get_session)):
    """Liveness + DB connectivity probe."""
    db.execute(text("SELECT 1"))
    return {"status": "ok", "version": __version__}


@app.get("/info", tags=["meta"])
def info(db: Session = Depends(get_session)):
    """Quick row-count overview — what's in the database right now."""
    return {
        "version":   __version__,
        "log_level": settings.log_level,
        "tables":    fetch_table_counts(db),
    }
