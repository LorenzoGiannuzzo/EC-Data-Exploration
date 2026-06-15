"""Engine and session factory.

The same `engine` is shared by the API (via dependency injection),
the CLI, and direct business-layer calls. No per-request engine creation.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from data_explorer.config import settings

engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,                 # silently drop dead connections
    pool_recycle=3600,                  # rotate every hour
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,             # keep objects usable after commit
    future=True,
)


def get_session() -> Iterator[Session]:
    """FastAPI dependency. Yields a session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
