# Data Explorer

Modular-monolithic service for electrical-load profile analytics.
Three layers — **core / db / api / cli** — wrapped in two containers
(`backend` + `frontend`) plus an external `postgres` container.

## Repository layout

```
data-exploration/
├── docker-compose.yml          # postgres + backend + frontend
├── .env.example                # copy to .env and edit
│
├── postgres/
│   └── init.sql                # schema bootstrap (runs once on first start)
│
├── backend/
│   ├── Dockerfile              # multi-stage, ~150 MB final image
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── alembic/
│   └── src/data_explorer/
│       ├── core/               # pure-python algorithms (no Streamlit, no FastAPI)
│       ├── db/                 # SQLAlchemy models, session, ingestion
│       ├── api/                # FastAPI routers
│       ├── cli/                # Typer commands
│       └── config.py           # Pydantic Settings
│
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                  # Streamlit GUI (calls backend over HTTP)
```

## Phase 1 — what works now

- PostgreSQL/PostGIS database container with full schema (PODs, measurements
  partitioned by year, ATECO lookup, GSE/ARERA reference profiles).
- Backend container with `/health` and `/info` endpoints, plus the CLI.
- Ingestion command that loads your existing `data/` folder into Postgres.
- Frontend container that confirms it can reach the backend.

Phases 2-4 add the analytical endpoints and rebuild the dashboard tabs on
top of them.

## Quickstart

```bash
# 1. Configure
cp .env.example .env
# edit .env if needed — at minimum change POSTGRES_PASSWORD

# 2. Build & start the stack
docker compose up -d --build

# 3. Verify
curl http://localhost:8000/health
# → {"status":"ok","version":"0.1.0"}

# 4. Ingest your existing data (CSV/Excel under ./data)
docker compose exec backend data-explorer ingest --data-dir /data/raw

# 5. Optional: ingest the official ATECO lookup
docker compose exec backend data-explorer ingest-ateco \
    /data/raw/Note-esplicative-ATECO-2025-italiano-inglese.xlsx

# 6. Check row counts
docker compose exec backend data-explorer db-check

# 7. Open the GUI
#    http://localhost:8501
```

The host folder pointed at by `HOST_DATA_DIR` in `.env` (default `./data`)
is mounted **read-only** into the backend container at `/data/raw`.

## Manual SQL access

```bash
# from the host machine (psql installed)
psql -h localhost -U data_explorer -d data_explorer

# or from within the postgres container
docker compose exec postgres psql -U data_explorer -d data_explorer
```

## Migrations (after Phase 1)

```bash
# generate a new migration based on model changes
docker compose exec backend alembic revision --autogenerate -m "add foo"

# apply
docker compose exec backend alembic upgrade head
```

## Tear down

```bash
docker compose down              # keep DB volume
docker compose down -v           # also wipe the database
```
