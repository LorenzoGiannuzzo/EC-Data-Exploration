"""Configuration loaded from environment variables (or a .env file)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — all variables are overridable via env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+psycopg://data_explorer:change_me_in_production"
        "@localhost:5432/data_explorer"
    )

    # ── API ──────────────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level:    str = "INFO"

    # ── Data ingestion ───────────────────────────────────────────────────
    # Path inside the container where the host folder is mounted.
    raw_data_dir: str = "/data/raw"


settings = Settings()
