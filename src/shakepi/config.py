"""Application configuration and filesystem layout."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from ``SHAKEPI_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="SHAKEPI_", env_file=".env", extra="ignore")

    data_root: Path = Field(default=Path("data"))
    database_url: str | None = None
    max_upload_bytes: int = 100 * 1024 * 1024
    cache_limit_bytes: int = 50 * 1024 * 1024 * 1024
    sqlite_busy_timeout_ms: int = 5_000
    # Production defaults to strict. Tests and local migration work can explicitly opt out.
    allow_unsafe_sqlite: bool = False
    job_workers: int = 1

    @cached_property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @cached_property
    def cache_root(self) -> Path:
        return self.data_root / "cache"

    @cached_property
    def quarantine_root(self) -> Path:
        return self.data_root / "quarantine"

    @cached_property
    def temp_root(self) -> Path:
        return self.data_root / "tmp"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.data_root / 'shakepi.sqlite3'}"

    def ensure_directories(self) -> None:
        for directory in (self.data_root, self.raw_root, self.cache_root, self.quarantine_root, self.temp_root):
            directory.mkdir(parents=True, exist_ok=True)
