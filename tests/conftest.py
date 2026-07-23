from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from shakepi.archive import ArchiveService
from shakepi.config import Settings
from shakepi.db import create_database_engine, initialize_database

UTC = timezone.utc


@pytest.fixture
def app_context(tmp_path: Path):
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    sessions = initialize_database(create_database_engine(settings))
    return settings, sessions, ArchiveService(settings, sessions)


def write_mseed(path: Path, channel: str = "EHZ", data: np.ndarray | None = None) -> None:
    samples = data if data is not None else np.arange(1_000, dtype=np.int32)
    trace = Trace(
        data=samples,
        header={
            "network": "AM",
            "station": "RF4E8",
            "location": "00",
            "channel": channel,
            "starttime": UTCDateTime(datetime(2026, 2, 2, tzinfo=UTC)),
            "sampling_rate": 100.0,
        },
    )
    Stream([trace]).write(str(path), format="MSEED")
