from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from shakepi.processing import ZarrCache
from shakepi.schemas import PreprocessingRecipe


def test_zarr_cache_stores_chunked_regenerable_stream(app_context) -> None:
    settings, sessions, _archive = app_context
    trace = Trace(
        np.arange(12_000, dtype=np.float32),
        header={
            "network": "AM",
            "station": "RF4E8",
            "location": "00",
            "channel": "EHZ",
            "starttime": UTCDateTime(datetime(2026, 2, 2, tzinfo=timezone.utc)),
            "sampling_rate": 100.0,
        },
    )
    cache = ZarrCache(settings, sessions)
    recipe = PreprocessingRecipe(bandpass_low_hz=1, bandpass_high_hz=20)
    key = cache.key_for(["source-checksum"], recipe, ["EHZ"])
    path = cache.store_stream(key, Stream([trace]), {"period_id": 1})
    assert (path / ".complete").exists()
    assert path.name == f"{key}.zarr"
    assert cache.store_stream(key, Stream([trace]), {"period_id": 1}) == path
    loaded = cache.load_stream(
        key,
        (
            datetime(2026, 2, 2, 0, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 2, 2, 0, 0, 20, tzinfo=timezone.utc),
        ),
    )
    assert loaded is not None
    assert loaded[0].stats.npts == 1_001
