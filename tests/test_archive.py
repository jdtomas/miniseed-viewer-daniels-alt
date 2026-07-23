from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from shakepi.archive import ParsedFilename

from conftest import write_mseed


def test_parse_sds_filename_and_doy() -> None:
    parsed = ParsedFilename.parse("AM.RF4E8.00.ENE.D.2026.033")
    assert parsed.station == "RF4E8"
    assert parsed.channel == "ENE"
    assert parsed.utc_day.isoformat() == "2026-02-02"


@pytest.mark.parametrize("filename", ["../AM.RF4E8.00.EHZ.D.2026.033", "not-a-miniseed", "AM.RF4E8.00.EHZ.D.2026.367"])
def test_rejects_invalid_filenames(filename: str) -> None:
    with pytest.raises(ValueError):
        ParsedFilename.parse(filename)


def test_ingest_duplicate_and_conflict(app_context, tmp_path: Path) -> None:
    settings, sessions, archive = app_context
    first = tmp_path / "first"
    write_mseed(first)
    checksum = hashlib.sha256(first.read_bytes()).hexdigest()
    result = archive.ingest_path(first, "AM.RF4E8.00.EHZ.D.2026.033", checksum, first.stat().st_size)
    assert result.status == "stored"
    expected = settings.raw_root / "2026" / "AM" / "RF4E8" / "EHZ.D" / "AM.RF4E8.00.EHZ.D.2026.033"
    assert expected.exists()

    duplicate = tmp_path / "duplicate"
    duplicate.write_bytes(expected.read_bytes())
    duplicate_result = archive.ingest_path(
        duplicate,
        "AM.RF4E8.00.EHZ.D.2026.033",
        checksum,
        duplicate.stat().st_size,
    )
    assert duplicate_result.status == "duplicate"

    conflict = tmp_path / "conflict"
    write_mseed(conflict, data=np.arange(1_000, dtype=np.int32) * 3)
    conflict_checksum = hashlib.sha256(conflict.read_bytes()).hexdigest()
    conflict_result = archive.ingest_path(
        conflict,
        "AM.RF4E8.00.EHZ.D.2026.033",
        conflict_checksum,
        conflict.stat().st_size,
    )
    assert conflict_result.status == "conflict"
    assert list(settings.quarantine_root.iterdir())


def test_ingest_bytes_accepts_sds_filename_without_file_extension(app_context, tmp_path: Path) -> None:
    _, _, archive = app_context
    source = tmp_path / "source"
    write_mseed(source)

    result = archive.ingest_bytes(source.read_bytes(), "AM.RF4E8.00.EHZ.D.2026.033")

    assert result.status == "stored"
