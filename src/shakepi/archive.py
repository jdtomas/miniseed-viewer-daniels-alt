"""Immutable MiniSEED ingestion and station-period discovery."""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from obspy import read
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import RawFile, StationChannelRole, StationPeriod
from .schemas import IngestResult

UTC = timezone.utc

SDS_FILENAME = re.compile(
    r"^(?P<network>[A-Za-z0-9]{1,8})\."
    r"(?P<station>[A-Za-z0-9_-]{1,32})\."
    r"(?P<location>[A-Za-z0-9_-]{0,8})\."
    r"(?P<channel>[A-Za-z0-9]{2,16})\."
    r"(?P<quality>[A-Za-z])\."
    r"(?P<year>\d{4})\."
    r"(?P<doy>\d{3})$"
)

DEFAULT_CHANNEL_ROLES = {"EHZ": "geophone", "ENZ": "mems_z", "ENN": "mems_n", "ENE": "mems_e"}


@dataclass(frozen=True)
class ParsedFilename:
    network: str
    station: str
    location: str
    channel: str
    quality: str
    year: int
    doy: int

    @property
    def utc_day(self) -> date:
        return datetime.strptime(f"{self.year}-{self.doy:03d}", "%Y-%j").date()

    @classmethod
    def parse(cls, filename: str) -> "ParsedFilename":
        if Path(filename).name != filename:
            raise ValueError("Upload filename must not contain a directory")
        match = SDS_FILENAME.fullmatch(filename)
        if not match:
            raise ValueError("Filename must match NET.STA.LOC.CHA.D.YEAR.DOY")
        values = match.groupdict()
        parsed = cls(
            network=values["network"],
            station=values["station"],
            location=values["location"],
            channel=values["channel"],
            quality=values["quality"],
            year=int(values["year"]),
            doy=int(values["doy"]),
        )
        # ``strptime`` validates leap days and day 366.
        _ = parsed.utc_day
        return parsed


class ArchiveService:
    def __init__(self, settings: Settings, session_factory: sessionmaker[Session]):
        self.settings = settings
        self.session_factory = session_factory

    def _archive_destination(self, parsed: ParsedFilename, filename: str) -> Path:
        return (
            self.settings.raw_root
            / str(parsed.year)
            / parsed.network
            / parsed.station
            / f"{parsed.channel}.D"
            / filename
        )

    def _validate_headers(self, temporary_path: Path, parsed: ParsedFilename) -> tuple[datetime, datetime, float, int]:
        try:
            stream = read(str(temporary_path), headonly=True)
        except Exception as error:  # ObsPy exposes several parser exceptions.
            raise ValueError(f"File is not readable MiniSEED: {error}") from error
        if not stream:
            raise ValueError("MiniSEED file does not contain any traces")

        first = stream[0]
        for trace in stream:
            stats = trace.stats
            identity = (stats.network, stats.station, stats.location or "", stats.channel)
            expected = (parsed.network, parsed.station, parsed.location, parsed.channel)
            if identity != expected:
                raise ValueError(
                    "Filename and MiniSEED headers disagree: "
                    f"expected {'.'.join(expected)}, found {'.'.join(identity)}"
                )
            if stats.starttime.year != parsed.year or stats.starttime.julday != parsed.doy:
                raise ValueError("MiniSEED trace start time does not match filename UTC year/day")

        start = min(trace.stats.starttime for trace in stream).datetime.replace(tzinfo=UTC)
        end = max(trace.stats.endtime for trace in stream).datetime.replace(tzinfo=UTC)
        return start, end, float(first.stats.sampling_rate), sum(int(trace.stats.npts) for trace in stream)

    def ingest_path(self, temporary_path: Path, filename: str, sha256: str, byte_count: int) -> IngestResult:
        parsed = ParsedFilename.parse(filename)
        start, end, sampling_rate, sample_count = self._validate_headers(temporary_path, parsed)
        destination = self._archive_destination(parsed, filename)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with self.session_factory.begin() as session:
            duplicate = session.scalar(select(RawFile).where(RawFile.sha256 == sha256))
            if duplicate:
                temporary_path.unlink(missing_ok=True)
                return IngestResult(
                    status="duplicate",
                    filename=filename,
                    sha256=sha256,
                    period_id=duplicate.period_id,
                    message="An identical file is already in the archive.",
                )

            period = session.scalar(
                select(StationPeriod).where(
                    StationPeriod.network == parsed.network,
                    StationPeriod.station == parsed.station,
                    StationPeriod.location == parsed.location,
                    StationPeriod.utc_day == parsed.utc_day,
                )
            )
            if period is None:
                period = StationPeriod(
                    network=parsed.network,
                    station=parsed.station,
                    location=parsed.location,
                    utc_day=parsed.utc_day,
                )
                session.add(period)
                session.flush()

            existing_channel = session.scalar(
                select(RawFile).where(RawFile.period_id == period.id, RawFile.channel == parsed.channel)
            )
            if existing_channel or destination.exists():
                quarantine_name = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}-{filename}"
                quarantine_path = self.settings.quarantine_root / quarantine_name
                temporary_path.replace(quarantine_path)
                return IngestResult(
                    status="conflict",
                    filename=filename,
                    sha256=sha256,
                    period_id=period.id,
                    message=f"A different file already exists for this station/day/channel; saved to quarantine: {quarantine_name}",
                )

            temporary_path.replace(destination)
            raw_file = RawFile(
                period_id=period.id,
                original_filename=filename,
                archive_path=str(destination.resolve()),
                sha256=sha256,
                bytes=byte_count,
                network=parsed.network,
                station=parsed.station,
                location=parsed.location,
                channel=parsed.channel,
                quality=parsed.quality,
                start_time=start,
                end_time=end,
                sampling_rate=sampling_rate,
                sample_count=sample_count,
            )
            session.add(raw_file)
            if session.scalar(
                select(StationChannelRole).where(
                    StationChannelRole.network == parsed.network,
                    StationChannelRole.station == parsed.station,
                    StationChannelRole.location == parsed.location,
                    StationChannelRole.channel == parsed.channel,
                )
            ) is None and parsed.channel in DEFAULT_CHANNEL_ROLES:
                session.add(
                    StationChannelRole(
                        network=parsed.network,
                        station=parsed.station,
                        location=parsed.location,
                        channel=parsed.channel,
                        role=DEFAULT_CHANNEL_ROLES[parsed.channel],
                    )
                )
            return IngestResult(
                status="stored",
                filename=filename,
                sha256=sha256,
                period_id=period.id,
                message="MiniSEED file stored in the immutable SDS archive.",
            )

    async def ingest_upload(self, upload: object) -> IngestResult:
        """Stream a FastAPI ``UploadFile`` to local temporary storage."""
        filename = getattr(upload, "filename", None)
        if not filename:
            raise ValueError("Upload is missing a filename")
        ParsedFilename.parse(filename)
        temporary_path = self.settings.temp_root / f"{uuid4().hex}.upload"
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with temporary_path.open("wb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    byte_count += len(chunk)
                    if byte_count > self.settings.max_upload_bytes:
                        raise ValueError("Upload exceeds configured size limit")
                    digest.update(chunk)
                    destination.write(chunk)
            return self.ingest_path(temporary_path, filename, digest.hexdigest(), byte_count)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

    def ingest_bytes(self, contents: bytes, filename: str) -> IngestResult:
        """Ingest bytes supplied by the Dash drag-and-drop uploader."""
        ParsedFilename.parse(filename)
        if len(contents) > self.settings.max_upload_bytes:
            raise ValueError("Upload exceeds configured size limit")

        temporary_path = self.settings.temp_root / f"{uuid4().hex}.upload"
        try:
            temporary_path.write_bytes(contents)
            return self.ingest_path(
                temporary_path,
                filename,
                hashlib.sha256(contents).hexdigest(),
                len(contents),
            )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def list_periods(self) -> list[dict[str, object]]:
        with self.session_factory() as session:
            periods = session.scalars(
                select(StationPeriod).order_by(StationPeriod.station.asc(), StationPeriod.utc_day.desc())
            ).all()
            response: list[dict[str, object]] = []
            for period in periods:
                files = session.scalars(select(RawFile).where(RawFile.period_id == period.id)).all()
                response.append(
                    {
                        "id": period.id,
                        "label": f"{period.network}.{period.station}.{period.location or '--'} — {period.utc_day.isoformat()}",
                        "channels": self.order_channels(period, [file.channel for file in files]),
                    }
                )
            return response

    def order_channels(self, period: StationPeriod, channels: list[str]) -> list[str]:
        with self.session_factory() as session:
            roles = {
                entry.channel: entry.role
                for entry in session.scalars(
                    select(StationChannelRole).where(
                        StationChannelRole.network == period.network,
                        StationChannelRole.station == period.station,
                        StationChannelRole.location == period.location,
                    )
                )
            }
        priority = {"geophone": 0, "mems_z": 1, "mems_n": 2, "mems_e": 3}
        return sorted(channels, key=lambda channel: (priority.get(roles.get(channel, ""), 4), channel))


def copy_stream_to_path(source: Path, destination: Path) -> str:
    """Utility used by CLI imports and tests; returns the streamed SHA-256."""
    digest = hashlib.sha256()
    with source.open("rb") as input_file, destination.open("wb") as output_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
            output_file.write(chunk)
    return digest.hexdigest()
