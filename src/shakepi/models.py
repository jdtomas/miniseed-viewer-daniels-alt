"""SQLAlchemy models for durable analysis metadata."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Actor(Base):
    __tablename__ = "actors"

    id: Mapped[int] = mapped_column(primary_key=True)
    display_name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class StationPeriod(Base):
    __tablename__ = "station_periods"
    __table_args__ = (
        UniqueConstraint("network", "station", "location", "utc_day", name="uq_station_period"),
        Index("ix_station_period_listing", "station", "utc_day"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    network: Mapped[str] = mapped_column(String(8))
    station: Mapped[str] = mapped_column(String(32))
    location: Mapped[str] = mapped_column(String(8), default="")
    utc_day: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    raw_files: Mapped[list["RawFile"]] = relationship(back_populates="period")


class RawFile(Base):
    __tablename__ = "raw_files"
    __table_args__ = (
        UniqueConstraint("sha256", name="uq_raw_file_sha256"),
        UniqueConstraint("period_id", "channel", name="uq_period_channel"),
        Index("ix_raw_files_period", "period_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    period_id: Mapped[int] = mapped_column(ForeignKey("station_periods.id"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255))
    archive_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    bytes: Mapped[int] = mapped_column(Integer)
    network: Mapped[str] = mapped_column(String(8))
    station: Mapped[str] = mapped_column(String(32))
    location: Mapped[str] = mapped_column(String(8), default="")
    channel: Mapped[str] = mapped_column(String(16))
    quality: Mapped[str] = mapped_column(String(4), default="D")
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sampling_rate: Mapped[float] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    period: Mapped[StationPeriod] = relationship(back_populates="raw_files")


class StationChannelRole(Base):
    __tablename__ = "station_channel_roles"
    __table_args__ = (UniqueConstraint("network", "station", "location", "channel", name="uq_channel_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    network: Mapped[str] = mapped_column(String(8))
    station: Mapped[str] = mapped_column(String(32))
    location: Mapped[str] = mapped_column(String(8), default="")
    channel: Mapped[str] = mapped_column(String(16))
    role: Mapped[str] = mapped_column(String(24))


class PreprocessingRecipeRecord(Base):
    __tablename__ = "preprocessing_recipes"

    id: Mapped[int] = mapped_column(primary_key=True)
    recipe_hash: Mapped[str] = mapped_column(String(64), unique=True)
    canonical_json: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[str] = mapped_column(String(20))
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("actors.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (Index("ix_analysis_runs_period_created", "period_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    period_id: Mapped[int] = mapped_column(ForeignKey("station_periods.id"), nullable=False)
    recipe_id: Mapped[int | None] = mapped_column(ForeignKey("preprocessing_recipes.id"), nullable=True)
    detector_name: Mapped[str] = mapped_column(String(64))
    detector_version: Mapped[str] = mapped_column(String(100))
    model_weight: Mapped[str | None] = mapped_column(String(100), nullable=True)
    channel_selection: Mapped[list[str]] = mapped_column(JSON)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON)
    search_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    search_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(24), default="queued")
    warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("actors.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    candidates: Mapped[list["DetectionCandidate"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    recipe: Mapped["PreprocessingRecipeRecord | None"] = relationship()


class DetectionCandidate(Base):
    __tablename__ = "detection_candidates"
    __table_args__ = (Index("ix_candidates_run_timestamp", "run_id", "timestamp"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(32))
    phase: Mapped[str | None] = mapped_column(String(8), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    score: Mapped[float] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    run: Mapped[AnalysisRun] = relationship(back_populates="candidates")
    assessment: Mapped["ManualAssessment | None"] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", uselist=False
    )


class ManualAssessment(Base):
    __tablename__ = "manual_assessments"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("detection_candidates.id"), unique=True)
    verdict: Mapped[str] = mapped_column(String(16))
    notes: Mapped[str] = mapped_column(Text, default="")
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("actors.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    candidate: Mapped[DetectionCandidate] = relationship(back_populates="assessment")


class CacheArtifact(Base):
    __tablename__ = "cache_artifacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(Text)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
