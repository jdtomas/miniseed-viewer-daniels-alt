"""Validated public service inputs and results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PreprocessingRecipe(BaseModel):
    """A reproducible preprocessing recipe. All durations are in seconds."""

    schema_version: str = "1"
    fill_short_gaps: bool = False
    max_gap_seconds: float = Field(default=1.0, gt=0, le=10)
    taper_fraction: float = Field(default=0.05, ge=0, le=0.5)
    deep_denoiser: bool = False
    deep_denoiser_weight: str = "original"
    bandpass_low_hz: float | None = Field(default=None, gt=0)
    bandpass_high_hz: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_bandpass_pair(self) -> "PreprocessingRecipe":
        if (self.bandpass_low_hz is None) != (self.bandpass_high_hz is None):
            raise ValueError("Bandpass needs both low and high cutoffs")
        if self.bandpass_low_hz and self.bandpass_high_hz and self.bandpass_low_hz >= self.bandpass_high_hz:
            raise ValueError("Bandpass low cutoff must be lower than high cutoff")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class IngestResult(BaseModel):
    status: Literal["stored", "duplicate", "conflict"]
    filename: str
    sha256: str
    period_id: int | None = None
    message: str


class CandidateResult(BaseModel):
    kind: Literal["stalta", "phase_pick"]
    timestamp: datetime
    end_time: datetime | None = None
    score: float
    phase: Literal["P", "S"] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class DetectionRunResult(BaseModel):
    detector_name: str
    detector_version: str
    model_weight: str | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    skipped_ranges: list[tuple[datetime, datetime]] = Field(default_factory=list)


class StaLtaParameters(BaseModel):
    sta_seconds: float = Field(default=1.0, gt=0)
    lta_seconds: float = Field(default=20.0, gt=0)
    trigger_on: float = Field(default=3.5, gt=0)
    trigger_off: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_windows(self) -> "StaLtaParameters":
        if self.sta_seconds >= self.lta_seconds:
            raise ValueError("STA duration must be shorter than LTA duration")
        if self.trigger_off is None:
            self.trigger_off = self.trigger_on / 2
        if self.trigger_off >= self.trigger_on:
            raise ValueError("Trigger-off ratio must be below trigger-on ratio")
        return self


class PhaseNetParameters(BaseModel):
    p_threshold: float = Field(default=0.3, gt=0, le=1)
    s_threshold: float = Field(default=0.3, gt=0, le=1)
    weight: str = "instance"


class AssessmentPayload(BaseModel):
    verdict: Literal["quake", "noise", "uncertain"]
    notes: str = Field(default="", max_length=10_000)


class ChannelSelection(BaseModel):
    channels: list[str] = Field(min_length=1, max_length=4)

    @field_validator("channels")
    @classmethod
    def unique_channels(cls, channels: list[str]) -> list[str]:
        if len(set(channels)) != len(channels):
            raise ValueError("Channel selection must not contain duplicates")
        return channels
