"""Pickle-safe process worker functions for preprocessing and detection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import sessionmaker

from .config import Settings
from .db import create_database_engine, initialize_database
from .detectors import DETECTORS, PhaseNetDetector
from .models import AnalysisRun, DetectionCandidate
from .processing import Preprocessor, ZarrCache
from .schemas import DetectionRunResult, PhaseNetParameters, PreprocessingRecipe, StaLtaParameters
from .waveforms import WaveformSource

UTC = timezone.utc


def _services(settings_values: dict[str, Any]):
    settings = Settings(**settings_values)
    sessions = initialize_database(create_database_engine(settings))
    return settings, sessions, WaveformSource(sessions), ZarrCache(settings, sessions)


def preprocess_job(
    settings_values: dict[str, Any], period_id: int, channels: list[str], recipe_values: dict[str, Any]
) -> str:
    _settings, _sessions, source, cache = _services(settings_values)
    recipe = PreprocessingRecipe(**recipe_values)
    cache_key = cache.key_for(cache.source_hashes(period_id, channels), recipe, channels)
    if cache.load_stream(cache_key, source.bounds(period_id)) is not None:
        return cache_key
    period = source.period(period_id)
    start = datetime.combine(period.utc_day, datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(period.utc_day, datetime.max.time(), tzinfo=UTC)
    stream = Preprocessor().apply(source.read(period_id, channels, (start, end)), recipe)
    cache.store_stream(cache_key, stream, {"period_id": period_id, "recipe": recipe.model_dump(mode="json")})
    return cache_key


def detection_job(
    settings_values: dict[str, Any],
    run_id: int,
    period_id: int,
    channels: list[str],
    search_start: datetime,
    search_end: datetime,
    recipe_values: dict[str, Any],
    detector_name: str,
    parameter_values: dict[str, Any],
) -> DetectionRunResult:
    _settings, sessions, source, cache = _services(settings_values)
    with sessions.begin() as session:
        run = session.get(AnalysisRun, run_id)
        if run is None:
            raise KeyError(f"Unknown analysis run {run_id}")
        run.status = "running"
    recipe = PreprocessingRecipe(**recipe_values)
    try:
        cache_key = cache.key_for(cache.source_hashes(period_id, channels), recipe, channels)
        processed = cache.load_stream(cache_key, (search_start, search_end))
        if processed is None:
            processed = Preprocessor().apply(source.read(period_id, channels, (search_start, search_end)), recipe)
        detector = DETECTORS[detector_name]
        parameters = StaLtaParameters(**parameter_values) if detector_name == "stalta" else PhaseNetParameters(**parameter_values)
        if detector_name == "phasenet":
            PhaseNetDetector.validate_components(processed)
        result = detector.run(processed, parameters)
        _persist_result(sessions, run_id, result)
        return result
    except Exception as error:
        with sessions.begin() as session:
            run = session.get(AnalysisRun, run_id)
            if run is not None:
                run.status = "failed"
                run.warning = str(error)
                run.completed_at = datetime.now(UTC)
        raise


def _persist_result(sessions: sessionmaker, run_id: int, result: DetectionRunResult) -> None:
    with sessions.begin() as session:
        run = session.get(AnalysisRun, run_id)
        if run is None:
            raise KeyError(f"Unknown analysis run {run_id}")
        for candidate in result.candidates:
            session.add(
                DetectionCandidate(
                    run_id=run_id,
                    kind=candidate.kind,
                    phase=candidate.phase,
                    timestamp=candidate.timestamp,
                    end_time=candidate.end_time,
                    score=candidate.score,
                    details=candidate.details,
                )
            )
        run.status = "complete"
        run.warning = "\n".join(result.warnings) or None
        run.completed_at = datetime.now(UTC)
