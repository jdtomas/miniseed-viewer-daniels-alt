"""Application orchestration for recipes, background runs, and assessments."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .detectors import DETECTORS, PhaseNetDetector
from .models import (
    Actor,
    AnalysisRun,
    DetectionCandidate,
    ManualAssessment,
    PreprocessingRecipeRecord,
)
from .jobs import detection_job, preprocess_job
from .processing import Preprocessor, ZarrCache
from .schemas import (
    AssessmentPayload,
    DetectionRunResult,
    PhaseNetParameters,
    PreprocessingRecipe,
    StaLtaParameters,
)
from .waveforms import WaveformSource, merge_contiguous_segments

UTC = timezone.utc


class LocalJobRunner:
    """A single-worker executor. Jobs are persisted before they run and after completion."""

    def __init__(self, max_workers: int = 1):
        try:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
            self.process_backed = True
        except (NotImplementedError, PermissionError):
            # Restricted local sandboxes can forbid POSIX semaphores. Production uses the
            # process pool; this fallback keeps local development and tests functional.
            self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="shakepi-analysis")
            self.process_backed = False
        self._futures: dict[str, Future[Any]] = {}

    def submit_once(self, key: str, function: Any, *args: Any) -> Future[Any]:
        existing = self._futures.get(key)
        if existing is not None and not existing.done():
            return existing
        future = self.executor.submit(function, *args)
        self._futures[key] = future
        return future

    def status(self, key: str) -> dict[str, object]:
        future = self._futures.get(key)
        if future is None:
            return {"status": "unknown"}
        if future.cancelled():
            return {"status": "failed", "message": "The background job was cancelled."}
        if not future.done():
            return {"status": "running" if future.running() else "queued"}
        error = future.exception()
        if error is not None:
            return {"status": "failed", "message": str(error)}
        return {"status": "complete", "result": future.result()}


class AnalysisService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        waveform_source: WaveformSource,
        cache: ZarrCache,
        jobs: LocalJobRunner,
    ):
        self.session_factory = session_factory
        self.waveform_source = waveform_source
        self.cache = cache
        self.jobs = jobs
        self.preprocessor = Preprocessor()

    def anonymous_actor_id(self, session: Session) -> int:
        actor = session.scalar(select(Actor).where(Actor.display_name == "anonymous"))
        if actor is None:
            raise RuntimeError("Database is missing the anonymous actor")
        return actor.id

    def save_recipe(self, recipe: PreprocessingRecipe) -> int:
        with self.session_factory.begin() as session:
            record = session.scalar(
                select(PreprocessingRecipeRecord).where(PreprocessingRecipeRecord.recipe_hash == recipe.fingerprint())
            )
            if record is None:
                record = PreprocessingRecipeRecord(
                    recipe_hash=recipe.fingerprint(),
                    canonical_json=recipe.canonical_json(),
                    schema_version=recipe.schema_version,
                    created_by_id=self.anonymous_actor_id(session),
                )
                session.add(record)
                session.flush()
            return record.id

    def queue_preprocessing(
        self,
        period_id: int,
        channels: list[str],
        recipe: PreprocessingRecipe,
    ) -> tuple[str, Future[str]]:
        cache_key = self.cache.key_for(self.cache.source_hashes(period_id, channels), recipe, channels)
        return (
            cache_key,
            self.jobs.submit_once(
                f"preprocess:{cache_key}",
                preprocess_job,
                self.cache.settings.model_dump(mode="json"),
                period_id,
                channels,
                recipe.model_dump(mode="json"),
            ),
        )

    def preprocessing_status(self, cache_key: str) -> dict[str, object]:
        if (self.cache.artifact_path(cache_key) / ".complete").exists():
            return {"status": "complete", "cache_key": cache_key}
        status = self.jobs.status(f"preprocess:{cache_key}")
        return {**status, "cache_key": cache_key}

    def prepared_stream(
        self,
        period_id: int,
        channels: list[str],
        utc_range: tuple[datetime, datetime],
        recipe: PreprocessingRecipe,
    ):
        """Use completed day cache when available; otherwise preprocess only the requested range."""
        cache_key = self.cache.key_for(self.cache.source_hashes(period_id, channels), recipe, channels)
        cached = self.cache.load_stream(cache_key, utc_range)
        if cached:
            return merge_contiguous_segments(cached)
        return self.preprocessor.apply(self.waveform_source.read(period_id, channels, utc_range), recipe)

    def queue_detection(
        self,
        period_id: int,
        channels: list[str],
        search_start: datetime,
        search_end: datetime,
        recipe: PreprocessingRecipe,
        detector_name: str,
        parameters: StaLtaParameters | PhaseNetParameters,
    ) -> tuple[int, Future[DetectionRunResult]]:
        if detector_name not in DETECTORS:
            raise ValueError(f"Unknown detector: {detector_name}")
        recipe_id = self.save_recipe(recipe)
        detector = DETECTORS[detector_name]
        with self.session_factory.begin() as session:
            run = AnalysisRun(
                period_id=period_id,
                recipe_id=recipe_id,
                detector_name=detector_name,
                detector_version=getattr(detector, "version", "unknown"),
                model_weight=getattr(parameters, "weight", None),
                channel_selection=channels,
                parameters=parameters.model_dump(mode="json"),
                search_start=search_start,
                search_end=search_end,
                status="queued",
                created_by_id=self.anonymous_actor_id(session),
            )
            session.add(run)
            session.flush()
            run_id = run.id

        return run_id, self.jobs.submit_once(
            f"run:{run_id}",
            detection_job,
            self.cache.settings.model_dump(mode="json"),
            run_id,
            period_id,
            channels,
            search_start,
            search_end,
            recipe.model_dump(mode="json"),
            detector_name,
            parameters.model_dump(mode="json"),
        )

    def detection_status(self, run_id: int) -> dict[str, object]:
        with self.session_factory() as session:
            run = session.get(AnalysisRun, run_id)
            if run is None:
                return {
                    "status": "unknown",
                    "run_id": run_id,
                    "message": "The detector run no longer exists.",
                }
            candidate_count = session.scalar(
                select(func.count(DetectionCandidate.id)).where(
                    DetectionCandidate.run_id == run_id
                )
            )
            return {
                "status": run.status,
                "run_id": run.id,
                "detector_name": run.detector_name,
                "candidate_count": int(candidate_count or 0),
                "message": run.warning,
            }

    def candidates_for_period(self, period_id: int) -> list[DetectionCandidate]:
        with self.session_factory() as session:
            candidates = session.scalars(
                select(DetectionCandidate)
                .join(AnalysisRun)
                .where(AnalysisRun.period_id == period_id, AnalysisRun.status == "complete")
                .order_by(DetectionCandidate.timestamp.desc())
            ).all()
            for candidate in candidates:
                session.expunge(candidate)
            return candidates

    def save_assessment(self, candidate_id: int, payload: AssessmentPayload) -> None:
        with self.session_factory.begin() as session:
            assessment = session.scalar(
                select(ManualAssessment).where(ManualAssessment.candidate_id == candidate_id)
            )
            if assessment is None:
                assessment = ManualAssessment(
                    candidate_id=candidate_id,
                    actor_id=self.anonymous_actor_id(session),
                    verdict=payload.verdict,
                    notes=payload.notes,
                )
                session.add(assessment)
            else:
                assessment.verdict = payload.verdict
                assessment.notes = payload.notes
