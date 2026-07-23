from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timezone

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from shakepi.detectors import PhaseNetDetector, StaLtaDetector
from shakepi.plotting import waveform_spectrogram_figure
from shakepi.processing import Preprocessor
from shakepi.processing import ZarrCache
from shakepi.schemas import PreprocessingRecipe, StaLtaParameters
from shakepi.services import AnalysisService, LocalJobRunner
from shakepi.waveforms import WaveformSource
from shakepi.waveforms import merge_contiguous_segments
from shakepi.waveforms import minmax_decimate

UTC = timezone.utc


def trace(data: np.ndarray, channel: str = "EHZ") -> Trace:
    return Trace(
        data=data.astype(np.float32),
        header={
            "network": "AM",
            "station": "RF4E8",
            "location": "00",
            "channel": channel,
            "sampling_rate": 100.0,
            "starttime": UTCDateTime(datetime(2026, 2, 2, tzinfo=UTC)),
        },
    )


def test_preprocessing_is_non_mutating_and_validates_nyquist() -> None:
    source = Stream([trace(np.sin(np.linspace(0, 100, 10_000)))])
    original = source[0].data.copy()
    processed = Preprocessor().apply(source, PreprocessingRecipe(bandpass_low_hz=1, bandpass_high_hz=20))
    assert np.array_equal(source[0].data, original)
    assert not np.array_equal(processed[0].data, original)
    with pytest.raises(ValueError, match="Nyquist"):
        Preprocessor().apply(source, PreprocessingRecipe(bandpass_low_hz=1, bandpass_high_hz=50))


def test_stalta_finds_synthetic_impulse() -> None:
    data = np.random.default_rng(3).normal(0, 0.02, 12_000)
    data[7_000:7_150] += 4
    result = StaLtaDetector().run(
        Stream([trace(data)]),
        StaLtaParameters(sta_seconds=0.2, lta_seconds=2, trigger_on=2, trigger_off=1),
    )
    assert result.candidates
    assert result.candidates[0].kind == "stalta"


def test_phasenet_requires_complete_components() -> None:
    with pytest.raises(ValueError, match="Z/N/E"):
        PhaseNetDetector.validate_components(Stream([trace(np.zeros(1_000), "EHZ")]))
    PhaseNetDetector.validate_components(
        Stream([trace(np.zeros(1_000), "ENZ"), trace(np.zeros(1_000), "ENN"), trace(np.zeros(1_000), "ENE")])
    )


def test_minmax_decimation_preserves_outlier() -> None:
    values = np.zeros(100_000)
    values[54_321] = 99
    _times, decimated = minmax_decimate(np.arange(len(values)), values, max_points=500)
    assert len(decimated) <= 500
    assert decimated.max() == 99


def test_contiguous_miniseed_records_are_merged_for_display() -> None:
    first = trace(np.arange(100))
    second = trace(np.arange(100, 200))
    second.stats.starttime = first.stats.endtime + first.stats.delta

    merged = merge_contiguous_segments(Stream([first, second]))

    assert len(merged) == 1
    assert merged[0].stats.npts == 200


def test_empty_waveform_figure_uses_the_dark_theme() -> None:
    figure = waveform_spectrogram_figure(Stream())

    assert figure.layout.paper_bgcolor == "#0b1020"
    assert figure.layout.plot_bgcolor == "#101827"
    assert figure.layout.font.color == "#e5edf9"


def test_waveform_figure_uses_date_axis_for_selected_samples() -> None:
    figure = waveform_spectrogram_figure(Stream([trace(np.sin(np.linspace(0, 10, 1_000)))]))

    assert figure.data[0].type == "scatter"
    assert isinstance(figure.data[0].x[0], datetime)
    assert figure.layout.xaxis.type == "date"


def test_local_job_runner_reports_honest_future_states() -> None:
    runner = object.__new__(LocalJobRunner)
    pending: Future[str] = Future()
    runner._futures = {"job": pending}

    assert runner.status("missing") == {"status": "unknown"}
    assert runner.status("job") == {"status": "queued"}

    pending.set_running_or_notify_cancel()
    assert runner.status("job") == {"status": "running"}

    pending.set_result("cache-key")
    assert runner.status("job") == {"status": "complete", "result": "cache-key"}

    failed: Future[str] = Future()
    failed.set_exception(RuntimeError("worker failed"))
    runner._futures["failed"] = failed
    assert runner.status("failed") == {"status": "failed", "message": "worker failed"}


def test_detection_run_is_persisted_by_background_worker(app_context, tmp_path) -> None:
    from hashlib import sha256

    from conftest import write_mseed

    settings, sessions, archive = app_context
    source_file = tmp_path / "event.mseed"
    data = np.random.default_rng(10).normal(0, 0.02, 12_000)
    data[7_000:7_150] += 4
    write_mseed(source_file, data=data.astype(np.int32))
    ingested = archive.ingest_path(
        source_file,
        "AM.RF4E8.00.EHZ.D.2026.033",
        sha256(source_file.read_bytes()).hexdigest(),
        source_file.stat().st_size,
    )
    service = AnalysisService(
        sessions,
        WaveformSource(sessions),
        ZarrCache(settings, sessions),
        LocalJobRunner(max_workers=1),
    )
    start, end = WaveformSource(sessions).bounds(ingested.period_id)
    cache_key, preprocessing_future = service.queue_preprocessing(
        ingested.period_id,
        ["EHZ"],
        PreprocessingRecipe(),
    )
    assert preprocessing_future.result(timeout=10) == cache_key
    assert service.preprocessing_status(cache_key)["status"] == "complete"

    run_id, future = service.queue_detection(
        ingested.period_id,
        ["EHZ"],
        start,
        end,
        PreprocessingRecipe(),
        "stalta",
        StaLtaParameters(sta_seconds=0.2, lta_seconds=2, trigger_on=2, trigger_off=1),
    )
    future.result(timeout=10)
    assert service.candidates_for_period(ingested.period_id)
    status = service.detection_status(run_id)
    assert status["status"] == "complete"
    assert int(status["candidate_count"]) > 0
