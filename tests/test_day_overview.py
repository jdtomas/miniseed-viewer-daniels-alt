import numpy as np
from obspy import Stream, Trace, UTCDateTime

from shakepi.day_overview import DayOverview
from shakepi.plotting import waveform_spectrogram_figure


def test_day_summary_preserves_spike_gap_and_reuses_disk_cache(app_context, tmp_path, monkeypatch):
    settings, sessions, archive = app_context
    data = np.zeros(6000, dtype=np.int32)
    data[1759] = 98765
    stream = Stream([
        Trace(data, header=dict(network="AM", station="RF4E8", location="00", channel="EHZ", sampling_rate=100, starttime=UTCDateTime("2026-02-02"))),
        Trace(data, header=dict(network="AM", station="RF4E8", location="00", channel="EHZ", sampling_rate=100, starttime=UTCDateTime("2026-02-02T01:00:00"))),
    ])
    path = tmp_path / "recording"
    stream.write(str(path), format="MSEED")
    result = archive.ingest_bytes(path.read_bytes(), "AM.RF4E8.00.EHZ.D.2026.033")
    day = DayOverview(settings, sessions)
    try:
        key = day.queue(result.period_id)
        summary = day.futures[key].result(timeout=15)
        assert max(v for v in summary["channels"][0]["y"] if v is not None) == 98765
        assert None in summary["channels"][0]["y"]
        assert len(summary["channels"][0]["x"]) <= 4001
        monkeypatch.setattr("shakepi.day_overview.read", lambda _: (_ for _ in ()).throw(AssertionError("cache must avoid raw reads")))
        day.futures.clear()
        assert day.futures[day.queue(result.period_id)].result(timeout=5) == summary
    finally:
        day.executor.shutdown()


def test_waveform_only_does_not_compute_spectrogram(monkeypatch):
    monkeypatch.setattr("shakepi.plotting.spectrogram_data", lambda _: (_ for _ in ()).throw(AssertionError("spectrogram was not requested")))
    figure = waveform_spectrogram_figure(Stream([Trace(np.arange(1000, dtype=float))]), show_spectrogram=False)
    assert len(figure.data) == 1
