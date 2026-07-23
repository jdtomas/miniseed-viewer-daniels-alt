"""Read only the requested MiniSEED time range and build plot-ready products."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime, read
from scipy.signal import spectrogram
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .models import RawFile, StationPeriod

UTC = timezone.utc


class WaveformSource:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def period_channels(self, period_id: int) -> list[str]:
        with self.session_factory() as session:
            return list(session.scalars(select(RawFile.channel).where(RawFile.period_id == period_id)))

    def period(self, period_id: int) -> StationPeriod:
        with self.session_factory() as session:
            period = session.get(StationPeriod, period_id)
            if period is None:
                raise KeyError(f"Unknown station period {period_id}")
            session.expunge(period)
            return period

    def bounds(self, period_id: int) -> tuple[datetime, datetime]:
        with self.session_factory() as session:
            files = session.scalars(select(RawFile).where(RawFile.period_id == period_id)).all()
            if not files:
                raise KeyError(f"Station period {period_id} has no files")
            start = min(file.start_time for file in files)
            end = max(file.end_time for file in files)
            return ensure_utc(start), ensure_utc(end)

    def read(self, period_id: int, channels: list[str], utc_range: tuple[datetime, datetime]) -> Stream:
        start, end = utc_range
        if start >= end:
            raise ValueError("Time range start must precede end")
        selected = set(channels)
        with self.session_factory() as session:
            files = session.scalars(
                select(RawFile).where(RawFile.period_id == period_id, RawFile.channel.in_(selected))
            ).all()
        stream = Stream()
        for file in files:
            stream += read(
                file.archive_path,
                starttime=UTCDateTime(start),
                endtime=UTCDateTime(end),
            )
        return merge_contiguous_segments(stream)


def merge_contiguous_segments(stream: Stream) -> Stream:
    """Join adjacent MiniSEED records while preserving actual data gaps."""
    if not stream:
        return Stream()
    merged = stream.copy()
    merged.merge(method=0, fill_value=None)
    return merged.split()


def minmax_decimate(times: np.ndarray, data: np.ndarray, max_points: int = 4_000) -> tuple[np.ndarray, np.ndarray]:
    """Min/max downsample without hiding short high-amplitude transients."""
    if len(data) <= max_points:
        return times, data
    bins = max(1, max_points // 2)
    edges = np.linspace(0, len(data), bins + 1, dtype=int)
    output_times: list[float] = []
    output_values: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        section = data[left:right]
        if not np.isfinite(section).any():
            continue
        low_index = left + int(np.nanargmin(section))
        high_index = left + int(np.nanargmax(section))
        for index in sorted((low_index, high_index)):
            output_times.append(float(times[index]))
            output_values.append(float(data[index]))
    return np.asarray(output_times), np.asarray(output_values)


def trace_plot_data(trace: object, max_points: int = 4_000) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(trace.data, dtype=np.float32)
    offsets = np.arange(len(data), dtype=np.float64) / float(trace.stats.sampling_rate)
    return minmax_decimate(offsets, data, max_points=max_points)


def spectrogram_data(trace: object, max_time_bins: int = 1_500, max_frequency_bins: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce an RX-style log-power spectrogram bounded for browser delivery."""
    sample_rate = float(trace.stats.sampling_rate)
    data = np.asarray(trace.data, dtype=np.float32)
    if len(data) < 16:
        return np.empty(0), np.empty(0), np.empty((0, 0))
    max_input_samples = 300_000
    decimation = max(1, int(np.ceil(len(data) / max_input_samples)))
    if decimation > 1:
        data = data[::decimation]
        sample_rate /= decimation
    target_window = max(32, min(2048, int(len(data) / max(1, max_time_bins // 2))))
    nperseg = min(len(data), 1 << (target_window - 1).bit_length())
    noverlap = int(nperseg * 0.75)
    frequencies, times, power = spectrogram(
        data,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=min(noverlap, nperseg - 1),
        scaling="density",
        mode="psd",
    )
    if len(frequencies) > max_frequency_bins:
        frequency_indices = np.linspace(0, len(frequencies) - 1, max_frequency_bins, dtype=int)
        frequencies = frequencies[frequency_indices]
        power = power[frequency_indices, :]
    if len(times) > max_time_bins:
        time_indices = np.linspace(0, len(times) - 1, max_time_bins, dtype=int)
        times = times[time_indices]
        power = power[:, time_indices]
    decibels = 10 * np.log10(np.maximum(power, np.finfo(np.float32).tiny))
    return times, frequencies, decibels


def ensure_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)
