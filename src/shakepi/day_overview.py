"""Regenerable raw-count summaries. Never used as detector input."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from threading import Lock
from uuid import uuid4

import numpy as np
from obspy import read
from sqlalchemy import select

from .models import RawFile, StationPeriod
from .waveforms import minmax_decimate


class DayOverview:
    def __init__(self, settings, sessions):
        self.root = settings.cache_root / "day-overviews"
        self.sessions = sessions
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="day-overview")
        self.futures = {}
        self.lock = Lock()

    def queue(self, period_id):
        with self.sessions() as session:
            period = session.get(StationPeriod, period_id)
            files = session.scalars(select(RawFile).where(RawFile.period_id == period_id)).all()
            if period is None or not files:
                raise ValueError("Choose an available recording first")
            sources = [(f.channel, f.sha256, f.archive_path) for f in files]
            start = datetime.combine(period.utc_day, datetime.min.time(), tzinfo=timezone.utc)
        key = hashlib.sha256(json.dumps({"schema": 1, "sources": sorted(sources)}).encode()).hexdigest()
        with self.lock:
            if key not in self.futures or self.futures[key].done() and self.futures[key].exception():
                self.futures[key] = self.executor.submit(self._build, key, sources, start)
        return key

    def status(self, key):
        future = self.futures.get(key)
        if future is None:
            return "failed", "Overview job unavailable. Load it again."
        if not future.done():
            return "running", None
        try:
            return "complete", future.result()
        except Exception as error:
            return "failed", str(error)

    def _build(self, key, sources, start):
        path = self.root / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        result = {"start": start.isoformat(), "end": (start + timedelta(days=1)).isoformat(), "channels": []}
        # Each MiniSEED segment contributes extrema at their actual times.
        # Bin once more globally; never merge or preprocess full-day arrays.
        for channel, _, filename in sorted(sources):
            points = []
            coverage = []
            for trace in read(filename):
                if not len(trace.data):
                    continue
                data = np.asarray(trace.data)
                coverage.append((float(trace.stats.starttime.timestamp), float(trace.stats.endtime.timestamp) + trace.stats.delta))
                for index in sorted({int(np.argmin(data)), int(np.argmax(data))}):
                    points.append((float(trace.stats.starttime.timestamp) + index / trace.stats.sampling_rate, float(data[index])))
            points.sort()
            if not points:
                continue
            times, values = np.asarray(points, dtype=float).T
            times, values = minmax_decimate(times, values, max_points=4000)
            # Keep actual recording gaps disconnected after downsampling.
            coverage.sort()
            gaps = []
            edge = coverage[0][1]
            for left, right in coverage[1:]:
                if left - edge > 0.00001:
                    gaps.append((edge, left))
                edge = max(edge, right)
            x, y, gap_index = [], [], 0
            for time, value in zip(times, values):
                while gap_index < len(gaps) and time >= gaps[gap_index][1]:
                    x.append(gaps[gap_index][0] * 1000)
                    y.append(None)
                    gap_index += 1
                x.append(float(time * 1000))
                y.append(float(value))
            result["channels"].append({"name": channel, "x": x, "y": y})
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f"{uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(result), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return result


def day_figure(summary, period_id, start, end):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    channels = summary["channels"]
    if not channels:
        raise ValueError("No samples available for this recording")
    figure = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for i, channel in enumerate(channels, 1):
        figure.add_trace(go.Scatter(x=channel["x"], y=channel["y"], mode="lines", name=channel["name"], connectgaps=False, line={"color": "#80d8ff", "width": 1}), row=i, col=1)
        figure.update_yaxes(title_text=channel["name"], row=i, col=1)
    figure.update_xaxes(type="date", range=[summary["start"], summary["end"]], tickformat="%H:%M")
    figure.add_shape(type="rect", xref="x", yref="paper", x0=start or summary["start"], x1=end or summary["end"], y0=0, y1=1, fillcolor="rgba(34,197,94,.18)", line={"color": "#86efac"}, editable=True)
    figure.update_layout(template="plotly_dark", paper_bgcolor="#0b1020", plot_bgcolor="#101827", height=max(250, len(channels)*90), margin={"l": 55, "r": 15, "t": 15, "b": 35}, showlegend=False, dragmode="pan", meta={"period_id": period_id, "day_start": summary["start"], "day_end": summary["end"]})
    return figure
