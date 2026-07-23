"""Plotly rendering from bounded waveform products rather than raw day traces."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from obspy import Stream

from .waveforms import spectrogram_data, trace_plot_data

UTC = timezone.utc


def _offsets_to_datetimes(start: datetime, offsets: np.ndarray) -> list[datetime]:
    return [start + timedelta(seconds=float(offset)) for offset in offsets]


def waveform_spectrogram_figure(
    stream: Stream,
    search_range: tuple[datetime, datetime] | None = None,
    title: str = "Waveform review",
) -> go.Figure:
    traces = list(stream)
    if not traces:
        return go.Figure(
            layout={
                "title": "No waveform data in this time range",
                "template": "plotly_dark",
                "paper_bgcolor": "#0b1020",
                "plot_bgcolor": "#101827",
                "font": {"color": "#e5edf9"},
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            }
        )
    figure = make_subplots(
        rows=len(traces) * 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.58, 0.42] * len(traces),
        subplot_titles=[label for trace in traces for label in (trace.id, f"{trace.id} spectrogram")],
    )
    for index, trace in enumerate(traces):
        row = index * 2 + 1
        offsets, values = trace_plot_data(trace)
        start = trace.stats.starttime.datetime.replace(tzinfo=UTC)
        times = _offsets_to_datetimes(start, offsets)
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                line={"width": 1, "color": "#80d8ff"},
                name=trace.id,
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S.%L UTC}<br>%{y:.4g} counts<extra></extra>",
            ),
            row=row,
            col=1,
        )
        spec_times, frequencies, decibels = spectrogram_data(trace)
        if spec_times.size:
            figure.add_trace(
                go.Heatmap(
                    x=_offsets_to_datetimes(start, spec_times),
                    y=frequencies,
                    z=decibels,
                    colorscale="Turbo",
                    colorbar={"title": "dB", "len": 0.2},
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S UTC}<br>%{y:.2f} Hz<br>%{z:.1f} dB<extra></extra>",
                ),
                row=row + 1,
                col=1,
            )
            figure.update_yaxes(title_text="Hz", row=row + 1, col=1)
        figure.update_yaxes(title_text="counts", row=row, col=1)
    if search_range:
        left, right = search_range
        figure.add_vrect(
            x0=left,
            x1=right,
            fillcolor="#22c55e",
            opacity=0.13,
            line_width=1,
            line_color="#22c55e",
            annotation_text="search region",
            annotation_position="top left",
        )
    figure.update_xaxes(
        title_text="UTC time",
        type="date",
        tickformat="%H:%M:%S",
        hoverformat="%Y-%m-%d %H:%M:%S.%L UTC",
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        height=max(600, 360 * len(traces)),
        margin={"l": 64, "r": 24, "t": 70, "b": 48},
        showlegend=False,
        dragmode="pan",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#101827",
    )
    return figure
