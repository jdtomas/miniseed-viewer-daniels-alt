"""FastAPI endpoints and Dash review interface."""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dash import ClientsideFunction, Dash, Input, Output, State, ctx, dcc, html, no_update
from fastapi import FastAPI, File, HTTPException, UploadFile
from obspy import Stream
from pydantic import ValidationError

from .archive import ArchiveService
from .config import Settings
from .db import create_database_engine, initialize_database
from .plotting import waveform_spectrogram_figure
from .processing import ZarrCache
from .schemas import AssessmentPayload, PhaseNetParameters, PreprocessingRecipe, StaLtaParameters
from .services import AnalysisService, LocalJobRunner
from .waveforms import WaveformSource

UTC = timezone.utc


def _ingest_dash_upload(archive: ArchiveService, contents: str, filename: str) -> str:
    """Decode a ``dcc.Upload`` payload and hand it to the archive service."""
    try:
        _, encoded_contents = contents.split(",", 1)
        file_contents = base64.b64decode(encoded_contents, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Upload data could not be decoded") from error
    return archive.ingest_bytes(file_contents, filename).message


def _parse_time(value: str | None) -> datetime:
    if not value:
        raise ValueError("A UTC time is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _recipe_from_controls(
    fill_short_gaps: bool,
    deep_denoiser: bool,
    low: float | None,
    high: float | None,
) -> PreprocessingRecipe:
    return PreprocessingRecipe(
        fill_short_gaps=fill_short_gaps,
        deep_denoiser=deep_denoiser,
        bandpass_low_hz=low or None,
        bandpass_high_hz=high or None,
    )


def _activity_status(
    message: str,
    detail: str | None = None,
    state: str = "busy",
) -> html.Div:
    children: list[object] = [html.Strong(message, className="activity-message")]
    if state == "busy":
        children.append(
            html.Div(
                html.Div(className="activity-progress-bar"),
                className="activity-progress",
                role="progressbar",
                **{"aria-label": message, "aria-valuetext": message},
            )
        )
    if detail:
        children.append(html.Span(detail, className="activity-detail"))
    return html.Div(
        children,
        className=f"activity-status activity-status--{state}",
        role="status",
        **{"aria-live": "polite"},
    )


def _graph_skeleton(message: str) -> html.Div:
    return html.Div(
        [
            html.Div(message, className="skeleton-label"),
            html.Div(className="skeleton-line skeleton-line--wide"),
            html.Div(className="skeleton-line skeleton-line--medium"),
            html.Div(className="skeleton-plot"),
            html.Div(className="skeleton-line skeleton-line--short"),
        ],
        className="graph-skeleton",
        role="status",
        **{"aria-live": "polite"},
    )


def create_application(settings: Settings | None = None) -> tuple[FastAPI, Dash]:
    settings = settings or Settings()
    engine = create_database_engine(settings)
    sessions = initialize_database(engine)
    archive = ArchiveService(settings, sessions)
    source = WaveformSource(sessions)
    analysis = AnalysisService(sessions, source, ZarrCache(settings, sessions), LocalJobRunner(settings.job_workers))

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield
        finally:
            day.executor.shutdown(wait=True, cancel_futures=True)
            analysis.jobs.executor.shutdown(wait=True, cancel_futures=True)
            engine.dispose()

    server = FastAPI(title="ShakePi MiniSEED Viewer", lifespan=lifespan)

    @server.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @server.get("/api/periods")
    def periods() -> list[dict[str, object]]:
        return archive.list_periods()

    @server.post("/api/uploads")
    async def upload(files: list[UploadFile] = File(...)) -> list[dict[str, object]]:
        results = []
        for file in files:
            try:
                results.append((await archive.ingest_upload(file)).model_dump(mode="json"))
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return results

    app = Dash(
        server=server,
        title="ShakePi MiniSEED Viewer",
        update_title=None,
        assets_folder=str(Path(__file__).parent / "assets"),
    )
    from .workbench import layout

    app.layout = layout()

    from .day_overview import DayOverview, day_figure
    day = DayOverview(settings, sessions)

    @app.callback(Output("day-operation", "data"), Input("load-day", "n_clicks"), State("period-select", "value"), prevent_initial_call=True)
    def load_day(_, period_id):
        try:
            return {"key": day.queue(period_id), "period_id": period_id}
        except Exception as error:
            return {"error": str(error), "period_id": period_id}

    @app.callback(Output("day-graph", "figure"), Output("day-status", "children"), Output("day-refresh", "disabled"),
                  Input("day-operation", "data"), Input("day-refresh", "n_intervals"), Input("period-select", "value"),
                  State("range-start", "value"), State("range-end", "value"))
    def monitor_day(operation, _, period_id, start, end):
        if not operation or operation.get("period_id") != period_id:
            return waveform_spectrogram_figure(Stream()), "Choose a recording, then Load day overview.", True
        if operation.get("error"):
            return no_update, operation["error"], True
        state, result = day.status(operation["key"])
        if state == "running":
            return no_update, "Preparing reduced day overview in the background…", False
        if state == "failed":
            return no_update, f"Overview failed: {result}", True
        return day_figure(result, period_id, start, end), "Day overview ready · reduced raw counts", True

    @app.callback(Output("range-start", "value", allow_duplicate=True), Output("range-end", "value", allow_duplicate=True),
                  Input("day-graph", "relayoutData"), State("day-graph", "figure"), State("period-select", "value"),
                  State("range-start", "value"), State("range-end", "value"), prevent_initial_call=True)
    def select_day_range(event, figure, period_id, current_start, current_end):
        meta = (figure or {}).get("layout", {}).get("meta", {})
        if not event or meta.get("period_id") != period_id:
            return no_update, no_update
        shape = event.get("shapes", [{}])[0] if event.get("shapes") else {}
        left = event.get("shapes[0].x0", shape.get("x0", current_start))
        right = event.get("shapes[0].x1", shape.get("x1", current_end))
        if not any(key.startswith("shapes") for key in event):
            return no_update, no_update
        try:
            start, end = _parse_time(str(left)), _parse_time(str(right))
            start, end = max(start, _parse_time(meta["day_start"])), min(end, _parse_time(meta["day_end"]))
            if start >= end:
                return no_update, no_update
            return start.isoformat(), end.isoformat()
        except (ValueError, KeyError):
            return no_update, no_update

    app.clientside_callback(
        """async function(start, end, figure, period) {
            const skip = window.dash_clientside.no_update;
            if (!start || !end || figure?.layout?.meta?.period_id !== period) return skip;
            const graph = document.querySelector('#day-graph .js-plotly-plot');
            if (!graph?.layout?.shapes?.length) return skip;
            const parse = x => Date.parse(/[Zz]|[+-]\\d\\d:\\d\\d$/.test(x) ? x : x + 'Z');
            const a = parse(start), b = parse(end), shape = graph.layout.shapes[0];
            if (!Number.isFinite(a) || !Number.isFinite(b) || a >= b) return skip;
            if (parse(shape.x0) === a && parse(shape.x1) === b) return skip;
            await Plotly.relayout(graph, {'shapes[0].x0': start, 'shapes[0].x1': end});
            return start + end;
        }""", Output("day-sync", "data"), Input("range-start", "value"), Input("range-end", "value"),
        State("day-graph", "figure"), State("period-select", "value"), prevent_initial_call=True)

    @app.callback(
        Output("upload-status", "children"),
        Input("upload-files", "contents"),
        State("upload-files", "filename"),
        prevent_initial_call=True,
        running=[(Output("upload-files", "disabled"), True, False)],
    )
    def ingest_dash_upload(
        contents: str | list[str] | None, filenames: str | list[str] | None
    ) -> str | list[html.Li]:
        if not contents or not filenames:
            return no_update

        content_items = contents if isinstance(contents, list) else [contents]
        filename_items = filenames if isinstance(filenames, list) else [filenames]
        if len(content_items) != len(filename_items):
            return "Upload did not include a filename for every file."

        messages = []
        for content, filename in zip(content_items, filename_items):
            try:
                messages.append(html.Li(_ingest_dash_upload(archive, content, filename)))
            except ValueError as error:
                messages.append(html.Li(f"{filename}: {error}"))
        return messages[0].children if len(messages) == 1 else html.Ul(messages)

    @app.callback(
        Output("period-select", "options"),
        Output("period-select", "value"),
        Input("refresh", "n_intervals"),
        Input("upload-status", "children"),
        State("period-select", "value"),
    )
    def refresh_periods(_: int, upload_status, current: int | None):
        choices = archive.list_periods()
        options = [
            {"label": f"{item['label']}  [{', '.join(item['channels'])}]", "value": item["id"]}
            for item in choices
        ]
        values = {item["id"] for item in choices}
        if current in values:
            return options, no_update
        return options, None

    @app.callback(
        Output("channel-select", "options"),
        Output("channel-select", "value"),
        Output("period-status", "children"),
        Output("range-start", "value"),
        Output("range-end", "value"),
        Input("period-select", "value"),
        State("channel-select", "value"),
    )
    def select_period(period_id: int | None, current_channels: list[str] | None):
        if not period_id:
            return [], [], "No station period selected.", "", ""
        period = next(item for item in archive.list_periods() if item["id"] == period_id)
        channels = list(period["channels"])
        selected = [channel for channel in (current_channels or []) if channel in channels][:4] or channels[:4]
        start, end = source.bounds(period_id)
        default_end = min(end, start + timedelta(minutes=5))
        return (
            [{"label": channel, "value": channel} for channel in channels],
            selected,
            (
                f"Available channels: {', '.join(channels)}. Raw amplitudes are uncorrected counts. "
                "Choose a time range, then click Load detail."
            ),
            start.isoformat(),
            default_end.isoformat(),
        )

    @app.callback(
        Output("trigger-channel-select", "options"),
        Output("trigger-channel-select", "value"),
        Output("detector-select", "options"),
        Output("detector-select", "value"),
        Input("channel-select", "value"),
        State("trigger-channel-select", "value"),
        State("detector-select", "value"),
    )
    def set_detector_compatibility(
        channels: list[str] | None,
        current_trigger_channel: str | None,
        current_detector: str | None,
    ):
        channels = channels or []
        channel_options = [{"label": channel, "value": channel} for channel in channels]
        trigger_channel = current_trigger_channel if current_trigger_channel in channels else (channels[0] if channels else None)
        phasenet_available = {"ENZ", "ENN", "ENE"}.issubset({channel.upper() for channel in channels})
        detector_options = [
            {"label": "STA/LTA (single channel)", "value": "stalta"},
            {
                "label": "PhaseNet (MEMS Z/N/E)",
                "value": "phasenet",
                "disabled": not phasenet_available,
            },
        ]
        if phasenet_available:
            detector = current_detector or "phasenet"
        else:
            detector = "stalta"
        return channel_options, trigger_channel, detector_options, detector

    app.clientside_callback(
        """function(period, channels) {
            const disabled = !period || !channels || !channels.length;
            return [!period, disabled, disabled];
        }""", Output("load-day", "disabled"), Output("load-overview", "disabled"), Output("search-button", "disabled"),
        Input("period-select", "value"), Input("channel-select", "value"))

    app.clientside_callback(
        """function(figure, period, channels) {
            const meta = figure?.layout?.meta;
            const stale = figure?.data?.length && (!meta || meta.period_id !== period ||
                JSON.stringify(meta.channels) !== JSON.stringify([...(channels || [])].sort()));
            return stale ? {display: 'none'} : {};
        }""", Output("overview-loading", "style"), Input("overview-graph", "figure"),
        Input("period-select", "value"), Input("channel-select", "value"))

    app.clientside_callback(
        ClientsideFunction(namespace="shakepi", function_name="viewButtonsDisabled"),
        Output("use-visible", "disabled"),
        Output("reset-view", "disabled"),
        Input("overview-graph", "figure"),
        Input("period-select", "value"),
        Input("channel-select", "value"),
    )

    app.clientside_callback(
        r"""async function(clicks, figure) {
            if (!clicks || !figure || !figure.data || !figure.data.length) {
                return window.dash_clientside.no_update;
            }
            const graph = document.querySelector('#overview-graph .js-plotly-plot');
            if (!graph || !window.Plotly) return window.dash_clientside.no_update;
            const update = {};
            for (const [name, axis] of Object.entries(figure.layout)) {
                if (!/^[xy]axis\d*$/.test(name)) continue;
                const range = name.startsWith('x')
                    ? figure.layout.meta?.default_time_range : null;
                if (range) {
                    update[name + '.autorange'] = false;
                    update[name + '.range[0]'] = range[0];
                    update[name + '.range[1]'] = range[1];
                } else {
                    update[name + '.autorange'] = true;
                }
            }
            await window.Plotly.relayout(graph, update);
            return clicks;
        }""",
        Output("reset-view-result", "data"),
        Input("reset-view", "n_clicks"),
        State("overview-graph", "figure"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="shakepi", function_name="useVisibleRange"),
        Output("range-start", "value", allow_duplicate=True),
        Output("range-end", "value", allow_duplicate=True),
        Output("visible-range-status", "children"),
        Input("use-visible", "n_clicks"),
        State("overview-graph", "figure"),
        State("period-select", "value"),
        State("channel-select", "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("overview-graph", "figure"),
        Output("overview-status", "children"),
        Input("load-overview", "n_clicks"),
        Input("candidate-select", "value"),
        Input("detail-left", "value"),
        Input("detail-right", "value"),
        Input("period-select", "value"),
        State("channel-select", "value"),
        State("range-start", "value"),
        State("range-end", "value"),
        State("preprocess-options", "value"),
        State("bandpass-low", "value"),
        State("bandpass-high", "value"),
        State("show-spectrogram", "value"),
        prevent_initial_call=True,
        running=[
            (
                Output("overview-range-controls", "className"),
                "workbench-toolbar controls-disabled",
                "workbench-toolbar",
            ),
            (Output("overview-region", "aria-busy"), "true", "false"),
            (Output("candidate-list-region", "className"), "controls-disabled", ""),
        ],
    )
    def render_overview(
        _load_clicks: int | None,
        candidate_id: int | None,
        left: float | None,
        right: float | None,
        period_id: int | None,
        channels: list[str] | None,
        start_value: str | None,
        end_value: str | None,
        options: list[str] | None,
        low: float | None,
        high: float | None,
        spectrogram: list[str] | None,
    ):
        if ctx.triggered_id == "period-select":
            return waveform_spectrogram_figure(Stream()), "Choose an interval, then Load detail."
        if ctx.triggered_id != "load-overview" and not candidate_id:
            return no_update, no_update
        if not period_id or not channels or not start_value or not end_value:
            return (
                waveform_spectrogram_figure(Stream(), title="Select a station period"),
                "No waveform range selected.",
            )
        try:
            start, end = _parse_time(start_value), _parse_time(end_value)
            if len(channels) > 4:
                raise ValueError("Select at most four channels for detail")
            recipe = _recipe_from_controls("fill" in (options or []), "denoise" in (options or []), low, high)
            candidate = None
            if ctx.triggered_id != "load-overview":
                from .models import DetectionCandidate
                with sessions() as session:
                    candidate = session.get(DetectionCandidate, candidate_id)
                    if candidate is None or candidate.run.period_id != period_id:
                        return no_update, no_update
                    start = candidate.timestamp.replace(tzinfo=UTC) - timedelta(seconds=30 if left is None else left)
                    end = (candidate.end_time or candidate.timestamp).replace(tzinfo=UTC) + timedelta(seconds=90 if right is None else right)
                    recipe = PreprocessingRecipe.model_validate_json(candidate.run.recipe.canonical_json) if candidate.run.recipe else PreprocessingRecipe()
            stream = analysis.prepared_stream(period_id, channels, (start, end), recipe)
            point_count = sum(len(trace.data) for trace in stream)
            plotted_channels = ", ".join(trace.id for trace in stream) or "none"
            figure = waveform_spectrogram_figure(stream, show_spectrogram="show" in (spectrogram or []))
            if candidate is not None:
                figure.add_shape(type="line", x0=candidate.timestamp, x1=candidate.timestamp, y0=0, y1=1, xref="x", yref="paper", line={"color": "#fbbf24"})
            if stream:
                figure.update_xaxes(range=[start.isoformat(), end.isoformat()], autorange=False)
                figure.update_layout(meta={
                    "default_time_range": [start.isoformat(), end.isoformat()],
                    "period_id": period_id,
                    "channels": sorted(channels),
                })
            return (
                figure,
                (
                    f"Plotted {point_count:,} samples {'around candidate ' + str(candidate_id) if candidate is not None else ''} from {plotted_channels} "
                    f"between {start.isoformat()} and {end.isoformat()}."
                ),
            )
        except Exception as error:
            return (
                waveform_spectrogram_figure(Stream(), title=f"Unable to render waveform: {error}"),
                f"Unable to render waveform: {error}",
            )

    @app.callback(
        Output("preprocess-operation", "data"),
        Input("apply-preprocess", "n_clicks"),
        State("period-select", "value"),
        State("channel-select", "value"),
        State("preprocess-options", "value"),
        State("bandpass-low", "value"),
        State("bandpass-high", "value"),
        prevent_initial_call=True,
    )
    def apply_preprocess(_: int, period_id: int | None, channels: list[str] | None, options: list[str] | None, low: float | None, high: float | None):
        if not period_id or not channels:
            return {"status": "failed", "message": "Choose a station period and channels first."}
        try:
            recipe = _recipe_from_controls("fill" in (options or []), "denoise" in (options or []), low, high)
            cache_key, _future = analysis.queue_preprocessing(period_id, channels, recipe)
            return {"status": "submitted", "cache_key": cache_key}
        except (ValidationError, ValueError) as error:
            return {"status": "failed", "message": str(error)}

    @app.callback(
        Output("preprocess-status", "children"),
        Output("preprocess-controls", "disabled"),
        Output("preprocess-refresh", "disabled"),
        Input("preprocess-operation", "data"),
        Input("preprocess-refresh", "n_intervals"),
    )
    def monitor_preprocessing(operation: dict[str, object] | None, _: int):
        if not operation:
            return "", False, True
        if operation.get("status") == "failed":
            return _activity_status(
                "Preprocessing could not start",
                str(operation.get("message") or "Unknown error"),
                "error",
            ), False, True
        cache_key = str(operation.get("cache_key") or "")
        if not cache_key:
            return _activity_status(
                "Preprocessing status is unavailable",
                "Apply preprocessing again to retry.",
                "error",
            ), False, True
        status = analysis.preprocessing_status(cache_key)
        state = str(status["status"])
        if state == "complete":
            return _activity_status(
                "Preprocessing cache ready",
                f"Cache {cache_key[:12]}",
                "success",
            ), False, True
        if state in {"queued", "running"}:
            label = "Preprocessing queued" if state == "queued" else "Building preprocessing cache"
            return _activity_status(
                label,
                "The waveform viewer remains available while this runs.",
            ), True, False
        return _activity_status(
            "Preprocessing failed",
            str(status.get("message") or "The background job is no longer available."),
            "error",
        ), False, True

    @app.callback(
        Output("detector-operation", "data"),
        Input("search-button", "n_clicks"),
        State("period-select", "value"),
        State("channel-select", "value"),
        State("range-start", "value"),
        State("range-end", "value"),
        State("preprocess-options", "value"),
        State("bandpass-low", "value"),
        State("bandpass-high", "value"),
        State("detector-select", "value"),
        State("trigger-channel-select", "value"),
        State("sta-window", "value"),
        State("lta-window", "value"),
        State("trigger-on", "value"),
        State("trigger-off", "value"),
        State("p-threshold", "value"),
        State("s-threshold", "value"),
        prevent_initial_call=True,
    )
    def search(
        _: int,
        period_id: int | None,
        channels: list[str] | None,
        start_value: str | None,
        end_value: str | None,
        options: list[str] | None,
        low: float | None,
        high: float | None,
        detector_name: str,
        trigger_channel: str | None,
        sta: float,
        lta: float,
        trigger_on: float,
        trigger_off: float,
        p_threshold: float,
        s_threshold: float,
    ):
        try:
            if not period_id or not channels:
                raise ValueError("Choose a station period and channel selection")
            if _parse_time(start_value) >= _parse_time(end_value):
                raise ValueError("Search start must precede search end")
            recipe = _recipe_from_controls("fill" in (options or []), "denoise" in (options or []), low, high)
            parameters = (
                StaLtaParameters(sta_seconds=sta, lta_seconds=lta, trigger_on=trigger_on, trigger_off=trigger_off)
                if detector_name == "stalta"
                else PhaseNetParameters(p_threshold=p_threshold, s_threshold=s_threshold)
            )
            selected_channels = [trigger_channel or channels[0]] if detector_name == "stalta" else []
            if detector_name == "phasenet":
                selected_channels = [channel for channel in channels if channel.upper() in {"ENZ", "ENN", "ENE"}]
            run_id, _future = analysis.queue_detection(
                period_id,
                selected_channels,
                _parse_time(start_value),
                _parse_time(end_value),
                recipe,
                detector_name,
                parameters,
            )
            return {"status": "submitted", "run_id": run_id}
        except (ValidationError, ValueError) as error:
            return {"status": "failed", "message": f"Cannot start search: {error}"}

    @app.callback(
        Output("search-status", "children"),
        Output("search-controls", "disabled"),
        Output("detector-refresh", "disabled"),
        Input("detector-operation", "data"),
        Input("detector-refresh", "n_intervals"),
    )
    def monitor_detection(operation: dict[str, object] | None, _: int):
        if not operation:
            return "", False, True
        if operation.get("status") == "failed":
            return _activity_status(
                "Detector search could not start",
                str(operation.get("message") or "Unknown error"),
                "error",
            ), False, True
        run_id = int(operation.get("run_id") or 0)
        status = analysis.detection_status(run_id)
        state = str(status["status"])
        detector_name = str(status.get("detector_name") or "Detector")
        if state == "complete":
            candidate_count = int(status.get("candidate_count") or 0)
            detail = f"Found {candidate_count:,} candidate{'s' if candidate_count != 1 else ''}."
            if status.get("message"):
                detail = f"{detail} {status['message']}"
            return _activity_status(
                f"{detector_name} run {run_id} complete",
                detail,
                "success",
            ), False, True
        if state in {"queued", "running"}:
            label = (
                f"{detector_name} run {run_id} queued"
                if state == "queued"
                else f"{detector_name} run {run_id} searching for candidates"
            )
            return _activity_status(
                label,
                "Detected candidates will appear automatically.",
            ), True, False
        return _activity_status(
            f"{detector_name} run {run_id} failed",
            str(status.get("message") or "The detector run is no longer available."),
            "error",
        ), False, True

    @app.callback(
        Output("candidate-select", "options"),
        Output("candidate-select", "value"),
        Input("refresh", "n_intervals"),
        Input("period-select", "value"),
        State("candidate-select", "value"),
    )
    def refresh_candidates(_: int, period_id: int | None, current: int | None):
        if not period_id:
            return [], None
        candidates = analysis.candidates_for_period(period_id)
        options = [
            {
                "label": f"{candidate.timestamp:%Y-%m-%d %H:%M:%S} — {candidate.kind} {candidate.phase or ''} ({candidate.score:.2f})",
                "value": candidate.id,
            }
            for candidate in candidates
        ]
        valid = {option["value"] for option in options}
        if current in valid:
            return options, no_update
        return options, None

    @app.callback(Output("review-controls", "disabled"), Output("review-target", "children"),
                  Output("assessment-verdict", "value"), Output("assessment-notes", "value"),
                  Input("candidate-select", "value"), Input("period-select", "value"))
    def select_review(candidate_id, period_id):
        from .models import DetectionCandidate
        with sessions() as session:
            candidate = session.get(DetectionCandidate, candidate_id) if candidate_id else None
            if candidate is None or candidate.run.period_id != period_id:
                return True, "Select a detected candidate.", "unreviewed", ""
            review = candidate.assessment
            return False, f"Candidate {candidate.id} · {candidate.timestamp:%H:%M:%S} UTC", review.verdict if review else "unreviewed", review.notes if review else ""

    @app.callback(
        Output("assessment-status", "children"),
        Input("save-assessment", "n_clicks"),
        State("candidate-select", "value"),
        State("assessment-verdict", "value"),
        State("assessment-notes", "value"),
        prevent_initial_call=True,
    )
    def save_assessment(_: int, candidate_id: int | None, verdict: str | None, notes: str | None):
        if not candidate_id or not verdict:
            return "Choose a candidate and verdict first."
        try:
            analysis.save_assessment(candidate_id, AssessmentPayload(verdict=verdict, notes=notes or ""))
            return "Assessment saved."
        except ValidationError as error:
            return str(error)

    return server, app
