"""FastAPI endpoints and Dash review interface."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html, no_update
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

    server = FastAPI(title="ShakePi MiniSEED Viewer")

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
    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div([html.H1("ShakePi MiniSEED Viewer"), html.P("Shared retrospective detector calibration")]),
                    html.Div(
                        [
                            dcc.Upload(
                                id="upload-files",
                                children=html.Div(
                                    [
                                        html.Strong("Drop MiniSEED files here"),
                                        html.Span(" or click to choose files"),
                                    ]
                                ),
                                className="upload-dropzone",
                                className_active="upload-dropzone-active",
                                multiple=True,
                            ),
                            dcc.Loading(
                                id="upload-loading",
                                children=html.Div(
                                    id="upload-status",
                                    className="subtle",
                                    role="status",
                                    **{"aria-live": "polite"},
                                ),
                                custom_spinner=_activity_status(
                                    "Uploading and validating MiniSEED files"
                                ),
                                delay_show=150,
                                delay_hide=200,
                                target_components={"upload-status": "children"},
                            ),
                        ],
                        className="upload-area",
                    ),
                ],
                className="topbar",
            ),
            dcc.Interval(id="refresh", interval=3_000, n_intervals=0),
            dcc.Store(id="preprocess-operation", storage_type="session"),
            dcc.Store(id="detector-operation", storage_type="session"),
            dcc.Interval(
                id="preprocess-refresh",
                interval=1_000,
                n_intervals=0,
                disabled=True,
            ),
            dcc.Interval(
                id="detector-refresh",
                interval=1_000,
                n_intervals=0,
                disabled=True,
            ),
            dcc.Store(id="viewport"),
            html.Main(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Label("Station period"),
                                    dcc.Dropdown(id="period-select", placeholder="Upload MiniSEED files to begin"),
                                ],
                                className="selector-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Channels (up to four)"),
                                    dcc.Dropdown(id="channel-select", multi=True),
                                ],
                                className="selector-control",
                            ),
                            html.Div(id="period-status", className="subtle"),
                        ],
                        className="panel selector-panel",
                    ),
                    html.Section(
                        [
                            html.H2("Preprocessing and search region"),
                            html.Fieldset(
                                [
                                    html.Div(
                                        dcc.Checklist(
                                            id="preprocess-options",
                                            options=[
                                                {"label": "Fill gaps up to 1 second", "value": "fill"},
                                                {"label": "DeepDenoiser (100 Hz per channel)", "value": "denoise"},
                                            ],
                                        ),
                                        className="preprocess-toggles",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Bandpass low Hz"),
                                            dcc.Input(id="bandpass-low", type="number", min=0.01, step=0.1),
                                        ],
                                        className="control-field",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Bandpass high Hz"),
                                            dcc.Input(id="bandpass-high", type="number", min=0.02, step=0.1),
                                        ],
                                        className="control-field",
                                    ),
                                    html.Button("Apply preprocessing", id="apply-preprocess", className="secondary-button"),
                                    html.Div(id="preprocess-status", className="subtle"),
                                ],
                                id="preprocess-controls",
                                className="preprocess-grid",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Search start (UTC ISO 8601)"),
                                            dcc.Input(id="range-start", type="text", debounce=True),
                                        ],
                                        className="control-field",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Search end (UTC ISO 8601)"),
                                            dcc.Input(id="range-end", type="text", debounce=True),
                                        ],
                                        className="control-field",
                                    ),
                                    html.Button("Use visible range", id="use-visible", className="secondary-button"),
                                ],
                                id="overview-range-controls",
                                className="range-grid",
                            ),
                            dcc.Loading(
                                id="overview-loading",
                                children=dcc.Graph(
                                    id="overview-graph",
                                    config={"displaylogo": False},
                                    className="waveform-graph",
                                ),
                                custom_spinner=_graph_skeleton(
                                    "Preparing waveform and spectrogram"
                                ),
                                delay_show=150,
                                delay_hide=200,
                                target_components={"overview-graph": "figure"},
                                parent_className="graph-loading-shell",
                            ),
                            html.Div(
                                id="overview-status",
                                className="subtle",
                                role="status",
                                **{"aria-live": "polite"},
                            ),
                        ],
                        id="overview-region",
                        className="panel",
                    ),
                    html.Section(
                        [
                            html.H2("Trigger search"),
                            html.Fieldset(
                                [
                                    dcc.Dropdown(
                                        id="detector-select",
                                        options=[
                                            {"label": "STA/LTA (single channel)", "value": "stalta"},
                                            {"label": "PhaseNet (MEMS Z/N/E)", "value": "phasenet"},
                                        ],
                                        value="stalta",
                                        clearable=False,
                                    ),
                                    dcc.Dropdown(id="trigger-channel-select", placeholder="STA/LTA channel"),
                                    dcc.Input(id="sta-window", type="number", value=1.0, min=0.01, step=0.1),
                                    dcc.Input(id="lta-window", type="number", value=20.0, min=0.1, step=0.5),
                                    dcc.Input(id="trigger-on", type="number", value=3.5, min=0.1, step=0.1),
                                    dcc.Input(id="trigger-off", type="number", value=1.75, min=0.1, step=0.1),
                                    dcc.Input(id="p-threshold", type="number", value=0.3, min=0.01, max=1, step=0.05),
                                    dcc.Input(id="s-threshold", type="number", value=0.3, min=0.01, max=1, step=0.05),
                                    html.Button("Search", id="search-button", className="search-button"),
                                ],
                                id="search-controls",
                                className="controls-grid",
                            ),
                            html.P(
                                "PhaseNet is enabled only for complete time-aligned MEMS Z/N/E channels. STA/LTA defaults to the geophone.",
                                className="subtle",
                            ),
                            html.Div(
                                id="search-status",
                                role="status",
                                **{"aria-live": "polite"},
                            ),
                        ],
                        className="panel",
                    ),
                    html.Section(
                        [
                            html.Aside(
                                [
                                    html.H2("Detected candidates"),
                                    html.Div(
                                        dcc.RadioItems(
                                            id="candidate-select",
                                            className="candidate-list",
                                        ),
                                        id="candidate-list-region",
                                    ),
                                    html.Label("Assessment"),
                                    dcc.Dropdown(
                                        id="assessment-verdict",
                                        options=[
                                            {"label": "Earthquake", "value": "quake"},
                                            {"label": "Noise", "value": "noise"},
                                            {"label": "Uncertain", "value": "uncertain"},
                                        ],
                                    ),
                                    dcc.Textarea(id="assessment-notes", placeholder="Review notes"),
                                    html.Button("Save assessment", id="save-assessment", className="secondary-button"),
                                    html.Div(id="assessment-status", className="subtle"),
                                ],
                                className="results-sidebar",
                            ),
                            html.Div(
                                [
                                    html.H2("Candidate detail"),
                                    html.Fieldset(
                                        [
                                            html.Label("Left padding s"),
                                            dcc.Input(id="detail-left", type="number", value=30, min=0),
                                            html.Label("Right padding s"),
                                            dcc.Input(id="detail-right", type="number", value=90, min=0),
                                        ],
                                        id="detail-controls",
                                        className="controls-grid small-controls",
                                    ),
                                    dcc.Loading(
                                        id="detail-loading",
                                        children=dcc.Graph(
                                            id="detail-graph",
                                            config={"displaylogo": False},
                                            className="waveform-graph",
                                        ),
                                        custom_spinner=_graph_skeleton(
                                            "Preparing candidate waveform"
                                        ),
                                        delay_show=150,
                                        delay_hide=200,
                                        target_components={"detail-graph": "figure"},
                                        parent_className="graph-loading-shell",
                                    ),
                                    html.Div(
                                        id="detail-status",
                                        className="subtle",
                                        role="status",
                                        **{"aria-live": "polite"},
                                    ),
                                ],
                                id="detail-region",
                                className="results-detail",
                            ),
                        ],
                        className="results-layout panel",
                    ),
                ]
            ),
        ]
    )

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
        State("period-select", "value"),
    )
    def refresh_periods(_: int, current: int | None):
        choices = archive.list_periods()
        options = [
            {"label": f"{item['label']}  [{', '.join(item['channels'])}]", "value": item["id"]}
            for item in choices
        ]
        values = {item["id"] for item in choices}
        if current in values:
            return options, no_update
        return options, options[0]["value"] if options else None

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
        return (
            [{"label": channel, "value": channel} for channel in channels],
            selected,
            f"Available channels: {', '.join(channels)}. Raw amplitudes are uncorrected counts.",
            start.isoformat(),
            end.isoformat(),
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

    @app.callback(
        Output("viewport", "data"),
        Input("overview-graph", "relayoutData"),
        State("viewport", "data"),
        prevent_initial_call=True,
    )
    def capture_viewport(relayout: dict[str, object] | None, existing: dict[str, object] | None):
        if not relayout:
            return existing
        left = relayout.get("xaxis.range[0]")
        right = relayout.get("xaxis.range[1]")
        if left is not None and right is not None:
            return {"start": left, "end": right}
        return existing

    @app.callback(
        Output("range-start", "value", allow_duplicate=True),
        Output("range-end", "value", allow_duplicate=True),
        Input("use-visible", "n_clicks"),
        State("viewport", "data"),
        prevent_initial_call=True,
    )
    def use_visible(_: int, viewport: dict[str, object] | None):
        if not viewport:
            return "", ""
        def as_iso(value: object) -> str:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value, tz=UTC).isoformat()
            return str(value)

        return as_iso(viewport.get("start", "")), as_iso(viewport.get("end", ""))

    @app.callback(
        Output("overview-graph", "figure"),
        Output("overview-status", "children"),
        Input("period-select", "value"),
        Input("channel-select", "value"),
        Input("range-start", "value"),
        Input("range-end", "value"),
        Input("preprocess-options", "value"),
        Input("bandpass-low", "value"),
        Input("bandpass-high", "value"),
        running=[
            (
                Output("overview-range-controls", "className"),
                "range-grid controls-disabled",
                "range-grid",
            ),
            (Output("overview-region", "aria-busy"), "true", "false"),
        ],
    )
    def render_overview(
        period_id: int | None,
        channels: list[str] | None,
        start_value: str | None,
        end_value: str | None,
        options: list[str] | None,
        low: float | None,
        high: float | None,
    ):
        if not period_id or not channels or not start_value or not end_value:
            return (
                waveform_spectrogram_figure(Stream(), title="Select a station period"),
                "No waveform range selected.",
            )
        try:
            start, end = _parse_time(start_value), _parse_time(end_value)
            recipe = _recipe_from_controls("fill" in (options or []), "denoise" in (options or []), low, high)
            stream = analysis.prepared_stream(period_id, channels, (start, end), recipe)
            point_count = sum(len(trace.data) for trace in stream)
            plotted_channels = ", ".join(trace.id for trace in stream) or "none"
            return (
                waveform_spectrogram_figure(stream, (start, end)),
                (
                    f"Plotted {point_count:,} samples from {plotted_channels} "
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
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("detail-graph", "figure"),
        Output("detail-status", "children"),
        Input("candidate-select", "value"),
        Input("detail-left", "value"),
        Input("detail-right", "value"),
        State("channel-select", "value"),
        running=[
            (Output("detail-controls", "disabled"), True, False),
            (
                Output("candidate-list-region", "className"),
                "controls-disabled",
                "",
            ),
            (Output("detail-region", "aria-busy"), "true", "false"),
        ],
    )
    def render_detail(candidate_id: int | None, left: float | None, right: float | None, channels: list[str] | None):
        if not candidate_id or not channels:
            return (
                waveform_spectrogram_figure(Stream(), title="Select a detected candidate"),
                "Select a detected candidate to inspect its waveform.",
            )
        try:
            with sessions() as session:
                from .models import DetectionCandidate

                candidate = session.get(DetectionCandidate, candidate_id)
                if candidate is None:
                    return (
                        waveform_spectrogram_figure(Stream(), title="Candidate no longer exists"),
                        "Candidate no longer exists.",
                    )
                run = candidate.run
                start = candidate.timestamp - timedelta(seconds=left or 30)
                end = (candidate.end_time or candidate.timestamp) + timedelta(seconds=right or 90)
                recipe = (
                    PreprocessingRecipe.model_validate_json(run.recipe.canonical_json)
                    if run.recipe
                    else PreprocessingRecipe()
                )
                stream = analysis.prepared_stream(run.period_id, channels, (start, end), recipe)
                point_count = sum(len(trace.data) for trace in stream)
                figure = waveform_spectrogram_figure(
                    stream,
                    (candidate.timestamp, candidate.end_time or candidate.timestamp),
                    "Candidate detail",
                )
                if candidate.phase:
                    figure.add_vline(
                        x=candidate.timestamp,
                        line_color="#fbbf24",
                        annotation_text=f"{candidate.phase} pick",
                    )
                return (
                    figure,
                    (
                        f"Plotted {point_count:,} samples around candidate {candidate_id} "
                        f"between {start.isoformat()} and {end.isoformat()}."
                    ),
                )
        except Exception as error:
            return (
                waveform_spectrogram_figure(Stream(), title=f"Unable to render candidate: {error}"),
                f"Unable to render candidate: {error}",
            )

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
