"""Compact workbench layout; expensive views are loaded explicitly."""
from dash import dcc, html
from obspy import Stream

from .plotting import waveform_spectrogram_figure


def layout():
    def status(id):
        return html.Div(id=id, className="subtle", role="status", **{"aria-live": "polite"})

    def field(label, id, value=None, **kwargs):
        return html.Label([label, dcc.Input(id=id, value=value, **kwargs)], className="control-field")

    def button(label, id, primary=False, **kwargs):
        return html.Button(label, id=id, className="search-button" if primary else "secondary-button", **kwargs)

    empty = waveform_spectrogram_figure(Stream())
    return html.Div([
        html.Header([html.Div([html.H1("ShakePi MiniSEED Viewer"), html.P("Seismic workbench · Local recordings · UTC")])], className="topbar"),
        dcc.Interval(id="refresh", interval=15_000),
        *[dcc.Store(id=id) for id in ["preprocess-operation", "detector-operation", "day-operation", "reset-view-result", "day-sync"]],
        *[dcc.Interval(id=id, interval=1000, disabled=True) for id in ["preprocess-refresh", "detector-refresh", "day-refresh"]],
        html.Div([
            html.Aside([
                html.H2("Recordings"),
                dcc.Upload(id="upload-files", children=html.Div([html.Strong("Upload MiniSEED"), html.Span(" · drop files here")]), multiple=True, className="upload-dropzone", className_active="upload-dropzone-active"),
                dcc.Loading(status("upload-status"), id="upload-loading", target_components={"upload-status": "children"}),
                html.P("Uploads add recordings without opening them.", className="subtle"),
                dcc.Dropdown(id="period-select", placeholder="Choose a recording"),
                status("period-status"), html.Hr(), html.H2("Detected candidates"),
                html.P("Select a candidate to inspect it in the interval detail view.", className="subtle"),
                html.Div(dcc.RadioItems(id="candidate-select", className="candidate-list"), id="candidate-list-region"),
            ], className="workbench-library"),
            html.Main([
                html.Section([
                    html.Div([html.H2("Day overview"), button("Load day overview", "load-day")], className="workbench-toolbar"),
                    status("day-status"),
                    dcc.Graph(id="day-graph", figure=empty, config={"displaylogo": False, "edits": {"shapePosition": True}}, className="day-graph"),
                    html.P("Drag the highlighted region or either edge to move or resize the search interval. Zoom the day graph for fine adjustments, or enter exact UTC times on the right. Overview uses reduced raw counts; detection uses full-resolution data.", className="subtle"),
                ], className="panel"),
                html.Section([
                    html.H2("Interval detail"),
                    html.Label(["Channels (up to four)", dcc.Dropdown(id="channel-select", multi=True)]),
                    html.Div([
                        button("Load detail", "load-overview", True),
                        dcc.Checklist(id="show-spectrogram", options=[{"label": "Include spectrogram on Load detail", "value": "show"}], value=[]),
                        button("Use visible range", "use-visible", disabled=True),
                        button("Reset view", "reset-view", disabled=True),
                    ], id="overview-range-controls", className="workbench-toolbar"),
                    status("visible-range-status"),
                    dcc.Loading(dcc.Graph(id="overview-graph", figure=empty, config={"displaylogo": False}, className="waveform-graph"), id="overview-loading", target_components={"overview-graph": "figure"}, parent_className="graph-loading-shell"),
                    status("overview-status"),
                    html.P("Change channels or preprocessing, then Load detail. Zooming does not change the search interval until Use visible range is selected.", className="subtle"),
                ], id="overview-region", className="panel"),
            ], className="workbench-main"),
            html.Aside([
                html.Section([
                    html.H2("Search interval"),
                    field("Start (UTC ISO 8601)", "range-start", type="text", debounce=True),
                    field("End (UTC ISO 8601)", "range-end", type="text", debounce=True),
                    html.P("Detection analyzes this interval, independently of graph zoom.", className="subtle"),
                ], className="panel"),
                html.Details([
                    html.Summary("Preprocessing"),
                    html.Fieldset([
                        dcc.Checklist(id="preprocess-options", options=[{"label": "Fill gaps up to 1 second", "value": "fill"}, {"label": "DeepDenoiser (100 Hz per channel)", "value": "denoise"}]),
                        field("Bandpass low Hz", "bandpass-low", type="number", min=0.01),
                        field("Bandpass high Hz", "bandpass-high", type="number", min=0.02),
                        button("Build day cache", "apply-preprocess"),
                        html.P("Optional background preparation for the whole selected day. Load detail previews settings on just the requested interval.", className="subtle"),
                        status("preprocess-status"),
                    ], id="preprocess-controls"),
                ], className="panel"),
                html.Section([
                    html.H2("Detector"),
                    html.Fieldset([
                        dcc.Dropdown(id="detector-select", options=[{"label": "STA/LTA", "value": "stalta"}], value="stalta", clearable=False),
                        html.Label(["STA/LTA channel", dcc.Dropdown(id="trigger-channel-select")]),
                        html.Details([html.Summary("Detector parameters"),
                            field("STA window (s)", "sta-window", 1.0, type="number", min=0.01),
                            field("LTA window (s)", "lta-window", 20.0, type="number", min=0.1),
                            field("Trigger on", "trigger-on", 3.5, type="number"),
                            field("Trigger off", "trigger-off", 1.75, type="number"),
                            field("P threshold", "p-threshold", 0.3, type="number"),
                            field("S threshold", "s-threshold", 0.3, type="number"),
                        ]), button("Run detection", "search-button", True),
                    ], id="search-controls"),
                    html.P("PhaseNet requires ENZ, ENN and ENE. STA/LTA uses one channel.", className="subtle"), status("search-status"),
                ], className="panel"),
                html.Section([
                    html.H2("Candidate review"), status("review-target"),
                    html.Fieldset([
                        html.Label(["Assessment", dcc.Dropdown(id="assessment-verdict", options=[{"label": label, "value": value} for label, value in [("Unreviewed", "unreviewed"), ("Earthquake", "quake"), ("Noise", "noise"), ("Uncertain", "uncertain")]], value="unreviewed", clearable=False)]),
                        dcc.Textarea(id="assessment-notes", placeholder="Review notes"),
                        button("Save review", "save-assessment"),
                    ], id="review-controls", disabled=True), status("assessment-status"),
                    html.P("Manual assessment, not a confirmed catalog event.", className="subtle"),
                    html.Details([html.Summary("Candidate display padding"), html.Fieldset([
                        field("Before (s)", "detail-left", 30, type="number", min=0),
                        field("After (s)", "detail-right", 90, type="number", min=0),
                    ], id="detail-controls")]),
                ], className="panel"),
            ], className="workbench-inspector"),
        ], className="workbench-grid"),
    ])
