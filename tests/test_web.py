from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from obspy import Stream, Trace, UTCDateTime

from conftest import write_mseed
from shakepi.config import Settings
from shakepi.services import AnalysisService
from shakepi.web import _ingest_dash_upload, create_application

UTC = timezone.utc


def _component_with_id(component, component_id: str):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None:
            found = _component_with_id(child, component_id)
            if found is not None:
                return found
    return None


def _dash_callback(
    client: TestClient,
    output: str,
    inputs: list[dict[str, object]],
    state: list[dict[str, object]] | None = None,
):
    component_id, property_name = output.split(".", maxsplit=1)
    return client.post(
        "/_dash-update-component",
        json={
            "output": output,
            "outputs": {"id": component_id, "property": property_name},
            "inputs": inputs,
            "state": state or [],
            "changedPropIds": [f"{item['id']}.{item['property']}" for item in inputs],
        },
    )


def _dash_multi_callback(
    client: TestClient,
    output: str,
    outputs: list[dict[str, str]],
    inputs: list[dict[str, object]],
    state: list[dict[str, object]] | None = None,
):
    return client.post(
        "/_dash-update-component",
        json={
            "output": output,
            "outputs": outputs,
            "inputs": inputs,
            "state": state or [],
            "changedPropIds": [f"{item['id']}.{item['property']}" for item in inputs],
        },
    )


def _plotly_values(values: list[object] | dict[str, str]) -> np.ndarray:
    if isinstance(values, dict):
        dtype = np.dtype(values["dtype"])
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=dtype)
    return np.asarray(values)


def test_upload_dropzone_allows_sds_filenames(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    _, app = create_application(settings)

    upload = _component_with_id(app.layout, "upload-files")

    assert upload is not None
    assert upload.multiple is True
    assert getattr(upload, "accept", None) is None


def test_layout_has_scoped_accessible_loading_feedback(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    _, app = create_application(settings)

    upload_loading = _component_with_id(app.layout, "upload-loading")
    overview_loading = _component_with_id(app.layout, "overview-loading")
    detail_loading = _component_with_id(app.layout, "detail-loading")
    preprocess_refresh = _component_with_id(app.layout, "preprocess-refresh")
    detector_refresh = _component_with_id(app.layout, "detector-refresh")
    preprocess_operation = _component_with_id(app.layout, "preprocess-operation")
    detector_operation = _component_with_id(app.layout, "detector-operation")

    assert upload_loading.target_components == {"upload-status": "children"}
    assert overview_loading.target_components == {"overview-graph": "figure"}
    assert overview_loading.custom_spinner.role == "status"
    assert detail_loading.target_components == {"detail-graph": "figure"}
    assert detail_loading.custom_spinner.role == "status"
    assert preprocess_refresh.disabled is True
    assert detector_refresh.disabled is True
    assert preprocess_operation.storage_type == "session"
    assert detector_operation.storage_type == "session"


def test_dash_upload_ingests_sds_filename(app_context, tmp_path: Path) -> None:
    _, _, archive = app_context
    source = tmp_path / "source"
    write_mseed(source)
    encoded = base64.b64encode(source.read_bytes()).decode()
    contents = "data:application/octet-stream;base64," + encoded

    message = _ingest_dash_upload(archive, contents, "AM.RF4E8.00.EHZ.D.2026.033")

    assert message == "MiniSEED file stored in the immutable SDS archive."


def test_dash_upload_rejects_invalid_base64(app_context) -> None:
    _, _, archive = app_context

    with pytest.raises(ValueError, match="could not be decoded"):
        _ingest_dash_upload(
            archive,
            "data:application/octet-stream;base64,not base64",
            "AM.RF4E8.00.EHZ.D.2026.033",
        )


def test_period_selection_defaults_to_actual_trace_bounds(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    server, _app = create_application(settings)
    trace_start = datetime(2026, 2, 2, 3, 4, 5, tzinfo=UTC)
    source = tmp_path / "source"
    Stream(
        [
            Trace(
                data=np.arange(1_000, dtype=np.int32),
                header={
                    "network": "AM",
                    "station": "RF4E8",
                    "location": "00",
                    "channel": "EHZ",
                    "starttime": UTCDateTime(trace_start),
                    "sampling_rate": 100.0,
                },
            )
        ]
    ).write(str(source), format="MSEED")

    with TestClient(server) as client:
        upload = client.post(
            "/api/uploads",
            files={
                "files": (
                    "AM.RF4E8.00.EHZ.D.2026.033",
                    source.read_bytes(),
                    "application/octet-stream",
                )
            },
        )
        assert upload.status_code == 200
        period_id = upload.json()[0]["period_id"]

        response = client.post(
            "/_dash-update-component",
            json={
                "output": (
                    "..channel-select.options...channel-select.value...period-status.children"
                    "...range-start.value...range-end.value.."
                ),
                "outputs": [
                    {"id": "channel-select", "property": "options"},
                    {"id": "channel-select", "property": "value"},
                    {"id": "period-status", "property": "children"},
                    {"id": "range-start", "property": "value"},
                    {"id": "range-end", "property": "value"},
                ],
                "inputs": [{"id": "period-select", "property": "value", "value": period_id}],
                "state": [{"id": "channel-select", "property": "value", "value": None}],
                "changedPropIds": ["period-select.value"],
            },
        )

    assert response.status_code == 200
    payload = response.json()["response"]
    assert payload["range-start"]["value"] == "2026-02-02T03:04:05+00:00"
    assert payload["range-end"]["value"] == "2026-02-02T03:04:14.990000+00:00"


def test_period_refresh_does_not_reemit_unchanged_selection(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    server, _app = create_application(settings)
    source = tmp_path / "source"
    write_mseed(source)

    with TestClient(server) as client:
        upload = client.post(
            "/api/uploads",
            files={
                "files": (
                    "AM.RF4E8.00.EHZ.D.2026.033",
                    source.read_bytes(),
                    "application/octet-stream",
                )
            },
        )
        assert upload.status_code == 200
        period_id = upload.json()[0]["period_id"]

        response = client.post(
            "/_dash-update-component",
            json={
                "output": "..period-select.options...period-select.value..",
                "outputs": [
                    {"id": "period-select", "property": "options"},
                    {"id": "period-select", "property": "value"},
                ],
                "inputs": [{"id": "refresh", "property": "n_intervals", "value": 1}],
                "state": [{"id": "period-select", "property": "value", "value": period_id}],
                "changedPropIds": ["refresh.n_intervals"],
            },
        )

    assert response.status_code == 200
    payload = response.json()["response"]
    assert payload["period-select"]["options"]
    assert "value" not in payload["period-select"]


def test_overview_graph_dash_endpoint_contains_waveform_points(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    server, _app = create_application(settings)
    source = tmp_path / "source"
    samples = np.round(np.sin(np.linspace(0, 40, 5_000)) * 1_000).astype(np.int32)
    write_mseed(source, data=samples)

    with TestClient(server) as client:
        upload = client.post(
            "/api/uploads",
            files={
                "files": (
                    "AM.RF4E8.00.EHZ.D.2026.033",
                    source.read_bytes(),
                    "application/octet-stream",
                )
            },
        )
        assert upload.status_code == 200
        period_id = upload.json()[0]["period_id"]
        periods = client.get("/api/periods").json()
        assert periods[0]["channels"] == ["EHZ"]

        response = _dash_multi_callback(
            client,
            "..overview-graph.figure...overview-status.children..",
            [
                {"id": "overview-graph", "property": "figure"},
                {"id": "overview-status", "property": "children"},
            ],
            [
                {"id": "period-select", "property": "value", "value": period_id},
                {"id": "channel-select", "property": "value", "value": ["EHZ"]},
                {"id": "range-start", "property": "value", "value": "2026-02-02T00:00:00+00:00"},
                {
                    "id": "range-end",
                    "property": "value",
                    "value": "2026-02-02T00:00:49.990000+00:00",
                },
                {"id": "preprocess-options", "property": "value", "value": []},
                {"id": "bandpass-low", "property": "value", "value": None},
                {"id": "bandpass-high", "property": "value", "value": None},
            ],
        )

    assert response.status_code == 200
    payload = response.json()["response"]
    figure = payload["overview-graph"]["figure"]
    status = payload["overview-status"]["children"]
    waveform = figure["data"][0]
    x_values = _plotly_values(waveform["x"])
    y_values = _plotly_values(waveform["y"])
    assert waveform["type"] == "scatter"
    assert waveform["mode"] == "lines"
    assert len(x_values) == len(y_values) > 0
    assert x_values[0] == "2026-02-02T00:00:00+00:00"
    assert np.ptp(y_values) > 0
    assert "Plotted 5,000 samples" in status


@pytest.mark.parametrize(
    ("job_status", "expected_text", "controls_disabled", "polling_disabled"),
    [
        ({"status": "queued", "cache_key": "abc"}, "Preprocessing queued", True, False),
        ({"status": "running", "cache_key": "abc"}, "Building preprocessing cache", True, False),
        ({"status": "complete", "cache_key": "abc"}, "Preprocessing cache ready", False, True),
        (
            {"status": "failed", "cache_key": "abc", "message": "filter failed"},
            "Preprocessing failed",
            False,
            True,
        ),
    ],
)
def test_preprocessing_monitor_reports_job_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_status: dict[str, object],
    expected_text: str,
    controls_disabled: bool,
    polling_disabled: bool,
) -> None:
    monkeypatch.setattr(
        AnalysisService,
        "preprocessing_status",
        lambda _self, _cache_key: job_status,
    )
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    server, _app = create_application(settings)

    with TestClient(server) as client:
        response = _dash_multi_callback(
            client,
            "..preprocess-status.children...preprocess-controls.disabled...preprocess-refresh.disabled..",
            [
                {"id": "preprocess-status", "property": "children"},
                {"id": "preprocess-controls", "property": "disabled"},
                {"id": "preprocess-refresh", "property": "disabled"},
            ],
            [
                {
                    "id": "preprocess-operation",
                    "property": "data",
                    "value": {"status": "submitted", "cache_key": "abc"},
                },
                {"id": "preprocess-refresh", "property": "n_intervals", "value": 1},
            ],
        )

    assert response.status_code == 200
    payload = response.json()["response"]
    assert expected_text in str(payload["preprocess-status"]["children"])
    assert payload["preprocess-controls"]["disabled"] is controls_disabled
    assert payload["preprocess-refresh"]["disabled"] is polling_disabled


@pytest.mark.parametrize(
    ("job_status", "expected_text", "controls_disabled", "polling_disabled"),
    [
        (
            {"status": "queued", "run_id": 12, "detector_name": "stalta"},
            "stalta run 12 queued",
            True,
            False,
        ),
        (
            {"status": "running", "run_id": 12, "detector_name": "stalta"},
            "searching for candidates",
            True,
            False,
        ),
        (
            {
                "status": "complete",
                "run_id": 12,
                "detector_name": "stalta",
                "candidate_count": 2,
            },
            "Found 2 candidates",
            False,
            True,
        ),
        (
            {
                "status": "failed",
                "run_id": 12,
                "detector_name": "stalta",
                "message": "detector failed",
            },
            "detector failed",
            False,
            True,
        ),
    ],
)
def test_detector_monitor_reports_persisted_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_status: dict[str, object],
    expected_text: str,
    controls_disabled: bool,
    polling_disabled: bool,
) -> None:
    monkeypatch.setattr(
        AnalysisService,
        "detection_status",
        lambda _self, _run_id: job_status,
    )
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    server, _app = create_application(settings)

    with TestClient(server) as client:
        response = _dash_multi_callback(
            client,
            "..search-status.children...search-controls.disabled...detector-refresh.disabled..",
            [
                {"id": "search-status", "property": "children"},
                {"id": "search-controls", "property": "disabled"},
                {"id": "detector-refresh", "property": "disabled"},
            ],
            [
                {
                    "id": "detector-operation",
                    "property": "data",
                    "value": {"status": "submitted", "run_id": 12},
                },
                {"id": "detector-refresh", "property": "n_intervals", "value": 1},
            ],
        )

    assert response.status_code == 200
    payload = response.json()["response"]
    assert expected_text in str(payload["search-status"]["children"])
    assert payload["search-controls"]["disabled"] is controls_disabled
    assert payload["detector-refresh"]["disabled"] is polling_disabled


def test_candidate_refresh_does_not_reemit_current_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = SimpleNamespace(
        id=7,
        timestamp=datetime(2026, 2, 2, 1, 2, 3, tzinfo=UTC),
        kind="trigger",
        phase=None,
        score=0.91,
    )
    monkeypatch.setattr(
        AnalysisService,
        "candidates_for_period",
        lambda _self, _period_id: [candidate],
    )
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    server, _app = create_application(settings)

    with TestClient(server) as client:
        response = _dash_multi_callback(
            client,
            "..candidate-select.options...candidate-select.value..",
            [
                {"id": "candidate-select", "property": "options"},
                {"id": "candidate-select", "property": "value"},
            ],
            [
                {"id": "refresh", "property": "n_intervals", "value": 1},
                {"id": "period-select", "property": "value", "value": 3},
            ],
            [{"id": "candidate-select", "property": "value", "value": 7}],
        )

    assert response.status_code == 200
    payload = response.json()["response"]
    assert payload["candidate-select"]["options"][0]["value"] == 7
    assert "value" not in payload["candidate-select"]
