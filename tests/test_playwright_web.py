from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pytest
import uvicorn

from conftest import write_mseed
from shakepi.archive import ArchiveService
from shakepi.config import Settings
from shakepi.services import AnalysisService
from shakepi.web import create_application

playwright = pytest.importorskip("playwright.sync_api")
Error = playwright.Error
sync_playwright = playwright.sync_playwright


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    original_ingest = ArchiveService.ingest_bytes
    original_prepared_stream = AnalysisService.prepared_stream

    def delayed_ingest(self, contents, filename):
        time.sleep(0.4)
        return original_ingest(self, contents, filename)

    def delayed_prepared_stream(self, period_id, channels, utc_range, recipe):
        time.sleep(0.4)
        return original_prepared_stream(self, period_id, channels, utc_range, recipe)

    monkeypatch.setattr(ArchiveService, "ingest_bytes", delayed_ingest)
    monkeypatch.setattr(AnalysisService, "prepared_stream", delayed_prepared_stream)
    settings = Settings(data_root=tmp_path / "data", allow_unsafe_sqlite=True)
    server, _dash = create_application(settings)
    port = _free_port()
    config = uvicorn.Config(
        server,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="off",
    )
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/api/health", timeout=0.25) as response:
                if response.status == 200:
                    break
        except (TimeoutError, URLError):
            time.sleep(0.05)
    else:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Test server did not start")

    try:
        yield base_url
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)


def _launch_chromium():
    manager = sync_playwright()
    started = manager.start()
    try:
        browser = started.chromium.launch()
    except Error as error:
        manager.__exit__(None, None, None)
        pytest.skip(f"Playwright Chromium is not installed: {error}")
    return manager, browser


def test_uploaded_waveform_renders_points_and_survives_refresh(
    live_app: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / "AM.RF4E8.00.EHZ.D.2026.033"
    samples = np.round(np.random.default_rng(42).normal(0, 8, 6_000)).astype(np.int32)
    samples[4_000:4_150] += 1_000
    write_mseed(source, data=samples)

    manager, browser = _launch_chromium()
    try:
        page = browser.new_page()
        page.goto(live_app, wait_until="domcontentloaded")
        page.get_by_role("heading", name="ShakePi MiniSEED Viewer").wait_for(timeout=10_000)
        page.set_input_files("input[type=file]", source)
        page.locator("#upload-loading .activity-status--busy").wait_for(
            state="visible",
            timeout=5_000,
        )
        assert page.locator("input[type=file]").is_disabled()
        page.get_by_text(
            "MiniSEED file stored in the immutable SDS archive."
        ).wait_for(timeout=10_000)
        page.locator("#overview-loading .graph-skeleton").wait_for(
            state="visible",
            timeout=10_000,
        )
        page.get_by_text("Plotted 6,000 samples").wait_for(timeout=10_000)
        page.locator("#overview-loading .graph-skeleton").wait_for(state="hidden")
        page.wait_for_function(
            """
            () => {
              const graph = document.querySelector("#overview-graph .js-plotly-plot");
              const trace = graph?._fullData?.find((item) => item.type === "scatter");
              return Boolean(trace && trace.x?.length > 0 && trace.y?.length > 0);
            }
            """
        )

        point_count = page.evaluate(
            """
            () => {
              const graph = document.querySelector("#overview-graph .js-plotly-plot");
              const trace = graph._fullData.find((item) => item.type === "scatter");
              return {x: trace.x.length, y: trace.y.length, title: document.title};
            }
            """
        )
        assert point_count["x"] == point_count["y"] > 0
        assert point_count["title"] == "ShakePi MiniSEED Viewer"

        page.get_by_role("button", name="Search", exact=True).click()
        page.locator("#search-status .activity-status--busy").wait_for(
            state="visible",
            timeout=10_000,
        )
        assert page.get_by_role("button", name="Search", exact=True).is_disabled()
        page.locator("#search-status .activity-status--success").wait_for(
            state="visible",
            timeout=15_000,
        )
        page.locator("#candidate-select input").first.wait_for(timeout=10_000)
        page.locator("#detail-loading .graph-skeleton").wait_for(
            state="visible",
            timeout=10_000,
        )
        page.get_by_text("around candidate").wait_for(timeout=10_000)
        page.locator("#detail-loading .graph-skeleton").wait_for(state="hidden")
        page.wait_for_function(
            """
            () => {
              const graph = document.querySelector("#detail-graph .js-plotly-plot");
              const trace = graph?._fullData?.find((item) => item.type === "scatter");
              return Boolean(trace && trace.x?.length > 0 && trace.y?.length > 0);
            }
            """
        )

        page.reload(wait_until="domcontentloaded")
        page.get_by_text("Plotted 6,000 samples").wait_for(timeout=10_000)
        page.wait_for_function("() => document.title === 'ShakePi MiniSEED Viewer'")
        assert page.title() == "ShakePi MiniSEED Viewer"
    finally:
        browser.close()
        manager.__exit__(None, None, None)
