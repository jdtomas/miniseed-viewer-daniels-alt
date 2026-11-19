# ShakePi MiniSEED Viewer

Shared, retrospective review of Raspberry Shake MiniSEED recordings. The app keeps raw files immutable, indexes the
SDS-style archive in SQLite, caches regenerated data products on disk, and compares STA/LTA with PhaseNet picks.

## Run locally

Production requires Python 3.12 linked against SQLite 3.51.3 or newer. For a development environment that has an
older SQLite build, set `SHAKEPI_ALLOW_UNSAFE_SQLITE=true` explicitly.

```bash
uv sync --extra dev
SHAKEPI_ALLOW_UNSAFE_SQLITE=true uv run shakepi-viewer
```

Open `http://localhost:8050`. Upload files named like `AM.RF4E8.00.EHZ.D.2026.033`. The server validates both the
name and MiniSEED headers, stores accepted files at `data/raw/{year}/{network}/{station}/{channel}.D/`, and quarantines
conflicting replacements rather than overwriting data.

## Workbench workflow

1. Upload MiniSEED files or choose an existing recording in the left column. Neither action automatically reads waveform samples.
2. Select **Load day overview** to prepare a reduced raw-count summary in the background. Summaries are cached on disk and preserve extrema and recording gaps; they are only for navigation, never detector input.
3. Move or resize the highlighted region, or edit the UTC start/end fields in the right column.
4. Choose channels and select **Load detail**. The initial interval is at most five minutes. Enable **Include spectrogram on Load detail** only when needed, then reload detail.
5. Zoom/pan the detail graph, **Reset view** to restore the last loaded range, or **Use visible range** to explicitly change the search interval.
6. Choose preprocessing and detector settings, then **Run detection**. Select a candidate on the left to inspect it in the same detail graph. Assess it as Unreviewed, Earthquake, Noise, or Uncertain, add notes, and select **Save review**.

**Build day cache** is optional background processing for the entire selected station day, not just the search interval. Changing preprocessing or channels does not silently reload detail. Refreshing the page starts with no selected recording. Detector model weights may require a first-use download by SeisBench.

The responsive workbench keeps recordings/candidates on the left, day overview/detail in the center, and search/settings/review on the right. On smaller screens the columns stack.

## Design notes

- Raw MiniSEED stays outside SQLite and is never modified.
- SQLite stores archive metadata, preprocessing recipes, detector runs, candidate picks/triggers, and manual reviews.
- Zarr stores regenerable processed waveform caches; Plotly figures are always rebuilt from numerical products.
- The viewer links waveform and spectrogram time axes. The green search region is explicit: panning only changes the view until **Use visible range** is selected.
- STA/LTA uses a selected channel (geophone first). PhaseNet is only allowed when the MEMS `ENZ`, `ENN`, and `ENE` trio is available.

## Test

```bash
uv run playwright install chromium
SHAKEPI_ALLOW_UNSAFE_SQLITE=true uv run pytest
```
