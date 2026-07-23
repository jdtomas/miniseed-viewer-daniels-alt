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
