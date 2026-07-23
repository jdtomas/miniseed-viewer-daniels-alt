"""Reproducible preprocessing and Zarr cache artifacts."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import zarr
from numcodecs import Blosc
from obspy import Stream, Trace, UTCDateTime
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import CacheArtifact, RawFile
from .schemas import PreprocessingRecipe
from .waveforms import merge_contiguous_segments

UTC = timezone.utc


class Preprocessor:
    def apply(self, stream: Stream, recipe: PreprocessingRecipe) -> Stream:
        """Apply the documented fixed ordering without changing source arrays."""
        work = stream.copy()
        work = self._handle_gaps(work, recipe)
        for trace in work:
            trace.detrend("demean")
            trace.detrend("linear")
            if recipe.taper_fraction:
                trace.taper(max_percentage=recipe.taper_fraction, type="hann")

        if recipe.deep_denoiser:
            work = self._deep_denoise(work, recipe)

        if recipe.bandpass_low_hz is not None and recipe.bandpass_high_hz is not None:
            for trace in work:
                nyquist = trace.stats.sampling_rate / 2
                if recipe.bandpass_high_hz >= nyquist:
                    raise ValueError(
                        f"Bandpass high cutoff {recipe.bandpass_high_hz} Hz is at or above {trace.id}'s Nyquist {nyquist} Hz"
                    )
                trace.filter(
                    "bandpass",
                    freqmin=recipe.bandpass_low_hz,
                    freqmax=recipe.bandpass_high_hz,
                    corners=4,
                    zerophase=True,
                )
        return work

    def _handle_gaps(self, stream: Stream, recipe: PreprocessingRecipe) -> Stream:
        if not recipe.fill_short_gaps:
            return stream.split()
        output = Stream()
        for trace_id in sorted(set(trace.id for trace in stream)):
            group = stream.select(id=trace_id).copy()
            gaps = group.get_gaps()
            if any(float(gap[6]) > recipe.max_gap_seconds for gap in gaps):
                output += group.split()
            else:
                group.merge(method=1, fill_value="interpolate")
                output += group
        return output

    def _deep_denoise(self, stream: Stream, recipe: PreprocessingRecipe) -> Stream:
        try:
            from seisbench.models import DeepDenoiser
        except ImportError as error:
            raise RuntimeError("DeepDenoiser requires the optional seisbench dependency") from error
        model = DeepDenoiser.from_pretrained(recipe.deep_denoiser_weight)
        # SeisBench's model is explicitly grouped by channel and resamples to 100 Hz as needed.
        return model.annotate(stream, strict=True)


class ZarrCache:
    """A regenerable filesystem cache; SQLite stores only the artifact index."""

    def __init__(self, settings: Settings, session_factory: sessionmaker[Session]):
        self.settings = settings
        self.session_factory = session_factory

    def key_for(self, source_hashes: list[str], recipe: PreprocessingRecipe, channels: list[str]) -> str:
        import hashlib
        import json

        payload = {
            "sources": sorted(source_hashes),
            "recipe": recipe.model_dump(mode="json"),
            "channels": sorted(channels),
            "schema": "1",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def artifact_path(self, cache_key: str) -> Path:
        return self.settings.cache_root / "processed" / f"{cache_key}.zarr"

    def store_stream(self, cache_key: str, stream: Stream, metadata: dict[str, object]) -> Path:
        final_path = self.artifact_path(cache_key)
        if final_path.exists():
            self.touch(cache_key)
            return final_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = final_path.with_name(f".{final_path.name}.{uuid4().hex}.tmp")
        group = zarr.open_group(str(temp_path), mode="w")
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        for index, trace in enumerate(stream):
            name = f"trace_{index}"
            chunk_size = max(1, min(len(trace.data), int(trace.stats.sampling_rate * 600)))
            dataset = group.create_dataset(
                name,
                data=np.asarray(trace.data, dtype=np.float32),
                chunks=(chunk_size,),
                compressor=compressor,
                overwrite=True,
            )
            dataset.attrs.update(
                {
                    "id": trace.id,
                    "starttime": str(trace.stats.starttime),
                    "sampling_rate": float(trace.stats.sampling_rate),
                }
            )
        group.attrs.update(metadata)
        (temp_path / ".complete").write_text(datetime.now(UTC).isoformat())
        os.replace(temp_path, final_path)
        byte_count = sum(path.stat().st_size for path in final_path.rglob("*") if path.is_file())
        with self.session_factory.begin() as session:
            artifact = session.scalar(select(CacheArtifact).where(CacheArtifact.cache_key == cache_key))
            if artifact is None:
                artifact = CacheArtifact(
                    cache_key=cache_key,
                    kind="processed_stream",
                    path=str(final_path.resolve()),
                    bytes=byte_count,
                    status="complete",
                    metadata_json=metadata,
                )
                session.add(artifact)
            else:
                artifact.path = str(final_path.resolve())
                artifact.bytes = byte_count
                artifact.status = "complete"
                artifact.metadata_json = metadata
        self.evict_if_needed()
        return final_path

    def load_stream(
        self,
        cache_key: str,
        utc_range: tuple[datetime, datetime],
    ) -> Stream | None:
        """Load only chunk-aligned samples requested from a completed Zarr artifact."""
        path = self.artifact_path(cache_key)
        if not (path / ".complete").exists():
            return None
        start, end = utc_range
        group = zarr.open_group(str(path), mode="r")
        stream = Stream()
        for name in group.array_keys():
            dataset = group[name]
            trace_start = UTCDateTime(dataset.attrs["starttime"])
            sample_rate = float(dataset.attrs["sampling_rate"])
            left = max(0, int(np.ceil((UTCDateTime(start) - trace_start) * sample_rate)))
            right = min(dataset.shape[0], int(np.floor((UTCDateTime(end) - trace_start) * sample_rate)) + 1)
            if right <= left:
                continue
            network, station, location, channel = str(dataset.attrs["id"]).split(".", maxsplit=3)
            stream.append(
                Trace(
                    data=np.asarray(dataset[left:right], dtype=np.float32),
                    header={
                        "network": network,
                        "station": station,
                        "location": location,
                        "channel": channel,
                        "starttime": trace_start + left / sample_rate,
                        "sampling_rate": sample_rate,
                    },
                )
            )
        self.touch(cache_key)
        return merge_contiguous_segments(stream)

    def touch(self, cache_key: str) -> None:
        with self.session_factory.begin() as session:
            artifact = session.scalar(select(CacheArtifact).where(CacheArtifact.cache_key == cache_key))
            if artifact:
                artifact.last_accessed_at = datetime.now(UTC)

    def evict_if_needed(self) -> None:
        with self.session_factory.begin() as session:
            artifacts = session.scalars(select(CacheArtifact).order_by(CacheArtifact.last_accessed_at.asc())).all()
            total = sum(artifact.bytes for artifact in artifacts)
            for artifact in artifacts:
                if total <= self.settings.cache_limit_bytes:
                    break
                path = Path(artifact.path)
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                total -= artifact.bytes
                session.delete(artifact)

    def source_hashes(self, period_id: int, channels: list[str]) -> list[str]:
        with self.session_factory() as session:
            return list(
                session.scalars(
                    select(RawFile.sha256).where(RawFile.period_id == period_id, RawFile.channel.in_(channels))
                )
            )
