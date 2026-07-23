"""Detector adapters with an intentionally small shared interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
from obspy import Stream
from obspy.signal.trigger import classic_sta_lta, trigger_onset

from .schemas import CandidateResult, DetectionRunResult, PhaseNetParameters, StaLtaParameters

UTC = timezone.utc


@dataclass(frozen=True)
class DetectorCapabilities:
    required_components: tuple[str, ...]
    default_channel_role: str | None
    description: str


class Detector(Protocol):
    name: str

    def capabilities(self) -> DetectorCapabilities: ...

    def run(self, stream: Stream, parameters: object) -> DetectionRunResult: ...


class StaLtaDetector:
    name = "stalta"
    version = "obspy-classic-stalta-v1"

    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            required_components=("single_channel",),
            default_channel_role="geophone",
            description="Classic STA/LTA uses one selected channel and defaults to the geophone.",
        )

    def run(self, stream: Stream, parameters: StaLtaParameters) -> DetectionRunResult:
        if len(set(trace.id for trace in stream)) > 1:
            raise ValueError("STA/LTA requires one selected channel")
        candidates: list[CandidateResult] = []
        skipped: list[tuple[datetime, datetime]] = []
        for trace in stream.split():
            sample_rate = float(trace.stats.sampling_rate)
            nsta = int(round(parameters.sta_seconds * sample_rate))
            nlta = int(round(parameters.lta_seconds * sample_rate))
            if len(trace.data) <= nlta:
                start = trace.stats.starttime.datetime.replace(tzinfo=UTC)
                end = trace.stats.endtime.datetime.replace(tzinfo=UTC)
                skipped.append((start, end))
                continue
            characteristic = classic_sta_lta(np.asarray(trace.data, dtype=np.float64), nsta, nlta)
            on_off = trigger_onset(characteristic, parameters.trigger_on, parameters.trigger_off or parameters.trigger_on / 2)
            for onset_sample, offset_sample in on_off:
                start = (trace.stats.starttime + onset_sample / sample_rate).datetime.replace(tzinfo=UTC)
                end = (trace.stats.starttime + offset_sample / sample_rate).datetime.replace(tzinfo=UTC)
                peak = float(np.max(characteristic[onset_sample : max(onset_sample + 1, offset_sample)]))
                candidates.append(
                    CandidateResult(
                        kind="stalta",
                        timestamp=start,
                        end_time=end,
                        score=peak,
                        details={"channel": trace.stats.channel, "duration_seconds": (end - start).total_seconds()},
                    )
                )
        return DetectionRunResult(
            detector_name=self.name,
            detector_version=self.version,
            candidates=candidates,
            skipped_ranges=skipped,
        )


class PhaseNetDetector:
    name = "phasenet"
    version = "seisbench-phasenet-v1"

    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            required_components=("Z", "N", "E"),
            default_channel_role=None,
            description="PhaseNet requires complete time-aligned MEMS Z/N/E channels; it does not use the geophone.",
        )

    @staticmethod
    def validate_components(stream: Stream) -> None:
        mems = [trace for trace in stream if trace.stats.channel.upper() in {"ENZ", "ENN", "ENE"}]
        components = {trace.stats.channel[-1].upper() for trace in mems}
        if len(mems) != 3 or components != {"Z", "N", "E"}:
            raise ValueError("PhaseNet requires a complete MEMS Z/N/E component group")
        sample_rates = {float(trace.stats.sampling_rate) for trace in mems}
        if len(sample_rates) != 1:
            raise ValueError("PhaseNet requires equal sampling rates for Z/N/E components")

    def run(self, stream: Stream, parameters: PhaseNetParameters) -> DetectionRunResult:
        self.validate_components(stream)
        try:
            from seisbench.models import PhaseNet
        except ImportError as error:
            raise RuntimeError("PhaseNet requires the optional seisbench dependency") from error
        model = PhaseNet.from_pretrained(parameters.weight)
        classified = model.classify(
            stream,
            P_threshold=parameters.p_threshold,
            S_threshold=parameters.s_threshold,
            strict=True,
        )
        picks = getattr(classified, "picks", classified)
        candidates: list[CandidateResult] = []
        for pick in picks:
            phase = str(getattr(pick, "phase", "")).upper()
            peak_time = getattr(pick, "peak_time", None)
            if peak_time is None:
                continue
            if hasattr(peak_time, "datetime"):
                timestamp = peak_time.datetime.replace(tzinfo=UTC)
            elif isinstance(peak_time, datetime):
                timestamp = peak_time if peak_time.tzinfo else peak_time.replace(tzinfo=UTC)
            else:
                continue
            candidates.append(
                CandidateResult(
                    kind="phase_pick",
                    phase=phase if phase in {"P", "S"} else None,
                    timestamp=timestamp,
                    score=float(getattr(pick, "peak_value", 0.0)),
                    details={"trace_id": str(getattr(pick, "trace_id", ""))},
                )
            )
        return DetectionRunResult(
            detector_name=self.name,
            detector_version=self.version,
            model_weight=parameters.weight,
            candidates=candidates,
        )


DETECTORS: dict[str, Detector] = {"stalta": StaLtaDetector(), "phasenet": PhaseNetDetector()}
