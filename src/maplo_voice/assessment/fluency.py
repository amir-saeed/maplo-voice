"""Objective fluency measures from raw audio (NumPy, vectorised).

The LLM only sees text, so it cannot hear hesitation. These measures come from the
signal itself and are blended into the fluency score:

* speech rate (words per minute over the whole response)
* articulation rate (words per minute of actual phonation)
* pause count / mean / longest pause and pause ratio (share of time silent)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

_WORD = re.compile(r"[A-Za-zÀ-ɏ']+")
_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class FluencyFeatures:
    word_count: int
    speech_duration_ms: int  # first to last voiced frame
    phonation_ms: int  # voiced frames only
    pause_count: int
    mean_pause_ms: int
    longest_pause_ms: int
    pause_ratio: float
    words_per_minute: float
    articulation_wpm: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"- words: {self.word_count}\n"
            f"- speaking time: {self.speech_duration_ms / 1000:.1f}s\n"
            f"- speech rate: {self.words_per_minute:.0f} words/min "
            f"(articulation {self.articulation_wpm:.0f} words/min)\n"
            f"- pauses >= 250ms: {self.pause_count}, mean {self.mean_pause_ms}ms, "
            f"longest {self.longest_pause_ms}ms, silent share {self.pause_ratio:.0%}"
        )


EMPTY = FluencyFeatures(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def frame_levels_dbfs(pcm16: bytes, sample_rate: int, frame_ms: int = 20) -> np.ndarray:
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    frame_len = sample_rate * frame_ms // 1000
    n_frames = samples.size // frame_len
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return (20.0 * np.log10(np.maximum(rms, _EPS))).astype(np.float32)


def _runs(mask: np.ndarray) -> np.ndarray:
    """Lengths (in frames) of consecutive True runs."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def analyse_fluency(
    pcm16: bytes,
    sample_rate: int,
    transcript: str,
    *,
    frame_ms: int = 20,
    min_pause_ms: int = 250,
    dynamic_range_db: float = 25.0,
) -> FluencyFeatures:
    levels = frame_levels_dbfs(pcm16, sample_rate, frame_ms)
    if levels.size == 0:
        return EMPTY
    # Relative threshold: robust to microphone gain differences between candidates.
    threshold = max(-50.0, float(np.percentile(levels, 95)) - dynamic_range_db)
    voiced = levels >= threshold
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        return EMPTY
    voiced = voiced[idx[0] : idx[-1] + 1]  # trim leading/trailing silence

    pause_frames = _runs(~voiced) * frame_ms
    pauses = pause_frames[pause_frames >= min_pause_ms]
    speech_ms = int(voiced.size * frame_ms)
    phonation_ms = int(voiced.sum() * frame_ms)
    words = count_words(transcript)
    return FluencyFeatures(
        word_count=words,
        speech_duration_ms=speech_ms,
        phonation_ms=phonation_ms,
        pause_count=int(pauses.size),
        mean_pause_ms=int(pauses.mean()) if pauses.size else 0,
        longest_pause_ms=int(pauses.max()) if pauses.size else 0,
        pause_ratio=round(float(pauses.sum()) / speech_ms, 3) if speech_ms else 0.0,
        words_per_minute=round(words / (speech_ms / 60_000), 1) if speech_ms else 0.0,
        articulation_wpm=round(words / (phonation_ms / 60_000), 1) if phonation_ms else 0.0,
    )


def acoustic_fluency_score(f: FluencyFeatures) -> int:
    """0-100 heuristic. Typical fluent conversational English is ~120-170 wpm."""
    if f.word_count == 0:
        return 0
    rate = float(np.interp(f.words_per_minute, [30, 110, 170, 260], [0, 100, 100, 50]))
    pausing = float(np.interp(f.pause_ratio, [0.10, 0.50], [100, 0]))
    long_pause = float(np.interp(f.longest_pause_ms, [1_000, 4_000], [100, 30]))
    return max(0, min(100, round(0.5 * rate + 0.35 * pausing + 0.15 * long_pause)))
