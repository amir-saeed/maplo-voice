"""Energy-based voice activity detection (NumPy).

Deliberately dependency-light and deterministic (easy to unit test):

* Frame energy in dBFS, compared against ``max(fixed threshold, noise floor + margin)``.
  The noise floor is calibrated from the first ~200 ms (median level) and then tracked
  outside speech — falling fast, rising slowly — so the detector adapts to a noisy room
  instead of triggering on fans or keyboards, without "learning" speech as noise.
* Hysteresis: speech must persist ``start_frames`` to begin, and silence must persist
  ``silence_ms`` to end, so short clicks and natural pauses don't split utterances.
* Pre-roll: the ~300 ms before onset is kept, so the first syllable isn't clipped.
* Utterances shorter than ``min_speech_ms`` are discarded; longer than ``max_utterance_s``
  are force-ended so a stuck microphone can't grow memory without bound.

For production at scale you might swap in Silero VAD (ONNX); the interface stays the same.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

_EPS = 1e-10
_INT16_FULL_SCALE = 32768.0


class VadEventKind(StrEnum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"  # ``audio`` holds the utterance
    SPEECH_DISCARDED = "speech_discarded"  # too short (cough, click)


@dataclass(frozen=True, slots=True)
class VadEvent:
    kind: VadEventKind
    audio: bytes = b""
    duration_ms: int = 0


def frame_dbfs(frame: np.ndarray) -> float:
    """RMS level of an int16 frame in dBFS (0 = full scale, ~-90 = digital silence)."""
    samples = frame.astype(np.float32) / _INT16_FULL_SCALE
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return float(20.0 * np.log10(max(rms, _EPS)))


class EnergyVAD:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        threshold_dbfs: float = -45.0,
        noise_margin_db: float = 12.0,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_s: float = 30.0,
        pre_roll_ms: int = 300,
        start_frames: int = 3,
        calibration_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._frame_bytes = sample_rate * frame_ms // 1000 * 2
        self._threshold = threshold_dbfs
        self._margin = noise_margin_db
        self._silence_frames = max(1, silence_ms // frame_ms)
        self._min_speech_frames = max(1, min_speech_ms // frame_ms)
        self._max_frames = int(max_utterance_s * 1000 // frame_ms)
        self._start_frames = start_frames
        self._calibration_frames = max(1, calibration_ms // frame_ms)
        self._pre_roll: deque[bytes] = deque(maxlen=max(1, pre_roll_ms // frame_ms))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._remainder = b""
        self._in_speech = False
        self._utterance: list[bytes] = []
        self._voiced_run = 0
        self._silent_run = 0
        self._voiced_total = 0
        self._noise_floor = -70.0
        self._calibration: list[float] = []
        self._pre_roll.clear()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def effective_threshold(self) -> float:
        return max(self._threshold, self._noise_floor + self._margin)

    # ------------------------------------------------------------------ api
    def process(self, pcm16: bytes) -> list[VadEvent]:
        """Feed arbitrary-sized PCM16 chunks; returns events in order."""
        data = self._remainder + pcm16
        usable = len(data) - (len(data) % self._frame_bytes)
        self._remainder = data[usable:]
        events: list[VadEvent] = []
        for offset in range(0, usable, self._frame_bytes):
            event = self._process_frame(data[offset : offset + self._frame_bytes])
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> VadEvent | None:
        """Force-end the current utterance (push-to-talk release / ``input.commit``)."""
        if not self._in_speech:
            return None
        return self._end_utterance()

    # ------------------------------------------------------------------ internals
    def _process_frame(self, frame: bytes) -> VadEvent | None:
        level = frame_dbfs(np.frombuffer(frame, dtype="<i2"))

        if len(self._calibration) < self._calibration_frames:
            # Assume the user isn't talking in the first few hundred ms after connecting.
            self._calibration.append(level)
            self._noise_floor = float(np.median(self._calibration))
            self._pre_roll.append(frame)
            return None

        voiced = level >= self.effective_threshold
        if not self._in_speech:
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            # Asymmetric tracking: quiet frames pull the floor down fast; loud frames push
            # it up slowly, so sustained noise is absorbed but a word onset is not.
            alpha = 0.2 if level < self._noise_floor else 0.01
            self._noise_floor += alpha * (level - self._noise_floor)
            self._pre_roll.append(frame)
            if self._voiced_run >= self._start_frames:
                self._in_speech = True
                self._utterance = list(self._pre_roll)
                self._pre_roll.clear()
                self._voiced_total = self._voiced_run
                self._silent_run = 0
                return VadEvent(VadEventKind.SPEECH_START)
            return None

        self._utterance.append(frame)
        if voiced:
            self._voiced_total += 1
            self._silent_run = 0
        else:
            self._silent_run += 1
        if self._silent_run >= self._silence_frames or len(self._utterance) >= self._max_frames:
            return self._end_utterance()
        return None

    def _end_utterance(self) -> VadEvent:
        # Trim trailing silence (keep a little tail so word endings aren't clipped).
        keep_tail = min(self._silent_run, 5)
        frames = self._utterance[: len(self._utterance) - self._silent_run + keep_tail]
        voiced_total = self._voiced_total
        self._in_speech = False
        self._utterance = []
        self._voiced_run = self._silent_run = self._voiced_total = 0
        duration_ms = len(frames) * self.frame_ms
        if voiced_total < self._min_speech_frames:
            return VadEvent(VadEventKind.SPEECH_DISCARDED, duration_ms=duration_ms)
        return VadEvent(VadEventKind.SPEECH_END, b"".join(frames), duration_ms)
