"""Unit tests: voice activity detection, fluency measures and CEFR scoring."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import noise, pcm, silence, speech

from maplo_voice.assessment.fluency import (
    EMPTY,
    acoustic_fluency_score,
    analyse_fluency,
    count_words,
)
from maplo_voice.assessment.rubric import (
    DEFAULT_TASK_ID,
    clamp_score,
    get_task,
    score_to_cefr,
    weighted_overall,
)
from maplo_voice.db.models import CEFRLevel
from maplo_voice.voice.vad import EnergyVAD, VadEventKind, frame_dbfs


def kinds(events: list) -> list[str]:  # type: ignore[type-arg]
    return [e.kind.value for e in events]


# --------------------------------------------------------------------------- VAD
class TestEnergyVAD:
    def test_detects_single_utterance(self) -> None:
        events = EnergyVAD().process(pcm(silence(0.5), speech(1.0), silence(1.0)))
        assert kinds(events) == ["speech_start", "speech_end"]
        assert 1000 <= events[1].duration_ms <= 1500
        assert len(events[1].audio) == events[1].duration_ms * 32  # 16 kHz * 2 bytes / 1000

    def test_ignores_steady_background_noise(self) -> None:
        assert EnergyVAD().process(pcm(noise(3.0))) == []

    def test_detects_speech_over_noise(self) -> None:
        mixed = (noise(1.0, seed=1).astype(np.int32) + speech(1.0)).clip(-32768, 32767)
        events = EnergyVAD().process(pcm(noise(0.5), mixed.astype("<i2"), noise(1.0, seed=2)))
        assert kinds(events) == ["speech_start", "speech_end"]

    def test_short_blip_is_discarded(self) -> None:
        events = EnergyVAD().process(pcm(silence(0.5), speech(0.1), silence(1.0)))
        assert kinds(events) == ["speech_start", "speech_discarded"]
        assert events[1].audio == b""

    def test_quiet_speech_is_detected(self) -> None:
        events = EnergyVAD().process(pcm(silence(0.5), speech(1.0, amplitude=0.045), silence(1.0)))
        assert kinds(events) == ["speech_start", "speech_end"]

    def test_max_utterance_forces_end(self) -> None:
        events = EnergyVAD(max_utterance_s=1.0).process(pcm(silence(0.3), speech(2.5)))
        assert events[0].kind is VadEventKind.SPEECH_START
        assert events[1].kind is VadEventKind.SPEECH_END
        assert events[1].duration_ms == 1000

    def test_short_pauses_do_not_split_utterance(self) -> None:
        audio = pcm(silence(0.5), speech(1.0), silence(0.3), speech(1.0), silence(1.0))
        assert kinds(EnergyVAD().process(audio)) == ["speech_start", "speech_end"]

    @pytest.mark.parametrize("chunk", [1, 320, 333, 640, 4096])
    def test_result_independent_of_chunk_size(self, chunk: int) -> None:
        audio = pcm(silence(0.5), speech(1.0), silence(1.0))
        vad = EnergyVAD()
        events = []
        for i in range(0, len(audio), chunk * 2):
            events += vad.process(audio[i : i + chunk * 2])
        assert kinds(events) == ["speech_start", "speech_end"]

    def test_flush_ends_utterance_and_is_noop_when_silent(self) -> None:
        vad = EnergyVAD()
        assert vad.flush() is None
        vad.process(pcm(silence(0.5), speech(1.0)))
        assert vad.in_speech
        event = vad.flush()
        assert event is not None
        assert event.kind is VadEventKind.SPEECH_END
        assert not vad.in_speech

    def test_frame_dbfs_reference_levels(self) -> None:
        assert frame_dbfs(np.zeros(320, dtype="<i2")) < -150
        full_scale_sine = speech(0.02, amplitude=1.0)
        assert frame_dbfs(full_scale_sine) == pytest.approx(-3.0, abs=0.2)


# --------------------------------------------------------------------------- fluency
class TestFluency:
    @staticmethod
    def _answer(talk: float, pause: float, repeats: int = 10) -> bytes:
        parts: list[np.ndarray] = []
        for _ in range(repeats):
            parts += [speech(talk), silence(pause)]
        return pcm(*parts)

    def test_fluent_vs_halting(self) -> None:
        text = " ".join(["word"] * 54)
        fluent = analyse_fluency(self._answer(2.0, 0.3), 16_000, text)
        halting = analyse_fluency(self._answer(0.8, 1.8), 16_000, text)
        assert fluent.pause_ratio < 0.2 < halting.pause_ratio
        assert halting.longest_pause_ms >= 1700
        assert acoustic_fluency_score(fluent) > acoustic_fluency_score(halting) + 20

    def test_speech_rate(self) -> None:
        f = analyse_fluency(pcm(speech(30.0)), 16_000, " ".join(["word"] * 70))
        assert f.words_per_minute == pytest.approx(140, rel=0.02)
        assert f.pause_count == 0

    def test_leading_and_trailing_silence_ignored(self) -> None:
        f = analyse_fluency(pcm(silence(3.0), speech(10.0), silence(3.0)), 16_000, "a b c")
        assert f.speech_duration_ms == pytest.approx(10_000, abs=40)

    def test_empty_inputs(self) -> None:
        assert analyse_fluency(b"", 16_000, "hello") == EMPTY
        assert acoustic_fluency_score(EMPTY) == 0

    def test_count_words_handles_accents_and_apostrophes(self) -> None:
        assert count_words("Don't stop — café, naïve résumé!") == 5


# --------------------------------------------------------------------------- scoring
@pytest.mark.parametrize(
    ("score", "level"),
    [
        (0, CEFRLevel.A1),
        (19, CEFRLevel.A1),
        (20, CEFRLevel.A2),
        (34, CEFRLevel.A2),
        (35, CEFRLevel.B1),
        (54, CEFRLevel.B1),
        (55, CEFRLevel.B2),
        (74, CEFRLevel.B2),
        (75, CEFRLevel.C1),
        (89, CEFRLevel.C1),
        (90, CEFRLevel.C2),
        (100, CEFRLevel.C2),
    ],
)
def test_score_to_cefr_boundaries(score: int, level: CEFRLevel) -> None:
    assert score_to_cefr(score) is level


def test_weighted_overall_and_clamp() -> None:
    assert weighted_overall({"fluency": 80, "grammar": 64, "vocabulary": 60, "coherence": 72}) == 69
    assert clamp_score(-5) == 0
    assert clamp_score(140) == 100
    assert clamp_score(67.5) == 68


def test_unknown_task_falls_back_to_default() -> None:
    assert get_task("does-not-exist").id == DEFAULT_TASK_ID
    assert get_task(None).id == DEFAULT_TASK_ID
    assert get_task("remote-work").target_level is CEFRLevel.B2
