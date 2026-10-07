"""Speaking assessment: acoustic measures + LLM examiner -> CEFR result.

Pipeline per response
---------------------
1. Objective fluency features from the audio (NumPy).
2. Guard: too little speech -> ask the candidate to retry (no LLM spend).
3. Examiner LLM returns a schema-validated judgement (Structured Outputs).
4. Fluency = 60% examiner + 40% acoustic; overall = weighted mean; overall -> CEFR band.
5. Persist, then return a structured result (UI) plus short spoken feedback (TTS).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from maplo_voice.assessment.fluency import acoustic_fluency_score, analyse_fluency
from maplo_voice.assessment.rubric import (
    CRITERIA,
    EXAMINER_INSTRUCTIONS,
    TASKS,
    SpeakingTask,
    build_examiner_input,
    clamp_score,
    get_task,
    score_to_cefr,
    weighted_overall,
)
from maplo_voice.assessment.schemas import AssessmentOut, CriterionScores, ExaminerJudgement
from maplo_voice.db.models import Assessment
from maplo_voice.observability import get_logger, tracer
from maplo_voice.voice.session import REDACTED, AssessmentReply

if TYPE_CHECKING:
    from fastapi import FastAPI

    from maplo_voice.assessment.fluency import FluencyFeatures
    from maplo_voice.config import Settings
    from maplo_voice.db.session import Database
    from maplo_voice.services.asr import Transcript
    from maplo_voice.services.llm import OpenAIResponder

log = get_logger(__name__)

MIN_WORDS = 12
LLM_FLUENCY_WEIGHT = 0.6


def to_assessment_out(row: Assessment) -> AssessmentOut:
    return AssessmentOut(
        id=row.id,
        session_id=row.session_id,
        task_id=row.task_id,
        cefr_level=row.cefr_level,
        overall_score=row.overall_score,
        scores=CriterionScores(
            fluency=row.fluency,
            grammar=row.grammar,
            vocabulary=row.vocabulary,
            coherence=row.coherence,
        ),
        words_per_minute=row.words_per_minute,
        speech_duration_ms=row.speech_duration_ms,
        feedback=row.feedback,
        transcript=row.transcript,
        model=row.model,
        created_at=row.created_at,
    )


class AssessmentService:
    """Implements the ``Assessor`` protocol used by ``VoiceAgentSession``."""

    def __init__(
        self, *, llm: OpenAIResponder, db: Database, model: str, store_transcripts: bool = True
    ) -> None:
        self._llm = llm
        self._db = db
        self.model = model
        self._store_transcripts = store_transcripts

    def task_prompt(self, task_id: str | None) -> tuple[str, str]:
        if task_id and task_id not in TASKS:
            log.warning("unknown_assessment_task", task_id=task_id)
        task = get_task(task_id)
        return task.id, task.prompt

    async def assess(
        self,
        *,
        session_id: uuid.UUID,
        tenant_id: str,
        task_id: str,
        transcript: Transcript,
        pcm16: bytes,
        sample_rate: int,
    ) -> AssessmentReply:
        task = get_task(task_id)
        features = analyse_fluency(pcm16, sample_rate, transcript.text)
        with tracer.start_as_current_span(
            "assessment.score",
            attributes={
                "assessment.task_id": task.id,
                "assessment.word_count": features.word_count,
                "assessment.speech_ms": features.speech_duration_ms,
            },
        ) as span:
            if (
                features.word_count < MIN_WORDS
                or features.speech_duration_ms < task.min_seconds * 1000
            ):
                span.set_attribute("assessment.status", "insufficient_sample")
                return self._retry_reply(task, features, "insufficient_sample")

            judgement, usage = await self._llm.structured(
                model=self.model,
                instructions=EXAMINER_INSTRUCTIONS,
                input_text=build_examiner_input(task, transcript.text, features.summary()),
                schema=ExaminerJudgement,
                stage="assessment",
            )
            if judgement.insufficient_sample or judgement.off_topic:
                status = "off_topic" if judgement.off_topic else "insufficient_sample"
                span.set_attribute("assessment.status", status)
                return self._retry_reply(task, features, status, judgement.spoken_feedback)

            acoustic = acoustic_fluency_score(features)
            llm_scores = {c: clamp_score(getattr(judgement, c).score) for c in CRITERIA}
            scores = dict(llm_scores)
            scores["fluency"] = clamp_score(
                LLM_FLUENCY_WEIGHT * llm_scores["fluency"] + (1 - LLM_FLUENCY_WEIGHT) * acoustic
            )
            overall = weighted_overall(scores)
            level = score_to_cefr(overall)
            span.set_attributes(
                {
                    "assessment.status": "scored",
                    "assessment.cefr": level.value,
                    "assessment.overall": overall,
                }
            )

            feedback: dict[str, Any] = {
                "strengths": judgement.strengths[:3],
                "improvements": judgement.improvements[:3],
                "corrections": [c.model_dump() for c in judgement.corrections[:3]],
                "evidence": {c: getattr(judgement, c).evidence for c in CRITERIA},
                "llm_scores": llm_scores,
                "acoustic_fluency": acoustic,
                "fluency_features": features.as_dict(),
                "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
            }
            row = Assessment(
                id=uuid.uuid4(),
                session_id=session_id,
                tenant_id=tenant_id,
                task_id=task.id,
                task_prompt=task.prompt,
                transcript=transcript.text,
                speech_duration_ms=features.speech_duration_ms,
                words_per_minute=features.words_per_minute,
                cefr_level=level,
                overall_score=overall,
                feedback=feedback,
                model=self.model,
                created_at=datetime.now(UTC),
                **scores,
            )
            result = to_assessment_out(row).model_dump(mode="json")  # full detail to the client
            await self._persist(row)

        spoken = f"{judgement.spoken_feedback.strip()} Your estimated level is {level.value}."
        return AssessmentReply(spoken_feedback=spoken, result={"status": "scored", **result})

    async def _persist(self, row: Assessment) -> None:
        if not self._store_transcripts:
            # Evidence and corrections quote the candidate, so they are dropped as well.
            row.transcript = REDACTED
            row.feedback = {
                k: v for k, v in row.feedback.items() if k not in ("evidence", "corrections")
            }
        try:
            async with self._db.session() as db:
                db.add(row)
        except Exception:
            log.exception("assessment_persist_failed", session_id=str(row.session_id))

    @staticmethod
    def _retry_reply(
        task: SpeakingTask, features: FluencyFeatures, status: str, spoken: str | None = None
    ) -> AssessmentReply:
        if spoken is None:
            spoken = (
                "Thanks. That answer was a little short to assess. Please try again and "
                f"speak for at least {task.min_seconds} seconds."
            )
        return AssessmentReply(
            spoken_feedback=spoken,
            result={
                "status": status,
                "task_id": task.id,
                "min_seconds": task.min_seconds,
                "fluency_features": features.as_dict(),
            },
        )


def register_assessment(app: FastAPI, settings: Settings) -> AssessmentService:
    """Enable assessment mode (requires AI services and database to be registered)."""
    service = AssessmentService(
        llm=app.state.ai.llm,
        db=app.state.db,
        model=settings.openai.assessment_model,
        store_transcripts=settings.voice.store_transcripts,
    )
    app.state.assessor = service
    return service
