"""Assessment schemas: the LLM's structured output and the public API models.

``ExaminerJudgement`` is sent to the model as a strict JSON schema (Structured Outputs),
so every field is required and there are no free-form dicts. Numeric bounds are
enforced after parsing (``rubric.clamp_score``) rather than trusted from the model.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from maplo_voice.db.models import CEFRLevel


# --------------------------------------------------------------------------- LLM output
class CriterionJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int = Field(description="0-100")
    evidence: str = Field(description="Short quote(s) from the transcript supporting the score")


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    original: str
    corrected: str
    explanation: str


class ExaminerJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fluency: CriterionJudgement
    grammar: CriterionJudgement
    vocabulary: CriterionJudgement
    coherence: CriterionJudgement
    strengths: list[str] = Field(description="Up to 3 specific strengths")
    improvements: list[str] = Field(description="Up to 3 specific, actionable next steps")
    corrections: list[Correction] = Field(description="Up to 3 corrected examples")
    off_topic: bool
    insufficient_sample: bool
    spoken_feedback: str


# --------------------------------------------------------------------------- API
class TaskOut(BaseModel):
    id: str
    title: str
    prompt: str
    target_level: CEFRLevel
    min_seconds: int
    recommended_seconds: int


class CriterionScores(BaseModel):
    fluency: int
    grammar: int
    vocabulary: int
    coherence: int


class AssessmentOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    task_id: str
    cefr_level: CEFRLevel
    overall_score: int
    scores: CriterionScores
    words_per_minute: float
    speech_duration_ms: int
    feedback: dict[str, Any]
    transcript: str
    model: str
    created_at: datetime


class AssessmentList(BaseModel):
    items: list[AssessmentOut]
    total: int


class LevelCount(BaseModel):
    cefr_level: CEFRLevel
    count: int


class AssessmentStats(BaseModel):
    total: int
    average_overall: float | None
    average_scores: CriterionScores | None
    distribution: list[LevelCount]
