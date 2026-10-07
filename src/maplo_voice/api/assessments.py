"""Assessment REST API: speaking tasks, results history and cohort statistics."""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves path param type at runtime
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query, status

from maplo_voice.api.documents import CurrentPrincipal  # noqa: TC001 - FastAPI resolves at runtime
from maplo_voice.assessment.rubric import TASKS
from maplo_voice.assessment.schemas import (
    AssessmentList,
    AssessmentOut,
    AssessmentStats,
    CriterionScores,
    LevelCount,
    TaskOut,
)
from maplo_voice.assessment.service import to_assessment_out
from maplo_voice.db.models import Assessment, CEFRLevel
from maplo_voice.db.session import DbSession  # noqa: TC001 - FastAPI resolves at runtime

router = APIRouter(prefix="/v1", tags=["assessment"])


@router.get("/assessment/tasks")
async def list_tasks(_: CurrentPrincipal) -> list[TaskOut]:
    return [
        TaskOut(
            id=t.id,
            title=t.title,
            prompt=t.prompt,
            target_level=t.target_level,
            min_seconds=t.min_seconds,
            recommended_seconds=t.recommended_seconds,
        )
        for t in TASKS.values()
    ]


@router.get("/assessments")
async def list_assessments(
    principal: CurrentPrincipal,
    db: DbSession,
    cefr_level: CEFRLevel | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AssessmentList:
    where = [Assessment.tenant_id == principal.tenant_id]
    if cefr_level is not None:
        where.append(Assessment.cefr_level == cefr_level)
    rows = (
        await db.scalars(
            sa.select(Assessment)
            .where(*where)
            .order_by(Assessment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    total = await db.scalar(sa.select(sa.func.count(Assessment.id)).where(*where)) or 0
    return AssessmentList(items=[to_assessment_out(r) for r in rows], total=total)


@router.get("/assessments/stats")
async def assessment_stats(principal: CurrentPrincipal, db: DbSession) -> AssessmentStats:
    tenant = Assessment.tenant_id == principal.tenant_id
    agg = (
        await db.execute(
            sa.select(
                sa.func.count(Assessment.id),
                sa.func.avg(Assessment.overall_score),
                sa.func.avg(Assessment.fluency),
                sa.func.avg(Assessment.grammar),
                sa.func.avg(Assessment.vocabulary),
                sa.func.avg(Assessment.coherence),
            ).where(tenant)
        )
    ).one()
    dist = (
        await db.execute(
            sa.select(Assessment.cefr_level, sa.func.count(Assessment.id))
            .where(tenant)
            .group_by(Assessment.cefr_level)
        )
    ).all()
    counts = {level: count for level, count in dist}
    total = int(agg[0])
    return AssessmentStats(
        total=total,
        average_overall=round(float(agg[1]), 1) if total else None,
        average_scores=CriterionScores(
            fluency=round(float(agg[2])),
            grammar=round(float(agg[3])),
            vocabulary=round(float(agg[4])),
            coherence=round(float(agg[5])),
        )
        if total
        else None,
        distribution=[LevelCount(cefr_level=lvl, count=counts.get(lvl, 0)) for lvl in CEFRLevel],
    )


@router.get("/assessments/{assessment_id}")
async def get_assessment(
    assessment_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession
) -> AssessmentOut:
    row = await db.scalar(
        sa.select(Assessment).where(
            Assessment.id == assessment_id, Assessment.tenant_id == principal.tenant_id
        )
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Assessment not found")
    return to_assessment_out(row)
