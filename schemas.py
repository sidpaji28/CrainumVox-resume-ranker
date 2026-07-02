from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RankRequest(BaseModel):
    job_description: str
    candidates: List[Dict[str, Any]]


class ScoreDimension(BaseModel):
    score: float  # 0–100
    rationale: str


class CandidateResult(BaseModel):
    candidate_id: str
    name: str
    rank: int
    total_score: float  # 0–100
    verdict: str        # e.g. "Strong Fit", "Moderate Fit", "Poor Fit"
    dimensions: Dict[str, ScoreDimension]
    recruiter_summary: str
    red_flags: List[str]
    green_flags: List[str]


class RankResponse(BaseModel):
    job_title_inferred: str
    total_candidates: int
    shortlist: List[CandidateResult]
    ranking_rationale: str