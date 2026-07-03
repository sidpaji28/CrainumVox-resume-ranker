import json
import os
from groq import Groq
from .behavioral_scorer import score_behavioral_signals

def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return Groq(api_key=api_key)

MODEL = "llama-3.3-70b-versatile"

SCORE_PROMPT = """You are a senior technical recruiter with deep domain expertise.
Evaluate this candidate against the parsed job requirements and return a structured assessment.

PARSED JOB REQUIREMENTS:
{jd_parsed}

CANDIDATE PROFILE:
{candidate}

Evaluate on THREE dimensions and return ONLY valid JSON, no markdown, no extra text:
{{
  "skill_match": {{
    "score": <0-100>,
    "rationale": "<2-3 sentences: which skills align, which are missing, depth of match>"
  }},
  "experience_relevance": {{
    "score": <0-100>,
    "rationale": "<2-3 sentences: how well their actual work history maps to what this role needs>"
  }},
  "career_trajectory": {{
    "score": <0-100>,
    "rationale": "<2-3 sentences: is their career arc pointing toward this role, or away from it?>"
  }},
  "recruiter_summary": "<1 paragraph: what a recruiter would say about this person — honest, specific, no filler>",
  "green_flags": ["<specific strength 1>", "<specific strength 2>"],
  "red_flags": ["<specific concern 1>", "<specific concern 2>"]
}}

Be specific. Reference actual details. If a candidate is a poor fit, say so clearly."""


def score_candidate_llm(candidate: dict, jd_parsed: dict) -> dict:
    client = _get_client()
    profile = candidate.get("profile", {})
    candidate_summary = {
        "name": profile.get("anonymized_name"),
        "headline": profile.get("headline"),
        "summary": profile.get("summary"),
        "years_of_experience": profile.get("years_of_experience"),
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "location": profile.get("location"),
        "career_history": [
            {
                "company": r.get("company"),
                "title": r.get("title"),
                "duration_months": r.get("duration_months"),
                "is_current": r.get("is_current"),
                "industry": r.get("industry"),
                "description": r.get("description")
            }
            for r in candidate.get("career_history", [])
        ],
        "skills": [
            {
                "name": s.get("name"),
                "proficiency": s.get("proficiency"),
                "endorsements": s.get("endorsements"),
                "duration_months": s.get("duration_months")
            }
            for s in candidate.get("skills", [])
        ],
        "education": [
            {
                "institution": e.get("institution"),
                "degree": e.get("degree"),
                "field": e.get("field_of_study"),
                "tier": e.get("tier")
            }
            for e in candidate.get("education", [])
        ],
        "skill_assessment_scores": candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    }

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a structured data extractor. Return only valid JSON, no markdown fences, no extra text."},
            {"role": "user", "content": SCORE_PROMPT.format(
                jd_parsed=json.dumps(jd_parsed, indent=2),
                candidate=json.dumps(candidate_summary, indent=2)
            )}
        ]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def score_logistics(candidate: dict, jd_parsed: dict) -> tuple:
    signals = candidate.get("redrob_signals", {})
    score = 50.0
    flags_good = []
    flags_bad = []

    jd_location = jd_parsed.get("location_requirement", "any").lower()
    candidate_mode = signals.get("preferred_work_mode", "flexible").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if jd_location in ["onsite", "remote", "hybrid"]:
        if candidate_mode == "flexible":
            score += 15
            flags_good.append("Flexible on work mode")
        elif candidate_mode == jd_location:
            score += 20
            flags_good.append(f"Work mode ({candidate_mode}) matches role requirement")
        elif not willing_to_relocate and jd_location == "onsite":
            score -= 20
            flags_bad.append("Not willing to relocate; role requires onsite")
    else:
        score += 10

    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max", 0)
    if sal_min > 0:
        score += 10
        flags_good.append(f"Salary expectation stated: ₹{sal_min}–{sal_max} LPA")

    score = max(0, min(100, score))
    rationale = f"Work mode: {candidate_mode} | Relocate: {willing_to_relocate} | Salary: ₹{sal_min}-{sal_max}L"
    return round(score, 1), rationale, flags_bad, flags_good


def score_candidate(candidate: dict, jd_parsed: dict) -> dict:
    llm_scores = score_candidate_llm(candidate, jd_parsed)

    signals = candidate.get("redrob_signals", {})
    beh_score, beh_rationale, beh_red, beh_green = score_behavioral_signals(signals, jd_parsed)
    log_score, log_rationale, log_red, log_green = score_logistics(candidate, jd_parsed)

    weights = {
        "skill_match": 0.30,
        "experience_relevance": 0.25,
        "career_trajectory": 0.15,
        "behavioral_signals": 0.20,
        "logistics": 0.10
    }

    total = (
        llm_scores["skill_match"]["score"] * weights["skill_match"] +
        llm_scores["experience_relevance"]["score"] * weights["experience_relevance"] +
        llm_scores["career_trajectory"]["score"] * weights["career_trajectory"] +
        beh_score * weights["behavioral_signals"] +
        log_score * weights["logistics"]
    )

    if total >= 75:   verdict = "Strong Fit"
    elif total >= 55: verdict = "Moderate Fit"
    elif total >= 35: verdict = "Weak Fit"
    else:             verdict = "Poor Fit"

    return {
        "candidate_id": candidate.get("candidate_id"),
        "name": candidate.get("profile", {}).get("anonymized_name", "Unknown"),
        "total_score": round(total, 1),
        "verdict": verdict,
        "dimensions": {
            "skill_match": {
                "score": llm_scores["skill_match"]["score"],
                "rationale": llm_scores["skill_match"]["rationale"],
                "weight": "30%"
            },
            "experience_relevance": {
                "score": llm_scores["experience_relevance"]["score"],
                "rationale": llm_scores["experience_relevance"]["rationale"],
                "weight": "25%"
            },
            "career_trajectory": {
                "score": llm_scores["career_trajectory"]["score"],
                "rationale": llm_scores["career_trajectory"]["rationale"],
                "weight": "15%"
            },
            "behavioral_signals": {
                "score": beh_score,
                "rationale": beh_rationale,
                "weight": "20%"
            },
            "logistics": {
                "score": log_score,
                "rationale": log_rationale,
                "weight": "10%"
            }
        },
        "recruiter_summary": llm_scores.get("recruiter_summary", ""),
        "red_flags": llm_scores.get("red_flags", []) + beh_red + log_red,
        "green_flags": llm_scores.get("green_flags", []) + beh_green + log_green
    }
