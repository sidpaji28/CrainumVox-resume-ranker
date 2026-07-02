import json
import anthropic
from .behavioral_scorer import score_behavioral_signals

client = anthropic.Anthropic()

SCORE_PROMPT = """You are a senior technical recruiter with deep domain expertise.
Evaluate this candidate against the parsed job requirements and return a structured assessment.

PARSED JOB REQUIREMENTS:
{jd_parsed}

CANDIDATE PROFILE:
{candidate}

Evaluate on THREE dimensions and return ONLY valid JSON:
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
  "recruiter_summary": "<1 paragraph: what a recruiter would say about this person for this role — honest, specific, no filler>",
  "green_flags": ["<specific strength 1>", "<specific strength 2>"],
  "red_flags": ["<specific concern 1>", "<specific concern 2>"]
}}

Be specific and honest. Reference actual details from the profile. Do NOT give generic praise.
If a candidate is a poor fit, say so clearly. Scores should reflect reality, not optimism."""


def score_candidate_llm(candidate: dict, jd_parsed: dict) -> dict:
    """Get LLM scores for skill match, experience relevance, career trajectory."""
   
    # Build a clean candidate summary to feed (exclude noise)
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
   
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
            for r in career
        ],
        "skills": [
            {
                "name": s.get("name"),
                "proficiency": s.get("proficiency"),
                "endorsements": s.get("endorsements"),
                "duration_months": s.get("duration_months")
            }
            for s in skills
        ],
        "education": [
            {
                "institution": e.get("institution"),
                "degree": e.get("degree"),
                "field": e.get("field_of_study"),
                "tier": e.get("tier")
            }
            for e in education
        ],
        "skill_assessment_scores": candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    }

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": SCORE_PROMPT.format(
                jd_parsed=json.dumps(jd_parsed, indent=2),
                candidate=json.dumps(candidate_summary, indent=2)
            )
        }]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def score_logistics(candidate: dict, jd_parsed: dict) -> tuple:
    """Rule-based logistics scoring: location, salary, work mode."""
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    score = 50.0  # neutral baseline
    flags_good = []
    flags_bad = []

    # Work mode alignment
    jd_location = jd_parsed.get("location_requirement", "any").lower()
    candidate_mode = signals.get("preferred_work_mode", "flexible").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if jd_location in ["onsite", "remote", "hybrid"]:
        if candidate_mode == "flexible":
            score += 15
            flags_good.append("Flexible on work mode")
        elif candidate_mode == jd_location:
            score += 20
            flags_good.append(f"Work mode preference ({candidate_mode}) matches role")
        elif not willing_to_relocate and jd_location == "onsite":
            score -= 20
            flags_bad.append("Not willing to relocate and role requires onsite")
    else:
        score += 10  # any mode acceptable

    # Salary range (if available — it's in INR LPA)
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max", 0)
    if sal_min > 0:
        score += 10  # candidate has stated expectations (transparency is good)
        flags_good.append(f"Salary expectation stated: ₹{sal_min}–{sal_max} LPA")

    score = max(0, min(100, score))
    rationale = f"Work mode: {candidate_mode}, relocate: {willing_to_relocate}, salary: ₹{sal_min}-{sal_max}L"
    return round(score, 1), rationale, flags_bad, flags_good


def score_candidate(candidate: dict, jd_parsed: dict) -> dict:
    """Full scoring pipeline: LLM (3 dims) + behavioral (rule-based) + logistics."""

    # LLM scoring
    llm_scores = score_candidate_llm(candidate, jd_parsed)

    # Behavioral signals
    signals = candidate.get("redrob_signals", {})
    beh_score, beh_rationale, beh_red, beh_green = score_behavioral_signals(signals, jd_parsed)

    # Logistics
    log_score, log_rationale, log_red, log_green = score_logistics(candidate, jd_parsed)

    # Weighted aggregate
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

    # Verdict buckets
    if total >= 75:
        verdict = "Strong Fit"
    elif total >= 55:
        verdict = "Moderate Fit"
    elif total >= 35:
        verdict = "Weak Fit"
    else:
        verdict = "Poor Fit"

    # Merge flags
    all_red = (
        llm_scores.get("red_flags", []) +
        beh_red +
        log_red
    )
    all_green = (
        llm_scores.get("green_flags", []) +
        beh_green +
        log_green
    )

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
        "red_flags": all_red,
        "green_flags": all_green
    }