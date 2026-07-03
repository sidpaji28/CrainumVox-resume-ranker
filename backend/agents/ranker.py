import json
import os
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed
from .jd_parser import parse_jd
from .candidate_scorer import score_candidate

MODEL = "llama-3.3-70b-versatile"

def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return Groq(api_key=api_key)

RATIONALE_PROMPT = """You're a senior recruiter. You just scored {n} candidates for:
Role: {job_title} | Domain: {domain} | Must-haves: {must_haves}

Top-ranked candidates:
{top_candidates}

Write 2-3 sentences explaining what separated the top candidates from the rest.
Reference names and concrete reasons. No filler."""


def generate_ranking_rationale(jd_parsed: dict, ranked_results: list) -> str:
    client = _get_client()
    top_3 = ranked_results[:3]
    top_summary = "\n".join([
        f"{r['rank']}. {r['name']} (score: {r['total_score']}, {r['verdict']}) — {r['recruiter_summary'][:200]}"
        for r in top_3
    ])
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a concise, honest senior recruiter. No filler words."},
            {"role": "user", "content": RATIONALE_PROMPT.format(
                n=len(ranked_results),
                job_title=jd_parsed.get("job_title", "Unknown"),
                domain=jd_parsed.get("domain", ""),
                must_haves=", ".join(jd_parsed.get("must_have_skills", [])[:6]),
                top_candidates=top_summary
            )}
        ]
    )
    return response.choices[0].message.content.strip()


def rank_candidates(jd_text: str, candidates: list) -> dict:
    # Step 1 — Parse JD
    jd_parsed = parse_jd(jd_text)

    # Step 2 — Score all candidates in parallel
    results = []
    max_workers = min(5, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cand = {
            executor.submit(score_candidate, cand, jd_parsed): cand
            for cand in candidates
        }
        for future in as_completed(future_to_cand):
            cand = future_to_cand[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    "candidate_id": cand.get("candidate_id", "unknown"),
                    "name": cand.get("profile", {}).get("anonymized_name", "Unknown"),
                    "total_score": 0.0,
                    "verdict": "Error",
                    "dimensions": {},
                    "recruiter_summary": f"Scoring failed: {str(e)}",
                    "red_flags": [f"Scoring error: {str(e)}"],
                    "green_flags": []
                })

    # Step 3 — Sort & rank
    results.sort(key=lambda x: x["total_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Step 4 — Rationale
    rationale = generate_ranking_rationale(jd_parsed, results)

    return {
        "job_title_inferred": jd_parsed.get("job_title", "Unknown Role"),
        "total_candidates": len(results),
        "shortlist": results,
        "ranking_rationale": rationale,
        "jd_parsed": jd_parsed
    }
