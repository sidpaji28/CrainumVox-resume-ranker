import json
import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from .jd_parser import parse_jd
from .candidate_scorer import score_candidate

client = anthropic.Anthropic()

RANKING_RATIONALE_PROMPT = """You're a senior recruiter. You've just scored {n} candidates for the following role:

{jd_summary}

Here are the top candidates in ranked order:
{top_candidates}

Write a 2-3 sentence paragraph explaining the key differentiators that determined this ranking.
Be specific — reference actual candidate names and concrete reasons. No filler."""


def generate_ranking_rationale(jd_parsed: dict, ranked_results: list) -> str:
    top_3 = ranked_results[:3]
    top_summary = "\n".join([
        f"{i+1}. {r['name']} (score: {r['total_score']}, verdict: {r['verdict']}) — {r['recruiter_summary'][:150]}..."
        for i, r in enumerate(top_3)
    ])

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": RANKING_RATIONALE_PROMPT.format(
                n=len(ranked_results),
                jd_summary=json.dumps(jd_parsed, indent=2),
                top_candidates=top_summary
            )
        }]
    )
    return response.content[0].text.strip()


def rank_candidates(jd_text: str, candidates: list) -> dict:
    """Main orchestrator: parse JD → score all candidates (parallel) → rank → explain."""

    # Step 1: Parse JD
    jd_parsed = parse_jd(jd_text)

    # Step 2: Score all candidates in parallel (LLM call per candidate)
    results = []
    with ThreadPoolExecutor(max_workers=min(5, len(candidates))) as executor:
        future_to_cand = {
            executor.submit(score_candidate, cand, jd_parsed): cand
            for cand in candidates
        }
        for future in as_completed(future_to_cand):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                cand = future_to_cand[future]
                results.append({
                    "candidate_id": cand.get("candidate_id", "unknown"),
                    "name": cand.get("profile", {}).get("anonymized_name", "Unknown"),
                    "total_score": 0.0,
                    "verdict": "Error",
                    "dimensions": {},
                    "recruiter_summary": f"Scoring failed: {str(e)}",
                    "red_flags": ["Scoring error"],
                    "green_flags": []
                })

    # Step 3: Sort by total score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)

    # Step 4: Assign ranks
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Step 5: Generate overall ranking rationale
    rationale = generate_ranking_rationale(jd_parsed, results)

    return {
        "job_title_inferred": jd_parsed.get("job_title", "Unknown Role"),
        "total_candidates": len(results),
        "shortlist": results,
        "ranking_rationale": rationale,
        "jd_parsed": jd_parsed
    }