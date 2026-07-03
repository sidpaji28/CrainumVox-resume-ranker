#!/usr/bin/env python3
"""
CLI tool for ranking candidates against a job description.

Usage:
  python scripts/rank_cli.py \
    --jd sample_data/sample_jd.txt \
    --candidates sample_data/candidates.json \
    --output output/ranked_output.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from agents.ranker import rank_candidates


def main():
    parser = argparse.ArgumentParser(
        description="AI Candidate Ranker — rank candidates against a JD using LLM scoring"
    )
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--candidates", required=True, help="Path to candidates JSON file (array)")
    parser.add_argument("--output", default="output/ranked_output.json", help="Output file path")
    parser.add_argument("--top", type=int, default=None, help="Only show top N candidates")
    parser.add_argument("--csv", action="store_true", help="Also export a CSV summary")
    args = parser.parse_args()

    # Validate API key
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set.\n"
              "  export GROQ_API_KEY=gsk_...\n"
              "  or add it to a .env file and run: source .env")
        sys.exit(1)

    # Load inputs
    print(f"Loading JD from: {args.jd}")
    with open(args.jd) as f:
        jd_text = f.read()

    print(f"Loading candidates from: {args.candidates}")
    with open(args.candidates) as f:
        candidates = json.load(f)

    if not isinstance(candidates, list):
        candidates = [candidates]

    n = len(candidates)
    print(f"\n→ Ranking {n} candidate(s) against the JD...")
    print("  (This makes 1 LLM call per candidate + 2 shared calls. ~20-40s for 5 candidates)\n")

    # Run ranking
    result = rank_candidates(jd_text, candidates)

    # Apply top-N filter
    if args.top:
        result["shortlist"] = result["shortlist"][:args.top]

    # Add metadata
    result["generated_at"] = datetime.utcnow().isoformat() + "Z"
    result["jd_source"] = args.jd
    result["candidates_source"] = args.candidates

    # Save JSON output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  ROLE: {result['job_title_inferred']}")
    print(f"  {result['total_candidates']} candidates evaluated")
    print(f"{'='*70}")
    print(f"  {'RANK':<6} {'NAME':<22} {'SCORE':<8} {'VERDICT'}")
    print(f"  {'-'*60}")
    for c in result["shortlist"]:
        print(f"  #{c['rank']:<5} {c['name']:<22} {c['total_score']:<8.1f} {c['verdict']}")

    print(f"\n  Ranking rationale:")
    for line in result["ranking_rationale"].split(". "):
        if line.strip():
            print(f"  · {line.strip()}.")

    print(f"\n  Full output saved to: {out_path}")

    # Optional CSV export
    if args.csv:
        csv_path = out_path.with_suffix(".csv")
        rows = ["rank,candidate_id,name,total_score,verdict,skill_match,experience,trajectory,engagement,logistics,recruiter_summary"]
        for c in result["shortlist"]:
            d = c.get("dimensions", {})
            summary = c.get("recruiter_summary", "").replace('"', "''")
            rows.append(",".join([
                str(c["rank"]),
                c["candidate_id"],
                c["name"],
                str(c["total_score"]),
                c["verdict"],
                str(d.get("skill_match", {}).get("score", "")),
                str(d.get("experience_relevance", {}).get("score", "")),
                str(d.get("career_trajectory", {}).get("score", "")),
                str(d.get("behavioral_signals", {}).get("score", "")),
                str(d.get("logistics", {}).get("score", "")),
                f'"{summary}"'
            ]))
        with open(csv_path, "w") as f:
            f.write("\n".join(rows))
        print(f"  CSV saved to: {csv_path}")

    print()


if __name__ == "__main__":
    main()
