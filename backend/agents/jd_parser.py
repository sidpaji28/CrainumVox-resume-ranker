import json
import os
from groq import Groq

def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Create a .env file with GROQ_API_KEY=gsk_... or export it."
        )
    return Groq(api_key=api_key)

MODEL = "llama-3.3-70b-versatile"

JD_PARSE_PROMPT = """You are an expert technical recruiter. Parse the following job description and extract structured requirements.

Return ONLY valid JSON with this exact structure:
{
  "job_title": "inferred title",
  "seniority": "junior|mid|senior|lead|principal",
  "domain": "e.g. ML Engineering, Data Engineering, Backend, etc.",
  "must_have_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill1"],
  "min_years_experience": <number or null>,
  "experience_domains": ["domain1"],
  "key_responsibilities": ["responsibility1"],
  "location_requirement": "remote|onsite|hybrid|any",
  "industry_context": "brief note on domain/industry fit needed",
  "seniority_signals": ["signals from JD that indicate seniority level"]
}

Job Description:
"""


def parse_jd(jd_text: str) -> dict:
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a structured data extractor. Return only valid JSON, no markdown fences, no extra text."},
            {"role": "user", "content": JD_PARSE_PROMPT + jd_text}
        ]
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
