import json
import anthropic

client = anthropic.Anthropic()

JD_PARSE_PROMPT = """You are an expert technical recruiter. Parse the following job description and extract structured requirements.

Return ONLY valid JSON with this exact structure:
{{
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
}}

Job Description:
{jd}"""


def parse_jd(jd_text: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": JD_PARSE_PROMPT.format(jd=jd_text)}]
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())