# AI Candidate Ranker

> Ranks candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role.
> Powered by **Groq API** + **Llama 3.3 70B** for ultra-fast inference.

## Architecture

```
Job Description (text)
        │
        ▼
┌───────────────────┐
│   JD Parser       │  Llama-3.3-70b via Groq
│   (LLM Agent)     │  → Extracts: must-have skills, seniority,
└───────────────────┘    domain, location, responsibilities
        │
        ▼
┌────────────────────────────────────────────────────┐
│          Candidate Scorer  (parallel, 1 call each) │
│                                                     │
│  ┌──────────────────────┐  ┌───────────────────┐   │
│  │  LLM Scoring (Groq)  │  │  Rule-based        │   │
│  │                      │  │                    │   │
│  │  • Skill Match  30%  │  │ • Behavioral  20%  │   │
│  │  • Experience   25%  │  │   (platform sigs)  │   │
│  │  • Trajectory   15%  │  │ • Logistics   10%  │   │
│  └──────────────────────┘  └───────────────────┘   │
└────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│   Ranker          │  Sort → Assign ranks → Explain
└───────────────────┘
        │
        ▼
  Ranked shortlist (JSON + optional CSV)
```

## Scoring Dimensions

| Dimension | Weight | How scored |
|---|---|---|
| Skill Match | 30% | LLM: semantic alignment of skills to JD requirements |
| Experience Relevance | 25% | LLM: does their actual work history map to this role? |
| Career Trajectory | 15% | LLM: is their arc pointing toward this role or away? |
| Behavioral Signals | 20% | Rule-based: response rate, interview completion, offer acceptance, GitHub, verification |
| Logistics | 10% | Rule-based: work mode match, relocation, salary transparency |

**Verdict:** Strong Fit (≥75) · Moderate Fit (≥55) · Weak Fit (≥35) · Poor Fit (<35)

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd candidate-ranker
pip install -r backend/requirements.txt
```

### 2. Get your Groq API key

Free at: https://console.groq.com

```bash
cp .env.example .env
# Add your key to .env:  GROQ_API_KEY=gsk_...
```

### 3a. Run the web app

```bash
./start.sh
# Open http://localhost:8000
```

### 3b. Run from CLI

```bash
python scripts/rank_cli.py \
  --jd sample_data/sample_jd.txt \
  --candidates sample_data/candidates.json \
  --output output/ranked_output.json \
  --csv
```

### 3c. REST API

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"job_description": "...", "candidates": [...]}'
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/rank` | Main ranking endpoint |
| GET | `/health` | Server + API key status |
| GET | `/sample-jd` | Sample job description |
| GET | `/sample-candidates` | Sample candidate profiles |
| GET | `/outputs` | List saved output files |
| GET | `/outputs/{filename}` | Retrieve a saved output |

## Project Structure

```
candidate-ranker/
├── backend/
│   ├── app.py                   ← FastAPI application
│   ├── requirements.txt
│   ├── models/schemas.py
│   └── agents/
│       ├── jd_parser.py         ← LLM: JD → structured requirements
│       ├── candidate_scorer.py  ← LLM: 3-dim scoring + logistics
│       ├── behavioral_scorer.py ← Rule-based: platform signals
│       └── ranker.py            ← Orchestrator: parallel → rank → explain
├── frontend/index.html          ← Single-page UI
├── scripts/rank_cli.py          ← CLI tool
├── sample_data/
│   ├── candidates.json          ← 5 sample candidate profiles
│   └── sample_jd.txt
├── output/
├── start.sh
├── .env.example
└── README.md
```

## Why Groq?

- **Speed**: Groq's LPU inference is 10–20x faster than typical LLM APIs — scoring 5 candidates takes ~5s instead of 40s
- **Free tier**: Generous free API access at console.groq.com
- **Llama 3.3 70B**: Strong reasoning, great at structured JSON output, open-weights model
