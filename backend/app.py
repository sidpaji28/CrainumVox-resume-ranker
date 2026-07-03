import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from agents.ranker import rank_candidates

app = FastAPI(
    title="AI Candidate Ranker",
    description="Ranks candidates the way a great recruiter would — not by keywords, but by actual fit.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
SAMPLE_DIR   = Path(__file__).parent.parent / "sample_data"
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Request / Response models ──────────────────────────────────────
class RankRequest(BaseModel):
    job_description: str
    candidates: List[Dict[str, Any]]
    save_output: Optional[bool] = False


# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "AI Candidate Ranker API. POST to /rank"}


@app.get("/health")
async def health():
    api_key_set = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "ok",
        "api_key_configured": api_key_set,
        "version": "1.0.0"
    }


@app.post("/rank")
async def rank(request: RankRequest):
    if not request.job_description.strip():
        raise HTTPException(400, "job_description cannot be empty")
    if not request.candidates:
        raise HTTPException(400, "candidates list cannot be empty")
    if len(request.candidates) > 100:
        raise HTTPException(400, "Maximum 100 candidates per request")

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            500,
            "GROQ_API_KEY not configured on the server. "
            "Set it via environment variable or .env file."
        )

    try:
        result = rank_candidates(request.job_description, request.candidates)

        if request.save_output:
            from datetime import datetime
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_file = OUTPUT_DIR / f"ranked_{ts}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            result["saved_to"] = str(out_file)

        return result

    except ValueError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ranking failed: {str(e)}")


@app.get("/sample-jd")
async def sample_jd():
    jd_file = SAMPLE_DIR / "sample_jd.txt"
    if jd_file.exists():
        return {"job_description": jd_file.read_text()}
    return {
        "job_description": (
            "Senior ML Engineer — Real-Time Inference Platform\n\n"
            "5+ years in data/ML engineering. Strong Python, PySpark, Kafka.\n"
            "Hands-on with model serving (BentoML/Triton). Snowflake/BigQuery.\n"
            "NLP / LLM fine-tuning a big plus. Vector DB experience (Milvus/Pinecone).\n"
            "Location: Toronto (onsite preferred)."
        )
    }


@app.get("/sample-candidates")
async def sample_candidates():
    cand_file = SAMPLE_DIR / "candidates.json"
    if cand_file.exists():
        return json.loads(cand_file.read_text())
    return []


@app.get("/outputs")
async def list_outputs():
    files = sorted(OUTPUT_DIR.glob("ranked_*.json"), reverse=True)
    return {
        "files": [
            {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in files[:20]
        ]
    }


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.suffix == ".json":
        raise HTTPException(404, "Output file not found")
    return JSONResponse(json.loads(path.read_text()))


# Mount frontend last (catches /static/*)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
