from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models.schemas import RankRequest
from agents.ranker import rank_candidates
import os

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

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Candidate Ranker API is running. POST to /rank"}


@app.post("/rank")
async def rank(request: RankRequest):
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty")
    if not request.candidates:
        raise HTTPException(status_code=400, detail="candidates list cannot be empty")
    if len(request.candidates) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 candidates per request")

    try:
        result = rank_candidates(request.job_description, request.candidates)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/sample-jd")
async def sample_jd():
    return {
        "job_description": """Senior ML Engineer — Real-Time Inference Platform

We're building the next generation of real-time ML infrastructure at a fast-growing AI company.
You'll own the feature pipeline, model serving layer, and MLOps tooling that powers our core product.

What you'll do:
- Design and build real-time feature pipelines (Kafka, Spark Streaming) for low-latency model inference
- Own the model serving infrastructure: BentoML, Triton, or similar
- Partner with data scientists to productionize models — you'll own the path from notebook to production
- Instrument pipelines with monitoring, alerting, and data quality checks
- Drive the architecture for our vector store (Milvus or Pinecone) for embedding-based retrieval

What we need:
- 5+ years in data/ML engineering
- Strong Python and SQL; PySpark/Spark Streaming for real-time pipelines
- Hands-on with at least one model serving framework
- Experience with warehouse design (Snowflake, BigQuery)
- Familiarity with LLM fine-tuning or NLP work is a big plus
- MLOps mindset: you think about drift, quality, and reliability from day one

Nice to have:
- Vector databases (Milvus, Pinecone, Weaviate)
- LoRA / PEFT fine-tuning experience
- Weights & Biases for experiment tracking

Location: Toronto (onsite preferred, hybrid considered)
"""
    }