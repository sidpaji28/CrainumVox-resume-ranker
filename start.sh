#!/bin/bash
set -e

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║        AI Candidate Ranker            ║"
echo "  ║        powered by Groq + Llama        ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

if [ -f .env ]; then
  echo "  → Loading .env..."
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$GROQ_API_KEY" ]; then
  echo "  ✗ GROQ_API_KEY is not set."
  echo ""
  echo "  Get your free key at: https://console.groq.com"
  echo ""
  echo "  Option 1: Create a .env file:"
  echo "    echo 'GROQ_API_KEY=gsk_...' > .env"
  echo ""
  echo "  Option 2: Export it:"
  echo "    export GROQ_API_KEY=gsk_..."
  echo ""
  exit 1
fi

echo "  ✓ Groq API key found"

if ! python -c "import fastapi, groq" 2>/dev/null; then
  echo "  → Installing dependencies..."
  pip install -r backend/requirements.txt -q
fi

echo "  ✓ Dependencies ready"
echo "  ✓ Model: llama-3.3-70b-versatile"
echo ""
echo "  Starting server on http://localhost:8000"
echo ""

cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
