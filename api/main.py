"""
FastAPI backend exposing the predictive maintenance model and the full RAG+LLM
assistant as HTTP endpoints.

Design note on structure: routes map conceptually to what ASP.NET Core MVC controllers
do (route -> request model -> handler -> response model), which is deliberate — it's
the fastest path to real depth in a new framework when there's an existing mental model
to map onto, rather than learning FastAPI's conventions from zero.

The LLM-backed assistant is constructed lazily (on first /ask request, via a cached
dependency) rather than at import time, so the server can start and serve /predict
without requiring an LLM provider API key to be set — useful for testing the ML-only path in
isolation, and a more honest reflection of which parts of this app actually need the
API key.
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from ml.predictor import predict_machine
from orchestration.chain import ManufacturingAssistant

app = FastAPI(
    title="Manufacturing AI Intelligence Platform",
    description="Predictive maintenance + RAG assistant for manufacturing engineers.",
    version="0.1.0",
)


# --- Request/response models -------------------------------------------------

class PredictResponse(BaseModel):
    machine_id: str
    found: bool
    machine_type: str | None = None
    failure_predicted_within_7d: bool | None = None
    failure_probability: float | None = None
    latest_reading_date: str | None = None
    days_since_maintenance: int | None = None
    top_contributing_features: dict | None = None
    error: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class RetrievedChunk(BaseModel):
    title: str
    doc_type: str
    machine_type: str
    distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    routing: dict | None = None
    prediction: dict | None = None
    retrieved_chunks: list[RetrievedChunk] = []


# --- Dependencies --------------------------------------------------------------

@lru_cache(maxsize=1)
def get_assistant() -> ManufacturingAssistant:
    # Cached so ChatOpenAI (and its API-key validation) only happens once, on first
    # use, not on every request and not at server startup.
    return ManufacturingAssistant()


# --- Routes ----------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(machine_id: str):
    if not machine_id or not machine_id.strip():
        raise HTTPException(status_code=422, detail="machine_id must not be empty")

    result = predict_machine(machine_id.strip().upper())
    if not result["found"]:
        # 404, not 200-with-found=false, since the resource being asked about
        # (this machine's data) genuinely doesn't exist — a cleaner REST contract
        # for API consumers than silently returning 200 for a missing resource.
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, assistant: ManufacturingAssistant = Depends(get_assistant)):
    try:
        result = assistant.answer(request.question)
    except Exception as e:
        # Covers missing/invalid API key (for whichever provider is configured) and any
        # other upstream API failure — surfaced as a 502 (this service failed calling an
        # upstream dependency), not a 500, so API consumers can distinguish "our bug"
        # from "the LLM call failed."
        raise HTTPException(status_code=502, detail=f"Assistant failed to generate an answer: {e}")

    return {
        "question": result["question"],
        "answer": result["answer"],
        "routing": result.get("routing"),
        "prediction": result.get("prediction"),
        "retrieved_chunks": result.get("retrieved_chunks", []),
    }
