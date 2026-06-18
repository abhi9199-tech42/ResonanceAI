"""
ResonanceAI REST API — FastAPI wrapper for production deployment.

Endpoints:
  POST /detect    — Hallucination detection
  POST /verify    — QA verification
  POST /learn     — One-shot concept learning
  GET  /health    — Health check
  GET  /version   — Version info
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("resonanceai.api")


# ── Request / Response Models ────────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to check for hallucination")
    top_k: int = Field(5, ge=1, le=50, description="Number of nearest matches to return")
    resonance_dim: int = Field(2048, ge=64, le=8192)


class DetectResponse(BaseModel):
    confidence: float = Field(..., ge=0, le=1)
    raw_cosine: float
    nn_label: str
    nearest_matches: List[dict]
    latency_ms: float


class VerifyRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    answer: str = Field(..., min_length=1, max_length=5000)
    choices: Optional[List[str]] = None
    resonance_dim: int = Field(2048, ge=64, le=8192)


class VerifyResponse(BaseModel):
    confidence: float = Field(..., ge=0, le=1)
    winner: str
    details: List[dict]
    latency_ms: float


class LearnRequest(BaseModel):
    concept: str = Field(..., min_length=1, max_length=500)
    definition: str = Field(..., min_length=1, max_length=5000)


class LearnResponse(BaseModel):
    success: bool
    concept: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float
    hippocampus_size: int


# ── App ──────────────────────────────────────────────────────────────────────

_system = None
_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _system, _start_time
    _start_time = time.time()
    from urcm.core.system import URCMSystem
    _system = URCMSystem(resonance_dim=2048)
    logger.info("URCM system initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ResonanceAI",
    version="0.2.0",
    description="Hallucination detection and QA verification via phoneme resonance",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version="0.2.0",
        uptime_s=round(time.time() - _start_time, 1) if _start_time else 0,
        hippocampus_size=len(_system.hippocampus) if _system else 0,
    )


@app.get("/version")
def version():
    return {"version": "0.2.0", "model": "URCM", "resonance_dim": 2048}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if _system is None:
        raise HTTPException(503, "System not initialized")
    t0 = time.time()
    try:
        result = _system.detect_hallucination(req.text, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(500, f"Detection failed: {e}")
    latency = (time.time() - t0) * 1000
    return DetectResponse(
        confidence=result["confidence"],
        raw_cosine=result["raw_cosine"],
        nn_label=result["nn_label"],
        nearest_matches=result.get("nearest_matches", []),
        latency_ms=round(latency, 1),
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    if _system is None:
        raise HTTPException(503, "System not initialized")
    t0 = time.time()
    try:
        result = _system.verify_qa(req.question, req.answer, choices=req.choices)
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(500, f"Verification failed: {e}")
    latency = (time.time() - t0) * 1000
    return VerifyResponse(
        confidence=result["confidence"],
        winner=result.get("winner", ""),
        details=result.get("details", []),
        latency_ms=round(latency, 1),
    )


@app.post("/learn", response_model=LearnResponse)
def learn(req: LearnRequest):
    if _system is None:
        raise HTTPException(503, "System not initialized")
    t0 = time.time()
    try:
        _system.learn_concept_oneshot(req.concept, req.definition)
    except Exception as e:
        logger.error(f"Learning failed: {e}")
        raise HTTPException(500, f"Learning failed: {e}")
    latency = (time.time() - t0) * 1000
    return LearnResponse(success=True, concept=req.concept, latency_ms=round(latency, 1))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
