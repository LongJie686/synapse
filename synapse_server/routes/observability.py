"""Observability and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from synapse_core.observability import get_hub

router = APIRouter()


@router.get("/metrics")
async def get_metrics() -> dict:
    hub = get_hub()
    return hub.get_metrics_summary()


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": "0.1.0",
        "services": {
            "api": "ok",
            "observability": "ok",
        },
    }
