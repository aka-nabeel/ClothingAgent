"""Health and readiness checks for the clothing application."""

import logging

from app.catalog.models import Branch, Product
from app.database import get_session_factory
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from clothing_agent.app.core.container import get_container

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    """Confirm that the FastAPI process is running and LLM client status."""
    container = get_container()
    return {
        "status": "ok",
        "service": "clothing-app",
        "llm_configured": container.llm.configured,
    }


@router.get("/health/ready")
async def readiness() -> dict[str, object]:
    """Verify database connectivity and LLM configuration readiness."""
    try:
        async with get_session_factory()() as db:
            await db.execute(select(Branch.branch_id).limit(1))
            await db.execute(select(Product.product_id).limit(1))
    except Exception as exc:
        logger.exception("database_readiness_failed", extra={"event": "database_readiness_failed"})
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "database": "unavailable_or_schema_mismatch"},
        ) from exc

    container = get_container()
    if not container.llm.configured and not container.config.allow_local_fallback:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "database": "connected", "llm": "not_configured"},
        )

    return {
        "status": "ready",
        "database": "connected",
        "llm": "configured" if container.llm.configured else "local_fallback",
    }
