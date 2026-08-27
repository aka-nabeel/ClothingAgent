"""Health and readiness checks for the agent API."""

import logging

from .config import get_config
from .container import get_container
from .errors import AgentError
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    """Confirm that the agent process and dependencies are initialized."""
    container = get_container()
    return {
        "status": "ok",
        "service": "clothing-agent",
        "agent": "monolithic",
        "llm_configured": container.llm.configured,
    }


@router.get("/health/ready")
async def readiness() -> dict[str, object]:
    """Verify the commerce backend connection and LLM configuration."""
    container = get_container()
    config = get_config()
    backend_status = "ok"
    try:
        branches = await container.tool_adapter.get_branches()
        if not branches:
            backend_status = "degraded"
    except Exception as exc:
        logger.warning(
            "readiness_dependency_failed",
            extra={"event": "readiness_dependency_failed", "error": str(exc)},
        )
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "commerce_backend": str(exc)},
        ) from exc

    if not container.llm.configured and not config.allow_local_fallback:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "llm": "not_configured"},
        )
    return {
        "status": "ready",
        "commerce_backend": backend_status,
        "llm": "configured" if container.llm.configured else "local_fallback",
    }
