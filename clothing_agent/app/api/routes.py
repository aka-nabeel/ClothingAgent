"""HTTP routes exposing Fitzy to the frontend/application team."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..agent.agent import FitzyAgent

router = APIRouter(prefix="/api/v1/agent", tags=["fitzy-agent"])
chat_router = APIRouter(prefix="/api/v1", tags=["fitzy-agent"])


from typing import Any, Optional

class ChatRequest(BaseModel):
    """Inbound customer message for one Fitzy session."""

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """Customer-facing Fitzy response mapped for mobile and web frontends."""

    session_id: str
    reply: str
    response: str
    state: Optional[dict[str, Any]] = None


def get_agent() -> FitzyAgent:
    """Resolve the configured Agent instance.

    The application bootstrap must replace this dependency with its singleton
    runtime instance. Keeping the dependency explicit makes testing easy.
    """

    raise RuntimeError("Fitzy Agent dependency is not configured")


@router.post("/chat", response_model=ChatResponse)
@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, agent: FitzyAgent = Depends(get_agent)) -> ChatResponse:
    """Process one customer message through the Fitzy runtime."""

    reply_text = await agent.process_message(session_id=request.session_id, message=request.message)
    state = agent.get_state(request.session_id)
    runtime_context = agent._build_runtime_context(state)

    return ChatResponse(
        session_id=request.session_id,
        reply=reply_text,
        response=reply_text,
        state=runtime_context,
    )
