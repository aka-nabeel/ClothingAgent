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
    language: Optional[str] = None


class ChatResponse(BaseModel):
    """Customer-facing Fitzy response mapped for mobile and web frontends."""

    session_id: str
    reply: str
    response: str
    state: Optional[dict[str, Any]] = None


class SessionResetRequest(BaseModel):
    """Inbound session reset payload."""

    session_id: str = Field(min_length=1)
    keep_cart: bool = True
    language: Optional[str] = None


class SessionResetResponse(BaseModel):
    """Session reset output."""

    session_id: str
    status: str = "reset_successful"
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

    reply_text = await agent.process_message(
        session_id=request.session_id,
        message=request.message,
        language=request.language,
    )
    state = agent.get_state(request.session_id)
    if request.language:
        state.set_language(request.language)
    runtime_context = agent._build_runtime_context(state)

    return ChatResponse(
        session_id=request.session_id,
        reply=reply_text,
        response=reply_text,
        state=runtime_context,
    )


@router.post("/session/reset", response_model=SessionResetResponse)
@chat_router.post("/session/reset", response_model=SessionResetResponse)
@router.post("/session/new", response_model=SessionResetResponse)
@chat_router.post("/session/new", response_model=SessionResetResponse)
async def reset_session_endpoint(
    request: SessionResetRequest,
    agent: FitzyAgent = Depends(get_agent),
) -> SessionResetResponse:
    """Flush session state while leaving cart items intact if keep_cart is True."""

    state = agent.get_state(request.session_id)
    state.displayed_products = []
    state.selected_product_id = None
    state.current_search.clear()
    if request.language:
        state.set_language(request.language)
    if not request.keep_cart:
        state.cart.cart_id = None
        state.cart.item_count = 0
    runtime_context = agent._build_runtime_context(state)
    return SessionResetResponse(session_id=request.session_id, state=runtime_context)


@router.delete("/session/{session_id}", response_model=SessionResetResponse)
@chat_router.delete("/session/{session_id}", response_model=SessionResetResponse)
async def delete_session_endpoint(
    session_id: str,
    agent: FitzyAgent = Depends(get_agent),
) -> SessionResetResponse:
    """Delete session state and clear cart when explicitly requested."""

    state = agent.get_state(session_id)
    state.displayed_products = []
    state.selected_product_id = None
    state.current_search.clear()
    state.cart.cart_id = None
    state.cart.item_count = 0
    runtime_context = agent._build_runtime_context(state)
    return SessionResetResponse(session_id=session_id, state=runtime_context)
