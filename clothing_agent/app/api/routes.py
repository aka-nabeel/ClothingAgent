"""HTTP routes exposing Fitzy to the frontend/application team."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..agent.agent import FitzyAgent

from typing import Any, Optional
from uuid import uuid4

from ..agent.agent import FitzyAgent
from ..agent.tool_tracing import reset_current_trace, set_current_trace
from ..agent.turn_contract import AgentTurnResponse, ContentType
from ..agent.turn_trace import TurnTrace

router = APIRouter(prefix="/api/v1/agent", tags=["fitzy-agent"])
chat_router = APIRouter(prefix="/api/v1", tags=["fitzy-agent"])


class ChatRequest(BaseModel):
    """Inbound customer message for one Fitzy session."""

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    language: Optional[str] = None


class ChatResponse(BaseModel):
    """Customer-facing Fitzy response mapped for mobile and web frontends."""

    session_id: str
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    reply: str
    response: str
    message: Optional[str] = None
    active_agent: str = "Fitzy"
    intent: Optional[str] = None
    content_type: ContentType = ContentType.GENERAL
    products: list[dict[str, Any]] = Field(default_factory=list)
    product: dict[str, Any] | None = None
    cart: dict[str, Any] | None = None
    checkout: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    suggested_actions: list[str] = Field(default_factory=list)
    ui_actions: list[str] = Field(default_factory=list)
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

    trace = TurnTrace(request.session_id, request.message)
    trace.input(request.language)
    token = set_current_trace(trace)
    try:
        turn = await agent.process_message(
            session_id=request.session_id,
            message=request.message,
            language=request.language,
        )
        state = agent.get_state(request.session_id)
        if request.language:
            state.set_language(request.language)
        runtime_context = agent._build_runtime_context(state)

        if isinstance(turn, AgentTurnResponse):
            reply_str = turn.reply
            content_type = turn.content_type
            products = turn.products
            product = turn.product
            cart = turn.cart
            checkout = turn.checkout
            order = turn.order
            suggested_actions = turn.suggested_actions
            ui_actions = turn.ui_actions
        else:
            reply_str = str(turn)
            content_type = ContentType.GENERAL
            products = []
            product = None
            cart = None
            checkout = None
            order = None
            suggested_actions = []
            ui_actions = []

        trace.response(content_type.value if hasattr(content_type, "value") else str(content_type), state.language.value if state.language else "english", reply_str)
        trace.end()

        return ChatResponse(
            session_id=request.session_id,
            conversation_id=request.session_id,
            message_id=uuid4().hex,
            reply=reply_str,
            response=reply_str,
            message=reply_str,
            active_agent="Fitzy",
            intent=getattr(state, "current_intent", None) or "general_chat",
            content_type=content_type,
            products=products,
            product=product,
            cart=cart,
            checkout=checkout,
            order=order,
            suggested_actions=suggested_actions,
            ui_actions=ui_actions,
            state=runtime_context,
        )
    except Exception as exc:
        trace.event("CHAT ERROR", error=str(exc))
        trace.end("error")
        raise
    finally:
        reset_current_trace(token)


@router.post("/session/reset", response_model=SessionResetResponse)
@chat_router.post("/session/reset", response_model=SessionResetResponse)
@router.post("/session/new", response_model=SessionResetResponse)
@chat_router.post("/session/new", response_model=SessionResetResponse)
async def reset_session_endpoint(
    request: SessionResetRequest,
    agent: FitzyAgent = Depends(get_agent),
) -> SessionResetResponse:
    """Flush session state while leaving cart items intact if keep_cart is True."""

    state = agent.reset_state(
        request.session_id,
        keep_cart=request.keep_cart,
        language=request.language,
    )
    runtime_context = agent._build_runtime_context(state)
    return SessionResetResponse(session_id=request.session_id, state=runtime_context)


@router.delete("/session/{session_id}", response_model=SessionResetResponse)
@chat_router.delete("/session/{session_id}", response_model=SessionResetResponse)
async def delete_session_endpoint(
    session_id: str,
    agent: FitzyAgent = Depends(get_agent),
) -> SessionResetResponse:
    """Delete session state and clear cart when explicitly requested."""

    state = agent.reset_state(session_id, keep_cart=False)
    runtime_context = agent._build_runtime_context(state)
    return SessionResetResponse(session_id=session_id, state=runtime_context)
