"""Automatic operational tracing for AgentTools without changing tool behavior."""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import time
from typing import Any, Callable
from uuid import uuid4

from .turn_trace import _compact_result, _safe

logger = logging.getLogger("fitzy.agent")
CURRENT_TURN_TRACE: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "fitzy_current_turn_trace", default=None
)


def set_current_trace(trace: Any) -> contextvars.Token:
    """Bind a turn trace to the current async context."""

    return CURRENT_TURN_TRACE.set(trace)


def reset_current_trace(token: contextvars.Token) -> None:
    """Restore the previous async trace context."""

    CURRENT_TURN_TRACE.reset(token)


def instrument_agent_tools(tools: Any) -> Any:
    """Wrap all public async AgentTools methods once."""

    if getattr(tools, "_fitzy_tracing_installed", False):
        return tools

    for name in dir(tools):
        if name.startswith("_"):
            continue
        method = getattr(tools, name, None)
        if method is None or not inspect.iscoroutinefunction(method):
            continue
        if getattr(method, "_fitzy_traced", False):
            continue
        setattr(tools, name, _wrap_tool(name, method))

    tools._fitzy_tracing_installed = True
    return tools


def _wrap_tool(name: str, method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        trace = CURRENT_TURN_TRACE.get()
        action_id = uuid4().hex
        state = next((arg for arg in args if hasattr(arg, "current_intent")), None)
        intent = getattr(state, "current_intent", None)
        parameters = {
            key: _safe(value)
            for key, value in kwargs.items()
            if key != "state"
        }
        start = time.perf_counter()
        if trace:
            trace.event(
                "TOOL START",
                action_id=action_id,
                intent=intent,
                tool=name,
                parameters=parameters,
            )
        else:
            logger.info(
                "[TOOL START] action_id=%s intent=%s tool=%s parameters=%r",
                action_id, intent, name, parameters,
            )
        try:
            result = await method(*args, **kwargs)
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 2)
            if trace:
                trace.event(
                    "TOOL ERROR",
                    action_id=action_id,
                    intent=intent,
                    tool=name,
                    duration_ms=duration,
                    error=str(exc),
                )
            else:
                logger.exception("[TOOL ERROR] action_id=%s tool=%s", action_id, name)
            raise
        duration = round((time.perf_counter() - start) * 1000, 2)
        summary = _compact_result(result)
        if trace:
            trace.event(
                "TOOL RESULT",
                action_id=action_id,
                intent=intent,
                tool=name,
                duration_ms=duration,
                result=summary,
            )
        else:
            logger.info(
                "[TOOL RESULT] action_id=%s intent=%s tool=%s duration_ms=%s result=%r",
                action_id, intent, name, duration, summary,
            )
        return result

    wrapper._fitzy_traced = True
    return wrapper
