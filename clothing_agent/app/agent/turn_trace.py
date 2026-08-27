"""Rich but safe operational logging for one Fitzy conversation turn."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger("fitzy.agent")

_SENSITIVE_KEYS = {
    "phone", "customer_name", "delivery_address", "address", "delivery_notes",
    "api_key", "authorization", "token", "password",
}


def _safe(value: Any) -> Any:
    """Recursively redact sensitive fields before logging."""

    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in _SENSITIVE_KEYS else _safe(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe(item) for item in value]
    return value


def _compact_result(result: Any) -> dict[str, Any]:
    """Summarize known commerce result types without dumping full payloads."""

    if result is None:
        return {"type": "none"}

    summary: dict[str, Any] = {"type": type(result).__name__}
    products = getattr(result, "products", None)
    if products is not None:
        summary.update(
            item_count=len(products),
            product_ids=[getattr(p, "product_id", None) for p in products],
            article_codes=[getattr(p, "article_code", None) for p in products],
        )
        return summary

    if getattr(result, "cart_id", None) is not None:
        summary.update(
            cart_id=str(result.cart_id),
            item_count=getattr(result, "total_quantity", getattr(result, "item_count", None)),
            subtotal=str(getattr(result, "subtotal", None)),
        )

    if getattr(result, "grand_total", None) is not None:
        summary.update(
            subtotal=str(getattr(result, "subtotal", None)),
            discount_total=str(getattr(result, "discount_total", None)),
            delivery_fee=str(getattr(result, "delivery_fee", None)),
            grand_total=str(getattr(result, "grand_total", None)),
        )

    if getattr(result, "order_number", None):
        summary.update(
            order_number=result.order_number,
            status=getattr(result, "status", None),
        )

    product = getattr(result, "product", None)
    if product is not None:
        summary.update(
            product_id=getattr(product, "product_id", None),
            article_code=getattr(product, "article_code", None),
            product_name=getattr(product, "product_name", None),
        )

    return summary


@dataclass
class TurnTrace:
    """Lifecycle logger for a single customer message."""

    session_id: str
    message: str
    request_id: str = field(default_factory=lambda: uuid4().hex)
    started: float = field(default_factory=time.perf_counter)

    def event(self, label: str, **fields: Any) -> None:
        field_strs = []
        for key, value in fields.items():
            if value in (None, "", [], {}):
                continue
            field_strs.append(f"{key}={_safe(value)!r}")
        formatted_fields = " | ".join(field_strs)
        logger.info(
            "[--- %s ---] request=%s session=%s%s",
            label,
            self.request_id[:8],
            self.session_id,
            f" | {formatted_fields}" if formatted_fields else "",
        )

    def input(self, explicit_language: str | None = None, log_raw: bool = False) -> None:
        if log_raw:
            self.event("CHAT INPUT", message=self.message, explicit_language=explicit_language)
        else:
            self.event("CHAT INPUT", message_length=len(self.message or ""), explicit_language=explicit_language)

    def language(self, value: str, source: str = "deterministic") -> None:
        self.event("LANGUAGE", language=value, source=source)

    def intent(self, intents: list[str], extracted: dict[str, Any] | None = None) -> None:
        self.event("INTENT EXTRACTED", intents=intents, extracted=extracted)

    def plan(self, plan_id: str, actions: list[Any]) -> None:
        self.event("PLAN", plan_id=plan_id, actions=actions)

    def requirements(self, action_id: str, required: list[str], missing: list[str]) -> None:
        self.event("REQUIREMENTS", action_id=action_id, required=required, missing=missing)

    def state(self, **summary: Any) -> None:
        self.event("STATE", **summary)

    def response(self, content_type: str, language: str, reply: str, log_raw: bool = False) -> None:
        if log_raw:
            self.event("CHAT REPLY", content_type=content_type, language=language, reply=reply)
        else:
            self.event("CHAT REPLY", content_type=content_type, language=language, reply_length=len(reply or ""))

    def end(self, status: str = "success") -> None:
        self.event(
            "CHAT END",
            status=status,
            total_duration_ms=round((time.perf_counter() - self.started) * 1000, 2),
        )
