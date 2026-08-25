"""Structured response contract returned by the Fitzy chat endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Frontend rendering hint for a customer-facing Agent turn."""

    GENERAL = "GENERAL"
    TEXT = "TEXT"
    PRODUCT_LIST = "PRODUCT_LIST"
    PRODUCT_DETAILS = "PRODUCT_DETAILS"
    CART = "CART"
    CHECKOUT = "CHECKOUT"
    ORDER = "ORDER"


class ToolTrace(BaseModel):
    """Optional safe diagnostic trace for a tool call."""

    action_id: str
    intent: str | None = None
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str
    result_summary: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None


class AgentTurnResponse(BaseModel):
    """Conversation result plus trusted structured UI payloads.

    The payloads are produced from commerce results; the LLM only supplies the
    conversational message.
    """

    session_id: str
    reply: str
    language: str = "english"
    content_type: ContentType = ContentType.GENERAL
    products: list[dict[str, Any]] = Field(default_factory=list)
    product: dict[str, Any] | None = None
    cart: dict[str, Any] | None = None
    checkout: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    suggested_actions: list[str] = Field(default_factory=list)
    ui_actions: list[str] = Field(default_factory=list)
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        return self.reply

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.reply == other
        return super().__eq__(other)

    def lower(self) -> str:
        return self.reply.lower()

    def strip(self, *args: Any, **kwargs: Any) -> str:
        return self.reply.strip(*args, **kwargs)

    def __contains__(self, item: object) -> bool:
        return str(item) in self.reply


def decimal_safe(value: Any) -> Any:
    """Convert Decimal values to stable JSON-friendly strings."""

    if value is None:
        return None
    if hasattr(value, "quantize") and hasattr(value, "__str__"):
        return str(value)
    return value


def _model_or_fields(value: Any, fields: dict[str, Any]) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return fields


def product_card(product: Any) -> dict[str, Any]:
    """Build a compact detail payload from a trusted ProductView."""

    variants = getattr(product, "variants", []) or getattr(product, "options", []) or []
    available_variants = [v for v in variants if getattr(v, "is_available", True)]
    colors = sorted({str(v.color) for v in variants if getattr(v, "color", None)})
    sizes = sorted({str(v.size) for v in variants if getattr(v, "size", None)})
    available_colors = sorted({str(v.color) for v in available_variants if getattr(v, "color", None)})
    available_sizes = sorted({str(v.size) for v in available_variants if getattr(v, "size", None)})

    return {
        "product_id": getattr(product, "product_id", None),
        "article_code": getattr(product, "article_code", ""),
        "product_name": getattr(product, "product_name", ""),
        "description": getattr(product, "description", None),
        "category": getattr(product, "category", None),
        "subcategory": getattr(product, "subcategory", None),
        "product_type": getattr(product, "product_type", None),
        "gender": getattr(product, "gender", None),
        "brand": getattr(product, "brand", None),
        "material": getattr(product, "material", None),
        "fit": getattr(product, "fit", None),
        "season": getattr(product, "season", None),
        "occasion": getattr(product, "occasion", None),
        "base_price": decimal_safe(getattr(product, "base_price", getattr(product, "price", None))),
        "final_price": decimal_safe(getattr(product, "final_price", getattr(product, "price", None))),
        "discount_amount": decimal_safe(getattr(product, "discount_amount", 0)),
        "applied_offer": product.applied_offer.model_dump(mode="json") if getattr(product, "applied_offer", None) else None,
        "images": list(getattr(product, "images", []) or getattr(product, "image_urls", []) or []),
        "colors": colors,
        "sizes": sizes,
        "available_colors": available_colors,
        "available_sizes": available_sizes,
        "variants": [
            {
                "variant_id": v.variant_id,
                "sku": getattr(v, "sku", ""),
                "color": getattr(v, "color", ""),
                "size": getattr(v, "size", ""),
                "price": decimal_safe(getattr(v, "price", None)),
                "final_price": decimal_safe(getattr(v, "final_price", getattr(v, "price", None))),
                "discount_amount": decimal_safe(getattr(v, "discount_amount", 0)),
                "is_available": getattr(v, "is_available", True),
                "branch_availability": [
                    _model_or_fields(
                        b,
                        {
                            "branch_code": getattr(b, "branch_code", None),
                            "branch_name": getattr(b, "branch_name", None),
                            "is_available": getattr(b, "is_available", False),
                            "available_quantity": getattr(b, "available_quantity", 0),
                        },
                    )
                    for b in (getattr(v, "branch_availability", []) or [])
                ],
            }
            for v in variants
        ],
    }
