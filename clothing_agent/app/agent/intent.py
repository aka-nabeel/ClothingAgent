"""Structured intent extraction and deterministic language classification for Fitzy."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .intents import IntentName, registered_intent_values


class LanguageCode(str, Enum):
    """Languages Fitzy can return to customers."""

    ENGLISH = "english"
    ROMAN_URDU = "roman_urdu"
    URDU_SCRIPT = "urdu_script"


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
ROMAN_URDU_HINTS = {
    "mujhe", "chahiye", "dikhao", "dikha", "karo", "kar do", "hain", "hai",
    "mera", "meri", "mere", "aap", "apka", "ke", "ki", "ka", "mein",
    "yeh", "ye", "woh", "wo", "kya", "kitna", "kitni", "sirf", "phir",
    "shadi", "shaadi", "kapray", "kapre", "acha", "achi", "do",
}



def classify_language(text: str, previous: LanguageCode | str | None = None) -> LanguageCode:
    """Classify user input into English, Roman Urdu, or Urdu script.

    This is deliberately deterministic. The LLM may enrich intent meaning,
    but it does not choose the response language independently.
    """
    text = (text or "").strip()
    prev_str = previous.value if hasattr(previous, "value") else str(previous) if previous else None
    prev_lang: LanguageCode | None = None
    if prev_str:
        if prev_str in ("urdu_script", "urdu"):
            prev_lang = LanguageCode.URDU_SCRIPT
        elif prev_str in ("roman_urdu", "roman urdu", "roman"):
            prev_lang = LanguageCode.ROMAN_URDU
        elif prev_str in ("english", "en"):
            prev_lang = LanguageCode.ENGLISH

    if not text:
        return prev_lang or LanguageCode.ENGLISH

    if DEVANAGARI_RE.search(text):
        # Unsupported Hindi/Devanagari never becomes a supported response language.
        return prev_lang or LanguageCode.ROMAN_URDU

    if URDU_SCRIPT_RE.search(text):
        return LanguageCode.URDU_SCRIPT

    words = {word.lower() for word in re.findall(r"[A-Za-z']+", text)}
    roman_hits = len(words & ROMAN_URDU_HINTS)
    if roman_hits >= 1:
        return LanguageCode.ROMAN_URDU

    # Preserve an already-established Urdu Script or Roman Urdu conversation
    if prev_lang in (LanguageCode.URDU_SCRIPT, LanguageCode.ROMAN_URDU):
        return prev_lang

    return LanguageCode.ENGLISH



class SearchOverrides(BaseModel):
    """Ephemeral filters extracted from the current turn."""

    model_config = ConfigDict(extra="ignore")

    categories: list[str] = Field(default_factory=list)
    product_types: list[str] = Field(default_factory=list)
    occasions: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    excluded_colors: list[str] = Field(default_factory=list)
    size_mapping: dict[str, str] = Field(default_factory=dict)
    materials: list[str] = Field(default_factory=list)
    fits: list[str] = Field(default_factory=list)
    semantic_tags: list[str] = Field(default_factory=list)
    minimum_price: float | None = None
    maximum_price: float | None = None
    branch_reference: str | None = None
    article_code: str | None = None
    sku: str | None = None


class DeliveryExtraction(BaseModel):
    """Delivery fields explicitly mentioned by the customer in this turn."""

    model_config = ConfigDict(extra="ignore")

    customer_name: str | None = None
    phone: str | None = None
    delivery_address: str | None = None
    city: str | None = None
    delivery_notes: str | None = None


class ProductReference(BaseModel):
    """A conversational or explicit product reference."""

    model_config = ConfigDict(extra="ignore")

    index: int | None = None
    product_id: int | None = None
    article_code: str | None = None
    sku: str | None = None
    text_reference: str | None = None


class StructuredIntent(BaseModel):
    """Closed, structured interpretation of one user message.

    `intents` supports multi-intent messages. `primary_intent` is the first
    action to consider when multiple actions exist. The model never decides
    backend truth; it only identifies user intent and explicit parameters.
    """

    model_config = ConfigDict(extra="ignore")

    intents: list[IntentName] = Field(min_length=1, max_length=8)
    primary_intent: IntentName | None = None

    language: LanguageCode | None = None

    search_overrides: SearchOverrides = Field(default_factory=SearchOverrides)
    delivery: DeliveryExtraction = Field(default_factory=DeliveryExtraction)

    product_reference: ProductReference | None = None
    cart_item_index: int | None = None
    quantity: int | None = Field(default=None, ge=1)

    explicit_confirmation: bool | None = None
    cancel_requested: bool = False
    change_topic: bool = False
    reset_shopping: bool = False

    unavailable_interest: ProductReference | None = None

    @field_validator("primary_intent")
    @classmethod
    def validate_primary_intent(
        cls, value: IntentName | None, info: Any
    ) -> IntentName | None:
        if value is None:
            return value
        intents = info.data.get("intents") or []
        if intents and value not in intents:
            raise ValueError("primary_intent must be one of intents")
        return value

    @field_validator("delivery", "search_overrides", mode="before")
    @classmethod
    def normalize_null_objects(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    @field_validator("product_reference", mode="before")
    @classmethod
    def normalize_product_reference(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            ref_str = value.lower().strip()
            word_map = {"first": 1, "1st": 1, "one": 1, "second": 2, "2nd": 2, "two": 2, "third": 3, "3rd": 3, "three": 3}
            idx = None
            for k, v in word_map.items():
                if k in ref_str:
                    idx = v
                    break
            return ProductReference(text_reference=value, index=idx)
        if isinstance(value, int):
            return ProductReference(index=value)
        return value

    @field_validator("intents", mode="before")
    @classmethod
    def normalize_intents(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            cleaned = []
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("intent") or item.get("intent_name") or item.get("value")
                    if name:
                        cleaned.append(name)
                elif hasattr(item, "value"):
                    cleaned.append(item.value)
                elif isinstance(item, str):
                    cleaned.append(item)
                else:
                    cleaned.append(item)
            return cleaned
        raise TypeError("intents must be a string or list of strings")


def build_intent_schema_description() -> str:
    """Build the closed intent instructions embedded in the LLM system prompt."""
    lines = [
        "You MUST select intents only from this registered vocabulary:",
        *[f"- {intent}" for intent in registered_intent_values()],
        "",
        "Multi-intent requests MUST be represented in `intents` as an ordered list.",
        "Do not invent new intent names.",
        "Use search_overrides for temporary product filters.",
        "Use product_reference for phrases such as 'first one', 'that shirt', or an article/SKU.",
        "Use delivery for customer delivery information explicitly supplied in the message.",
        "Use explicit_confirmation only when the customer clearly confirms a pending order.",
        "Never infer backend inventory, price, promotion, branch stock, or order truth.",
    ]
    return "\n".join(lines)
