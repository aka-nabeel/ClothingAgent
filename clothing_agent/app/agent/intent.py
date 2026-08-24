"""Structured intent models for Fitzy's Phase 2 planning layer.

The LLM populates these models. This module normalizes what the customer
appears to want into semantic intents that the planner can turn into executable actions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import ToolName
from .state import LanguageMode


class IntentType(StrEnum):
    """Customer-level intent names understood by the V1 planner."""

    STORE_CONTEXT = "store_context"
    PRODUCT_SEARCH = "product_search"
    PRODUCT_DETAILS = "product_details"
    BRANCH_INFORMATION = "branch_information"
    AVAILABILITY_CHECK = "availability_check"
    CREATE_CART = "create_cart"
    VIEW_CART = "view_cart"
    ADD_TO_CART = "add_to_cart"
    UPDATE_CART = "update_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    CLEAR_CART = "clear_cart"
    CHECKOUT = "checkout"
    PLACE_ORDER = "place_order"
    GENERAL_CONVERSATION = "general_conversation"


class IntentRequest(BaseModel):
    """One normalized customer intent produced by the intent extractor."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    intent_id: str = Field(default="intent-1", min_length=1)
    intent_type: IntentType = Field(validation_alias=AliasChoices("intent_type", "intent", "name", "type", "tool"))
    parameters: dict[str, Any] = Field(default_factory=dict)
    explicit_confirmation: bool | None = None
    customer_choice_required: bool = False

    @field_validator("intent_type", mode="before")
    @classmethod
    def normalize_intent_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            val = value.lower().strip()
            aliases = {
                "get_products": IntentType.PRODUCT_SEARCH,
                "search_products": IntentType.PRODUCT_SEARCH,
                "product_search": IntentType.PRODUCT_SEARCH,
                "search": IntentType.PRODUCT_SEARCH,
                "get_product_details": IntentType.PRODUCT_DETAILS,
                "product_details": IntentType.PRODUCT_DETAILS,
                "details": IntentType.PRODUCT_DETAILS,
                "get_branches": IntentType.BRANCH_INFORMATION,
                "branch_information": IntentType.BRANCH_INFORMATION,
                "check_availability": IntentType.AVAILABILITY_CHECK,
                "availability_check": IntentType.AVAILABILITY_CHECK,
                "preview_checkout": IntentType.CHECKOUT,
                "checkout": IntentType.CHECKOUT,
                "get_cart": IntentType.VIEW_CART,
                "view_cart": IntentType.VIEW_CART,
                "add_to_cart": IntentType.ADD_TO_CART,
                "update_cart": IntentType.UPDATE_CART,
                "remove_from_cart": IntentType.REMOVE_FROM_CART,
                "clear_cart": IntentType.CLEAR_CART,
                "get_store_context": IntentType.STORE_CONTEXT,
                "store_context": IntentType.STORE_CONTEXT,
                "what_products": IntentType.STORE_CONTEXT,
                "catalog": IntentType.STORE_CONTEXT,
                "categories": IntentType.STORE_CONTEXT,
                "available_products": IntentType.STORE_CONTEXT,
                "product_catalog": IntentType.STORE_CONTEXT,
                "place_order": IntentType.PLACE_ORDER,
                "general_conversation": IntentType.GENERAL_CONVERSATION,
            }
            if val in aliases:
                return aliases[val]
        return value


class IntentExtraction(BaseModel):
    """Complete normalized result for one customer message."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    language: LanguageMode = Field(default=LanguageMode.ENGLISH)
    intents: list[IntentRequest] = Field(default_factory=list)

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, value: Any) -> Any:
        if isinstance(value, str):
            val = value.lower().strip()
            if "roman" in val:
                return LanguageMode.ROMAN_URDU
            if "urdu" in val or val in ("ur", "pk", "urdu_script"):
                return LanguageMode.URDU_SCRIPT
            if "english" in val or val in ("en", "us", "uk", "eng"):
                return LanguageMode.ENGLISH
        return value or LanguageMode.ENGLISH

    @model_validator(mode="before")
    @classmethod
    def wrap_single_intent(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "intents" not in data and ("intent" in data or "name" in data or "intent_type" in data):
                data["intents"] = [data]
        return data

    @property
    def is_empty(self) -> bool:
        """Return True when no actionable intent was extracted."""
        return not self.intents


def intent_to_tool(intent_type: IntentType) -> ToolName | None:
    """Map a semantic customer intent to the canonical tool contract."""

    mapping: dict[IntentType, ToolName] = {
        IntentType.STORE_CONTEXT: ToolName.GET_STORE_CONTEXT,
        IntentType.PRODUCT_SEARCH: ToolName.GET_PRODUCTS,
        IntentType.PRODUCT_DETAILS: ToolName.GET_PRODUCT_DETAILS,
        IntentType.BRANCH_INFORMATION: ToolName.GET_BRANCHES,
        IntentType.AVAILABILITY_CHECK: ToolName.CHECK_AVAILABILITY,
        IntentType.CREATE_CART: ToolName.CREATE_CART,
        IntentType.VIEW_CART: ToolName.GET_CART,
        IntentType.ADD_TO_CART: ToolName.ADD_TO_CART,
        IntentType.UPDATE_CART: ToolName.UPDATE_CART,
        IntentType.REMOVE_FROM_CART: ToolName.REMOVE_FROM_CART,
        IntentType.CLEAR_CART: ToolName.CLEAR_CART,
        IntentType.CHECKOUT: ToolName.PREVIEW_CHECKOUT,
        IntentType.PLACE_ORDER: ToolName.PLACE_ORDER,
    }
    return mapping.get(intent_type)


def classify_language(message: str, current_language: LanguageMode | None = None) -> LanguageMode:
    """Deterministically classify input language into English, Roman Urdu, Urdu script, or fallback from unsupported Hindi."""
    import re

    # 1. Devanagari / Hindi check: Never adopt Devanagari. Retain established language or default to English.
    if re.search(r"[\u0900-\u097F]", message):
        return current_language or LanguageMode.ENGLISH

    # 2. Explicit Urdu Script check: Contains Arabic/Urdu unicode range
    if re.search(r"[\u0600-\u06FF]", message):
        return LanguageMode.URDU_SCRIPT

    # 3. Explicit Roman Urdu check: Check for distinctive Roman Urdu tokens
    roman_urdu_words = {
        "mujhe", "chahiye", "kuch", "shadi", "karo", "dikhao", "apna", "hai",
        "hain", "kya", "batao", "kaunsa", "kitne", "pehan", "kapray", "bhej",
        "do", "kardo", "aacha", "bhi", "wala", "wali", "wale", "karni", "hote",
        "sub", "sab", "yeh", "woh", "kis", "kisi", "mere", "meri", "humara", "humari"
    }
    tokens = set(re.findall(r"\b\w+\b", message.lower()))
    if tokens & roman_urdu_words:
        return LanguageMode.ROMAN_URDU

    # 4. If a session language is already established (e.g. URDU_SCRIPT or ROMAN_URDU),
    # retain established session language on priority instead of drifting to English.
    if current_language is not None:
        return current_language

    # 5. Default fallback for initial turn without established language
    return LanguageMode.ENGLISH
