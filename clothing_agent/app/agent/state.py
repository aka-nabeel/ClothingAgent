"""Conversation state for Fitzy's deterministic execution layer.

The state is deliberately separated from the LLM.  It stores facts already
established during the conversation so the Agent can reuse them without asking
the customer again, while avoiding duplicated business logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ToolName


class LanguageMode(StrEnum):
    """Supported V1 response modes for English and Urdu conversations."""

    ENGLISH = "english"
    URDU_SCRIPT = "urdu_script"
    ROMAN_URDU = "roman_urdu"


class CustomerPreferences(BaseModel):
    """Longer-lived customer preferences explicitly established in conversation."""

    model_config = ConfigDict(extra="forbid")

    preferred_colors: list[str] = Field(default_factory=list)
    excluded_colors: list[str] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    preferred_product_types: list[str] = Field(default_factory=list)
    preferred_occasions: list[str] = Field(default_factory=list)
    preferred_materials: list[str] = Field(default_factory=list)
    preferred_fits: list[str] = Field(default_factory=list)
    size_mapping: dict[str, str] = Field(default_factory=dict)
    minimum_price: Decimal | None = None
    maximum_price: Decimal | None = None
    branch_preference: str | None = None


class SearchContext(BaseModel):
    """Temporary filters describing the currently active product search."""

    model_config = ConfigDict(extra="forbid")

    query_text: str | None = None
    categories: list[str] = Field(default_factory=list)
    product_types: list[str] = Field(default_factory=list)
    occasions: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    excluded_colors: list[str] = Field(default_factory=list)
    size_mapping: dict[str, str] = Field(default_factory=dict)
    minimum_price: Decimal | None = None
    maximum_price: Decimal | None = None
    branch_code: str | None = None
    article_code: str | None = None
    sku: str | None = None
    in_stock_only: bool = True
    limit: int = 4

    def update_from_mapping(self, values: dict[str, Any]) -> None:
        """Apply only explicitly supplied search values without losing unrelated filters."""

        for field_name, value in values.items():
            if value is None:
                continue
            if field_name not in type(self).model_fields:
                continue
            if isinstance(value, list) and not value:
                continue
            if isinstance(value, dict) and not value:
                continue
            setattr(self, field_name, value)

    def clear(self) -> None:
        """Reset temporary search filters while leaving customer/cart state intact."""

        defaults = type(self)()
        for field_name in type(self).model_fields:
            setattr(self, field_name, getattr(defaults, field_name))


class DisplayedProductReference(BaseModel):
    """Stable reference used to resolve conversational phrases like 'the second one'."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=1)
    product_id: int
    article_code: str | None = None
    product_name: str


class DeliveryContext(BaseModel):
    """Customer delivery information collected before final order placement."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str | None = None
    phone: str | None = None
    delivery_address: str | None = None
    city: str | None = None
    delivery_notes: str | None = None


class CartContext(BaseModel):
    """Minimal cart facts needed by the Agent; authoritative contents remain in the backend."""

    model_config = ConfigDict(extra="forbid")

    cart_id: UUID | None = None
    item_count: int = 0
    subtotal: Decimal = Decimal("0.00")


class ActionStatus(StrEnum):
    """String values used for action lifecycle state."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlannedAction(BaseModel):
    """One semantic action in a future dependency-aware action plan."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(default_factory=lambda: uuid4().hex)
    tool_name: ToolName
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependency_ids: list[str] = Field(default_factory=list)
    status: ActionStatus = ActionStatus.PENDING
    missing_parameters: list[str] = Field(default_factory=list)
    result_reference: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActionPlan(BaseModel):
    """Current multi-intent execution graph owned by the Agent runtime."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default_factory=lambda: uuid4().hex)
    actions: list[PlannedAction] = Field(default_factory=list)

    def get(self, action_id: str) -> PlannedAction | None:
        """Return one action by ID, or ``None`` when it is not present."""

        return next((action for action in self.actions if action.action_id == action_id), None)

    def ready_actions(self) -> list[PlannedAction]:
        """Return actions whose dependencies are complete and are not already running."""

        completed_ids = {
            action.action_id
            for action in self.actions
            if action.status == ActionStatus.COMPLETED
        }
        ready: list[PlannedAction] = []
        for action in self.actions:
            if action.status != ActionStatus.PENDING:
                continue
            if all(dependency in completed_ids for dependency in action.dependency_ids):
                ready.append(action)
        return ready


class ConversationState(BaseModel):
    """Complete runtime state for one customer conversation with Fitzy."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID = Field(default_factory=uuid4)
    language: LanguageMode | None = None
    preferences: CustomerPreferences = Field(default_factory=CustomerPreferences)
    current_search: SearchContext = Field(default_factory=SearchContext)
    displayed_products: list[DisplayedProductReference] = Field(default_factory=list)
    selected_product_id: int | None = None
    delivery: DeliveryContext = Field(default_factory=DeliveryContext)
    cart: CartContext = Field(default_factory=CartContext)
    action_plan: ActionPlan = Field(default_factory=ActionPlan)
    pending_action_id: str | None = None
    last_tool_results: dict[str, Any] = Field(default_factory=dict)

    def set_language(self, language: str | LanguageMode) -> None:
        """Set the normalized V1 language mode used for response generation.

        ``urdu`` is accepted as a compatibility alias for Urdu-script mode.
        Roman Urdu is kept distinct so response formatting can preserve the
        customer's conversational script rather than drifting into Hindi.
        """

        if isinstance(language, LanguageMode):
            self.language = language
            return

        normalized = language.strip().lower()
        aliases = {
            "english": LanguageMode.ENGLISH,
            "en": LanguageMode.ENGLISH,
            "us": LanguageMode.ENGLISH,
            "uk": LanguageMode.ENGLISH,
            "urdu": LanguageMode.URDU_SCRIPT,
            "ur": LanguageMode.URDU_SCRIPT,
            "pk": LanguageMode.URDU_SCRIPT,
            "urdu_script": LanguageMode.URDU_SCRIPT,
            "roman_urdu": LanguageMode.ROMAN_URDU,
            "roman urdu": LanguageMode.ROMAN_URDU,
            "roman": LanguageMode.ROMAN_URDU,
        }
        try:
            self.language = aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "language must be english, urdu_script, or roman_urdu"
            ) from exc

    def remember_displayed_products(self, products: list[DisplayedProductReference]) -> None:
        """Replace display references with the latest ordered result set."""

        self.displayed_products = products

    def remember_selected_product(self, product_id: int | None) -> None:
        """Record the product currently selected by the customer."""

        self.selected_product_id = product_id

    def reset_current_search(self) -> None:
        """Clear temporary search state without losing persistent preferences or cart."""

        self.current_search.clear()

    def record_tool_result(self, tool_name: ToolName, result: Any) -> None:
        """Store the latest normalized result for runtime continuity and diagnostics."""

        self.last_tool_results[tool_name.value] = result

    def set_delivery_field(self, field_name: str, value: Any) -> None:
        """Set one known delivery field without allowing arbitrary state mutation."""

        if field_name not in type(self.delivery).model_fields:
            raise ValueError(f"Unknown delivery field: {field_name}")
        if value not in (None, ""):
            setattr(self.delivery, field_name, value)

    def has_complete_delivery_details(self) -> bool:
        """Return whether all required anonymous delivery fields are present."""

        return all(
            getattr(self.delivery, field_name) not in (None, "")
            for field_name in ("customer_name", "phone", "delivery_address", "city")
        )
