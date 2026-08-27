"""Dependency-aware action planning for Fitzy.

The planner converts canonical Fitzy intents into semantic tool actions.
It queries the authoritative Intent Registry for intent definitions and requirements.
"""

from __future__ import annotations

from typing import Any

from .contracts import ToolName
from .intent import StructuredIntent
from .intents import (
    ActionName,
    IntentDefinition,
    IntentName,
    RequirementName,
    get_intent_definition,
)
from .state import ActionPlan, ActionStatus, ConversationState, PlannedAction

ACTION_TO_TOOL: dict[ActionName, ToolName | None] = {
    ActionName.RESPOND_GENERAL: None,
    ActionName.GET_PRODUCTS: ToolName.GET_PRODUCTS,
    ActionName.GET_PRODUCT_DETAILS: ToolName.GET_PRODUCT_DETAILS,
    ActionName.GET_BRANCHES: ToolName.GET_BRANCHES,
    ActionName.CHECK_AVAILABILITY: ToolName.CHECK_AVAILABILITY,
    ActionName.GET_CART: ToolName.GET_CART,
    ActionName.CREATE_CART: ToolName.CREATE_CART,
    ActionName.ADD_TO_CART: ToolName.ADD_TO_CART,
    ActionName.UPDATE_CART: ToolName.UPDATE_CART,
    ActionName.REMOVE_FROM_CART: ToolName.REMOVE_FROM_CART,
    ActionName.CLEAR_CART: ToolName.CLEAR_CART,
    ActionName.PREVIEW_CHECKOUT: ToolName.PREVIEW_CHECKOUT,
    ActionName.PLACE_ORDER: ToolName.PLACE_ORDER,
}


def action_to_tool(action: ActionName) -> ToolName | None:
    """Return the semantic ToolName for an internal action, or None if non-tool action."""
    return ACTION_TO_TOOL.get(action)


class ActionPlanner:
    """Build a dependency-aware graph without executing any tool."""

    def build_plan(
        self,
        extraction: StructuredIntent | Any,
        state: ConversationState | None = None,
    ) -> ActionPlan:
        """Build a plan from multiple intents and known conversation state.

        Independent reads stay independent and can later run concurrently.
        Internal prerequisites such as cart creation or checkout preview are
        inserted only when they are necessary for a requested action and are
        not already satisfied by state or another action in the same plan.
        """
        state = state or ConversationState()
        actions: list[PlannedAction] = []
        last_search_id: str | None = None
        last_cart_write_id: str | None = None
        last_checkout_id: str | None = None
        has_cart = state.cart.cart_id is not None
        has_checkout = "preview_checkout" in state.last_tool_results

        intent_names: list[IntentName] = []
        if isinstance(extraction, StructuredIntent):
            intent_names = list(extraction.intents)
        elif hasattr(extraction, "intents"):
            raw_intents = getattr(extraction, "intents")
            for item in raw_intents:
                if isinstance(item, IntentName):
                    intent_names.append(item)
                elif hasattr(item, "intent_type"):
                    # Compatibility with legacy test objects
                    val = getattr(item, "intent_type")
                    val_str = val.value if hasattr(val, "value") else str(val)
                    try:
                        intent_names.append(IntentName(val_str))
                    except ValueError:
                        pass
                elif isinstance(item, str):
                    try:
                        intent_names.append(IntentName(item))
                    except ValueError:
                        pass
        elif isinstance(extraction, (list, tuple)):
            for item in extraction:
                if isinstance(item, IntentName):
                    intent_names.append(item)
                elif isinstance(item, str):
                    try:
                        intent_names.append(IntentName(item))
                    except ValueError:
                        pass

        for idx, intent_name in enumerate(intent_names):
            definition = get_intent_definition(intent_name)
            tool_name = action_to_tool(definition.action)

            # Build parameter dictionary for this intent action
            params = self._extract_params_for_intent(intent_name, extraction, idx)

            if tool_name is None:
                continue

            dependencies: list[str] = []

            # Product reference dependency check
            if RequirementName.PRODUCT_REFERENCE in definition.required:
                if not self._has_direct_product_reference(params):
                    if last_search_id:
                        dependencies.append(last_search_id)
                    elif state.selected_product_id is not None:
                        params.setdefault("product_id", state.selected_product_id)

            # Cart dependency check
            if RequirementName.CART_ID in definition.required and not self._has_cart_reference(params) and not has_cart:
                if intent_name == IntentName.ADD_TO_CART:
                    if last_cart_write_id and actions[last_cart_index(actions, last_cart_write_id)].tool_name == ToolName.CREATE_CART:
                        dependencies.append(last_cart_write_id)
                    else:
                        create_cart = PlannedAction(tool_name=ToolName.CREATE_CART, status=ActionStatus.PENDING)
                        actions.append(create_cart)
                        last_cart_write_id = create_cart.action_id
                        has_cart = True
                        dependencies.append(create_cart.action_id)

            # Checkout preview dependency check
            if RequirementName.CHECKOUT_PREVIEW in definition.required and not has_checkout:
                if last_checkout_id:
                    dependencies.append(last_checkout_id)
                else:
                    checkout = PlannedAction(
                        tool_name=ToolName.PREVIEW_CHECKOUT,
                        parameters=params.get("cart_id") and {"cart_id": params["cart_id"]} or {},
                        dependency_ids=[last_cart_write_id] if last_cart_write_id else [],
                        status=ActionStatus.PENDING,
                    )
                    actions.append(checkout)
                    last_checkout_id = checkout.action_id
                    has_checkout = True
                    dependencies.append(checkout.action_id)

            if tool_name in {
                ToolName.CREATE_CART,
                ToolName.ADD_TO_CART,
                ToolName.UPDATE_CART,
                ToolName.REMOVE_FROM_CART,
                ToolName.CLEAR_CART,
            } and last_cart_write_id and last_cart_write_id not in dependencies:
                dependencies.append(last_cart_write_id)

            if definition.confirmation_required:
                params.setdefault("explicit_confirmation", self._get_explicit_confirmation(extraction))

            action = PlannedAction(
                tool_name=tool_name,
                parameters=params,
                dependency_ids=self._deduplicate(dependencies),
                status=ActionStatus.PENDING,
            )
            actions.append(action)

            if intent_name == IntentName.PRODUCT_SEARCH:
                last_search_id = action.action_id
            if tool_name in {
                ToolName.CREATE_CART,
                ToolName.ADD_TO_CART,
                ToolName.UPDATE_CART,
                ToolName.REMOVE_FROM_CART,
                ToolName.CLEAR_CART,
            }:
                last_cart_write_id = action.action_id
                if tool_name == ToolName.CREATE_CART:
                    has_cart = True
            if tool_name == ToolName.PREVIEW_CHECKOUT:
                last_checkout_id = action.action_id
                has_checkout = True

        return ActionPlan(actions=actions)

    def _extract_params_for_intent(self, intent_name: IntentName, extraction: Any, index: int) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if isinstance(extraction, StructuredIntent):
            if intent_name == IntentName.PRODUCT_SEARCH:
                params.update(extraction.search_overrides.model_dump(exclude_none=True))
                if extraction.change_topic:
                    params["change_topic"] = True
                if extraction.reset_shopping:
                    params["reset_shopping"] = True
            elif intent_name in (IntentName.PRODUCT_DETAILS, IntentName.PRODUCT_AVAILABILITY, IntentName.BRANCH_AVAILABILITY):
                if extraction.product_reference:
                    ref = extraction.product_reference
                    if ref.product_id is not None:
                        params["product_id"] = ref.product_id
                    if ref.index is not None:
                        params["product_reference"] = ref.index
                    if ref.text_reference:
                        params["product_reference"] = ref.text_reference
                    if ref.article_code:
                        params["article_code"] = ref.article_code
                    if ref.sku:
                        params["sku"] = ref.sku
                if extraction.search_overrides.branch_reference:
                    params["branch_reference"] = extraction.search_overrides.branch_reference
            elif intent_name == IntentName.ADD_TO_CART:
                if extraction.product_reference:
                    ref = extraction.product_reference
                    if ref.product_id is not None:
                        params["product_id"] = ref.product_id
                    if ref.index is not None:
                        params["product_reference"] = ref.index
                    if ref.text_reference:
                        params["product_reference"] = ref.text_reference
                if extraction.quantity is not None:
                    params["quantity"] = extraction.quantity
                else:
                    params["quantity"] = 1
                if extraction.search_overrides.branch_reference:
                    params["branch_reference"] = extraction.search_overrides.branch_reference
            elif intent_name == IntentName.UPDATE_CART:
                if extraction.cart_item_index is not None:
                    params["cart_item_reference"] = extraction.cart_item_index
                if extraction.quantity is not None:
                    params["quantity"] = extraction.quantity
            elif intent_name == IntentName.REMOVE_FROM_CART:
                if extraction.cart_item_index is not None:
                    params["cart_item_reference"] = extraction.cart_item_index
            elif intent_name == IntentName.DELIVERY_INFO:
                params.update(extraction.delivery.model_dump(exclude_none=True))
            elif intent_name == IntentName.PLACE_ORDER:
                params.update(extraction.delivery.model_dump(exclude_none=True))
                if extraction.explicit_confirmation is not None:
                    params["explicit_confirmation"] = extraction.explicit_confirmation

        # Backward compatibility for legacy intent objects in tests
        if hasattr(extraction, "intents"):
            raw_intents = getattr(extraction, "intents")
            if index < len(raw_intents):
                item = raw_intents[index]
                if hasattr(item, "parameters") and isinstance(item.parameters, dict):
                    for k, v in item.parameters.items():
                        params.setdefault(k, v)
                if hasattr(item, "explicit_confirmation") and getattr(item, "explicit_confirmation") is not None:
                    params.setdefault("explicit_confirmation", getattr(item, "explicit_confirmation"))

        return params

    @staticmethod
    def _get_explicit_confirmation(extraction: Any) -> bool | None:
        if isinstance(extraction, StructuredIntent):
            return extraction.explicit_confirmation
        if hasattr(extraction, "explicit_confirmation"):
            return getattr(extraction, "explicit_confirmation")
        return None

    @staticmethod
    def _has_direct_product_reference(parameters: dict[str, Any]) -> bool:
        """Return True when an intent already identifies a product directly."""
        return any(parameters.get(name) not in (None, "", [], {}) for name in (
            "product_id", "product_reference", "article_code", "sku", "variant_id"
        ))

    @staticmethod
    def _has_cart_reference(parameters: dict[str, Any]) -> bool:
        """Return True when the intent includes a cart identifier."""
        return parameters.get("cart_id") not in (None, "")

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        """Preserve dependency order while removing duplicate action IDs."""
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                output.append(value)
        return output


def action_index(actions: list[PlannedAction], action_id: str) -> int:
    """Return an action index; raise a clear error for an unknown dependency."""
    for index, action in enumerate(actions):
        if action.action_id == action_id:
            return index
    raise KeyError(f"Unknown action dependency: {action_id}")


def last_cart_index(actions: list[PlannedAction], action_id: str) -> int:
    """Compatibility alias used by planner dependency checks."""
    return action_index(actions, action_id)

