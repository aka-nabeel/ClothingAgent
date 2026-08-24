"""Authoritative Fitzy intent registry and operational contracts.

This module defines the closed set of customer-facing intents understood by Fitzy.
Each intent declares:
- what the intent means,
- what operation(s) it requests,
- whether it requires customer clarification,
- what state/context it consumes,
- what tool/action is expected,
- what successful completion looks like.

The LLM should select from this registry rather than inventing intent names.
Business truth remains in the commerce backend; this registry only describes
Agent orchestration semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class IntentName(str, Enum):
    """Closed customer-facing intent vocabulary for Fitzy."""

    GENERAL_CHAT = "general_chat"

    PRODUCT_SEARCH = "product_search"
    PRODUCT_DETAILS = "product_details"
    PRODUCT_AVAILABILITY = "product_availability"
    BRANCH_INFORMATION = "branch_information"
    BRANCH_AVAILABILITY = "branch_availability"

    CART_VIEW = "cart_view"
    ADD_TO_CART = "add_to_cart"
    UPDATE_CART = "update_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    CLEAR_CART = "clear_cart"

    CHECKOUT_PREVIEW = "checkout_preview"
    DELIVERY_INFO = "delivery_info"
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"


class ActionName(str, Enum):
    """Internal executable actions produced by the planner."""

    RESPOND_GENERAL = "respond_general"
    GET_PRODUCTS = "get_products"
    GET_PRODUCT_DETAILS = "get_product_details"
    GET_BRANCHES = "get_branches"
    CHECK_AVAILABILITY = "check_availability"

    GET_CART = "get_cart"
    CREATE_CART = "create_cart"
    ADD_TO_CART = "add_to_cart"
    UPDATE_CART = "update_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    CLEAR_CART = "clear_cart"

    PREVIEW_CHECKOUT = "preview_checkout"
    PLACE_ORDER = "place_order"


class RequirementName(str, Enum):
    """Named requirements the runtime can check before executing an action."""

    PRODUCT_REFERENCE = "product_reference"
    PRODUCT_ID = "product_id"
    VARIANT_ID = "variant_id"
    CART_ID = "cart_id"
    CART_ITEM_REFERENCE = "cart_item_reference"
    QUANTITY = "quantity"

    BRANCH_ID = "branch_id"
    BRANCH_REFERENCE = "branch_reference"

    DELIVERY_NAME = "delivery_name"
    DELIVERY_PHONE = "delivery_phone"
    DELIVERY_ADDRESS = "delivery_address"
    DELIVERY_CITY = "delivery_city"

    CHECKOUT_PREVIEW = "checkout_preview"
    ORDER_CONFIRMATION = "order_confirmation"


@dataclass(frozen=True)
class IntentDefinition:
    """Executable semantic contract for one customer-facing intent.

    `required` lists values that must be available before the intent can
    complete. `action` identifies the backend-facing operation. `context`
    identifies the state surfaces used to satisfy requirements.
    """

    name: IntentName
    description: str
    examples: tuple[str, ...]
    action: ActionName
    required: FrozenSet[RequirementName] = frozenset()
    optional: FrozenSet[RequirementName] = frozenset()
    uses_state: FrozenSet[str] = frozenset()
    customer_choice_boundary: bool = False
    confirmation_required: bool = False
    can_execute_without_backend: bool = False


INTENT_REGISTRY: dict[IntentName, IntentDefinition] = {
    IntentName.GENERAL_CHAT: IntentDefinition(
        name=IntentName.GENERAL_CHAT,
        description="Casual conversation, greeting, thanks, or non-commerce chat.",
        examples=("Hi Fitzy", "Thanks", "How are you?"),
        action=ActionName.RESPOND_GENERAL,
        uses_state=frozenset({"language", "preferences"}),
        can_execute_without_backend=True,
    ),
    IntentName.PRODUCT_SEARCH: IntentDefinition(
        name=IntentName.PRODUCT_SEARCH,
        description=(
            "Find products using the user's current request plus persistent "
            "preferences and current search context."
        ),
        examples=(
            "Show me black shirts",
            "I need something for a wedding",
            "Show me jackets under 5000",
        ),
        action=ActionName.GET_PRODUCTS,
        optional=frozenset({
            RequirementName.BRANCH_REFERENCE,
        }),
        uses_state=frozenset({
            "preferences",
            "current_search",
            "displayed_products",
        }),
    ),
    IntentName.PRODUCT_DETAILS: IntentDefinition(
        name=IntentName.PRODUCT_DETAILS,
        description="Explain the details of one product already identified or displayed.",
        examples=(
            "Tell me more about the first one",
            "What material is this?",
            "What sizes does this have?",
        ),
        action=ActionName.GET_PRODUCT_DETAILS,
        required=frozenset({RequirementName.PRODUCT_REFERENCE}),
        uses_state=frozenset({"displayed_products", "selected_product"}),
        customer_choice_boundary=True,
    ),
    IntentName.PRODUCT_AVAILABILITY: IntentDefinition(
        name=IntentName.PRODUCT_AVAILABILITY,
        description="Check whether a specific product/variant is available for purchase.",
        examples=(
            "Is this available?",
            "Do you have it in large?",
            "Is black L available?",
        ),
        action=ActionName.CHECK_AVAILABILITY,
        required=frozenset({RequirementName.PRODUCT_REFERENCE}),
        optional=frozenset({RequirementName.BRANCH_REFERENCE}),
        uses_state=frozenset({"displayed_products", "selected_product"}),
    ),
    IntentName.BRANCH_INFORMATION: IntentDefinition(
        name=IntentName.BRANCH_INFORMATION,
        description="Tell the customer about Northstar branch locations/count.",
        examples=(
            "How many branches do you have?",
            "Where are your branches?",
        ),
        action=ActionName.GET_BRANCHES,
        uses_state=frozenset({"store_context"}),
    ),
    IntentName.BRANCH_AVAILABILITY: IntentDefinition(
        name=IntentName.BRANCH_AVAILABILITY,
        description="Check availability of a product at a specific Northstar branch.",
        examples=(
            "Is this available at F-7?",
            "Do you have this at Islamabad F7?",
        ),
        action=ActionName.CHECK_AVAILABILITY,
        required=frozenset({
            RequirementName.PRODUCT_REFERENCE,
            RequirementName.BRANCH_REFERENCE,
        }),
        uses_state=frozenset({"displayed_products", "selected_product", "branches"}),
    ),
    IntentName.CART_VIEW: IntentDefinition(
        name=IntentName.CART_VIEW,
        description="Show the customer's current cart contents and totals.",
        examples=("Show my cart", "What's in my cart?"),
        action=ActionName.GET_CART,
        required=frozenset({RequirementName.CART_ID}),
        uses_state=frozenset({"cart"}),
    ),
    IntentName.ADD_TO_CART: IntentDefinition(
        name=IntentName.ADD_TO_CART,
        description="Add a clearly identified sellable variant to the cart.",
        examples=("Add the first one", "Put this in my cart"),
        action=ActionName.ADD_TO_CART,
        required=frozenset({
            RequirementName.CART_ID,
            RequirementName.PRODUCT_REFERENCE,
            RequirementName.VARIANT_ID,
        }),
        optional=frozenset({RequirementName.QUANTITY, RequirementName.BRANCH_REFERENCE}),
        uses_state=frozenset({
            "displayed_products",
            "selected_product",
            "current_search",
            "preferences",
            "cart",
        }),
        customer_choice_boundary=True,
    ),
    IntentName.UPDATE_CART: IntentDefinition(
        name=IntentName.UPDATE_CART,
        description="Change quantity or other supported properties of an existing cart item.",
        examples=("Make that two", "Change the quantity to 3"),
        action=ActionName.UPDATE_CART,
        required=frozenset({RequirementName.CART_ID, RequirementName.CART_ITEM_REFERENCE}),
        optional=frozenset({RequirementName.QUANTITY}),
        uses_state=frozenset({"cart"}),
    ),
    IntentName.REMOVE_FROM_CART: IntentDefinition(
        name=IntentName.REMOVE_FROM_CART,
        description="Remove one identified item from the cart.",
        examples=("Remove the first one", "Take the shirt out"),
        action=ActionName.REMOVE_FROM_CART,
        required=frozenset({RequirementName.CART_ID, RequirementName.CART_ITEM_REFERENCE}),
        uses_state=frozenset({"cart"}),
        customer_choice_boundary=True,
    ),
    IntentName.CLEAR_CART: IntentDefinition(
        name=IntentName.CLEAR_CART,
        description="Remove every item from the current cart.",
        examples=("Clear my cart", "Empty the cart"),
        action=ActionName.CLEAR_CART,
        required=frozenset({RequirementName.CART_ID}),
        uses_state=frozenset({"cart"}),
    ),
    IntentName.CHECKOUT_PREVIEW: IntentDefinition(
        name=IntentName.CHECKOUT_PREVIEW,
        description="Show the authoritative checkout total without placing an order.",
        examples=("What's my total?", "Show checkout total"),
        action=ActionName.PREVIEW_CHECKOUT,
        required=frozenset({RequirementName.CART_ID}),
        uses_state=frozenset({"cart", "checkout"}),
    ),
    IntentName.DELIVERY_INFO: IntentDefinition(
        name=IntentName.DELIVERY_INFO,
        description="Provide or update delivery information in conversation state.",
        examples=(
            "My name is Ahmed",
            "My phone is 0300...",
            "DHA Lahore",
        ),
        action=ActionName.RESPOND_GENERAL,
        uses_state=frozenset({"delivery"}),
        can_execute_without_backend=True,
    ),
    IntentName.PLACE_ORDER: IntentDefinition(
        name=IntentName.PLACE_ORDER,
        description="Place the currently confirmed checkout as an order.",
        examples=("Place the order", "Yes, go ahead with the order"),
        action=ActionName.PLACE_ORDER,
        required=frozenset({
            RequirementName.CART_ID,
            RequirementName.CHECKOUT_PREVIEW,
            RequirementName.DELIVERY_NAME,
            RequirementName.DELIVERY_PHONE,
            RequirementName.DELIVERY_ADDRESS,
            RequirementName.DELIVERY_CITY,
            RequirementName.ORDER_CONFIRMATION,
        }),
        uses_state=frozenset({"cart", "checkout", "delivery", "confirmation"}),
        confirmation_required=True,
    ),
    IntentName.CANCEL_ORDER: IntentDefinition(
        name=IntentName.CANCEL_ORDER,
        description="Cancel an active order-placement/confirmation flow before placement.",
        examples=("Don't place it", "Cancel the order"),
        action=ActionName.RESPOND_GENERAL,
        uses_state=frozenset({"checkout", "confirmation"}),
        can_execute_without_backend=True,
    ),
}


def get_intent_definition(intent: IntentName | str) -> IntentDefinition:
    """Return the authoritative definition for an intent.

    Raises:
        ValueError: when the caller tries to use an unregistered intent.
    """
    normalized = IntentName(intent)
    return INTENT_REGISTRY[normalized]


def registered_intents() -> tuple[IntentDefinition, ...]:
    """Return all registered customer-facing intents in stable order."""
    return tuple(INTENT_REGISTRY.values())


def registered_intent_values() -> tuple[str, ...]:
    """Return the values presented to the LLM as the closed intent vocabulary."""
    return tuple(item.value for item in IntentName)
