from decimal import Decimal

import pytest

from clothing_agent.app.agent.agent import FitzyAgent
from clothing_agent.app.agent.contracts import ToolName
from clothing_agent.app.agent.intent import LanguageCode, SearchOverrides, StructuredIntent
from clothing_agent.app.agent.intents import IntentName
from clothing_agent.app.agent.state import ConversationState, LanguageMode


class StubLLM:
    async def generate_structured(self, *, system_prompt, user_message, response_model):
        return StructuredIntent(
            language=LanguageCode.ENGLISH,
            intents=[IntentName.PRODUCT_SEARCH],
            search_overrides=SearchOverrides(colors=["blue"]),
        )

    async def generate_text(self, *, system_prompt, user_message):
        return "Here are the available options."


@pytest.mark.asyncio
async def test_effective_search_preserves_existing_context():
    agent = object.__new__(FitzyAgent)
    state = ConversationState()
    state.current_search.colors = ["black"]
    state.current_search.maximum_price = Decimal("5000")
    values = agent._build_effective_product_search(state, {"colors": ["blue"]})
    assert values["colors"] == ["blue"]
    assert values["maximum_price"] == Decimal("5000")
    assert values.get("size_mapping", {}) == {}

@pytest.mark.asyncio
async def test_delivery_information_accumulates_across_turns():
    agent = object.__new__(FitzyAgent)
    state = ConversationState()
    agent._apply_delivery_fields({"customer_name": "Ahmed"}, state)
    agent._apply_delivery_fields({"phone": "0300"}, state)
    agent._apply_delivery_fields({"delivery_address": "DHA Lahore", "city": "Lahore"}, state)
    assert state.delivery.customer_name == "Ahmed"
    assert state.delivery.phone == "0300"
    assert state.delivery.delivery_address == "DHA Lahore"
    assert state.delivery.city == "Lahore"
    assert state.has_complete_delivery_details()

@pytest.mark.asyncio
async def test_waiting_action_reopens_on_new_turn():
    agent = object.__new__(FitzyAgent)
    state = ConversationState()
    from clothing_agent.app.agent.state import PlannedAction, ActionStatus
    pending = PlannedAction(tool_name=ToolName.PLACE_ORDER, status=ActionStatus.WAITING_FOR_INPUT, missing_parameters=["city"])
    state.action_plan.actions.append(pending)
    state.pending_action_id = pending.action_id
    agent._reopen_waiting_actions_for_new_input(state)
    assert pending.status == ActionStatus.PENDING
    assert pending.missing_parameters == []


@pytest.mark.asyncio
async def test_variant_resolution_from_latest_search():
    agent = object.__new__(FitzyAgent)
    state = ConversationState()

    from clothing_agent.app.integration.schemas import ProductSearchResponse, ProductOption
    from clothing_agent.app.agent.state import DisplayedProductReference

    p1 = ProductOption(
        product_id=1, variant_id=101, branch_id=1, article_code="NS-SH-001",
        product_name="Oxford Shirt", category="shirts", color="Black", size="M",
        price=Decimal("4500.00"), branch_code="ISB-F7", branch_name="F7", city="Islamabad", available_quantity=5,
    )
    p2 = ProductOption(
        product_id=1, variant_id=102, branch_id=1, article_code="NS-SH-001",
        product_name="Oxford Shirt", category="shirts", color="Black", size="L",
        price=Decimal("4500.00"), branch_code="ISB-F7", branch_name="F7", city="Islamabad", available_quantity=8,
    )
    state.last_tool_results[ToolName.GET_PRODUCTS.value] = ProductSearchResponse(products=[p1, p2], result_count=2)
    state.displayed_products = [DisplayedProductReference(index=1, product_id=1, product_name="Oxford Shirt")]

    resolved = agent._resolve_variant_from_latest_search(state, 1, {"color": "Black", "size": "L"})
    assert resolved is not None
    assert resolved.variant_id == 102
    assert resolved.size == "L"


@pytest.mark.asyncio
async def test_agent_checkout_confirmation_invalidation_on_cart_mutation():
    agent = object.__new__(FitzyAgent)
    state = ConversationState()

    state.last_tool_results[ToolName.PREVIEW_CHECKOUT.value] = {"cart_id": "cart-123", "grand_total": 5000}
    state.last_tool_results["explicit_confirmation"] = True

    # Extracted intent for cart addition
    extraction = StructuredIntent(
        language=LanguageCode.ENGLISH,
        intents=[IntentName.ADD_TO_CART],
    )
    agent._apply_intent_to_state(extraction, state)

    assert state.last_tool_results.get("explicit_confirmation") is None
    assert ToolName.PREVIEW_CHECKOUT.value not in state.last_tool_results


