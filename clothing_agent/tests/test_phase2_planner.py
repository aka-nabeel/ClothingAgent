from __future__ import annotations

import asyncio
from typing import Any

from clothing_agent.app.agent.contracts import ToolName
from clothing_agent.app.agent.execution import ActionExecutionCoordinator
from clothing_agent.app.agent.intent import DeliveryExtraction, LanguageCode, SearchOverrides, StructuredIntent
from clothing_agent.app.agent.intents import IntentName
from clothing_agent.app.agent.planner import ActionPlanner
from clothing_agent.app.agent.requirements import ToolRequirementChecker
from clothing_agent.app.agent.state import ActionStatus, ConversationState, LanguageMode


def test_search_and_add_to_cart_are_dependency_ordered() -> None:
    extraction = StructuredIntent(
        language=LanguageCode.ENGLISH,
        intents=[IntentName.PRODUCT_SEARCH, IntentName.ADD_TO_CART],
        search_overrides=SearchOverrides(categories=["shirts"]),
        quantity=1,
    )
    plan = ActionPlanner().build_plan(extraction)
    search, create_cart, add = plan.actions
    assert search.tool_name == ToolName.GET_PRODUCTS
    assert create_cart.tool_name == ToolName.CREATE_CART
    assert add.tool_name == ToolName.ADD_TO_CART
    assert search.dependency_ids == []
    assert create_cart.dependency_ids == []
    assert add.dependency_ids == [search.action_id, create_cart.action_id]


def test_independent_reads_remain_parallelizable() -> None:
    extraction = StructuredIntent(
        language=LanguageCode.ENGLISH,
        intents=[IntentName.PRODUCT_SEARCH, IntentName.BRANCH_INFORMATION],
    )
    plan = ActionPlanner().build_plan(extraction)
    assert all(not action.dependency_ids for action in plan.actions)


def test_place_order_inserts_checkout_prerequisite() -> None:
    state = ConversationState()
    extraction = StructuredIntent(
        language=LanguageCode.ENGLISH,
        intents=[IntentName.PLACE_ORDER],
        explicit_confirmation=True,
        delivery=DeliveryExtraction(
            customer_name="Ahmed",
            phone="0300",
            delivery_address="DHA",
            city="Lahore",
        ),
    )
    plan = ActionPlanner().build_plan(extraction, state)
    assert plan.actions[0].tool_name == ToolName.PREVIEW_CHECKOUT
    assert plan.actions[1].tool_name == ToolName.PLACE_ORDER
    assert plan.actions[1].dependency_ids == [plan.actions[0].action_id]


def test_language_modes_distinguish_english_urdu_script_and_roman_urdu() -> None:
    state = ConversationState()
    state.set_language("english")
    assert state.language == LanguageMode.ENGLISH
    state.set_language("urdu")
    assert state.language == LanguageMode.URDU_SCRIPT
    state.set_language("roman_urdu")
    assert state.language == LanguageMode.ROMAN_URDU


def test_order_requires_explicit_confirmation() -> None:
    from clothing_agent.app.agent.contracts import PLACE_ORDER

    result = ToolRequirementChecker().check(
        PLACE_ORDER,
        provided_parameters={
            "cart_id": "cart-1",
            "customer_name": "Ahmed",
            "phone": "0300",
            "delivery_address": "DHA",
            "city": "Lahore",
            "explicit_confirmation": False,
        },
    )
    assert not result.ready
    assert result.invalid_parameters == ("explicit_confirmation",)


async def fake_executor(tool_name: ToolName, parameters: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(0.001)
    return {"tool": tool_name.value, "parameters": dict(parameters)}


def test_ready_independent_actions_execute() -> None:
    async def scenario() -> None:
        state = ConversationState()
        extraction = StructuredIntent(
            language=LanguageCode.ENGLISH,
            intents=[IntentName.PRODUCT_SEARCH, IntentName.BRANCH_INFORMATION],
        )
        state.action_plan = ActionPlanner().build_plan(extraction)
        result = await ActionExecutionCoordinator().run_ready_actions(state, tool_executor=fake_executor)
        assert len(result.completed_action_ids) == 2
        assert all(action.status == ActionStatus.COMPLETED for action in state.action_plan.actions)

    asyncio.run(scenario())


def test_missing_required_parameter_waits_instead_of_calling_api() -> None:
    async def scenario() -> None:
        state = ConversationState()
        extraction = StructuredIntent(
            language=LanguageCode.ENGLISH,
            intents=[IntentName.ADD_TO_CART],
            quantity=1,
        )
        state.action_plan = ActionPlanner().build_plan(extraction, state)
        result = await ActionExecutionCoordinator().run_ready_actions(state, tool_executor=fake_executor)
        # CREATE_CART is an internal prerequisite. The first pass executes it.
        assert len(result.executed_action_ids) == 1
        assert state.action_plan.actions[0].tool_name == ToolName.CREATE_CART
        assert state.action_plan.actions[0].status == ActionStatus.COMPLETED

        # The second pass sees the dependent add-to-cart action. It must stop
        # before the API because the required variant is still unknown.
        result = await ActionExecutionCoordinator().run_ready_actions(state, tool_executor=fake_executor)
        assert result.executed_action_ids == ()
        assert result.waiting_action_id is not None
        assert "variant_id" in result.missing_parameters
        assert state.action_plan.actions[-1].status == ActionStatus.WAITING_FOR_INPUT

    asyncio.run(scenario())

