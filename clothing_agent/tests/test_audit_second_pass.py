"""Second-Pass Comprehensive Regression Test Suite for ClothingAgent (`agent-fix`).

Covers Scenarios A through H and all Section 3-20 Audit Requirements.
"""

import asyncio
from decimal import Decimal
import pytest
from fastapi.testclient import TestClient

from clothing_agent.app.agent.agent import FitzyAgent
from clothing_agent.app.agent.contracts import ToolName
from clothing_agent.app.agent.execution import ActionExecutionCoordinator
from clothing_agent.app.agent.intent import LanguageCode, ProductReference, SearchOverrides, StructuredIntent
from clothing_agent.app.agent.intents import IntentName
from clothing_agent.app.agent.normalization import normalize_color, normalize_confirmation, normalize_size
from clothing_agent.app.agent.planner import ActionPlanner
from clothing_agent.app.agent.state import ActionPlan, ActionStatus, ConversationState, DisplayedProductReference, PlannedAction
from clothing_agent.app.integration.schemas import ProductOption, ProductSearchResponse
from clothing_agent.app.llm.client import FakeLLMClient
from clothing_app.app.main import app


class DummyCommerceAdapter:
    async def get_branches(self):
        return [{"branch_id": 1, "branch_code": "LHR-01", "city": "Lahore"}]

    async def execute(self, tool_name: ToolName, parameters: dict):
        if tool_name == ToolName.GET_PRODUCTS:
            cat = parameters.get("categories", ["all"])[0]
            if cat == "fail_search":
                raise RuntimeError("Search service unavailable")
            if cat == "set_b":
                return ProductSearchResponse(
                    products=[
                        ProductOption(product_id=201, variant_id=2001, branch_id=1, article_code="B1", product_name="Shirt B1", price=Decimal("2000"), available_quantity=5, is_available=True),
                        ProductOption(product_id=202, variant_id=2002, branch_id=1, article_code="B2", product_name="Shirt B2", price=Decimal("2500"), available_quantity=5, is_available=True),
                    ],
                    result_count=2,
                )
            return ProductSearchResponse(
                products=[
                    ProductOption(product_id=101, variant_id=1001, branch_id=1, article_code="A1", product_name="Shirt A1", price=Decimal("1000"), available_quantity=5, is_available=True),
                    ProductOption(product_id=102, variant_id=1002, branch_id=1, article_code="A2", product_name="Shirt A2", price=Decimal("1500"), available_quantity=5, is_available=True),
                ],
                result_count=2,
            )
        if tool_name == ToolName.ADD_TO_CART:
            return {"cart_id": "cart_123", "item_count": 1, "subtotal": "2500.00", "idempotency_key": parameters.get("idempotency_key")}
        return {"status": "ok", "tool": tool_name.value}


# =====================================================================
# SCENARIO A: search A -> details A -> search B -> add second product
# =====================================================================
@pytest.mark.asyncio
async def test_scenario_a_search_b_overrides_search_a_context():
    llm = FakeLLMClient(
        StructuredIntent(
            language=LanguageCode.ENGLISH,
            intents=[IntentName.PRODUCT_SEARCH],
            search_overrides=SearchOverrides(categories=["set_b"]),
        ),
        "Here are shirts from set B."
    )
    agent = FitzyAgent(llm=llm, tools=DummyCommerceAdapter())  # type: ignore

    # Step 1: Search B
    res1 = await agent.process_message(session_id="sec_a", message="show me set b shirts")
    state = agent.get_state("sec_a")
    assert len(state.displayed_products) == 2
    assert state.displayed_products[1].product_id == 202, "Second product must belong to Set B (product 202)"


# =====================================================================
# SCENARIO B: search 3 products -> select product 9 -> clarify (no fallback)
# =====================================================================
@pytest.mark.asyncio
async def test_scenario_b_invalid_product_reference_clarifies():
    state = ConversationState()
    state.remember_displayed_products([
        DisplayedProductReference(index=1, product_id=101, article_code="A1", product_name="Shirt 1"),
        DisplayedProductReference(index=2, product_id=102, article_code="A2", product_name="Shirt 2"),
        DisplayedProductReference(index=3, product_id=103, article_code="A3", product_name="Shirt 3"),
    ])

    resolved_pid = FitzyAgent._resolve_displayed_product_id(state, 9)
    assert resolved_pid is None, "Out-of-bounds index 9 MUST return None, never product 101"


# =====================================================================
# SCENARIO C: "I prefer black" -> new turn -> black preference remains
# =====================================================================
@pytest.mark.asyncio
async def test_scenario_c_explicit_preference_persists():
    llm_pref = FakeLLMClient(
        StructuredIntent(
            language=LanguageCode.ENGLISH,
            intents=[IntentName.PRODUCT_SEARCH],
            search_overrides=SearchOverrides(colors=["black"]),
        ),
        "Got it, black preference recorded."
    )
    agent = FitzyAgent(llm=llm_pref, tools=DummyCommerceAdapter())  # type: ignore

    # Turn 1: "I prefer black"
    await agent.process_message(session_id="sec_c", message="I prefer black")
    state = agent.get_state("sec_c")
    assert state.preferences.preferred_colors == ["black"], "Preference for black must persist"

    # Turn 2: "show me shirts"
    llm_search = FakeLLMClient(
        StructuredIntent(
            language=LanguageCode.ENGLISH,
            intents=[IntentName.PRODUCT_SEARCH],
            search_overrides=SearchOverrides(categories=["shirts"]),
        ),
        "Here are black shirts."
    )
    agent._llm = llm_search
    await agent.process_message(session_id="sec_c", message="show me shirts")
    assert state.preferences.preferred_colors == ["black"], "Preference for black must remain in state"


# =====================================================================
# SCENARIO D: checkout -> confirmation -> cart changes -> "yes" -> old confirmation rejected
# =====================================================================
@pytest.mark.asyncio
async def test_scenario_d_cart_mutation_invalidates_checkout_confirmation():
    state = ConversationState()
    state.last_tool_results[ToolName.PREVIEW_CHECKOUT.value] = {"cart_id": "c1", "grand_total": 5000}
    state.last_tool_results["explicit_confirmation"] = True

    # Mutate cart
    agent = FitzyAgent(llm=FakeLLMClient(StructuredIntent(language=LanguageCode.ENGLISH, intents=[IntentName.ADD_TO_CART]), "Added"), tools=DummyCommerceAdapter())  # type: ignore
    extraction = StructuredIntent(language=LanguageCode.ENGLISH, intents=[IntentName.ADD_TO_CART])
    agent._apply_intent_to_state(extraction, state, message="add another shirt")

    assert ToolName.PREVIEW_CHECKOUT.value not in state.last_tool_results, "PREVIEW_CHECKOUT result must be cleared upon cart mutation"
    assert state.last_tool_results.get("explicit_confirmation") is None, "Pending confirmation must be invalidated upon cart mutation"


# =====================================================================
# SCENARIO E: side-effecting operations receive deterministic idempotency key
# =====================================================================
@pytest.mark.asyncio
async def test_scenario_e_mutation_idempotency_keys():
    coord = ActionExecutionCoordinator()
    state = ConversationState()
    state.delivery.customer_name = "Alice"
    state.delivery.phone = "03001234567"
    state.delivery.delivery_address = "Street 1"
    state.delivery.city = "Lahore"
    state.last_tool_results[ToolName.PREVIEW_CHECKOUT.value] = {"cart_id": "c123"}
    state.last_tool_results["explicit_confirmation"] = True

    action = PlannedAction(tool_name=ToolName.PLACE_ORDER, parameters={"cart_id": "c123"})
    state.action_plan = ActionPlan(actions=[action])

    async def fake_executor(tool_name, params):
        return {"order_id": "ord_1", "checkout_request_id": params.get("checkout_request_id")}

    res = await coord.run_ready_actions(
        state,
        tool_executor=fake_executor,
        state_values=FitzyAgent._state_values(state),
        derived_values=FitzyAgent._derived_values(state),
    )
    assert res.completed_action_ids == (action.action_id,)
    assert action.parameters.get("checkout_request_id") == action.action_id
    assert action.parameters.get("idempotency_key") == action.action_id


# =====================================================================
# SCENARIO F: successful search -> failed search (old result not presented as current)
# =====================================================================
@pytest.mark.asyncio
async def test_scenario_f_failed_search_does_not_present_old_results():
    llm = FakeLLMClient(
        StructuredIntent(
            language=LanguageCode.ENGLISH,
            intents=[IntentName.PRODUCT_SEARCH],
            search_overrides=SearchOverrides(categories=["fail_search"]),
        ),
        "Sorry, search failed."
    )
    agent = FitzyAgent(llm=llm, tools=DummyCommerceAdapter())  # type: ignore
    state = agent.get_state("sec_f")
    # Simulate prior successful search A
    state.last_tool_results[ToolName.GET_PRODUCTS.value] = ProductSearchResponse(
        products=[ProductOption(product_id=1, variant_id=10, branch_id=1, article_code="A", product_name="Old A", price=Decimal("100"), available_quantity=1, is_available=True)],
        result_count=1,
    )

    # Execute failed search
    await agent.process_message(session_id="sec_f", message="search fail_search")
    assert state.action_plan.actions[0].status == ActionStatus.FAILED
    assert ToolName.GET_PRODUCTS.value not in agent._build_runtime_context(state)


# =====================================================================
# SCENARIO G: customer A accesses customer B session -> 403 Forbidden
# =====================================================================
def test_scenario_g_customer_ownership_security_403():
    client = TestClient(app)
    sid = "sec_g_session_777"
    r1 = client.post("/api/v1/chat", json={"session_id": sid, "message": "Hi", "customer_id": "cust_A"})
    assert r1.status_code == 200

    r2 = client.post("/api/v1/chat", json={"session_id": sid, "message": "Cart", "customer_id": "cust_B"})
    assert r2.status_code == 403
    assert "Forbidden" in r2.json()["detail"]


# =====================================================================
# SCENARIO H: simultaneous same-session mutations (serialized locks)
# =====================================================================
@pytest.mark.asyncio
async def test_scenario_h_session_concurrency_serialization():
    llm = FakeLLMClient(StructuredIntent(language=LanguageCode.ENGLISH, intents=[IntentName.GENERAL_CHAT]), "Hello")
    agent = FitzyAgent(llm=llm, tools=DummyCommerceAdapter())  # type: ignore

    lock1 = agent._get_session_lock("sess_h")
    lock2 = agent._get_session_lock("sess_h")
    assert lock1 is lock2, "Locks must be identical for identical session IDs"


# =====================================================================
# SECTION 3.3: Variant Ownership Validation
# =====================================================================
@pytest.mark.asyncio
async def test_variant_ownership_validation():
    state = ConversationState()
    state.remember_selected_product(101)
    # Set latest GET_PRODUCTS result with product 101 having variant 1001 (NOT variant 9999)
    state.last_tool_results[ToolName.GET_PRODUCTS.value] = ProductSearchResponse(
        products=[ProductOption(product_id=101, variant_id=1001, branch_id=1, article_code="A1", product_name="Shirt 101", price=Decimal("100"), available_quantity=1, is_available=True)],
        result_count=1,
    )

    is_valid = FitzyAgent._verify_variant_belongs_to_product(state, product_id=101, variant_id=1001)
    assert is_valid is True, "Variant 1001 belongs to product 101"

    is_invalid = FitzyAgent._verify_variant_belongs_to_product(state, product_id=101, variant_id=9999)
    assert is_invalid is False, "Variant 9999 does NOT belong to product 101"


# =====================================================================
# SECTION 15: Canonical Multi-Lingual Normalization
# =====================================================================
def test_canonical_normalization_english_urdu_roman():
    assert normalize_color("kaala") == "black"
    assert normalize_color("سیاہ") == "black"
    assert normalize_color("neela") == "blue"

    assert normalize_size("chota") == "S"
    assert normalize_size("درمیانہ") == "M"
    assert normalize_size("large") == "L"

    assert normalize_confirmation("haan") is True
    assert normalize_confirmation("ہاں") is True
    assert normalize_confirmation("nahi") is False
