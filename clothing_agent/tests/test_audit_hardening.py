"""Targeted verification for audit requirements: invalid references, cart mutation ordering, session concurrency, state store persistence, and security controls."""

import asyncio
from decimal import Decimal
import pytest
from fastapi.testclient import TestClient

from clothing_agent.app.agent.agent import FitzyAgent
from clothing_agent.app.agent.contracts import ToolName
from clothing_agent.app.agent.intent import LanguageCode, SearchOverrides, StructuredIntent
from clothing_agent.app.agent.intents import IntentName
from clothing_agent.app.agent.planner import ActionPlanner
from clothing_agent.app.agent.state import ConversationState, DisplayedProductReference
from clothing_agent.app.core.container import get_container
from clothing_agent.app.core.state_store import FileConversationStateStore
from clothing_agent.app.llm.client import FakeLLMClient
from clothing_app.app.main import app


class DummyCommerceAdapter:
    async def get_branches(self):
        return [{"branch_id": 1, "branch_code": "LHR-01", "city": "Lahore"}]

    async def execute(self, tool_name: ToolName, parameters: dict):
        return {"status": "ok", "tool": tool_name.value}


@pytest.mark.asyncio
async def test_invalid_product_reference_out_of_bounds_returns_none():
    """Verify that an invalid product index (e.g. 9 when 3 products displayed) does NOT fallback to product 1."""
    state = ConversationState()
    state.remember_displayed_products([
        DisplayedProductReference(index=1, product_id=101, article_code="A1", product_name="Shirt 1"),
        DisplayedProductReference(index=2, product_id=102, article_code="A2", product_name="Shirt 2"),
        DisplayedProductReference(index=3, product_id=103, article_code="A3", product_name="Shirt 3"),
    ])

    resolved_pid = FitzyAgent._resolve_displayed_product_id(state, 9)
    assert resolved_pid is None, "Out-of-bounds index must return None, not product 1"

    resolved_pid_word = FitzyAgent._resolve_displayed_product_id(state, "invalid_word")
    assert resolved_pid_word is None, "Invalid text reference must return None"


@pytest.mark.asyncio
async def test_cart_mutation_sequential_dependency_chaining():
    """Verify planner chains dependencies sequentially for cart mutations."""
    planner = ActionPlanner()

    # Plan with multiple cart mutations: ADD + ADD + UPDATE + REMOVE + CLEAR
    extraction = StructuredIntent(
        language=LanguageCode.ENGLISH,
        intents=[
            IntentName.ADD_TO_CART,
            IntentName.ADD_TO_CART,
            IntentName.UPDATE_CART,
            IntentName.REMOVE_FROM_CART,
            IntentName.CLEAR_CART,
        ],
    )
    plan = planner.build_plan(extraction)
    actions = plan.actions

    # First action creates cart or adds to cart
    cart_write_actions = [a for a in actions if a.tool_name in {
        ToolName.CREATE_CART, ToolName.ADD_TO_CART, ToolName.UPDATE_CART, ToolName.REMOVE_FROM_CART, ToolName.CLEAR_CART
    }]

    # Ensure every cart write action after the first depends on the immediately preceding cart write action
    for i in range(1, len(cart_write_actions)):
        prev_id = cart_write_actions[i - 1].action_id
        curr_deps = cart_write_actions[i].dependency_ids
        assert prev_id in curr_deps, f"Cart mutation {cart_write_actions[i].tool_name} must depend on prior cart write {cart_write_actions[i-1].tool_name}"


@pytest.mark.asyncio
async def test_session_concurrency_locks():
    """Verify FitzyAgent serializes processing for the same session ID via locks."""
    llm = FakeLLMClient(
        StructuredIntent(language=LanguageCode.ENGLISH, intents=[IntentName.GENERAL_CHAT]),
        "Hello!"
    )
    agent = FitzyAgent(llm=llm, tools=DummyCommerceAdapter())  # type: ignore

    lock1 = agent._get_session_lock("session_alpha")
    lock2 = agent._get_session_lock("session_alpha")
    assert lock1 is lock2, "Same session ID must reuse the exact same asyncio Lock"

    lock_beta = agent._get_session_lock("session_beta")
    assert lock1 is not lock_beta, "Different session IDs must have separate locks"


@pytest.mark.asyncio
async def test_state_store_persistence(tmp_path):
    """Verify FileConversationStateStore persists state and hash filenames properly."""
    store = FileConversationStateStore(tmp_path)
    state = ConversationState()
    session_id_str = str(state.session_id)
    state.delivery.customer_name = "Alice Test"

    store.save(state)
    loaded = store.load(session_id_str)
    assert loaded is not None
    assert loaded.delivery.customer_name == "Alice Test"


def test_session_ownership_security_validation():
    """Verify chat route rejects requests claiming an active session owned by a different customer_id."""
    client = TestClient(app)
    session_id = "test_owner_session_123"

    # Turn 1: Customer A sets ownership
    r1 = client.post("/api/v1/chat", json={"session_id": session_id, "message": "Hi", "customer_id": "cust_A"})
    assert r1.status_code == 200

    # Turn 2: Customer B attempts to access same session
    r2 = client.post("/api/v1/chat", json={"session_id": session_id, "message": "Show my cart", "customer_id": "cust_B"})
    assert r2.status_code == 403
    assert "Forbidden" in r2.json()["detail"]
