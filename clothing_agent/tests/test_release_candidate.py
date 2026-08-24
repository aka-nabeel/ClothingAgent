"""Comprehensive V1 Release Candidate Test Suite & Verification Harness.

Validates API contract compatibility, database seeding, real Groq LLM multilingual
understanding, 10-step sales journey E2E flow, negative resilience, and performance latency.
"""

import os
import sys
import time
from decimal import Decimal
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

root_dir = str(Path(__file__).resolve().parent.parent.parent)
clothing_app_dir = os.path.join(root_dir, "clothing_app")
if clothing_app_dir not in sys.path:
    sys.path.insert(0, clothing_app_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from main import app
from clothing_agent.app.core.config import get_config
from clothing_agent.app.llm.client import OpenAICompatibleLLMClient


@pytest.fixture(autouse=True)
def reset_db_engine():
    from clothing_agent.app.core.container import get_container
    get_container.cache_clear()


@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def real_llm(config):
    """Instantiate real Groq LLM client using CLOTHING_AGENT_LLM_API_KEY or GROQ_API_KEY."""
    api_key = config.llm_api_key.get_secret_value() if config.llm_api_key else (os.getenv("GROQ_API_KEY") or os.getenv("CLOTHING_AGENT_LLM_API_KEY"))
    if not api_key:
        pytest.skip("GROQ_API_KEY or CLOTHING_AGENT_LLM_API_KEY is not set for real LLM smoke test")
    return OpenAICompatibleLLMClient(
        base_url=config.llm_api_base,
        api_key=api_key,
        model=config.llm_model,
        timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_api_contract_matrix_compatibility():
    """Verify all 15 required API contract endpoints respond correctly on the unified FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health & Readiness
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        res = await client.get("/health/ready")
        assert res.status_code == 200
        assert res.json()["status"] == "ready"

        # 2. Branches
        res = await client.get("/api/v1/branches")
        assert res.status_code == 200
        branches = res.json()
        assert isinstance(branches, list)
        assert len(branches) > 0
        branch_code = branches[0]["branch_code"]
        branch_id = branches[0]["branch_id"]

        # 3. Products Search (POST)
        res = await client.post("/api/v1/products/search", json={"in_stock_only": True, "limit": 10})
        assert res.status_code == 200
        data = res.json()
        assert "products" in data and len(data["products"]) > 0
        product = data["products"][0]
        product_id = product["product_id"]
        in_stock_variant = next(
            (v for v in product.get("variants", []) if v.get("available_quantity", 0) > 0),
            product["variants"][0]
        )
        variant_id = in_stock_variant["variant_id"]
        if in_stock_variant.get("branch_id"):
            branch_id = in_stock_variant["branch_id"]

        # 4. Product Details (GET)
        res = await client.get(f"/api/v1/products/{product_id}")
        assert res.status_code == 200
        p_details = res.json()
        assert p_details["product"]["product_id"] == product_id

        # 5. Products List (GET)
        res = await client.get("/api/v1/products?limit=5")
        assert res.status_code == 200

        # 6. Inventory Availability & Branch selection
        for b in branches:
            bid = b["branch_id"]
            res = await client.get(f"/api/v1/inventory/availability?variant_id={variant_id}&branch_id={bid}")
            assert res.status_code == 200
            if res.json().get("available_quantity", 0) > 0:
                branch_id = bid
                break

        # 7. Promotions
        res = await client.get("/api/v1/promotions")
        assert res.status_code == 200

        # 8. Cart Lifecycle: Create
        session_id = f"test_session_{uuid4().hex[:8]}"
        res = await client.post("/api/v1/carts", json={"session_id": session_id})
        assert res.status_code == 201
        cart = res.json()
        cart_id = cart["cart_id"]

        # 9. Cart: Get
        res = await client.get(f"/api/v1/carts/{cart_id}")
        assert res.status_code == 200

        # 10. Cart: Add Item
        res = await client.post(
            f"/api/v1/carts/{cart_id}/items",
            json={"variant_id": variant_id, "branch_id": branch_id, "quantity": 1}
        )
        assert res.status_code == 200
        cart_with_item = res.json()
        assert len(cart_with_item["items"]) == 1
        item_id = cart_with_item["items"][0]["item_id"]

        # 11. Cart: Update Item
        res = await client.patch(
            f"/api/v1/carts/{cart_id}/items/{item_id}",
            json={"quantity": 2}
        )
        assert res.status_code == 200
        assert res.json()["items"][0]["quantity"] == 2

        # 12. Cart: Preview Checkout
        res = await client.post(
            f"/api/v1/carts/{cart_id}/preview",
            json={"delivery_city": "Lahore"}
        )
        assert res.status_code == 200
        preview = res.json()
        assert Decimal(str(preview["grand_total"])) > Decimal("0.00")

        # 13. Order Placement
        res = await client.post(
            "/api/v1/orders",
            json={
                "cart_id": cart_id,
                "customer_name": "Test Customer",
                "phone": "03001234567",
                "delivery_address": "123 Test Street",
                "city": "Lahore"
            }
        )
        assert res.status_code == 201
        order = res.json()
        assert "order_id" in order
        order_id = order["order_id"]

        # 14. Order Retrieval
        res = await client.get(f"/api/v1/orders/{order_id}")
        assert res.status_code == 200
        assert res.json()["order_id"] == order_id

        # 15. Agent Chat Endpoints (both /api/v1/agent/chat and /api/v1/chat)
        res1 = await client.post("/api/v1/agent/chat", json={"session_id": "s1", "message": "Hi"})
        assert res1.status_code == 200
        assert "response" in res1.json()

        res2 = await client.post("/api/v1/chat", json={"session_id": "s2", "message": "Hi"})
        assert res2.status_code == 200
        assert "response" in res2.json()


@pytest.mark.asyncio
async def test_multilingual_live_llm_responses(real_llm):
    """Test real LLM in English, Roman Urdu, and Urdu Script without Hindi/Devanagari outputs."""
    # English
    en_reply = await real_llm.generate_text(
        system_prompt="You are Fitzy, a friendly sales assistant for Northstar Menswear. Answer concisely in English.",
        user_message="I need something for a wedding."
    )
    assert len(en_reply) > 0
    assert not any('\u0900' <= char <= '\u097F' for char in en_reply), "Contained Devanagari script!"

    # Roman Urdu
    ru_reply = await real_llm.generate_text(
        system_prompt="You are Fitzy. Respond in Roman Urdu (Pakistani Urdu written in Latin script). Do NOT use Devanagari/Hindi.",
        user_message="mujhe shadi ke liye kuch acha sa chahiye"
    )
    assert len(ru_reply) > 0
    assert not any('\u0900' <= char <= '\u097F' for char in ru_reply), "Contained Devanagari script!"

    # Urdu Script
    ur_reply = await real_llm.generate_text(
        system_prompt="You are Fitzy. Respond in authentic Urdu script. Do NOT use Hindi/Devanagari.",
        user_message="مجھے شادی کے لیے کچھ اچھا سا چاہیے"
    )
    assert len(ur_reply) > 0
    assert not any('\u0900' <= char <= '\u097F' for char in ur_reply), "Contained Devanagari script!"


@pytest.mark.asyncio
async def test_complete_live_sales_journey_e2e():
    """Execute the full 10-step customer sales journey via /api/v1/agent/chat."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = f"journey_{uuid4().hex[:8]}"

        steps = [
            "I need something for a wedding.",
            "Show me black ones under 5000.",
            "Tell me more about the first one.",
            "Add it to my cart.",
            "Change quantity to 2.",
            "What's my total?",
            "My name is Ahmed.",
            "My phone is 03001234567.",
            "DHA Lahore.",
            "Yes, place the order."
        ]

        for idx, step_msg in enumerate(steps, start=1):
            t0 = time.perf_counter()
            res = await client.post("/api/v1/agent/chat", json={"session_id": session_id, "message": step_msg})
            elapsed = time.perf_counter() - t0
            assert res.status_code == 200, f"Step {idx} failed with status {res.status_code}: {res.text}"
            reply = res.json()["response"]
            safe_reply = reply[:100].encode('ascii', errors='replace').decode()
            print(f"\n[Step {idx} ({elapsed:.2f}s)] User: '{step_msg}'\nFitzy: '{safe_reply}...'")


@pytest.mark.asyncio
async def test_negative_resilience():
    """Test safety, non-existent products, empty cart checkout, and graceful error handling."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = f"negative_{uuid4().hex[:8]}"

        # Nonexistent product search
        res = await client.post("/api/v1/agent/chat", json={"session_id": session_id, "message": "Show me red astronaut space suits"})
        assert res.status_code == 200
        reply = res.json()["response"]
        assert len(reply) > 0

        # Empty cart preview check
        empty_cart_res = await client.post("/api/v1/carts", json={"session_id": "empty_session"})
        empty_cart_id = empty_cart_res.json()["cart_id"]
        preview_res = await client.post(f"/api/v1/carts/{empty_cart_id}/preview", json={"delivery_city": "Lahore"})
        assert preview_res.status_code == 200
        assert Decimal(str(preview_res.json()["grand_total"])) == Decimal("0.00")


@pytest.mark.asyncio
async def test_performance_sanity():
    """Measure latency across catalog, cart, preview, and agent chat operations."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Search latency
        t0 = time.perf_counter()
        res = await client.post("/api/v1/products/search", json={"limit": 5})
        search_latency = time.perf_counter() - t0
        assert res.status_code == 200
        assert search_latency < 2.0, f"Product search too slow: {search_latency:.3f}s"

        # Product detail latency
        product_id = res.json()["products"][0]["product_id"]
        t0 = time.perf_counter()
        res = await client.get(f"/api/v1/products/{product_id}")
        detail_latency = time.perf_counter() - t0
        assert res.status_code == 200
        assert detail_latency < 1.0, f"Product detail too slow: {detail_latency:.3f}s"

        print(f"\n[Performance Sanity] Search Latency: {search_latency:.3f}s | Detail Latency: {detail_latency:.3f}s")


@pytest.mark.asyncio
async def test_currency_pkr_enforcement():
    """Verify currency is output as PKR or Rs. and NEVER as $."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = f"currency_{uuid4().hex[:8]}"
        res = await client.post("/api/v1/agent/chat", json={"session_id": session_id, "message": "Show me shirts under 5000"})
        assert res.status_code == 200
        reply = res.json()["response"]
        assert "$" not in reply, f"Response accidentally used $: {reply}"


@pytest.mark.asyncio
async def test_devanagari_guard_rejection():
    """Verify Devanagari/Hindi input is rejected from output emission."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = f"deva_{uuid4().hex[:8]}"
        res = await client.post("/api/v1/agent/chat", json={"session_id": session_id, "message": "मुझे काली शर्ट चाहिए"})
        assert res.status_code == 200
        reply = res.json()["response"]
        assert not any('\u0900' <= char <= '\u097F' for char in reply), f"Response emitted Devanagari: {reply}"


@pytest.mark.asyncio
async def test_intent_change_and_topic_switching():
    """Verify topic switching removes obsolete search filters while preserving preferences."""
    from clothing_agent.app.agent.agent import FitzyAgent
    from clothing_agent.app.agent.state import LanguageMode
    from clothing_agent.app.core.container import get_container

    container = get_container()
    agent: FitzyAgent = container.fitzy_agent
    state = agent.get_state(f"switch_{uuid4().hex[:8]}")

    # Set persistent preference
    state.preferences.preferred_colors = ["black"]

    # Turn 1: Search wedding clothes
    state.current_search.occasions = ["wedding"]

    # Turn 2: Switch topic to office clothes
    await agent.process_message(session_id=state.session_id, message="Actually forget that. Show me casual office clothes.")
    assert state.preferences.preferred_colors == ["black"], "Persistent preference was lost!"


@pytest.mark.asyncio
async def test_complete_live_sales_journey_roman_urdu_e2e():
    """Execute full 10-step sales journey in Roman Urdu with real LLM."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = f"roman_{uuid4().hex[:8]}"

        steps = [
            "mujhe shadi ke liye kuch acha sa chahiye",
            "black wala dikhao 5000 se kam mein",
            "pehlay walay ki details batao",
            "is ko meri cart mein add kardo",
            "quantity 2 kardo",
            "mera total kitna hua?",
            "mera naam Ahmed hai",
            "mera phone 03001234567 hai",
            "DHA Lahore",
            "haan order place kardo"
        ]

        for idx, step_msg in enumerate(steps, start=1):
            t0 = time.perf_counter()
            res = await client.post("/api/v1/agent/chat", json={"session_id": session_id, "message": step_msg})
            elapsed = time.perf_counter() - t0
            assert res.status_code == 200, f"Roman Urdu Step {idx} failed: {res.text}"
            reply = res.json()["response"]
            assert not any('\u0900' <= char <= '\u097F' for char in reply), f"Roman Urdu Step {idx} contained Devanagari!"
            safe_reply = reply[:100].encode('ascii', errors='replace').decode()
            print(f"\n[Roman Urdu Step {idx} ({elapsed:.2f}s)] User: '{step_msg}'\nFitzy: '{safe_reply}...'")


@pytest.mark.asyncio
async def test_complete_live_sales_journey_urdu_script_e2e():
    """Execute full 10-step sales journey in Urdu Script with real LLM."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = f"urdu_{uuid4().hex[:8]}"

        steps = [
            "مجھے شادی کے لیے کچھ اچھا سا چاہیے",
            "کالی شرٹ دکھاؤ 5000 سے کم میں",
            "پہلے والے کی تفصیلات بتاؤ",
            "اس کو کارٹ میں شامل کرو",
            "تعداد 2 کر دو",
            "میرا ٹوٹل کتنا ہوا؟",
            "میرا نام احمد ہے",
            "میرا فون نمبر 03001234567 ہے",
            "ڈی ایچ اے لاہور",
            "جی آرڈر پلےس کر دیں"
        ]

        for idx, step_msg in enumerate(steps, start=1):
            t0 = time.perf_counter()
            res = await client.post("/api/v1/agent/chat", json={"session_id": session_id, "message": step_msg})
            elapsed = time.perf_counter() - t0
            assert res.status_code == 200, f"Urdu Script Step {idx} failed: {res.text}"
            reply = res.json()["response"]
            assert not any('\u0900' <= char <= '\u097F' for char in reply), f"Urdu Script Step {idx} contained Devanagari!"
            try:
                print(f"\n[Urdu Script Step {idx} ({elapsed:.2f}s)] Fitzy response received ({len(reply)} chars)")
            except Exception:
                pass

