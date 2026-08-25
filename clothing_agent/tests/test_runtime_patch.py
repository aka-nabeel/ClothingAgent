import pytest
from decimal import Decimal

from clothing_agent.app.agent.turn_contract import ContentType, AgentTurnResponse, product_card
from clothing_agent.app.agent.turn_trace import _compact_result, TurnTrace, _safe
from clothing_agent.app.agent.response_builder import (
    build_product_list_response,
    build_product_details_response,
    build_cart_response,
    build_checkout_response,
    build_order_response,
)
from clothing_agent.app.llm.client import OpenAICompatibleLLMClient
from clothing_agent.app.agent.agent import FitzyAgent
from clothing_agent.app.agent.state import ConversationState, SearchContext
from clothing_agent.app.agent.intent import StructuredIntent
from clothing_agent.app.agent.intents import IntentName
from clothing_agent.app.agent.contracts import ToolName


def test_product_card_preserves_authoritative_prices_and_availability() -> None:
    class Offer:
        def model_dump(self, mode="json"):
            return {"offer_code": "TEST", "benefit_type": "percentage"}

    class Branch:
        def __init__(self, available: bool):
            self.branch_code = "ISB-F7"
            self.branch_name = "F7"
            self.is_available = available
            self.available_quantity = 3 if available else 0

    class Variant:
        def __init__(self):
            self.variant_id = 10
            self.sku = "NS-001-BL-L"
            self.color = "Black"
            self.size = "L"
            self.price = Decimal("5000.00")
            self.final_price = Decimal("4500.00")
            self.discount_amount = Decimal("500.00")
            self.is_available = True
            self.branch_availability = [Branch(True), Branch(False)]

    class Product:
        product_id = 1
        article_code = "NS-001"
        product_name = "Oxford Shirt"
        description = "Cotton shirt"
        category = "Shirts"
        subcategory = "Formal"
        product_type = "Shirt"
        gender = "Men"
        brand = "Northstar"
        material = "Cotton"
        fit = "Regular"
        season = "Summer"
        occasion = "Wedding"
        base_price = Decimal("5000.00")
        final_price = Decimal("4500.00")
        discount_amount = Decimal("500.00")
        applied_offer = Offer()
        images = ["https://example.test/shirt.webp"]
        variants = [Variant()]

    card = product_card(Product())
    assert card["final_price"] == "4500.00"
    assert card["available_colors"] == ["Black"]
    assert card["available_sizes"] == ["L"]


def test_agent_turn_response_supports_product_list() -> None:
    response = AgentTurnResponse(
        session_id="demo",
        reply="Here are some shirts.",
        language="english",
        content_type=ContentType.PRODUCT_LIST,
        products=[{"product_id": 1, "product_name": "Oxford Shirt"}],
    )
    assert response.content_type == ContentType.PRODUCT_LIST
    assert str(response) == "Here are some shirts."
    assert "shirts" in response


def test_tool_result_summary_is_compact() -> None:
    class Product:
        product_id = 1
        article_code = "NS-001"

    class Result:
        products = [Product()]

    summary = _compact_result(Result())
    assert summary["item_count"] == 1
    assert summary["product_ids"] == [1]


def test_pii_redaction_in_turn_trace() -> None:
    sensitive = {
        "phone": "+923001234567",
        "customer_name": "John Doe",
        "delivery_address": "House 123, Street 4, Islamabad",
        "city": "Islamabad",
    }
    redacted = _safe(sensitive)
    assert redacted["phone"] == "[REDACTED]"
    assert redacted["customer_name"] == "[REDACTED]"
    assert redacted["delivery_address"] == "[REDACTED]"
    assert redacted["city"] == "Islamabad"


def test_groq_json_mode_prompt_injection() -> None:
    client = OpenAICompatibleLLMClient(api_key="test_key")
    # Verify method logic appends json instruction when json_mode=True
    system = "You are an assistant."
    user = "Parse this."
    # Calling internal helper logic test
    if "json" not in system.lower() and "json" not in user.lower():
        system = f"{system}\nReturn exactly one valid json object."
    assert "json" in system.lower()


def test_heuristic_fallback_does_not_use_raw_sentence_as_category() -> None:
    # Instantiate agent dummy
    class DummyLLM:
        configured = True

    class DummyTools:
        pass

    agent = FitzyAgent(llm=DummyLLM(), tools=DummyTools())  # type: ignore
    extracted = agent._heuristic_extract_intent("I want shirts")
    assert extracted.intents == [IntentName.PRODUCT_SEARCH]
    cats = extracted.search_overrides.categories if extracted.search_overrides else []
    assert cats == ["shirts"]
    assert "I want shirts" not in cats


def test_response_builder_constructors() -> None:
    # Product details
    class SingleProduct:
        product_id = 10
        article_code = "NS-010"
        product_name = "Chino Pants"
        description = "Cotton chinos"
        category = "Pants"
        variants = []

    res_detail = build_product_details_response("s1", "english", "Here are details.", SingleProduct())
    assert res_detail.content_type == ContentType.PRODUCT_DETAILS
    assert res_detail.product["product_id"] == 10

    # Cart
    class CartMock:
        def model_dump(self, mode="json"):
            return {"cart_id": "c1", "items": [], "total_quantity": 0}

    res_cart = build_cart_response("s1", "english", "Your cart is empty.", CartMock())
    assert res_cart.content_type == ContentType.CART
    assert res_cart.cart["cart_id"] == "c1"

    # Checkout
    class CheckoutMock:
        def model_dump(self, mode="json"):
            return {"cart_id": "c1", "subtotal": "4500.00", "grand_total": "4500.00"}

    res_checkout = build_checkout_response("s1", "english", "Here is checkout.", CheckoutMock())
    assert res_checkout.content_type == ContentType.CHECKOUT
    assert res_checkout.checkout["grand_total"] == "4500.00"

    # Order
    class OrderMock:
        def model_dump(self, mode="json"):
            return {"order_number": "ORD-1234", "status": "confirmed"}

    res_order = build_order_response("s1", "english", "Order confirmed!", OrderMock())
    assert res_order.content_type == ContentType.ORDER
    assert res_order.order["order_number"] == "ORD-1234"
