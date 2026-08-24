from __future__ import annotations

from decimal import Decimal

import pytest

from clothing_agent.app.agent.agent import FitzyAgent
from clothing_agent.app.agent.contracts import ToolName
from clothing_agent.app.agent.intent import IntentExtraction, IntentRequest, IntentType
from clothing_agent.app.agent.state import LanguageMode
from clothing_agent.app.integration.client import CommerceToolAdapter
from clothing_agent.app.integration.schemas import CartView, ProductOption, ProductSearchResponse
from clothing_agent.app.llm.client import FakeLLMClient


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[ToolName, dict]] = []
        self.cart_counter = 0

    async def execute(self, tool_name: ToolName, parameters):
        self.calls.append((tool_name, dict(parameters)))
        if tool_name == ToolName.GET_PRODUCTS:
            return ProductSearchResponse(
                products=[
                    ProductOption(
                        product_id=1,
                        variant_id=10,
                        branch_id=100,
                        article_code="NS-SH-001",
                        product_name="Classic Oxford Shirt",
                        category="shirts",
                        color="Black",
                        size="L",
                        price=Decimal("4500.00"),
                        branch_code="ISB-F7",
                        branch_name="F-7 Islamabad",
                        city="Islamabad",
                        available_quantity=4,
                        image_url="https://example.com/shirt.jpg",
                        is_available=True,
                    )
                ],
                result_count=1,
            )
        if tool_name == ToolName.CREATE_CART:
            self.cart_counter += 1
            return CartView(cart_id=f"00000000-0000-0000-0000-{self.cart_counter:012d}", item_count=0, subtotal=Decimal("0"))
        if tool_name == ToolName.ADD_TO_CART:
            return CartView(cart_id=parameters["cart_id"], item_count=1, subtotal=Decimal("4500"))
        if tool_name == ToolName.GET_CART:
            return CartView(cart_id=parameters["cart_id"], item_count=1, subtotal=Decimal("4500"))
        raise AssertionError(f"Unexpected tool call: {tool_name}")


@pytest.mark.asyncio
async def test_search_uses_llm_intent_and_existing_tool_adapter() -> None:
    extraction = IntentExtraction(
        language=LanguageMode.ENGLISH,
        intents=[
            IntentRequest(
                intent_id="1",
                intent_type=IntentType.PRODUCT_SEARCH,
                parameters={"categories": ["shirts"], "colors": ["black"], "maximum_price": Decimal("5000")},
            )
        ],
    )
    llm = FakeLLMClient(extraction, "Here is a suitable black shirt.")
    adapter = FakeAdapter()
    agent = FitzyAgent(llm=llm, tools=adapter)  # type: ignore[arg-type]

    response = await agent.process_message(session_id="s1", message="Show me black shirts under 5000")

    assert "shirt" in response.lower()
    assert adapter.calls[0][0] == ToolName.GET_PRODUCTS
    assert adapter.calls[0][1]["colors"] == ["black"]


@pytest.mark.asyncio
async def test_add_to_cart_can_use_existing_cart_and_displayed_variant() -> None:
    search = IntentExtraction(
        language=LanguageMode.ENGLISH,
        intents=[IntentRequest(intent_id="1", intent_type=IntentType.PRODUCT_SEARCH, parameters={"categories": ["shirts"]})],
    )
    add = IntentExtraction(
        language=LanguageMode.ENGLISH,
        intents=[IntentRequest(intent_id="2", intent_type=IntentType.ADD_TO_CART, parameters={"product_reference": 1})],
    )

    class SequenceLLM(FakeLLMClient):
        def __init__(self):
            super().__init__(search, "")
            self.responses = [search, add]

        async def generate_structured(self, *, system_prompt, user_message, response_model):
            response = self.responses.pop(0)
            return response_model.model_validate(response.model_dump())

        async def generate_text(self, *, system_prompt, user_message):
            return "Added it to your cart."

    llm = SequenceLLM()
    adapter = FakeAdapter()
    agent = FitzyAgent(llm=llm, tools=adapter)  # type: ignore[arg-type]

    await agent.process_message(session_id="s2", message="show me shirts")
    await agent.process_message(session_id="s2", message="add the first one")

    tool_names = [call[0] for call in adapter.calls]
    assert ToolName.GET_PRODUCTS in tool_names
    assert ToolName.CREATE_CART in tool_names
    assert ToolName.ADD_TO_CART in tool_names


@pytest.mark.asyncio
async def test_urdu_response_rejects_devanagari() -> None:
    extraction = IntentExtraction(
        language=LanguageMode.ROMAN_URDU,
        intents=[IntentRequest(intent_id="1", intent_type=IntentType.GENERAL_CONVERSATION)],
    )

    class UnsafeLLM(FakeLLMClient):
        def __init__(self):
            super().__init__(extraction, "")
            self.responses = ["मجھے سمجھ آ گیا", "Bilkul, main madad karta hoon."]

        async def generate_text(self, *, system_prompt, user_message):
            return self.responses.pop(0)

    agent = FitzyAgent(llm=UnsafeLLM(), tools=FakeAdapter())  # type: ignore[arg-type]
    result = await agent.process_message(session_id="s3", message="mujhe help chahiye")
    assert result == "Bilkul, main madad karta hoon."


@pytest.mark.asyncio
async def test_explicit_language_override_and_config_toggle() -> None:
    extraction = IntentExtraction(
        language=LanguageMode.URDU_SCRIPT,
        intents=[IntentRequest(intent_id="1", intent_type=IntentType.GENERAL_CONVERSATION)],
    )
    llm = FakeLLMClient(extraction, "Hello, how can I help you?")
    adapter = FakeAdapter()
    agent = FitzyAgent(llm=llm, tools=adapter)  # type: ignore[arg-type]

    await agent.process_message(session_id="s4", message="who are you", language="english")
    state = agent.get_state("s4")
    assert state.language == LanguageMode.ENGLISH
