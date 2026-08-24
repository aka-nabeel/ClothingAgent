"""Tests for Fitzy's closed intent vocabulary and deterministic language policy."""
from clothing_agent.app.agent.intent import (
    LanguageCode,
    StructuredIntent,
    classify_language,
)
from clothing_agent.app.agent.intents import (
    ActionName,
    IntentName,
    RequirementName,
    get_intent_definition,
    registered_intent_values,
)


def test_registry_is_closed_and_has_expected_core_intents():
    values = set(registered_intent_values())
    expected = {
        "general_chat",
        "product_search",
        "product_details",
        "product_availability",
        "branch_information",
        "branch_availability",
        "cart_view",
        "add_to_cart",
        "update_cart",
        "remove_from_cart",
        "clear_cart",
        "checkout_preview",
        "delivery_info",
        "place_order",
        "cancel_order",
    }
    assert expected <= values


def test_each_registered_intent_has_an_action():
    from clothing_agent.app.agent.intents import registered_intents

    for definition in registered_intents():
        assert definition.action in ActionName


def test_place_order_declares_all_critical_requirements():
    definition = get_intent_definition(IntentName.PLACE_ORDER)
    required = definition.required
    assert RequirementName.CART_ID in required
    assert RequirementName.CHECKOUT_PREVIEW in required
    assert RequirementName.DELIVERY_NAME in required
    assert RequirementName.DELIVERY_PHONE in required
    assert RequirementName.DELIVERY_ADDRESS in required
    assert RequirementName.DELIVERY_CITY in required
    assert RequirementName.ORDER_CONFIRMATION in required
    assert definition.confirmation_required is True


def test_product_details_requires_product_reference():
    definition = get_intent_definition(IntentName.PRODUCT_DETAILS)
    assert RequirementName.PRODUCT_REFERENCE in definition.required


def test_branch_availability_requires_product_and_branch():
    definition = get_intent_definition(IntentName.BRANCH_AVAILABILITY)
    assert RequirementName.PRODUCT_REFERENCE in definition.required
    assert RequirementName.BRANCH_REFERENCE in definition.required


def test_language_classifier():
    assert classify_language("I need black shirts") == LanguageCode.ENGLISH
    assert classify_language("mujhe black shirts chahiye") == LanguageCode.ROMAN_URDU
    assert classify_language("مجھے کالی شرٹس چاہیے") == LanguageCode.URDU_SCRIPT


def test_roman_urdu_short_turn_preserves_language():
    assert classify_language("haan", previous=LanguageCode.ROMAN_URDU) == LanguageCode.ROMAN_URDU


def test_devanagari_does_not_become_supported_language():
    language = classify_language("मुझे काली शर्ट चाहिए")
    assert language != LanguageCode.URDU_SCRIPT


def test_multi_intent_is_supported():
    intent = StructuredIntent(
        intents=[IntentName.PRODUCT_SEARCH, IntentName.BRANCH_INFORMATION],
        primary_intent=IntentName.PRODUCT_SEARCH,
    )
    assert len(intent.intents) == 2
