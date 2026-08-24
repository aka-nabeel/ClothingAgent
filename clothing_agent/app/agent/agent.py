"""Fitzy's V1 runtime orchestrator."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from .contracts import ToolName
from .execution import ActionExecutionCoordinator
from .intent import IntentExtraction, IntentRequest, IntentType, classify_language
from .planner import ActionPlanner
from .response import ResponseGuard
from .state import (
    ActionStatus,
    ConversationState,
    DisplayedProductReference,
    LanguageMode,
)
from ..core.config import AgentConfig, get_config
from ..integration.client import CommerceToolAdapter
from ..integration.schemas import ProductSearchResponse
from ..llm.client import LLMClient
from ..llm.prompts import build_intent_system_prompt

logger = logging.getLogger("fitzy.agent")


class FitzyAgent:
    """Coordinate language understanding, planning, requirements and tool execution."""

    def __init__(self, *, llm: LLMClient, tools: CommerceToolAdapter, config: AgentConfig | None = None) -> None:
        self._llm = llm
        self._tools = tools
        self._config = config or get_config()
        self._planner = ActionPlanner()
        self._execution = ActionExecutionCoordinator()
        self._responses = ResponseGuard(llm)
        self._states: dict[str, ConversationState] = {}

    def get_state(self, session_id: str) -> ConversationState:
        """Return an existing conversation state or create a fresh one."""

        if session_id not in self._states:
            self._states[session_id] = ConversationState()
        return self._states[session_id]

    async def process_message(
        self,
        *,
        session_id: str,
        message: str,
        language: str | None = None,
    ) -> str:
        """Process one customer message through the full V1 runtime pipeline."""

        state = self.get_state(session_id)
        logger.info("[CHAT INPUT] session=%s | message=%r | explicit_language=%r", session_id, message, language)

        # Language Priority Rule:
        # 1. If explicit `language` parameter is passed in API request, update session state.
        if language:
            state.set_language(language)
        # 2. If session language is uninitialized, detect once to set default session language.
        elif not state.language:
            det_language = classify_language(message, None)
            state.set_language(det_language)
        # 3. If session language is already set (e.g. from UI header or session setup),
        # state.language remains unchanged as the authoritative priority session language.
        
        extraction = await self._extract_intent(message)

        extracted_intents_summary = [
            f"{i.intent_type.value}(params={i.parameters}, confirmation={i.explicit_confirmation})"
            for i in extraction.intents
        ]
        logger.info(
            "[INTENT EXTRACTED] session=%s | language=%s | intents=%s",
            session_id,
            state.language.value if state.language else "unknown",
            extracted_intents_summary,
        )

        waiting_actions = [action.model_copy(deep=True) for action in state.action_plan.actions if action.status == ActionStatus.WAITING_FOR_INPUT]
        self._apply_intent_to_state(extraction, state)
        self._reopen_waiting_actions_for_new_input(state)
        plan = self._planner.build_plan(extraction, state)
        self._merge_waiting_actions(plan, waiting_actions)
        state.action_plan = plan
        
        actions_summary = [
            f"{a.action_id}:{a.tool_name.value}(params={a.parameters}, missing={a.missing_parameters}, status={a.status.value})"
            for a in plan.actions
        ]
        logger.info(
            "[PLAN & STATE] session=%s | plan_id=%s | active_search=%s | delivery=%s | cart_items=%d | actions=%s",
            session_id,
            plan.plan_id,
            state.current_search.model_dump(exclude_none=True),
            state.delivery.model_dump(exclude_none=True),
            state.cart.item_count,
            actions_summary,
        )

        self._resolve_known_parameters(state)
        await self._execute_until_waiting(state)

        runtime_context = self._build_runtime_context(state)
        reply = await self._responses.generate(
            language=state.language or LanguageMode.ENGLISH,
            user_message=message,
            runtime_context=runtime_context,
        )
        logger.info("[CHAT REPLY] session=%s | reply=%r", session_id, reply)
        return reply

    async def _extract_intent(self, message: str) -> IntentExtraction:
        """Use the LLM for semantic intent extraction with heuristic fallback."""
        try:
            return await self._llm.generate_structured(
                system_prompt=build_intent_system_prompt(),
                user_message=message,
                response_model=IntentExtraction,
            )
        except Exception as exc:
            logger.warning("llm.extraction_failed error=%s falling_back_to_heuristic", exc)
            return self._heuristic_extract_intent(message)

    def _heuristic_extract_intent(self, message: str) -> IntentExtraction:
        msg = message.lower()
        intents = []
        if any(w in msg for w in ["kaun kaun", "kya kya", "what products", "all products", "categories", "range", "collection", "kya hai"]):
            intents.append(IntentRequest(intent_id="intent-1", intent_type=IntentType.STORE_CONTEXT, parameters={}))
        elif any(w in msg for w in ["search", "find", "shirt", "pant", "kurta", "denim", "dress", "show", "buy", "oxford"]):
            intents.append(IntentRequest(intent_id="intent-1", intent_type=IntentType.PRODUCT_SEARCH, parameters={"query_text": message}))
        elif any(w in msg for w in ["cart", "add"]):
            intents.append(IntentRequest(intent_id="intent-1", intent_type=IntentType.ADD_TO_CART, parameters={}))
        elif any(w in msg for w in ["checkout", "order", "place"]):
            intents.append(IntentRequest(intent_id="intent-1", intent_type=IntentType.CHECKOUT, parameters={}))
        else:
            intents.append(IntentRequest(intent_id="intent-1", intent_type=IntentType.GENERAL_CONVERSATION, parameters={"query": message}))
        return IntentExtraction(language=LanguageMode.ENGLISH, intents=intents)

    def _apply_intent_to_state(self, extraction: IntentExtraction, state: ConversationState) -> None:
        """Persist turn facts into the correct long-lived or action-scoped state.

        Delivery information is accumulated across turns, explicit order
        confirmation is retained for the current execution cycle, and search
        filters update the current search without erasing unrelated preferences.
        """

        for intent in extraction.intents:
            params = intent.parameters

            self._apply_delivery_fields(params, state)

            if intent.intent_type in {
                IntentType.STORE_CONTEXT,
                IntentType.GENERAL_CONVERSATION,
            }:
                state.displayed_products = []
                state.selected_product_id = None
                state.current_search.clear()

            if intent.intent_type in {
                IntentType.ADD_TO_CART,
                IntentType.UPDATE_CART,
                IntentType.REMOVE_FROM_CART,
                IntentType.CLEAR_CART,
            }:
                state.last_tool_results["explicit_confirmation"] = None
                state.last_tool_results.pop(ToolName.PREVIEW_CHECKOUT.value, None)

            if intent.explicit_confirmation is not None and intent.intent_type == IntentType.PLACE_ORDER:
                if ToolName.PREVIEW_CHECKOUT.value in state.last_tool_results:
                    state.last_tool_results["explicit_confirmation"] = intent.explicit_confirmation
                else:
                    state.last_tool_results["explicit_confirmation"] = None

            if intent.intent_type != IntentType.PRODUCT_SEARCH:
                continue

            # Normalize parameter aliases from LLM extractions
            if "max_price" in params and "maximum_price" not in params:
                params["maximum_price"] = params["max_price"]
            if "min_price" in params and "minimum_price" not in params:
                params["minimum_price"] = params["min_price"]
            if "color" in params and "colors" not in params:
                val = params["color"]
                params["colors"] = [val] if isinstance(val, str) else val
            if "category" in params and "categories" not in params:
                val = params["category"]
                params["categories"] = [val] if isinstance(val, str) else val
            if "query" in params and "query_text" not in params:
                params["query_text"] = params["query"]

            state.current_search.update_from_mapping({
                key: value
                for key, value in params.items()
                if key in type(state.current_search).model_fields
            })

            # Explicit preference language may be represented by the extractor.
            if params.get("remember_preference") is True:
                for field_name in (
                    "colors",
                    "excluded_colors",
                    "categories",
                    "product_types",
                    "occasions",
                    "materials",
                    "fits",
                    "size_mapping",
                    "minimum_price",
                    "maximum_price",
                    "branch_code",
                ):
                    value = params.get(field_name)
                    if value in (None, [], {}, ""):
                        continue
                    mapping = {
                        "colors": "preferred_colors",
                        "excluded_colors": "excluded_colors",
                        "categories": "preferred_categories",
                        "product_types": "preferred_product_types",
                        "occasions": "preferred_occasions",
                        "materials": "preferred_materials",
                        "fits": "preferred_fits",
                        "size_mapping": "size_mapping",
                        "minimum_price": "minimum_price",
                        "maximum_price": "maximum_price",
                        "branch_code": "branch_preference",
                    }
                    setattr(state.preferences, mapping[field_name], value)

    @staticmethod
    def _apply_delivery_fields(params: dict[str, Any], state: ConversationState) -> None:
        """Merge explicitly extracted delivery fields into conversation state."""

        field_names = (
            "customer_name",
            "phone",
            "delivery_address",
            "city",
            "delivery_notes",
        )
        for field_name in field_names:
            value = params.get(field_name)
            if value in (None, ""):
                continue
            setattr(state.delivery, field_name, value)

    def _reopen_waiting_actions_for_new_input(self, state: ConversationState) -> None:
        """Re-evaluate previously blocked actions after the customer supplied new facts.

        A waiting action is not discarded: once the next turn adds missing
        requirements to state, it becomes eligible for the same execution plan.
        """

        for action in state.action_plan.actions:
            if action.status == ActionStatus.WAITING_FOR_INPUT:
                action.status = ActionStatus.PENDING
                action.missing_parameters = []

    @staticmethod
    def _merge_waiting_actions(plan: Any, waiting_actions: list[Any]) -> None:
        """Carry forward blocked actions so follow-up messages can satisfy them.

        A customer may answer a missing field without repeating the original
        intent, for example: ``"Lahore"`` after Fitzy asked for a city. The
        previously blocked action therefore remains part of the active plan.
        """

        existing_tools = {action.tool_name for action in plan.actions}
        for waiting in waiting_actions:
            if waiting.tool_name in existing_tools:
                continue
            waiting.status = ActionStatus.PENDING
            waiting.missing_parameters = []
            plan.actions.insert(0, waiting)

    def _resolve_known_parameters(self, state: ConversationState) -> None:
        """Resolve values already known from state or previous tool results.

        This step never invents customer choices.  It only derives values that
        are deterministic from prior results, such as a cart ID or a product
        reference that already exists in the displayed product set.
        """

        cart_result = state.last_tool_results.get(ToolName.CREATE_CART.value)
        if cart_result is not None and getattr(cart_result, "cart_id", None):
            state.cart.cart_id = cart_result.cart_id

        latest_cart = state.last_tool_results.get(ToolName.GET_CART.value)
        if latest_cart is not None:
            self._apply_cart_result(state, latest_cart)

        add_result = state.last_tool_results.get(ToolName.ADD_TO_CART.value)
        if add_result is not None:
            self._apply_cart_result(state, add_result)

        preview = state.last_tool_results.get(ToolName.PREVIEW_CHECKOUT.value)
        if preview is not None and getattr(preview, "cart_id", None):
            state.cart.cart_id = preview.cart_id

        for action in state.action_plan.actions:
            self._resolve_action_parameters(state, action)

    def _resolve_action_parameters(self, state: ConversationState, action: Any) -> None:
        """Fill action parameters from deterministic conversation state."""

        params = action.parameters
        if action.tool_name == ToolName.GET_PRODUCTS:
            action.parameters = self._build_effective_product_search(state, action.parameters)

        if action.tool_name in {ToolName.GET_CART, ToolName.ADD_TO_CART, ToolName.UPDATE_CART, ToolName.REMOVE_FROM_CART, ToolName.CLEAR_CART, ToolName.PREVIEW_CHECKOUT, ToolName.PLACE_ORDER}:
            if "cart_id" not in params and state.cart.cart_id:
                params["cart_id"] = str(state.cart.cart_id)

        if action.tool_name == ToolName.PLACE_ORDER:
            params.setdefault("customer_name", state.delivery.customer_name)
            params.setdefault("phone", state.delivery.phone)
            params.setdefault("delivery_address", state.delivery.delivery_address)
            params.setdefault("city", state.delivery.city)
            params.setdefault("delivery_notes", state.delivery.delivery_notes)

        if action.tool_name == ToolName.GET_PRODUCT_DETAILS:
            product_id = params.get("product_id")
            reference = (
                params.get("product_reference")
                or params.get("display_index")
                or params.get("product_index")
                or params.get("index")
                or params.get("reference")
            )
            if product_id is None and reference is not None:
                product_id = self._resolve_displayed_product_id(state, reference)
            if product_id is None and state.displayed_products:
                product_id = state.displayed_products[0].product_id
            if product_id is None and state.selected_product_id:
                product_id = state.selected_product_id
            if product_id is not None:
                params["product_id"] = product_id
                state.remember_selected_product(product_id)

        if action.tool_name == ToolName.ADD_TO_CART:
            reference = params.get("product_reference") or params.get("display_index")
            if params.get("variant_id") is None and reference is not None:
                option = self._resolve_variant_from_latest_search(state, reference, params)
                if option is not None:
                    params.setdefault("variant_id", option.variant_id)
                    params.setdefault("branch_id", option.branch_id)
                    params.setdefault("selected_product_id", option.product_id)
            params.setdefault("quantity", 1)

    @staticmethod
    def _build_effective_product_search(state: ConversationState, turn_parameters: dict[str, Any]) -> dict[str, Any]:
        """Build one canonical search request from preferences, active search, and turn overrides.

        Precedence is explicit: current-turn values override the active search,
        active search values override persistent preferences, and empty optional
        values are ignored. This prevents conversational refinement from dropping
        earlier constraints such as an occasion or budget.
        """

        preference_map = {
            "colors": state.preferences.preferred_colors,
            "excluded_colors": state.preferences.excluded_colors,
            "categories": state.preferences.preferred_categories,
            "product_types": state.preferences.preferred_product_types,
            "occasions": state.preferences.preferred_occasions,
            "materials": state.preferences.preferred_materials,
            "fits": state.preferences.preferred_fits,
            "size_mapping": state.preferences.size_mapping,
            "minimum_price": state.preferences.minimum_price,
            "maximum_price": state.preferences.maximum_price,
            "branch_code": state.preferences.branch_preference,
        }
        active = state.current_search.model_dump(exclude_none=True)
        effective: dict[str, Any] = {}
        for key, value in preference_map.items():
            if value not in (None, [], {}, ""):
                effective[key] = value
        for key, value in active.items():
            if value not in (None, [], {}, ""):
                effective[key] = value
        for key, value in turn_parameters.items():
            if value not in (None, [], {}, ""):
                effective[key] = value
        effective.setdefault("in_stock_only", state.current_search.in_stock_only)
        effective["limit"] = min(int(effective.get("limit", state.current_search.limit)), 20)
        return effective

    @staticmethod
    def _resolve_displayed_product_id(state: ConversationState, reference: Any) -> int | None:
        """Resolve '1', '2', 'first', etc. against the latest displayed product set."""

        if isinstance(reference, str):
            ref_str = reference.lower().strip()
            word_map = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}
            if ref_str in word_map:
                reference = word_map[ref_str]
        try:
            index = int(reference)
        except (TypeError, ValueError):
            index = 1
        match = next((item for item in state.displayed_products if item.index == index), None)
        if match:
            return match.product_id
        if state.displayed_products:
            return state.displayed_products[0].product_id
        return None

    def _resolve_variant_from_latest_search(self, state: ConversationState, reference: Any, params: dict[str, Any]) -> Any | None:
        """Resolve a displayed product to one unambiguous available variant."""

        result = state.last_tool_results.get(ToolName.GET_PRODUCTS.value)
        if not isinstance(result, ProductSearchResponse):
            return None
        try:
            index = int(reference)
        except (TypeError, ValueError):
            return None
        displayed = next((item for item in state.displayed_products if item.index == index), None)
        if displayed is None:
            return None

        candidates = [item for item in result.products if item.product_id == displayed.product_id]
        if params.get("color"):
            candidates = [item for item in candidates if (item.color or "").lower() == str(params["color"]).lower()]
        if params.get("size"):
            candidates = [item for item in candidates if (item.size or "").lower() == str(params["size"]).lower()]
        available = [item for item in candidates if item.available_quantity > 0 and item.is_available is not False]
        if not available:
            return None

        preferred_branch = params.get("branch_code") or state.current_search.branch_code or state.preferences.branch_preference
        if preferred_branch:
            preferred = [item for item in available if (item.branch_code or "").lower() == str(preferred_branch).lower()]
            if preferred:
                available = preferred

        # Branch is an internal fulfillment dimension for ordinary shopping.
        # When multiple valid branches remain, choose deterministically without
        # asking the customer to select one. Prefer larger available quantity,
        # then stable branch code/id ordering.
        available.sort(key=lambda item: (
            -int(item.available_quantity),
            (item.branch_code or "").lower(),
            int(item.branch_id),
            int(item.variant_id),
        ))
        first = available[0]
        return first if all(item.variant_id == first.variant_id and item.size == first.size and item.color == first.color for item in available) else first

    async def _execute_until_waiting(self, state: ConversationState) -> None:
        """Execute ready actions, refresh deterministic state, and continue dependencies."""

        for _ in range(10):
            self._resolve_known_parameters(state)
            result = await self._execution.run_ready_actions(
                state,
                tool_executor=self._tools.execute,
                state_values=self._state_values(state),
                derived_values=self._derived_values(state),
            )

            for action_id in result.completed_action_ids:
                action = state.action_plan.get(action_id)
                if action:
                    logger.info("action.completed action=%s tool=%s", action_id, action.tool_name.value)
                    self._apply_completed_action(state, action)

            if result.waiting_action_id:
                logger.info("action.waiting action=%s missing=%s", result.waiting_action_id, result.missing_parameters)
                return
            if result.failed_action_ids:
                logger.warning("action.failed actions=%s", result.failed_action_ids)
                return
            if not result.executed_action_ids:
                return

            if not state.action_plan.ready_actions():
                return

    def _apply_completed_action(self, state: ConversationState, action: Any) -> None:
        """Apply normalized tool results to state without duplicating backend logic."""

        result = state.last_tool_results.get(action.tool_name.value)
        if result is None:
            return

        if action.tool_name == ToolName.GET_PRODUCTS and isinstance(result, ProductSearchResponse):
            references: list[DisplayedProductReference] = []
            seen_products: set[int] = set()
            index = 1
            for option in result.products:
                if option.product_id in seen_products:
                    continue
                seen_products.add(option.product_id)
                references.append(
                    DisplayedProductReference(
                        index=index,
                        product_id=option.product_id,
                        article_code=option.article_code,
                        product_name=option.product_name,
                    )
                )
                index += 1
            state.remember_displayed_products(references)
        elif action.tool_name in {ToolName.ADD_TO_CART, ToolName.UPDATE_CART, ToolName.REMOVE_FROM_CART, ToolName.CLEAR_CART}:
            state.last_tool_results["explicit_confirmation"] = None
            state.last_tool_results.pop(ToolName.PREVIEW_CHECKOUT.value, None)
            self._apply_cart_result(state, result)
        elif action.tool_name in {ToolName.CREATE_CART, ToolName.GET_CART}:
            self._apply_cart_result(state, result)

    @staticmethod
    def _apply_cart_result(state: ConversationState, result: Any) -> None:
        """Update local cart metadata from an authoritative cart response."""

        if getattr(result, "cart_id", None):
            state.cart.cart_id = result.cart_id
        if getattr(result, "item_count", None) is not None:
            state.cart.item_count = int(result.item_count or 0)
        if getattr(result, "subtotal", None) is not None:
            state.cart.subtotal = Decimal(str(result.subtotal or 0))

    @staticmethod
    def _state_values(state: ConversationState) -> dict[str, Any]:
        """Flatten deterministic state values for generic requirement checking."""

        return {
            "cart_id": str(state.cart.cart_id) if state.cart.cart_id else None,
            "customer_name": state.delivery.customer_name,
            "phone": state.delivery.phone,
            "delivery_address": state.delivery.delivery_address,
            "city": state.delivery.city,
            "delivery_notes": state.delivery.delivery_notes,
        }

    @staticmethod
    def _derived_values(state: ConversationState) -> dict[str, Any]:
        """Provide deterministic defaults that do not invent customer choices."""

        return {
            "quantity": 1,
            "explicit_confirmation": state.last_tool_results.get("explicit_confirmation"),
        }

    def _build_runtime_context(self, state: ConversationState) -> dict[str, Any]:
        """Build compact response context focused on the current execution cycle."""

        relevant: dict[str, Any] = {}
        completed_tools = []
        for action in reversed(state.action_plan.actions):
            if action.status != ActionStatus.COMPLETED:
                continue
            tool_key = action.tool_name.value
            if tool_key in completed_tools:
                continue
            if tool_key in state.last_tool_results:
                value = state.last_tool_results[tool_key]
                relevant[tool_key] = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
                completed_tools.append(tool_key)

        product_cards = []
        if state.displayed_products and ToolName.GET_PRODUCTS.value in state.last_tool_results:
            search_res = state.last_tool_results[ToolName.GET_PRODUCTS.value]
            if hasattr(search_res, "products") and search_res.products:
                seen_ids = set()
                for ref in state.displayed_products:
                    if ref.product_id in seen_ids:
                        continue
                    seen_ids.add(ref.product_id)
                    p_match = next((p for p in search_res.products if p.product_id == ref.product_id), None)
                    if p_match:
                        product_cards.append({
                            "product": {
                                "product_id": p_match.product_id,
                                "article_code": p_match.article_code,
                                "product_name": p_match.product_name,
                                "description": getattr(p_match, "description", None),
                                "category": getattr(p_match, "category", ""),
                                "subcategory": None,
                                "product_type": getattr(p_match, "product_type", ""),
                                "gender": getattr(p_match, "gender", "MEN"),
                                "brand": getattr(p_match, "brand", "Northstar"),
                                "material": getattr(p_match, "material", None),
                                "fit": getattr(p_match, "fit", None),
                                "season": getattr(p_match, "season", None),
                                "occasion": getattr(p_match, "occasion", None),
                                "base_price": float(getattr(p_match, "price", 0)),
                                "final_price": float(getattr(p_match, "price", 0)),
                                "discount_amount": 0,
                                "applied_offer": None,
                                "images": [p_match.image_url] if getattr(p_match, "image_url", None) else [],
                                "variants": [{
                                    "variant_id": p_match.variant_id,
                                    "sku": getattr(p_match, "sku", p_match.article_code),
                                    "color": getattr(p_match, "color", ""),
                                    "size": getattr(p_match, "size", ""),
                                    "price": float(getattr(p_match, "price", 0)),
                                    "final_price": float(getattr(p_match, "price", 0)),
                                    "discount_amount": 0,
                                    "applied_offer": None,
                                    "is_available": getattr(p_match, "available_quantity", 0) > 0,
                                    "branch_availability": []
                                }]
                            }
                        })

        return {
            "language": state.language.value if state.language else None,
            "preferences": state.preferences.model_dump(mode="json"),
            "current_search": state.current_search.model_dump(mode="json"),
            "displayed_products": [item.model_dump(mode="json") for item in state.displayed_products],
            "product_cards": product_cards,
            "selected_product_id": state.selected_product_id,
            "delivery": state.delivery.model_dump(mode="json"),
            "cart": state.cart.model_dump(mode="json"),
            "pending_action_id": state.pending_action_id,
            "pending_action": (
                state.action_plan.get(state.pending_action_id).model_dump(mode="json")
                if state.pending_action_id and state.action_plan.get(state.pending_action_id)
                else None
            ),
            "current_tool_results": relevant,
        }

    async def close(self) -> None:
        """Close the underlying commerce adapter transport when supported."""

        close = getattr(self._tools, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result
