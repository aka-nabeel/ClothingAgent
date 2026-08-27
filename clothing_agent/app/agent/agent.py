"""Fitzy's V1 runtime orchestrator."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_DNS

from .contracts import ToolName
from .execution import ActionExecutionCoordinator
from .intent import ProductReference, SearchOverrides, StructuredIntent, classify_language
from .intent_prompts import INTENT_EXTRACTION_SYSTEM_PROMPT
from .intents import IntentName
from .planner import ActionPlanner
from .response import ResponseGuard
from .state import (
    ActionStatus,
    ConversationState,
    DisplayedProductReference,
    LanguageMode,
    SearchContext,
)
from ..core.config import AgentConfig, get_config
from ..integration.client import CommerceToolAdapter
from ..integration.schemas import ProductOption, ProductSearchResponse
from ..llm.client import LLMClient
from .tool_tracing import CURRENT_TURN_TRACE
from .turn_contract import AgentTurnResponse, ContentType
from .response_builder import (
    build_product_list_response,
    build_product_details_response,
    build_cart_response,
    build_checkout_response,
    build_order_response,
)

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
        self._sessions: dict[str, ConversationState] = {}

    def _to_uuid(self, session_id: str | UUID) -> UUID:
        if isinstance(session_id, UUID):
            return session_id
        try:
            return UUID(str(session_id))
        except ValueError:
            return uuid5(NAMESPACE_DNS, str(session_id))

    def get_state(self, session_id: str | UUID) -> ConversationState:
        """Fetch or initialize the long-lived conversation state for a session."""

        key = str(session_id)
        if key not in self._sessions:
            self._sessions[key] = ConversationState(session_id=self._to_uuid(session_id))
        return self._sessions[key]

    def reset_state(
        self,
        session_id: str | UUID,
        *,
        keep_cart: bool = False,
        language: str | None = None,
    ) -> ConversationState:
        """Completely clear all conversation state, preferences, tool caches, and search filters."""

        key = str(session_id)
        state = ConversationState(session_id=self._to_uuid(session_id))
        if language:
            state.set_language(language)
        if keep_cart and key in self._sessions:
            existing_cart = self._sessions[key].cart
            state.cart = existing_cart
        self._sessions[key] = state
        return state

    async def process_message(
        self,
        session_id: str | None = None,
        message: str = "",
        language: str | None = None,
        state: ConversationState | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AgentTurnResponse | str:
        """Process one customer message through the full V1 runtime pipeline."""

        # Flexible parameter resolution for all caller signatures
        if state is not None:
            state_obj = state
            resolved_session_id = str(state_obj.session_id)
            resolved_message = message if message else (session_id if isinstance(session_id, str) else "")
        else:
            resolved_session_id = session_id or kwargs.get("session_id", "default_session")
            resolved_message = message or kwargs.get("message", "")
            state_obj = self.get_state(resolved_session_id)

        # Language Resolution Rule:
        if language:
            state_obj.set_language(language)
        else:
            det_language = classify_language(resolved_message, state_obj.language)
            state_obj.set_language(det_language)
        
        extraction = await self._extract_intent(resolved_message)

        extracted_intents_summary = [
            f"{i.value if hasattr(i, 'value') else str(i)}"
            for i in getattr(extraction, "intents", [])
        ]
        
        trace = CURRENT_TURN_TRACE.get()
        if trace:
            trace.intent(extracted_intents_summary)
        else:
            logger.info(
                "[--- INTENT EXTRACTED ---] session=%s | language=%s | intents=%s",
                resolved_session_id,
                state_obj.language.value if state_obj.language else "unknown",
                extracted_intents_summary,
            )

        waiting_actions = [action.model_copy(deep=True) for action in state_obj.action_plan.actions if action.status == ActionStatus.WAITING_FOR_INPUT]
        self._apply_intent_to_state(extraction, state_obj, message=resolved_message)
        self._reopen_waiting_actions_for_new_input(state_obj)
        plan = self._planner.build_plan(extraction, state_obj)
        self._merge_waiting_actions(plan, waiting_actions)
        state_obj.action_plan = plan
        
        actions_summary = [
            f"{a.action_id}:{a.tool_name.value}(params={a.parameters}, missing={a.missing_parameters}, status={a.status.value})"
            for a in plan.actions
        ]
        if trace:
            trace.event(
                "PLAN & STATE",
                plan_id=plan.plan_id[:8] if hasattr(plan, "plan_id") and plan.plan_id else None,
                active_search=state_obj.current_search.model_dump(exclude_none=True),
                delivery=state_obj.delivery.model_dump(exclude_none=True),
                cart_items=state_obj.cart.item_count,
                actions=actions_summary,
            )
        else:
            logger.info(
                "[--- PLAN & STATE ---] session=%s | plan_id=%s | active_search=%s | delivery=%s | cart_items=%d | actions=%s",
                resolved_session_id,
                plan.plan_id,
                state_obj.current_search.model_dump(exclude_none=True),
                state_obj.delivery.model_dump(exclude_none=True),
                state_obj.cart.item_count,
                actions_summary,
            )

        self._resolve_known_parameters(state_obj)
        await self._execute_until_waiting(state_obj, user_message=resolved_message)

        runtime_context = self._build_runtime_context(state_obj, user_message=resolved_message)
        reply = await self._responses.generate(
            language=state_obj.language or LanguageMode.ENGLISH,
            user_message=resolved_message,
            runtime_context=runtime_context,
        )

        lang_str = state_obj.language.value if state_obj.language else "english"

        # Construct authoritative AgentTurnResponse envelope based on current turn action priority
        executed_tools = [a.tool_name.value for a in state_obj.action_plan.actions if a.status == ActionStatus.COMPLETED]

        if ToolName.PLACE_ORDER.value in executed_tools:
            order_res = state_obj.last_tool_results[ToolName.PLACE_ORDER.value]
            return build_order_response(session_id=resolved_session_id, language=lang_str, reply=reply, result=order_res)

        if ToolName.PREVIEW_CHECKOUT.value in executed_tools:
            checkout_res = state_obj.last_tool_results[ToolName.PREVIEW_CHECKOUT.value]
            return build_checkout_response(session_id=resolved_session_id, language=lang_str, reply=reply, result=checkout_res)

        if any(k in executed_tools for k in (ToolName.ADD_TO_CART.value, ToolName.UPDATE_CART.value, ToolName.REMOVE_FROM_CART.value, ToolName.CLEAR_CART.value, ToolName.GET_CART.value)):
            cart_tool = next(k for k in (ToolName.ADD_TO_CART.value, ToolName.UPDATE_CART.value, ToolName.REMOVE_FROM_CART.value, ToolName.CLEAR_CART.value, ToolName.GET_CART.value) if k in executed_tools)
            cart_res = state_obj.last_tool_results[cart_tool]
            return build_cart_response(session_id=resolved_session_id, language=lang_str, reply=reply, result=cart_res)

        if ToolName.GET_PRODUCT_DETAILS.value in executed_tools:
            detail_res = state_obj.last_tool_results[ToolName.GET_PRODUCT_DETAILS.value]
            return build_product_details_response(session_id=resolved_session_id, language=lang_str, reply=reply, result=detail_res)

        if ToolName.GET_PRODUCTS.value in executed_tools and not self._is_broad_category_search(state_obj.current_search, user_message=resolved_message):
            search_res = state_obj.last_tool_results[ToolName.GET_PRODUCTS.value]
            return build_product_list_response(session_id=resolved_session_id, language=lang_str, reply=reply, result=search_res, context=context)

        return AgentTurnResponse(
            session_id=resolved_session_id,
            reply=reply,
            language=lang_str,
            content_type=ContentType.GENERAL,
        )

    async def _extract_intent(self, message: str) -> StructuredIntent:
        """Use the LLM for semantic intent extraction with heuristic fallback."""
        try:
            return await self._llm.generate_structured(
                system_prompt=INTENT_EXTRACTION_SYSTEM_PROMPT,
                user_message=message,
                response_model=StructuredIntent,
            )
        except Exception as exc:
            logger.warning("llm.extraction_failed error=%s falling_back_to_heuristic", exc)
            return self._heuristic_extract_intent(message)

    def _heuristic_extract_intent(self, message: str) -> StructuredIntent:
        msg = message.lower()
        if any(w in msg for w in ["kaun kaun", "kya kya", "what products", "all products", "categories", "range", "collection", "kya hai"]) and not any(w in msg for w in ["pant", "pants", "shirt", "shirts", "t-shirt", "tshirts", "trouser", "trousers"]):
            return StructuredIntent(intents=[IntentName.BRANCH_INFORMATION])
        elif any(w in msg for w in ["search", "find", "shirt", "pant", "trouser", "denim", "dress", "show", "buy", "oxford", "chino", "option", "want"]):
            cats = []
            types = []
            if "shirt" in msg or "oxford" in msg:
                cats.append("shirts")
                if "formal" in msg:
                    types.append("formal shirts")
                elif "casual" in msg:
                    types.append("casual shirts")
            elif "pant" in msg or "trouser" in msg or "chino" in msg:
                cats.append("pants")
                if "formal" in msg:
                    types.append("formal pants")
                elif "chino" in msg:
                    types.append("chinos")
            elif "t-shirt" in msg or "tshirt" in msg or "polo" in msg:
                cats.append("t-shirts")
            elif "kurta" in msg:
                cats.append("traditional")
            elif "jacket" in msg or "outerwear" in msg:
                cats.append("outerwear")
            
            overrides = {}
            if cats:
                overrides["categories"] = cats
            if types:
                overrides["product_types"] = types
            if not overrides:
                overrides["query_text"] = message
            return StructuredIntent(intents=[IntentName.PRODUCT_SEARCH], search_overrides=SearchOverrides.model_validate(overrides))
        elif any(w in msg for w in ["detail", "details", "tell me about", "about the", "more info", "first one", "second one", "this shirt", "this pant", "this product", "material", "fabric"]):
            idx = 1
            if "second" in msg or "2nd" in msg or "two" in msg:
                idx = 2
            elif "third" in msg or "3rd" in msg or "three" in msg:
                idx = 3
            return StructuredIntent(intents=[IntentName.PRODUCT_DETAILS], product_reference=ProductReference(index=idx, text_reference=message))
        elif any(w in msg for w in ["cart", "add"]):
            return StructuredIntent(intents=[IntentName.ADD_TO_CART], product_reference={"index": 1})
        elif any(w in msg for w in ["checkout", "order", "place"]):
            return StructuredIntent(intents=[IntentName.CHECKOUT_PREVIEW])
        else:
            return StructuredIntent(intents=[IntentName.GENERAL_CHAT])

    def _apply_intent_to_state(self, extraction: StructuredIntent | Any, state: ConversationState, message: str = "") -> None:
        """Persist turn facts into the correct long-lived or action-scoped state.

        Delivery information is accumulated across turns, explicit order
        confirmation is retained for the current execution cycle, and search
        filters update the current search without erasing unrelated preferences.
        """
        if isinstance(extraction, StructuredIntent):
            if extraction.delivery:
                del_dict = extraction.delivery.model_dump(exclude_none=True)
                self._apply_delivery_fields(del_dict, state, message=message)
            else:
                self._apply_delivery_fields({}, state, message=message)

            for intent_name in extraction.intents:
                if intent_name in {IntentName.GENERAL_CHAT, IntentName.BRANCH_INFORMATION} or extraction.change_topic or extraction.reset_shopping:
                    state.displayed_products = []
                    state.selected_product_id = None
                    state.current_search.clear()

                if intent_name in {
                    IntentName.ADD_TO_CART,
                    IntentName.UPDATE_CART,
                    IntentName.REMOVE_FROM_CART,
                    IntentName.CLEAR_CART,
                }:
                    state.last_tool_results["explicit_confirmation"] = None
                    state.last_tool_results.pop(ToolName.PREVIEW_CHECKOUT.value, None)

                if intent_name == IntentName.PLACE_ORDER and extraction.explicit_confirmation is not None:
                    if ToolName.PREVIEW_CHECKOUT.value in state.last_tool_results:
                        state.last_tool_results["explicit_confirmation"] = extraction.explicit_confirmation
                    else:
                        state.last_tool_results["explicit_confirmation"] = None

                if intent_name == IntentName.PRODUCT_SEARCH:
                    overrides = extraction.search_overrides
                    params = overrides.model_dump(exclude_none=True)

                    specific_filters = (
                        overrides.colors
                        or overrides.product_types
                        or overrides.occasions
                        or overrides.minimum_price
                        or overrides.maximum_price
                        or overrides.article_code
                        or overrides.sku
                    )
                    vague_query_words = {"casual", "formal", "party", "something", "clothes", "wear", "items", "stuff", "options", "menswear"}
                    cats = overrides.categories or []
                    query_str = " ".join(cats).lower().strip()
                    is_vague_query_text = query_str in vague_query_words or query_str.startswith("i want ") or query_str.startswith("show me ")

                    if not specific_filters or is_vague_query_text or extraction.change_topic:
                        state.displayed_products = []
                        state.selected_product_id = None
                    state.current_search.update_from_mapping({
                        key: value
                        for key, value in params.items()
                        if key in type(state.current_search).model_fields
                    })
        elif hasattr(extraction, "intents"):
            for intent in getattr(extraction, "intents"):
                params = getattr(intent, "parameters", {})
                self._apply_delivery_fields(params, state, message=message)

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
    def _apply_delivery_fields(params: dict[str, Any], state: ConversationState, message: str = "") -> None:
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
            if value not in (None, ""):
                setattr(state.delivery, field_name, value)

        if message:
            import re
            m_lower = message.lower()
            
            # Extract Phone
            phone_match = re.search(r'\b(03\d{9}|\+?923\d{9})\b', message)
            if phone_match:
                state.delivery.phone = phone_match.group(1)
            
            # Extract Name
            name_match = re.search(r'(?:my name is|i am|name:?)\s+([A-Za-z\s]+)', message, re.IGNORECASE)
            if name_match:
                state.delivery.customer_name = name_match.group(1).strip(". ")
            
            # Extract City / Address
            for c in ("lahore", "islamabad", "karachi", "rawalpindi", "faisalabad", "multan", "peshawar", "quetta"):
                if c in m_lower:
                    state.delivery.city = c.title()
                    if not state.delivery.delivery_address:
                        state.delivery.delivery_address = message.strip()
                    break

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

        if action.tool_name == ToolName.GET_PRODUCTS:
            action.parameters = self._build_effective_product_search(state, action.parameters)

        params = action.parameters

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
            idx = 1
            if isinstance(reference, ProductReference):
                idx = reference.index or 1
            elif isinstance(reference, int):
                idx = reference
            elif isinstance(reference, str) and reference.isdigit():
                idx = int(reference)
            
            if product_id is None and state.displayed_products:
                if 1 <= idx <= len(state.displayed_products):
                    product_id = state.displayed_products[idx - 1].product_id
                else:
                    product_id = state.displayed_products[0].product_id
            if product_id is None and state.selected_product_id:
                product_id = state.selected_product_id
            if product_id is not None:
                params["product_id"] = int(product_id)
                state.remember_selected_product(int(product_id))

        if action.tool_name == ToolName.ADD_TO_CART:
            reference = params.get("product_reference") or params.get("display_index") or params.get("index") or 1
            if isinstance(reference, ProductReference):
                ref_val = reference.index or reference.text_reference or 1
            else:
                ref_val = reference

            # Prioritize GET_PRODUCT_DETAILS if available to get precise variant_id & branch_id
            if ToolName.GET_PRODUCT_DETAILS.value in state.last_tool_results:
                det = state.last_tool_results[ToolName.GET_PRODUCT_DETAILS.value]
                opts = getattr(det, "options", None) or (det.get("options") if isinstance(det, dict) else []) or getattr(det, "variants", None) or (det.get("variants") if isinstance(det, dict) else [])
                abundant_opts = []
                in_stock_opts = []
                for opt in opts:
                    qty = getattr(opt, "available_quantity", 0) if not isinstance(opt, dict) else opt.get("available_quantity", 0)
                    avail = getattr(opt, "is_available", True) if not isinstance(opt, dict) else opt.get("is_available", True)
                    if avail and qty >= 2:
                        abundant_opts.append(opt)
                    elif avail and qty > 0:
                        in_stock_opts.append(opt)
                target_opts = abundant_opts if abundant_opts else (in_stock_opts if in_stock_opts else opts)
                for opt in target_opts:
                    vid = getattr(opt, "variant_id", None) or (opt.get("variant_id") if isinstance(opt, dict) else None)
                    bid = getattr(opt, "branch_id", None) or (opt.get("branch_id") if isinstance(opt, dict) else None)
                    pid = getattr(opt, "product_id", None) or (opt.get("product_id") if isinstance(opt, dict) else None)
                    if vid and bid:
                        params["variant_id"] = vid
                        params["branch_id"] = bid
                        if pid:
                            params["selected_product_id"] = pid
                        break

            if params.get("variant_id") is None:
                option = self._resolve_variant_from_latest_search(state, ref_val, params)
                if option is not None:
                    vid = getattr(option, "variant_id", None) or (option.get("variant_id") if isinstance(option, dict) else None)
                    bid = getattr(option, "branch_id", None) or (option.get("branch_id") if isinstance(option, dict) else None)
                    pid = getattr(option, "product_id", None) or (option.get("product_id") if isinstance(option, dict) else None)
                    if vid:
                        params["variant_id"] = vid
                    if bid:
                        params["branch_id"] = bid
                    if pid:
                        params["selected_product_id"] = pid

            if not params.get("branch_id") or int(params.get("branch_id", 0)) <= 0:
                params["branch_id"] = 55

            params.setdefault("quantity", 1)

        if action.tool_name in (ToolName.UPDATE_CART, ToolName.REMOVE_FROM_CART):
            if getattr(state, "cart", None) and getattr(state.cart, "cart_id", None):
                params.setdefault("cart_id", str(state.cart.cart_id))
            if not params.get("item_id"):
                cart_res = (
                    state.last_tool_results.get(ToolName.ADD_TO_CART.value)
                    or state.last_tool_results.get(ToolName.GET_CART.value)
                    or state.last_tool_results.get(ToolName.CREATE_CART.value)
                )
                items = []
                if isinstance(cart_res, dict):
                    items = cart_res.get("items", [])
                elif hasattr(cart_res, "items"):
                    items = getattr(cart_res, "items", [])
                if items:
                    idx = 0
                    ref = params.get("product_reference") or params.get("item_reference") or 1
                    if isinstance(ref, int) and 1 <= ref <= len(items):
                        idx = ref - 1
                    item = items[idx]
                    iid = item.get("item_id") if isinstance(item, dict) else getattr(item, "item_id", None)
                    if iid:
                        params["item_id"] = str(iid)

            if action.tool_name == ToolName.UPDATE_CART and not params.get("quantity"):
                qty = params.get("target_quantity") or params.get("new_quantity") or 2
                params["quantity"] = int(qty)

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
        if result is None:
            return None

        raw_products = getattr(result, "products", None) or (result.get("products") if isinstance(result, dict) else [])
        if not raw_products:
            return None

        if isinstance(reference, ProductReference):
            index = reference.index or 1
        else:
            try:
                index = int(reference)
            except (TypeError, ValueError):
                index = 1

        displayed = next((item for item in state.displayed_products if item.index == index), None)
        if displayed is None:
            return None

        candidates = []
        for p in raw_products:
            pid = getattr(p, "product_id", None) or (p.get("product_id") if isinstance(p, dict) else None)
            if pid == displayed.product_id:
                variants = getattr(p, "variants", None) or (p.get("variants") if isinstance(p, dict) else None)
                if variants:
                    for v in variants:
                        vid = getattr(v, "variant_id", None) or (v.get("variant_id") if isinstance(v, dict) else None)
                        bid = getattr(v, "branch_id", 1) or (v.get("branch_id", 1) if isinstance(v, dict) else 1)
                        color = getattr(v, "color", "") or (v.get("color", "") if isinstance(v, dict) else "")
                        size = getattr(v, "size", "") or (v.get("size", "") if isinstance(v, dict) else "")
                        qty = getattr(v, "available_quantity", 1) if not isinstance(v, dict) else v.get("available_quantity", 1)
                        avail = getattr(v, "is_available", True) if not isinstance(v, dict) else v.get("is_available", True)

                        b_avails = getattr(v, "branch_availability", None) or (v.get("branch_availability") if isinstance(v, dict) else [])
                        if b_avails:
                            for ba in b_avails:
                                b_id = getattr(ba, "branch_id", bid) or (ba.get("branch_id", bid) if isinstance(ba, dict) else bid)
                                b_code = getattr(ba, "branch_code", "") or (ba.get("branch_code", "") if isinstance(ba, dict) else "")
                                b_qty = getattr(ba, "available_quantity", qty) or (ba.get("available_quantity", qty) if isinstance(ba, dict) else qty)
                                if b_qty > 0:
                                    candidates.append(ProductOption(
                                        product_id=pid, variant_id=vid, branch_id=b_id, branch_code=b_code,
                                        article_code=displayed.article_code or "", product_name=displayed.product_name,
                                        color=color, size=size, available_quantity=b_qty, is_available=True
                                    ))
                        else:
                            if avail and qty > 0:
                                candidates.append(ProductOption(
                                    product_id=pid, variant_id=vid, branch_id=bid,
                                    article_code=displayed.article_code or "", product_name=displayed.product_name,
                                    color=color, size=size, available_quantity=qty, is_available=True
                                ))
                else:
                    vid = getattr(p, "variant_id", None) or (p.get("variant_id") if isinstance(p, dict) else None)
                    bid = getattr(p, "branch_id", 1) or (p.get("branch_id", 1) if isinstance(p, dict) else 1)
                    if vid:
                        if isinstance(p, ProductOption):
                            candidates.append(p)
                        elif isinstance(p, dict):
                            candidates.append(ProductOption.model_validate(p))

        if not candidates:
            return None

        if params.get("color"):
            filtered = [item for item in candidates if (item.color or "").lower() == str(params["color"]).lower()]
            if filtered:
                candidates = filtered
        if params.get("size"):
            filtered = [item for item in candidates if (item.size or "").lower() == str(params["size"]).lower()]
            if filtered:
                candidates = filtered

        return candidates[0]

    async def _execute_until_waiting(self, state: ConversationState, user_message: str = "") -> None:
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
                    self._apply_completed_action(state, action, user_message=user_message)

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

    @staticmethod
    def _is_broad_category_search(search: SearchContext, user_message: str = "") -> bool:
        """Deterministic Business Rule: Return True only if search is a vague query without category/product_type/filters."""

        msg_lower = user_message.lower().strip()
        specific_filters = (
            search.colors
            or search.product_types
            or search.occasions
            or search.minimum_price
            or search.maximum_price
        )
        if specific_filters:
            return False

        vague_terms = {
            "casual", "formal", "party", "something", "clothes", "clothing", "wear",
            "items", "stuff", "menswear", "collection", "categories", "catalog"
        }

        cats = [c.lower().strip() for c in (search.categories or [])]

        if any(c in vague_terms for c in cats):
            return True

        clothing_categories = {"shirt", "shirts", "t-shirt", "t-shirts", "tshirt", "tshirts", "pant", "pants", "trouser", "trousers", "outerwear", "traditional", "chinos", "jeans"}
        if any(c in clothing_categories for c in cats):
            return False

        intent_keywords = {"show", "display", "option", "options", "bring", "want", "have", "some", "see", "look", "looking"}
        if any(w in msg_lower for w in intent_keywords):
            return False

        query_str = str(search.query_text or "").lower().strip()
        if query_str in vague_terms or (not query_str and not cats):
            return True

        return False

    def _apply_completed_action(self, state: ConversationState, action: Any, user_message: str = "") -> None:
        """Apply normalized tool results to state without duplicating backend logic."""

        result = state.last_tool_results.get(action.tool_name.value)
        if result is None:
            return

        if action.tool_name == ToolName.GET_PRODUCTS:
            prods = getattr(result, "products", None) or (result.get("products") if isinstance(result, dict) else [])
            references: list[DisplayedProductReference] = []
            seen_products: set[int] = set()
            index = 1
            for option in prods:
                pid = getattr(option, "product_id", None) or (option.get("product_id") if isinstance(option, dict) else None)
                code = getattr(option, "article_code", None) or (option.get("article_code") if isinstance(option, dict) else None)
                name = getattr(option, "product_name", None) or (option.get("product_name") if isinstance(option, dict) else None)
                if not pid or pid in seen_products:
                    continue
                seen_products.add(pid)
                references.append(
                    DisplayedProductReference(
                        index=index,
                        product_id=int(pid),
                        article_code=code,
                        product_name=name or "",
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

    def _build_runtime_context(self, state: ConversationState, user_message: str = "") -> dict[str, Any]:
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
        if self._is_broad_category_search(state.current_search, user_message=user_message):
            state.displayed_products = []

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
