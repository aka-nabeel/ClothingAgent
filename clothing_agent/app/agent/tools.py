"""Agent-facing tool contracts for semantic interactions with the backend."""

import logging
from typing import Optional

from .state import ConversationState
from ..clients.clothing_app.client import ClothingAppClient
from ..clients.clothing_app.schemas import (
    ProductSearchRequest,
    ProductSearchResponse,
    ProductDetails,
    AddCartItemRequest,
    CartView,
    PreviewCartRequest,
    StoreOrderPreview,
    PlaceOrderRequest,
    OrderView,
)

logger = logging.getLogger(__name__)


class AgentTools:
    """DEPRECATED legacy tool wrapper.

    Canonical tool execution in Fitzy is performed via ``CommerceToolAdapter``
    located in ``clothing_agent.app.integration.tools.CommerceToolAdapter``.
    This class is retained for backward compatibility with early unit tests.
    """

    def __init__(self, client: ClothingAppClient) -> None:
        self._client = client

    async def get_products(
        self,
        state: ConversationState,
        search_query: Optional[str] = None
    ) -> ProductSearchResponse:
        """Search products based on the current conversational state and query."""
        
        # Merge persistent preferences with temporary overrides
        categories = state.current_search.get("categories", state.categories)
        product_types = state.current_search.get("product_types", state.product_types)
        occasions = state.current_search.get("occasions", state.occasions)
        preferred_colors = state.current_search.get("preferred_colors", state.preferred_colors)
        excluded_colors = state.current_search.get("excluded_colors", state.excluded_colors)
        materials = state.current_search.get("materials", state.materials)
        fits = state.current_search.get("fits", state.fits)
        branch_preference = state.current_search.get("branch_preference", state.branch_preference)
        
        budget_min = state.budget.minimum
        budget_max = state.budget.maximum
        if "budget" in state.current_search:
            temp_budget = state.current_search["budget"]
            budget_min = temp_budget.get("minimum", budget_min)
            budget_max = temp_budget.get("maximum", budget_max)
            
        size_preferences = dict(state.size_preferences)
        if "size_preferences" in state.current_search:
            size_preferences = dict(state.current_search["size_preferences"])
            
        limit = 4 if len(categories) <= 1 else 12  # Fetch more if multiple categories
        
        # Ensure we never request more than 20 items per the prompt constraints
        limit = min(limit, 20)

        request = ProductSearchRequest(
            query_text=search_query,
            categories=categories,
            product_types=product_types,
            occasions=occasions,
            colors=preferred_colors,
            excluded_colors=excluded_colors,
            size_mapping=size_preferences,
            materials=materials,
            fits=fits,
            minimum_price=budget_min,
            maximum_price=budget_max,
            branch_code=branch_preference,
            in_stock_only=True,
            limit=limit,
            article_code=state.current_search.get("specific_article"),
        )

        logger.info(
            "agent_tool_get_products",
            extra={"event": "agent_tool_get_products", "filters": request.model_dump(exclude_defaults=True)},
        )

        response = await self._client.search_products(request)
        
        # Keep track of what we displayed
        state.record_displayed_products(response.products)
        return response

    async def get_product_details(self, product_id: int) -> ProductDetails:
        """Retrieve complete product details for an exact product."""
        
        logger.info(
            "agent_tool_get_product_details",
            extra={"event": "agent_tool_get_product_details", "product_id": product_id},
        )
        return await self._client.get_product(product_id)
        
    async def get_product_details_by_index(
        self, 
        index: int, 
        state: ConversationState
    ) -> Optional[ProductDetails]:
        """Retrieve complete product details based on conversational index."""
        if index < 1 or index > len(state.displayed_products):
            logger.warning("agent_tool_invalid_product_index", extra={"event": "invalid_product_index", "index": index})
            return None
            
        product_id = state.displayed_products[index - 1].product_id
        return await self.get_product_details(product_id)

    async def _ensure_cart(self, state: ConversationState) -> None:
        """Create a cart for the session if it doesn't exist."""
        if not state.cart.cart_id:
            cart = await self._client.create_cart(session_id=state.session_id)
            state.cart.cart_id = cart.cart_id
            state.cart.item_count = cart.total_quantity
            state.cart.subtotal = float(cart.subtotal)

    async def add_cart_item(self, state: ConversationState, variant_id: int, branch_id: int, quantity: int = 1) -> CartView:
        await self._ensure_cart(state)
        req = AddCartItemRequest(variant_id=variant_id, branch_id=branch_id, quantity=quantity)
        cart = await self._client.add_cart_item(state.cart.cart_id, req)
        state.cart.item_count = cart.total_quantity
        state.cart.subtotal = float(cart.subtotal)
        return cart

    async def remove_cart_item(self, state: ConversationState, item_id: str) -> CartView | None:
        if not state.cart.cart_id:
            return None
        from uuid import UUID
        cart = await self._client.remove_cart_item(state.cart.cart_id, UUID(item_id))
        state.cart.item_count = cart.total_quantity
        state.cart.subtotal = float(cart.subtotal)
        return cart

    async def preview_checkout(self, state: ConversationState, offer_code: Optional[str] = None) -> StoreOrderPreview | None:
        if not state.cart.cart_id:
            return None
        req = PreviewCartRequest(offer_code=offer_code)
        return await self._client.preview_cart(state.cart.cart_id, req)

    async def place_order(
        self, 
        state: ConversationState, 
        customer_name: str,
        phone: str,
        delivery_address: str,
        city: str,
        delivery_notes: Optional[str] = None,
        offer_code: Optional[str] = None
    ) -> OrderView | None:
        if not state.cart.cart_id:
            return None
            
        req = PlaceOrderRequest(
            cart_id=state.cart.cart_id,
            checkout_request_id=str(__import__("uuid").uuid4()),
            offer_code=offer_code,
            customer_name=customer_name,
            phone=phone,
            delivery_address=delivery_address,
            city=city,
            delivery_notes=delivery_notes
        )
        order = await self._client.place_order(req)
        # Clear cart state after successful order
        state.cart.cart_id = None
        state.cart.item_count = 0
        state.cart.subtotal = 0.0
        return order
