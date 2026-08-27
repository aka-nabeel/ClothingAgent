"""Semantic client adapter for the existing prototype commerce APIs.

The Agent never calls raw HTTP endpoints directly. This class maps the stable
semantic tool contracts to the concrete Northstar prototype API. If another
brand exposes different routes later, only this adapter/integration map needs
to change; Agent intent and planning logic should remain untouched.
"""

import logging

logger = logging.getLogger(__name__)

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from ..agent.contracts import ToolName
from .errors import CommerceValidationError
from .http import AsyncJSONTransport
from .schemas import (
    AvailabilityView,
    BranchView,
    CartView,
    CheckoutPreview,
    OrderView,
    ProductDetails,
    ProductOption,
    ProductSearchRequest,
    ProductSearchResponse,
)


class CommerceAPIClient:
    """Expose semantic commerce operations over the existing HTTP API."""

    def __init__(self, transport: AsyncJSONTransport, api_map: Mapping[str, Mapping[str, Any]]) -> None:
        self._transport = transport
        self._api_map = api_map

    async def get_products(self, request: ProductSearchRequest) -> ProductSearchResponse:
        """Search existing product APIs using the semantic request contract.

        The current prototype supports one ``category`` and a flat ``sizes``
        list. When the semantic request contains multiple categories, the
        adapter fans out deterministic API requests and merges the results.
        This keeps that incompatibility inside the adapter instead of leaking
        prototype limitations into the Agent planner.
        """

        categories = request.categories or [None]
        if len(categories) == 1:
            return await self._search_once(request, categories[0])

        # The prototype endpoint supports a single category. Execute one
        # request per requested category and merge deterministically.
        responses = [await self._search_once(request, category) for category in categories]
        merged: list[ProductOption] = []
        max_len = max((len(response.products) for response in responses), default=0)
        for index in range(max_len):
            for response in responses:
                if index < len(response.products):
                    merged.append(response.products[index])
                    if len(merged) >= request.limit:
                        break
            if len(merged) >= request.limit:
                break

        return ProductSearchResponse(
            products=merged,
            result_count=len(merged),
            relaxed_constraints=[
                "categories were fanned out because the existing prototype search endpoint accepts one category"
            ],
        )

    async def _search_once(self, request: ProductSearchRequest, category: str | None) -> ProductSearchResponse:
        """Map one semantic request to the current prototype search payload."""

        semantic_tags = [*request.product_types, *request.occasions]
        sizes = list(request.size_mapping.values())
        payload = {
            "query_text": request.query_text,
            "category": category,
            "colors": request.colors,
            "excluded_colors": request.excluded_colors,
            "sizes": sizes,
            "minimum_price": float(request.minimum_price) if request.minimum_price is not None else None,
            "maximum_price": float(request.maximum_price) if request.maximum_price is not None else None,
            "branch_code": request.branch_code,
            "materials": [],
            "fits": [],
            "semantic_tags": semantic_tags,
            "in_stock_only": request.in_stock_only,
            "allow_relaxation": True,
            "limit": min(request.limit, 20),
        }

        result = await self._transport.request("POST", "/api/v1/products/search", json=payload)
        return ProductSearchResponse.model_validate(result)

    async def get_product_details(self, product_id: int) -> ProductDetails:
        """Retrieve authoritative details for one product."""

        if product_id <= 0:
            raise CommerceValidationError("product_id must be greater than zero")
        result = await self._transport.request("GET", f"/api/v1/products/{product_id}")
        if isinstance(result, dict):
            p = result.get("product") if isinstance(result.get("product"), dict) else result
            pid = p.get("product_id", product_id)
            code = p.get("article_code", "")
            name = p.get("product_name", "")
            opts = []
            for opt in p.get("variants", []):
                if isinstance(opt, dict):
                    vid = opt.get("variant_id")
                    color = opt.get("color", "")
                    size = opt.get("size", "")
                    price_str = str(opt.get("final_price", opt.get("price", 0)))
                    base_price_str = str(opt.get("price", 0))
                    disc_str = str(opt.get("discount_amount", 0))
                    b_avails = opt.get("branch_availability") or []
                    if b_avails and isinstance(b_avails, list):
                        for ba in b_avails:
                            if isinstance(ba, dict):
                                bid = ba.get("branch_id")
                                bcode = ba.get("branch_code", "")
                                bname = ba.get("branch_name", "")
                                bqty = ba.get("available_quantity", 0)
                                is_avail = ba.get("is_available", True) and bqty > 0
                                opts.append(ProductOption(
                                    product_id=pid,
                                    variant_id=vid,
                                    branch_id=bid,
                                    branch_code=bcode,
                                    branch_name=bname,
                                    article_code=code,
                                    product_name=name,
                                    color=color,
                                    size=size,
                                    price=price_str,
                                    base_price=base_price_str,
                                    discount_amount=disc_str,
                                    available_quantity=bqty,
                                    is_available=is_avail,
                                ))
                    else:
                        opt_dict = dict(opt)
                        opt_dict.setdefault("product_id", pid)
                        opt_dict.setdefault("article_code", code)
                        opt_dict.setdefault("product_name", name)
                        opts.append(ProductOption.model_validate(opt_dict))
            return ProductDetails(
                product_id=pid,
                article_code=code,
                product_name=name,
                category=p.get("category"),
                description=p.get("description"),
                attributes=p.get("attributes", {}),
                image_urls=[img if isinstance(img, str) else img.get("image_url", "") for img in p.get("images", []) if img],
                options=opts
            )
        return ProductDetails.model_validate(result)

    async def get_branches(self) -> list[BranchView]:
        """Retrieve all active branches exposed by the existing application."""

        result = await self._transport.request("GET", "/api/v1/branches")
        return [BranchView.model_validate(item) for item in result]

    async def check_availability(self, *, variant_id: int, branch_id: int) -> AvailabilityView:
        """Check exact variant availability at one branch."""

        if variant_id <= 0 or branch_id <= 0:
            raise CommerceValidationError("variant_id and branch_id must be greater than zero")
        result = await self._transport.request(
            "GET",
            "/api/v1/inventory/availability",
            params={"variant_id": variant_id, "branch_id": branch_id},
        )
        return AvailabilityView.model_validate(result)

    async def create_cart(self, *, session_id: UUID | str | None = None) -> CartView:
        """Create a persistent cart using the existing cart endpoint.

        The prototype cart contract is intentionally isolated here because its
        exact request payload is application-specific. The Agent only needs a
        normalized CartView and should not know the transport details.
        """

        payload: dict[str, Any] = {}
        if session_id is not None:
            payload["session_id"] = str(session_id)
        path = "/api/v1/carts"
        result = await self._transport.request("POST", path, json=payload)
        return CartView.model_validate(result)

    async def get_cart(self, cart_id: UUID | str) -> CartView:
        """Retrieve one persistent cart."""

        result = await self._transport.request("GET", f"/api/v1/carts/{cart_id}")
        return CartView.model_validate(result)

    async def add_to_cart(
        self,
        *,
        cart_id: UUID | str,
        variant_id: int,
        branch_id: int,
        quantity: int,
    ) -> CartView:
        """Add a sellable variant to the existing cart."""

        if variant_id <= 0 or branch_id <= 0 or quantity <= 0:
            raise CommerceValidationError("variant_id, branch_id and quantity must be positive")
        payload = {"variant_id": variant_id, "branch_id": branch_id, "quantity": quantity}
        result = await self._transport.request("POST", f"/api/v1/carts/{cart_id}/items", json=payload)
        return CartView.model_validate(result)

    async def update_cart(self, *, cart_id: UUID | str, item_id: UUID | str, quantity: int) -> CartView:
        """Update the quantity of an existing cart item."""

        if quantity <= 0:
            raise CommerceValidationError("quantity must be positive")
        result = await self._transport.request(
            "PATCH",
            f"/api/v1/carts/{cart_id}/items/{item_id}",
            json={"quantity": quantity},
        )
        return CartView.model_validate(result)

    async def remove_from_cart(self, *, cart_id: UUID | str, item_id: UUID | str) -> CartView:
        """Remove one item from the existing cart."""

        result = await self._transport.request("DELETE", f"/api/v1/carts/{cart_id}/items/{item_id}")
        return CartView.model_validate(result)

    async def clear_cart(self, *, cart_id: UUID | str) -> CartView:
        """Clear all items from the existing cart."""

        result = await self._transport.request("DELETE", f"/api/v1/carts/{cart_id}/items")
        return CartView.model_validate(result)

    async def preview_checkout(self, *, cart_id: UUID | str) -> CheckoutPreview:
        """Request authoritative checkout totals without mutating the cart."""

        result = await self._transport.request("POST", f"/api/v1/carts/{cart_id}/preview", json={})
        return CheckoutPreview.model_validate(result)

    async def place_order(self, payload: Mapping[str, Any]) -> OrderView:
        """Submit the final order through the existing order endpoint.

        The adapter intentionally forwards only the semantic delivery/order
        payload. It does not calculate totals or bypass the backend's order
        validation.
        """

        required = ("cart_id", "customer_name", "phone", "delivery_address", "city")
        missing = [field for field in required if payload.get(field) in (None, "")]
        if missing:
            raise CommerceValidationError(f"Missing order fields: {', '.join(missing)}")

        result = await self._transport.request("POST", "/api/v1/orders", json=dict(payload))
        return OrderView.model_validate(result)


class CommerceToolAdapter:
    """Dispatch semantic tool calls to the concrete commerce API client."""

    def __init__(self, client: CommerceAPIClient) -> None:
        self._client = client

    async def execute(self, tool_name: ToolName, parameters: Mapping[str, Any]) -> Any:
        """Execute one semantic tool against the mapped application API."""

        if tool_name == ToolName.GET_PRODUCTS:
            return await self._client.get_products(ProductSearchRequest.model_validate(parameters))
        if tool_name == ToolName.GET_PRODUCT_DETAILS:
            return await self._client.get_product_details(int(parameters["product_id"]))
        if tool_name == ToolName.GET_BRANCHES:
            return await self._client.get_branches()
        if tool_name == ToolName.CHECK_AVAILABILITY:
            return await self._client.check_availability(
                variant_id=int(parameters["variant_id"]),
                branch_id=int(parameters["branch_id"]),
            )
        if tool_name == ToolName.CREATE_CART:
            return await self._client.create_cart(session_id=parameters.get("session_id"))
        if tool_name == ToolName.GET_CART:
            return await self._client.get_cart(parameters["cart_id"])
        if tool_name == ToolName.ADD_TO_CART:
            logger.info("[--- EXECUTE ADD_TO_CART ---] params=%s", dict(parameters))
            return await self._client.add_to_cart(
                cart_id=parameters["cart_id"],
                variant_id=int(parameters["variant_id"]),
                branch_id=int(parameters["branch_id"]),
                quantity=int(parameters["quantity"]),
            )
        if tool_name == ToolName.UPDATE_CART:
            return await self._client.update_cart(
                cart_id=parameters["cart_id"],
                item_id=parameters["item_id"],
                quantity=int(parameters["quantity"]),
            )
        if tool_name == ToolName.REMOVE_FROM_CART:
            return await self._client.remove_from_cart(cart_id=parameters["cart_id"], item_id=parameters["item_id"])
        if tool_name == ToolName.CLEAR_CART:
            return await self._client.clear_cart(cart_id=parameters["cart_id"])
        if tool_name == ToolName.PREVIEW_CHECKOUT:
            return await self._client.preview_checkout(cart_id=parameters["cart_id"])
        if tool_name == ToolName.PLACE_ORDER:
            return await self._client.place_order(parameters)
        if tool_name == ToolName.GET_STORE_CONTEXT:
            return {
                "store_name": "Northstar Menswear",
                "categories": ["Shirts", "T-Shirts", "Pants & Trousers", "Traditional", "Outerwear"],
                "status": "available",
            }
        raise CommerceValidationError(f"Unsupported tool: {tool_name}")
