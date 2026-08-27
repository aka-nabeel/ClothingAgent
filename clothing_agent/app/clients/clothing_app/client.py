"""HTTP client for the clothing application's existing APIs."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, TypeVar
from uuid import UUID

import httpx
from .errors import ClothingAppUnavailableError
from .schemas import (
    AddCartItemRequest,
    CreateCartRequest,
    AvailabilityView,
    BranchView,
    CartView,
    ProductDetails,
    ProductSearchRequest,
    ProductSearchResponse,
    StoreContext,
    UpdateCartItemRequest,
    PreviewCartRequest,
    StoreOrderPreview,
    PlaceOrderRequest,
    OrderView,
)
from ...core.config import AgentConfig
from ...core.errors import AgentError
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ClothingAppClient:
    """DEPRECATED legacy HTTP client.

    Canonical HTTP transport in Fitzy is performed via ``CommerceAPIClient``
    located in ``clothing_agent.app.integration.client.CommerceAPIClient``.

    The agent never imports clothing-application repositories, SQLAlchemy
    models, or database sessions. This client is the sole integration boundary.
    """

    def __init__(self, config: AgentConfig, http: httpx.AsyncClient) -> None:
        self._config = config
        self._http = http
        self._base_url = config.clothing_app_base_url.rstrip("/")

    async def health(self) -> dict[str, Any]:
        """Return the clothing application's readiness response."""

        return await self._request("GET", "/health/ready", response_model=None)

    async def search_products(
        self,
        request: ProductSearchRequest,
    ) -> ProductSearchResponse:
        """Search products through the deployed-app retrieval contract."""

        return await self._request(
            "POST",
            "/api/v1/products/search",
            json=request.model_dump(mode="json"),
            response_model=ProductSearchResponse,
        )

    async def get_product(self, product_id: int) -> ProductDetails:
        """Retrieve complete product details."""

        return await self._request(
            "GET",
            f"/api/v1/products/{product_id}",
            response_model=ProductDetails,
        )

    async def list_branches(self) -> list[BranchView]:
        """Retrieve active branches."""

        data = await self._request("GET", "/api/v1/branches", response_model=None)
        return [BranchView.model_validate(item) for item in data]

    async def get_store_context(self) -> StoreContext:
        """Return general capabilities and structure of the store for the Agent."""
        return await self._request("GET", "/api/v1/store/context", response_model=StoreContext)

    async def get_menu(self) -> dict:
        """Retrieve the catalog menu."""
        return await self._request("GET", "/api/v1/menu", response_model=None)

    async def get_availability(
        self,
        variant_id: int,
        branch_id: int,
    ) -> AvailabilityView:
        """Retrieve live availability for an exact selection."""

        return await self._request(
            "GET",
            "/api/v1/inventory/availability",
            params={"variant_id": variant_id, "branch_id": branch_id},
            response_model=AvailabilityView,
        )

    async def create_cart(self, session_id: str | None = None, store_id: str | None = None) -> CartView:
        """Create or reuse the application cart for a conversation session."""
        body = CreateCartRequest(session_id=session_id, store_id=store_id)
        return await self._request(
            "POST", "/api/v1/carts", json=body.model_dump(exclude_none=True, mode="json"), response_model=CartView
        )

    async def get_cart(self, cart_id: UUID) -> CartView:
        """Return the current application cart."""

        return await self._request(
            "GET", f"/api/v1/carts/{cart_id}", response_model=CartView
        )

    async def add_cart_item(
        self,
        cart_id: UUID,
        request: AddCartItemRequest,
    ) -> CartView:
        """Add an exact product option after application-side validation."""

        return await self._request(
            "POST",
            f"/api/v1/carts/{cart_id}/items",
            json=request.model_dump(mode="json"),
            response_model=CartView,
        )

    async def update_cart_item(
        self,
        cart_id: UUID,
        item_id: UUID,
        request: UpdateCartItemRequest,
    ) -> CartView:
        """Update one cart-item quantity."""

        return await self._request(
            "PATCH",
            f"/api/v1/carts/{cart_id}/items/{item_id}",
            json=request.model_dump(mode="json"),
            response_model=CartView,
        )

    async def remove_cart_item(self, cart_id: UUID, item_id: UUID) -> CartView:
        """Remove one cart item."""

        return await self._request(
            "DELETE",
            f"/api/v1/carts/{cart_id}/items/{item_id}",
            response_model=CartView,
        )

    async def clear_cart(self, cart_id: UUID) -> CartView:
        """Clear all cart items while retaining the cart identity."""

        return await self._request(
            "DELETE", f"/api/v1/carts/{cart_id}/items", response_model=CartView
        )

    async def preview_cart(self, cart_id: UUID, request: PreviewCartRequest) -> StoreOrderPreview:
        """Preview checkout with discounts and delivery fees applied."""
        return await self._request(
            "POST",
            f"/api/v1/carts/{cart_id}/preview",
            json=request.model_dump(mode="json"),
            response_model=StoreOrderPreview,
        )

    async def place_order(self, request: PlaceOrderRequest) -> OrderView:
        """Submit an order and convert the temporary cart to a persistent state."""
        return await self._request(
            "POST",
            "/api/v1/orders",
            json=request.model_dump(mode="json"),
            response_model=OrderView,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        response_model: type[T] | None,
        **kwargs: Any,
    ) -> T | dict[str, Any] | list[Any]:
        """Execute a request and translate dependency errors consistently."""

        started = perf_counter()
        logger.info(
            "clothing_app_request_started",
            extra={
                "event": "clothing_app_request_started",
                "dependency_method": method,
                "dependency_path": path,
            },
        )
        try:
            response = await self._http.request(
                method,
                self._base_url + path,
                timeout=self._config.clothing_app_timeout_seconds,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            logger.exception(
                "clothing_app_unavailable",
                extra={
                    "event": "clothing_app_unavailable",
                    "dependency_method": method,
                    "dependency_path": path,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )
            raise ClothingAppUnavailableError() from exc

        logger.info(
            "clothing_app_request_completed",
            extra={
                "event": "clothing_app_request_completed",
                "dependency_method": method,
                "dependency_path": path,
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        if response.is_error:
            message = "The clothing application rejected the request."
            code = "CLOTHING_APP_REQUEST_FAILED"
            try:
                payload = response.json()
                error = payload.get("error", {})
                detail = payload.get("detail")
                if isinstance(error, dict):
                    message = str(error.get("message") or message)
                    code = str(error.get("code") or code)
                elif detail:
                    message = str(detail)
            except ValueError:
                pass
            raise AgentError(
                message,
                code=code,
                status_code=response.status_code,
            )

        payload = response.json()
        if response_model is None:
            return payload
        return response_model.model_validate(payload)
