"""Typed transport and normalized response models for the commerce API adapter.

The models in this module deliberately mirror the existing application's API
surface rather than inventing a new commerce protocol. The adapter uses these
models to normalize responses for the Agent while keeping HTTP details outside
Agent logic.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ProductSearchRequest(BaseModel):
    """Normalized semantic search request accepted by the commerce adapter."""

    model_config = ConfigDict(extra="forbid")

    query_text: str | None = None
    categories: list[str] = Field(default_factory=list)
    product_types: list[str] = Field(default_factory=list)
    occasions: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    excluded_colors: list[str] = Field(default_factory=list)
    size_mapping: dict[str, str] = Field(default_factory=dict)
    minimum_price: Decimal | None = None
    maximum_price: Decimal | None = None
    branch_code: str | None = None
    article_code: str | None = None
    sku: str | None = None
    in_stock_only: bool = True
    limit: int = Field(default=4, ge=1, le=20)


class ProductOption(BaseModel):
    """Normalized product option returned by the existing search API."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    product_id: int = 0
    variant_id: int = 0
    branch_id: int = 0
    article_code: str = ""
    product_name: str = ""
    category: str | None = None
    color: str | None = None
    size: str | None = None
    price: Decimal = Field(default=Decimal("0.00"), validation_alias=AliasChoices("price", "base_selling_price"))
    branch_code: str | None = None
    branch_name: str | None = None
    city: str | None = None
    available_quantity: int = 0
    image_url: str | None = None
    material: str | None = None
    fit: str | None = None
    season: str | None = None
    tags: list[str] = Field(default_factory=list)
    description: str | None = None
    is_available: bool | None = None


class ProductSearchResponse(BaseModel):
    """Normalized response from the existing product search endpoint."""

    model_config = ConfigDict(extra="allow")

    products: list[ProductOption] = Field(default_factory=list)
    result_count: int = 0
    relaxed_constraints: list[str] = Field(default_factory=list)


class BranchView(BaseModel):
    """Normalized branch information."""

    model_config = ConfigDict(extra="allow")

    branch_id: int
    branch_code: str
    branch_name: str
    city: str
    address: str


class ProductDetails(BaseModel):
    """Normalized detailed product response including sellable options."""

    model_config = ConfigDict(extra="allow")

    product_id: int
    article_code: str
    product_name: str
    category: str | None = None
    description: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    image_urls: list[str] = Field(default_factory=list)
    options: list[ProductOption] = Field(default_factory=list)


class AvailabilityView(BaseModel):
    """Normalized availability response for one exact variant and branch."""

    model_config = ConfigDict(extra="allow")

    product_id: int
    variant_id: int
    branch_id: int
    branch_code: str
    branch_name: str | None = None
    color: str | None = None
    size: str | None = None
    price: Decimal | None = None
    available_quantity: int = 0
    in_transit_quantity: int = 0
    is_available: bool = False


class CartView(BaseModel):
    """Loose normalized cart response.

    The prototype cart API is treated as an existing contract. Unknown fields
    are preserved so changes in the application response do not force Agent
    business logic changes.
    """

    model_config = ConfigDict(extra="allow")

    cart_id: UUID | None = None
    item_count: int = 0
    subtotal: Decimal = Decimal("0.00")


class CheckoutPreview(BaseModel):
    """Normalized checkout preview returned by the commerce API."""

    model_config = ConfigDict(extra="allow")

    cart_id: UUID | None = None
    subtotal: Decimal = Decimal("0.00")
    discount_total: Decimal = Decimal("0.00")
    delivery_fee: Decimal = Decimal("0.00")
    grand_total: Decimal = Decimal("0.00")
    applied_offers: list[dict[str, Any]] = Field(default_factory=list)


class OrderView(BaseModel):
    """Loose normalized order response from the existing order API."""

    model_config = ConfigDict(extra="allow")

    order_number: str | None = None
    status: str | None = None
    grand_total: Decimal | None = None
