"""Build frontend-compatible structured payloads from trusted commerce results."""

from __future__ import annotations

from typing import Any

from .turn_contract import AgentTurnResponse, ContentType, product_card


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _branch_map(context: Any) -> dict[str, Any]:
    if not context:
        return {}
    branches = _get(context, "branches", []) or []
    return {str(_get(b, "branch_code", "")).lower(): b for b in branches}


def calculate_authoritative_search_limit(categories: list[str] | None = None) -> int:
    """Authoritative result-limit policy across Fitzy.

    - ONE CATEGORY (or unspecified): default 3-4 products (4)
    - TWO CATEGORIES: ~3 per category (6)
    - THREE+ CATEGORIES: balanced allocation (min 3 * len(cats), 20)
    - HARD MAXIMUM: 20
    """
    if not categories:
        return 4
    count = len(categories)
    if count <= 1:
        return 4
    elif count == 2:
        return 6
    else:
        return min(count * 3, 20)


def product_options_for_frontend(result: Any, context: Any = None) -> list[dict[str, Any]]:
    """Flatten authoritative ProductView or ProductOption variants into the existing mobile card contract."""

    products: list[dict[str, Any]] = []
    branches = _branch_map(context)

    raw_products = _get(result, "products", []) or []
    for product in raw_products:
        images = _get(product, "images", None) or []
        image_url = (images[0] if isinstance(images, list) and images else None) or _get(product, "image_url", None)
        variants = _get(product, "variants", None) or _get(product, "options", None) or []

        if not variants:
            # product is already a flat ProductOption / product card dict or object
            products.append({
                "product_id": _get(product, "product_id", 0),
                "variant_id": _get(product, "variant_id", 0),
                "branch_id": _get(product, "branch_id", 0),
                "article_code": _get(product, "article_code", ""),
                "product_name": _get(product, "product_name", ""),
                "category": _get(product, "category", ""),
                "gender": _get(product, "gender", "MEN"),
                "brand": _get(product, "brand", "Northstar"),
                "color": _get(product, "color", ""),
                "size": _get(product, "size", ""),
                "price": str(_get(product, "price", _get(product, "final_price", 0))),
                "base_price": str(_get(product, "base_price", _get(product, "price", 0))),
                "discount_amount": str(_get(product, "discount_amount", 0)),
                "branch_code": _get(product, "branch_code", ""),
                "branch_name": _get(product, "branch_name", ""),
                "city": _get(product, "city", ""),
                "available_quantity": _get(product, "available_quantity", 0),
                "in_transit_quantity": 0,
                "image_url": image_url,
                "material": _get(product, "material", None),
                "fit": _get(product, "fit", None),
                "season": _get(product, "season", None),
                "tags": [],
                "description": _get(product, "description", None),
                "match_score": 0.0,
                "match_reasons": [],
            })
            continue

        for variant in variants:
            availabilities = _get(variant, "branch_availability", None) or []
            if not availabilities:
                products.append(
                    {
                        "product_id": _get(product, "product_id", 0),
                        "variant_id": _get(variant, "variant_id", 0),
                        "branch_id": _get(variant, "branch_id", 0),
                        "article_code": _get(product, "article_code", ""),
                        "product_name": _get(product, "product_name", ""),
                        "category": _get(product, "category", ""),
                        "gender": _get(product, "gender", "MEN"),
                        "brand": _get(product, "brand", "Northstar"),
                        "color": _get(variant, "color", ""),
                        "size": _get(variant, "size", ""),
                        "price": str(_get(variant, "final_price", _get(variant, "price", 0))),
                        "base_price": str(_get(variant, "price", 0)),
                        "discount_amount": str(_get(variant, "discount_amount", 0)),
                        "branch_code": _get(variant, "branch_code", ""),
                        "branch_name": _get(variant, "branch_name", ""),
                        "city": "",
                        "available_quantity": _get(variant, "available_quantity", 1 if _get(variant, "is_available", True) else 0),
                        "in_transit_quantity": 0,
                        "image_url": image_url,
                        "material": _get(product, "material", None),
                        "fit": _get(product, "fit", None),
                        "season": _get(product, "season", None),
                        "tags": [],
                        "description": _get(product, "description", None),
                        "match_score": 0.0,
                        "match_reasons": [],
                    }
                )
            else:
                for availability in availabilities:
                    branch = branches.get(str(_get(availability, "branch_code", "")).lower())
                    products.append(
                        {
                            "product_id": _get(product, "product_id", 0),
                            "variant_id": _get(variant, "variant_id", 0),
                            "branch_id": _get(branch, "branch_id", 0),
                            "article_code": _get(product, "article_code", ""),
                            "product_name": _get(product, "product_name", ""),
                            "category": _get(product, "category", ""),
                            "gender": _get(product, "gender", "MEN"),
                            "brand": _get(product, "brand", "Northstar"),
                            "color": _get(variant, "color", ""),
                            "size": _get(variant, "size", ""),
                            "price": str(_get(variant, "final_price", _get(variant, "price", 0))),
                            "base_price": str(_get(variant, "price", 0)),
                            "discount_amount": str(_get(variant, "discount_amount", 0)),
                            "branch_code": _get(availability, "branch_code", ""),
                            "branch_name": _get(availability, "branch_name", ""),
                            "city": _get(branch, "city", ""),
                            "available_quantity": _get(availability, "available_quantity", 0),
                            "in_transit_quantity": 0,
                            "image_url": image_url,
                            "material": _get(product, "material", None),
                            "fit": _get(product, "fit", None),
                            "season": _get(product, "season", None),
                            "tags": [],
                            "description": _get(product, "description", None),
                            "match_score": 0.0,
                            "match_reasons": [],
                        }
                    )

    products.sort(key=lambda item: (item["available_quantity"] <= 0, item["product_id"], item["variant_id"]))
    
    seen_pids = set()
    unique_products = []
    for item in products:
        pid = item["product_id"]
        if pid not in seen_pids:
            seen_pids.add(pid)
            unique_products.append(item)

    cats = None
    if context:
        if isinstance(context, dict):
            cats = context.get("categories") or (context.get("current_search") or {}).get("categories")
        else:
            cats = getattr(context, "categories", None)

    limit = calculate_authoritative_search_limit(cats)
    return unique_products[:limit]


def build_product_list_response(
    session_id: str | Any,
    language: str,
    reply: str,
    result: Any,
    context: Any = None,
    *,
    intent: str | None = None,
) -> AgentTurnResponse:
    """Build a product-card response directly from trusted ProductView data."""

    cards = product_options_for_frontend(result, context)
    return AgentTurnResponse(
        session_id=str(session_id),
        reply=reply,
        language=language,
        content_type=ContentType.PRODUCT_LIST,
        products=cards,
        suggested_actions=["view_details", "add_to_cart"] if cards else [],
        ui_actions=["show_product_carousel"] if cards else [],
    )


def build_product_details_response(session_id: str | Any, language: str, reply: str, result: Any) -> AgentTurnResponse:
    """Build a product detail payload from the authoritative backend result."""

    product = getattr(result, "product", result)
    return AgentTurnResponse(
        session_id=str(session_id),
        reply=reply,
        language=language,
        content_type=ContentType.PRODUCT_DETAILS,
        product=product_card(product),
        suggested_actions=["add_to_cart"],
        ui_actions=["open_product_details"],
    )


def build_cart_response(session_id: str | Any, language: str, reply: str, result: Any) -> AgentTurnResponse:
    """Build a cart UI payload from the backend CartView."""

    return AgentTurnResponse(
        session_id=str(session_id),
        reply=reply,
        language=language,
        content_type=ContentType.CART,
        cart=result.model_dump(mode="json") if hasattr(result, "model_dump") else (result if isinstance(result, dict) else None),
        suggested_actions=["checkout"],
        ui_actions=["open_cart"],
    )


def build_checkout_response(session_id: str | Any, language: str, reply: str, result: Any) -> AgentTurnResponse:
    """Build a checkout summary from authoritative backend totals."""

    return AgentTurnResponse(
        session_id=str(session_id),
        reply=reply,
        language=language,
        content_type=ContentType.CHECKOUT,
        checkout=result.model_dump(mode="json") if hasattr(result, "model_dump") else (result if isinstance(result, dict) else None),
        suggested_actions=["confirm_order"],
        ui_actions=["show_checkout_summary"],
    )


def build_order_response(session_id: str | Any, language: str, reply: str, result: Any) -> AgentTurnResponse:
    """Build an order confirmation payload from authoritative backend data."""

    return AgentTurnResponse(
        session_id=str(session_id),
        reply=reply,
        language=language,
        content_type=ContentType.ORDER,
        order=result.model_dump(mode="json") if hasattr(result, "model_dump") else (result if isinstance(result, dict) else None),
        ui_actions=["show_order_confirmation"],
    )
