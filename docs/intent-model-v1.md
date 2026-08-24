# Fitzy V1 Intent Model

## Purpose

Fitzy uses a closed-world intent vocabulary. The LLM selects registered intents and
extracts explicit parameters; the backend remains the source of truth.

## Registered customer-facing intents

### Conversational

- `general_chat`

### Catalog

- `product_search`
- `product_details`
- `product_availability`
- `branch_information`
- `branch_availability`

### Cart

- `cart_view`
- `add_to_cart`
- `update_cart`
- `remove_from_cart`
- `clear_cart`

### Checkout/order

- `checkout_preview`
- `delivery_info`
- `place_order`
- `cancel_order`

## Multi-intent

`multi-intent` is **not itself an intent value**.

A message containing multiple requests is represented as an ordered `intents`
array and then converted into an execution plan with dependencies.

Example:

> Show me black shirts and tell me how many Northstar branches you have.

```text
[
  PRODUCT_SEARCH,
  BRANCH_INFORMATION
]
```

These actions can execute independently.

Example:

> Show me black shirts and add the first one to my cart.

```text
[
  PRODUCT_SEARCH,
  ADD_TO_CART
]
```

The second action depends on the first action's displayed product result.

## Search refinement

Filters such as:

- wedding
- black
- blue
- size L
- under PKR 5000

are **not separate intents**. They are `search_overrides` attached to
`product_search`.

The state precedence is:

```text
current turn
    >
active current search
    >
persistent preferences
```

## Confirmation

A plain `"yes"` is not intrinsically `place_order`.

The runtime must have a pending order-confirmation context before `"yes"` can
satisfy the order confirmation requirement.

## User-facing behavior

Intent names are internal. Never expose intent names, tool names, API paths,
database IDs, or schema names to customers.
