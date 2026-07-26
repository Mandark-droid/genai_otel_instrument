"""Shared fixtures for MCP semantic-convention tests.

The tool surface used here mirrors the real three-server Swiggy schema map:
14 food + 8 dineout + 13 instamart = 35 tools, with the properties that matter
for the tests - duplicate tool names across servers (``get_addresses``,
``report_error``), a tool whose name would be mangled by a naive prefix split
(``search_restaurants_dineout``), and a Support stage that offers error
reporting but no cancellation, refund or replacement write API.
"""

import pytest

from genai_otel.mcp_semconv import MCPToolRegistry


def _tool(name, stage, behaviour):
    """Build a tool entry in the schema map's native ``details`` shape."""
    return {
        "name": name,
        "details": {
            "Field": "Value",
            "**Name**": name,
            "**Stage**": stage,
            "**Behaviour**": behaviour,
        },
    }


FOOD_SCHEMA = {
    "server": "food",
    "tool_count": 14,
    "tools": [
        _tool("apply_food_coupon", "Cart", "mutating"),
        _tool("fetch_food_coupons", "Cart", "read-only"),
        _tool("flush_food_cart", "Cart", "mutating"),
        _tool("get_addresses", "Discover", "read-only"),
        _tool("get_food_cart", "Cart", "read-only"),
        _tool("get_food_order_details", "Track", "read-only"),
        _tool("get_food_orders", "Track", "read-only"),
        _tool("get_restaurant_menu", "Discover", "read-only"),
        _tool("place_food_order", "Order", "mutating"),
        _tool("report_error", "Support", "mutating"),
        _tool("search_menu", "Discover", "read-only"),
        _tool("search_restaurants", "Discover", "read-only"),
        _tool("track_food_order", "Track", "read-only"),
        _tool("update_food_cart", "Cart", "mutating"),
    ],
}

DINEOUT_SCHEMA = {
    "server": "dineout",
    "tool_count": 8,
    "tools": [
        _tool("book_table", "Reserve", "mutating"),
        _tool("create_cart", "Reserve", "mutating"),
        _tool("get_available_slots", "Reserve", "read-only"),
        _tool("get_booking_status", "Manage", "read-only"),
        _tool("get_restaurant_details", "Find", "read-only"),
        _tool("get_saved_locations", "Find", "read-only"),
        _tool("report_error", "Support", "mutating"),
        # Name ends in a token that also names a server-ish word; a naive
        # split-on-separator prefix stripper mangles this one.
        _tool("search_restaurants_dineout", "Find", "read-only"),
    ],
}

INSTAMART_SCHEMA = {
    "server": "instamart",
    "tool_count": 13,
    "tools": [
        _tool("checkout", "Order", "mutating"),
        _tool("clear_cart", "Cart", "mutating"),
        _tool("create_address", "Discover", "mutating"),
        _tool("delete_address", "Discover", "mutating"),
        _tool("get_addresses", "Discover", "read-only"),
        _tool("get_cart", "Cart", "read-only"),
        _tool("get_order_details", "Track", "read-only"),
        _tool("get_orders", "Track", "read-only"),
        _tool("report_error", "Support", "mutating"),
        _tool("search_products", "Discover", "read-only"),
        _tool("track_order", "Track", "read-only"),
        _tool("update_cart", "Cart", "mutating"),
        _tool("your_go_to_items", "Discover", "read-only"),
    ],
}

ALL_SCHEMAS = [FOOD_SCHEMA, DINEOUT_SCHEMA, INSTAMART_SCHEMA]


@pytest.fixture
def registry():
    """The full 35-tool three-server registration surface."""
    return MCPToolRegistry.from_dicts(ALL_SCHEMAS)


@pytest.fixture
def registry_with_booking(registry):
    """The same surface with dineout's Reserve-stage placement tool marked."""
    registry.mark_placement(["book_table"])
    return registry
