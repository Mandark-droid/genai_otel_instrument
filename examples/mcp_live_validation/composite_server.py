"""A real FastMCP composite-proxy server built from MCP tool schema maps.

Three domain servers (food / instamart / dineout) are mounted into one parent
under a namespace, which is how a composite proxy actually presents a
multi-server surface to an agent: tools arrive prefixed
(``food_search_restaurants``), and that prefix is the only server attribution on
the wire.

Run standalone over stdio::

    python composite_server.py

The tool bodies are deliberately small canned responses. This server exists to
exercise the transport and the naming surface, not to simulate a business.
"""

import argparse
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP

#: Where the schema maps live, overridable for a different tool surface.
DEFAULT_SCHEMA_DIR = os.environ.get(
    "MCP_SCHEMA_DIR", r"REDACTED-PATH\schemas"
)

#: Tools that return a canned error, so the error-attribution path is exercised
#: against a real JSON-RPC round trip rather than a mock.
ERROR_TOOLS = {"report_error"}

#: Tools returning a cart, so cart-hash and drift detection have real input.
CART_TOOLS = {"get_food_cart", "update_food_cart", "get_cart", "update_cart", "create_cart"}

#: Tools that place an order or booking.
PLACEMENT_TOOLS = {"place_food_order", "checkout", "book_table"}


def load_schema(path: str) -> Dict[str, Any]:
    """Load one schema map."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _canned_response(server: str, tool: str, arguments: Dict[str, Any]) -> Any:
    """Build a small, deterministic response for one tool."""
    if tool in CART_TOOLS:
        return {
            "cart": {
                "cartId": "cart-{0}-001".format(server),
                "items": [
                    {"itemId": "item-101", "name": "Margherita", "qty": 1, "price": 249},
                    {"itemId": "item-102", "name": "Garlic Bread", "qty": 2, "price": 99},
                ],
                "total": 447,
            }
        }
    if tool in PLACEMENT_TOOLS:
        return {
            "orderId": "order-{0}-9001".format(server),
            "status": "CONFIRMED",
            "amountCharged": 447,
        }
    if tool == "get_addresses" or tool == "get_saved_locations":
        return {"addresses": [{"addressId": "addr-77", "label": "Home"}]}
    if tool in ("search_restaurants", "search_restaurants_dineout"):
        return {
            "restaurants": [
                {"restaurantId": "rest-501", "name": "Pizza Place", "rating": 4.3},
                {"restaurantId": "rest-502", "name": "Curry House", "rating": 4.1},
            ]
        }
    if tool == "get_restaurant_menu":
        return {
            "restaurantId": arguments.get("restaurantId", "rest-501"),
            "menu": [{"itemId": "item-101", "name": "Margherita", "price": 249}],
        }
    if tool in ("get_food_orders", "get_orders"):
        return {"orders": [{"orderId": "order-{0}-9001".format(server), "status": "DELIVERED"}]}
    return {"ok": True, "tool": tool, "server": server}


#: JSON type name -> Python annotation, for rebuilding declared tool signatures.
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def build_domain_server(schema: Dict[str, Any]) -> FastMCP:
    """Build one domain server exposing every tool in a schema map."""
    server_name = str(schema.get("server") or "unknown")
    app: FastMCP = FastMCP(name=server_name)

    for entry in schema.get("tools") or []:
        tool_name = entry.get("name")
        if not tool_name:
            continue
        description = (entry.get("short_description") or "")[:200]
        _register(app, server_name, tool_name, description, entry.get("parameters") or [])

    return app


def _build_signature(parameters: List[Dict[str, Any]]) -> Tuple[inspect.Signature, Dict[str, Any]]:
    """Rebuild a tool's declared signature so FastMCP validates real arguments.

    The schema map lists each tool's real parameters, so the mock server exposes
    the same argument surface the documented tool does. Every parameter is made
    optional: this server is exercising transport and naming, and a required
    argument would only add validation noise.

    Both the signature and ``__annotations__`` are returned - pydantic builds its
    argument schema from ``get_type_hints``, so a ``__signature__`` alone is not
    enough.
    """
    params = []
    annotations: Dict[str, Any] = {}
    seen = set()
    for spec in parameters:
        name = str(spec.get("name") or "").strip()
        if not name.isidentifier() or name in seen:
            continue
        seen.add(name)
        annotation = Optional[_TYPE_MAP.get(str(spec.get("type") or "string").lower(), str)]
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=annotation,
            )
        )
        annotations[name] = annotation
    annotations["return"] = Any
    return inspect.Signature(params, return_annotation=Any), annotations


def _register(
    app: FastMCP,
    server_name: str,
    tool_name: str,
    description: str,
    parameters: List[Dict[str, Any]],
) -> None:
    """Register one tool. Bound in a helper so each closure captures its own name."""

    async def handler(**kwargs: Any) -> Any:
        """Handle one tool call with a canned response."""
        if tool_name in ERROR_TOOLS:
            # A real tool-level failure: raised so it crosses the wire as a
            # JSON-RPC error the client instrumentor has to capture itself.
            raise ValueError(
                "Support ticket API unavailable: no cancellation endpoint exists "
                "for server '{0}'".format(server_name)
            )
        return _canned_response(server_name, tool_name, kwargs)

    signature, annotations = _build_signature(parameters)
    handler.__name__ = tool_name
    handler.__signature__ = signature
    handler.__annotations__ = annotations
    app.tool(name=tool_name, description=description or tool_name)(handler)


def build_composite(schema_dir: str = DEFAULT_SCHEMA_DIR) -> FastMCP:
    """Build the parent composite server with all domains mounted under a namespace."""
    parent: FastMCP = FastMCP(name="swiggy-composite")
    names: List[str] = sorted(n for n in os.listdir(schema_dir) if n.endswith(".json"))
    for name in names:
        schema = load_schema(os.path.join(schema_dir, name))
        server_name = str(schema.get("server") or name[:-5])
        parent.mount(build_domain_server(schema), namespace=server_name)
    return parent


def main() -> None:
    """Run the composite server over stdio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-dir", default=DEFAULT_SCHEMA_DIR)
    args = parser.parse_args()
    build_composite(args.schema_dir).run()


if __name__ == "__main__":
    main()
