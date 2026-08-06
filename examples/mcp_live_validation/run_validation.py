"""Live validation: real MCP client -> real MCP server -> real spans.

Starts the composite server as a subprocess, connects the official MCP SDK
client over stdio, and drives a sequence of real tool calls through the
instrumented ``ClientSession.call_tool``. Every span printed below was produced
by a real JSON-RPC round trip - nothing here is a mock or a hand-written span.

Run::

    python run_validation.py

It exits non-zero if any assertion about the emitted attributes fails, so it is
usable as a gate rather than only as a demo.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.mcp_semconv import (
    CommerceAttributes,
    MCPAttributes,
    MCPClientInstrumentor,
    MCPToolRegistry,
    TerminalStateClassifier,
    mcp_session,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.environ.get("MCP_SCHEMA_DIR", r"D:\Projects\food-delivery-chaos-lab\schemas")

_failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record a pass/fail assertion without aborting the run."""
    mark = "PASS" if condition else "FAIL"
    print("  [{0}] {1}{2}".format(mark, label, (" - " + detail) if detail else ""))
    if not condition:
        _failures.append(label)


def span_to_dict(span: Any) -> Dict[str, Any]:
    """Render a finished span as a plain dict for printing."""
    return {
        "name": span.name,
        "status": span.status.status_code.name,
        "attributes": {k: v for k, v in sorted((span.attributes or {}).items())},
        "events": [e.name for e in (span.events or [])],
    }


async def drive_session(instrumentor: MCPClientInstrumentor, exporter: InMemorySpanExporter):
    """Run one realistic agent session against the live server."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(HERE, "composite_server.py"), "--schema-dir", SCHEMA_DIR],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            wire_names = sorted(t.name for t in listed.tools)
            print("\n== Live tool surface ({0} tools) ==".format(len(wire_names)))
            print("  sample:", wire_names[:6])

            # Arguments below use each tool's real declared parameter names, so
            # the calls are valid against the documented surface.
            address = "addr-77"

            # Session A: searches, reads the cart, orders. Ends cleanly.
            # The address is seeded: the user supplied it, so using it is not a
            # hallucination. Without the seed every call below would be flagged.
            with mcp_session(
                "sess-live-success",
                user_id="+91-98200-12345",
                seed_identifiers=[address],
            ):
                await session.call_tool(
                    "food_search_restaurants", {"addressId": address, "query": "pizza"}
                )
                await session.call_tool("dineout_search_restaurants_dineout", {"query": "grill"})
                await session.call_tool(
                    "food_get_restaurant_menu",
                    {"addressId": address, "restaurantId": "rest-501"},
                )
                await session.call_tool("food_get_food_cart", {"addressId": address})
                await session.call_tool(
                    "food_place_food_order",
                    {"addressId": address, "paymentMethod": "UPI"},
                )
            success_spans = list(exporter.get_finished_spans())

            # Session B: looks up the order, then tries to cancel it. The surface
            # has no cancellation tool, so the agent can only file an error report
            # and then invent a tool that does not exist.
            with mcp_session(
                "sess-live-blocked",
                user_id="+91-98200-12345",
                requested_resolution="cancel_order",
            ):
                await session.call_tool("food_get_food_orders", {"addressId": address})
                try:
                    await session.call_tool(
                        "food_report_error",
                        {
                            "tool": "cancel_order",
                            "errorMessage": "user asked to cancel order-food-9001",
                        },
                    )
                except Exception:  # noqa: BLE001 - the error path is the point
                    pass
                try:
                    await session.call_tool("food_cancel_my_order", {"orderId": "order-food-9001"})
                except Exception:  # noqa: BLE001 - nonexistent tool
                    pass
            blocked_spans = [s for s in exporter.get_finished_spans() if s not in success_spans]

    return exporter.get_finished_spans(), wire_names, success_spans, blocked_spans


def main() -> int:
    """Run the live validation and report."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    registry = MCPToolRegistry.from_directory(SCHEMA_DIR)
    # dineout's placement tool is stage Reserve, not Order.
    registry.mark_placement(["book_table"])

    instrumentor = MCPClientInstrumentor(registry=registry)
    instrumentor.tracer = provider.get_tracer("mcp-live-validation")

    from genai_otel.config import OTelConfig

    instrumentor.instrument(OTelConfig(service_name="mcp-live-validation"))

    spans, wire_names, success_spans, blocked_spans = asyncio.run(
        drive_session(instrumentor, exporter)
    )

    print("\n== Emitted spans ({0}) ==".format(len(spans)))
    for span in spans:
        print(json.dumps(span_to_dict(span), indent=2, default=str))

    by_raw = {(s.attributes or {}).get(MCPAttributes.TOOL_RAW_NAME): s for s in spans}

    print("\n== Trap 1: composite-proxy prefix attribution ==")
    food = by_raw.get("food_search_restaurants")
    check("food_search_restaurants span exists", food is not None)
    if food is not None:
        attributes = food.attributes
        check(
            "mcp.tool == search_restaurants",
            attributes.get(MCPAttributes.TOOL) == "search_restaurants",
            str(attributes.get(MCPAttributes.TOOL)),
        )
        check(
            "mcp.server == food",
            attributes.get(MCPAttributes.SERVER) == "food",
            str(attributes.get(MCPAttributes.SERVER)),
        )
        check(
            "mcp.tool.raw_name preserved",
            attributes.get(MCPAttributes.TOOL_RAW_NAME) == "food_search_restaurants",
        )
        check(
            "span name carries server",
            food.name == "mcp.call_tool food.search_restaurants",
            food.name,
        )
        check(
            "stage/behaviour from schema map",
            attributes.get(MCPAttributes.STAGE) == "Discover"
            and attributes.get(MCPAttributes.BEHAVIOUR) == "read-only",
        )

    # The separator trap: a real tool name that itself contains the separator.
    dineout = by_raw.get("dineout_search_restaurants_dineout")
    check("dineout span exists", dineout is not None)
    if dineout is not None:
        check(
            "mcp.tool == search_restaurants_dineout (not mangled)",
            dineout.attributes.get(MCPAttributes.TOOL) == "search_restaurants_dineout",
            str(dineout.attributes.get(MCPAttributes.TOOL)),
        )
        check(
            "mcp.server == dineout",
            dineout.attributes.get(MCPAttributes.SERVER) == "dineout",
        )

    print("\n== Identifier provenance ==")
    check(
        "seeded address is not flagged as hallucinated",
        all(
            (s.attributes or {}).get(MCPAttributes.IDENTIFIER_HALLUCINATED) is False
            for s in success_spans
        ),
        str(
            [(s.attributes or {}).get(MCPAttributes.IDENTIFIER_HALLUCINATED) for s in success_spans]
        ),
    )
    orders = by_raw.get("food_get_food_orders")
    cancel = by_raw.get("food_cancel_my_order")
    check(
        "orderId echoed from a prior response is not flagged",
        cancel is not None
        and cancel.attributes.get(MCPAttributes.IDENTIFIER_HALLUCINATED) is False,
    )
    check("prior response that supplied the orderId was observed", orders is not None)

    print("\n== Invented tool: server attributed, metadata absent ==")
    if cancel is not None:
        check(
            "mcp.server still attributed from prefix",
            cancel.attributes.get(MCPAttributes.SERVER) == "food",
        )
        check(
            "mcp.tool is the unprefixed invented name",
            cancel.attributes.get(MCPAttributes.TOOL) == "cancel_my_order",
        )
        check(
            "no stage/behaviour invented for an unknown tool",
            MCPAttributes.STAGE not in cancel.attributes
            and MCPAttributes.BEHAVIOUR not in cancel.attributes,
        )
        check(
            "unknown-tool error captured verbatim",
            "Unknown tool" in str(cancel.attributes.get(MCPAttributes.ERROR_MESSAGE_RAW)),
        )

    print("\n== Trap 2: error capture at the client boundary ==")
    err = by_raw.get("food_report_error")
    check("error span exists", err is not None)
    if err is not None:
        raw = err.attributes.get(MCPAttributes.ERROR_MESSAGE_RAW)
        check("mcp.error.message_raw captured", bool(raw), str(raw))
        check(
            "message is verbatim upstream text",
            bool(raw) and "no cancellation endpoint exists" in str(raw),
        )
        check("span status is ERROR", err.status.status_code.name == "ERROR")

    print("\n== Privacy: no bodies, no plaintext identifiers ==")
    dumped = json.dumps([span_to_dict(s) for s in spans], default=str)
    check("no raw phone number on any span", "+91-98200-12345" not in dumped)
    check("no plaintext '9820012345'", "9820012345" not in dumped)
    check(
        "user id present only as salted hash",
        any(
            str((s.attributes or {}).get(MCPAttributes.USER_ID_HASH, "")).startswith("sha256:")
            for s in spans
        ),
    )
    check(
        "no request/response body keys on spans",
        "Margherita" not in dumped and "input.value" not in dumped and "output.value" not in dumped,
    )
    check(
        "session id recorded",
        any(
            (s.attributes or {}).get(MCPAttributes.SESSION_ID) == "sess-live-success" for s in spans
        ),
    )
    hashes = {
        (s.attributes or {}).get(MCPAttributes.USER_ID_HASH)
        for s in spans
        if (s.attributes or {}).get(MCPAttributes.USER_ID_HASH)
    }
    check("one user hashes to one value within a process", len(hashes) == 1, str(hashes))

    print("\n== Classifier on the spans this run actually produced ==")
    classifier = TerminalStateClassifier(registry=registry)

    print("\n-- Session A ({0} spans) --".format(len(success_spans)))
    success = classifier.classify(success_spans)
    print(success.explain())
    check(
        "order-only session -> COMPLETED_SUCCESS",
        success.state.value == "COMPLETED_SUCCESS",
        success.state.value,
    )
    check("order_placed recorded on success", success.order_placed is True)
    check(
        "commerce.order.placed on the placement span",
        any((s.attributes or {}).get(CommerceAttributes.ORDER_PLACED) for s in success_spans),
    )

    print("\n-- Session B ({0} spans) --".format(len(blocked_spans)))
    blocked = classifier.classify(blocked_spans, requested_resolution="cancel_order")
    print(blocked.explain())
    check(
        "blocked session -> BLOCKED_NO_TOOL",
        blocked.state.value == "BLOCKED_NO_TOOL",
        blocked.state.value,
    )
    check(
        "resolution == cancel_order",
        blocked.resolution is not None and blocked.resolution.value == "cancel_order",
    )
    check("evidence is non-empty", len(blocked.evidence) > 0)

    print("\n== Measured surface gaps (35-tool surface) ==")
    print("  missing capabilities:", registry.missing_capabilities())
    check("cancel_order has no tool", not registry.has_capability("cancel_order"))
    check("refund has no tool", not registry.has_capability("refund"))
    check("order_history_lookup HAS a tool", registry.has_capability("order_history_lookup"))

    print("\n== Result ==")
    if _failures:
        print("  {0} FAILED: {1}".format(len(_failures), _failures))
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
