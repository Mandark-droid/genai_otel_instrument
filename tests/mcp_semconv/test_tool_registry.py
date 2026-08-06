"""Tests for the MCP tool schema-map registry."""

import json
import os
import tempfile

import pytest

from genai_otel.mcp_semconv import MCPToolRegistry
from genai_otel.mcp_semconv.tool_registry import KNOWN_CAPABILITIES

from .conftest import ALL_SCHEMAS, FOOD_SCHEMA


class TestSchemaParsing:
    """Parsing the native ``details`` shape and normalised overrides."""

    def test_loads_all_three_servers(self, registry):
        assert registry.servers == ("dineout", "food", "instamart")
        assert registry.candidate_count == 35

    def test_reads_stage_and_behaviour_from_details_block(self, registry):
        meta = registry.get("place_food_order", "food")
        assert meta.stage == "Order"
        assert meta.behaviour == "mutating"

    def test_normalised_keys_override_details(self):
        registry = MCPToolRegistry.from_dicts(
            [
                {
                    "server": "food",
                    "tools": [
                        {
                            "name": "weird_tool",
                            "stage": "Support",
                            "behaviour": "read-only",
                            "details": {"**Stage**": "Cart", "**Behaviour**": "mutating"},
                        }
                    ],
                }
            ]
        )
        meta = registry.get("weird_tool", "food")
        assert meta.stage == "Support"
        assert meta.behaviour == "read-only"

    def test_explicit_idempotent_flag_wins_over_derivation(self):
        registry = MCPToolRegistry.from_dicts(
            [
                {
                    "server": "food",
                    "tools": [
                        {"name": "set_quantity", "behaviour": "mutating", "idempotent": True}
                    ],
                }
            ]
        )
        assert registry.get("set_quantity", "food").idempotent is True

    def test_idempotency_derived_from_behaviour_when_absent(self, registry):
        # The schema map carries no idempotency field, so it is derived. The
        # derivation must never mark an order-placement tool repeatable.
        assert registry.get("search_restaurants", "food").idempotent is True
        assert registry.get("place_food_order", "food").idempotent is False
        assert registry.get("checkout", "instamart").idempotent is False

    def test_placement_derived_from_order_stage(self, registry):
        assert registry.get("place_food_order", "food").placement is True
        assert registry.get("checkout", "instamart").placement is True
        # Reserve-stage booking is not an Order stage, so it is not auto-derived.
        assert registry.get("book_table", "dineout").placement is False

    def test_mark_placement_covers_reserve_stage_booking(self, registry_with_booking):
        assert registry_with_booking.get("book_table", "dineout").placement is True

    def test_malformed_entries_are_skipped_not_fatal(self):
        registry = MCPToolRegistry.from_dicts(
            [{"server": "food", "tools": [{"no_name": True}, "not-a-dict", {"name": "ok"}]}]
        )
        assert registry.candidate_count == 1
        assert registry.get("ok", "food") is not None

    def test_from_files_and_from_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for schema in ALL_SCHEMAS:
                path = os.path.join(directory, schema["server"] + ".json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(schema, handle)
            assert MCPToolRegistry.from_directory(directory).candidate_count == 35

    def test_unreadable_paths_degrade_gracefully(self):
        registry = MCPToolRegistry.from_files(["/definitely/not/a/path.json"])
        assert registry.candidate_count == 0
        assert MCPToolRegistry.from_directory("/definitely/not/a/dir").candidate_count == 0


class TestPrefixStripping:
    """Composite-proxy prefix handling.

    Getting this wrong does not raise - it silently scores every ground-truth
    comparison 0%, which is why each case is pinned explicitly.
    """

    def test_strips_known_server_prefix(self, registry):
        resolution = registry.resolve("food_search_restaurants")
        assert resolution.tool == "search_restaurants"
        assert resolution.server == "food"
        assert resolution.prefix_stripped is True

    def test_keeps_raw_name_alongside_stripped_name(self, registry):
        resolution = registry.resolve("instamart_checkout")
        assert resolution.tool == "checkout"
        assert resolution.raw_name == "instamart_checkout"

    def test_unprefixed_name_is_left_alone(self, registry):
        resolution = registry.resolve("search_restaurants")
        assert resolution.tool == "search_restaurants"
        assert resolution.prefix_stripped is False

    def test_does_not_mangle_name_that_merely_contains_a_server_word(self, registry):
        # A naive split on the first separator, or a contains-check on server
        # names, would wreck this real tool name.
        resolution = registry.resolve("search_restaurants_dineout")
        assert resolution.tool == "search_restaurants_dineout"
        assert resolution.server == "dineout"
        assert resolution.prefix_stripped is False

    def test_strips_prefix_from_that_same_tool_when_actually_prefixed(self, registry):
        resolution = registry.resolve("dineout_search_restaurants_dineout")
        assert resolution.tool == "search_restaurants_dineout"
        assert resolution.server == "dineout"
        assert resolution.prefix_stripped is True

    @pytest.mark.parametrize("separator", ["_", "-", ".", "/", ":"])
    def test_supports_each_proxy_separator(self, registry, separator):
        resolution = registry.resolve("food" + separator + "search_restaurants")
        assert resolution.tool == "search_restaurants"
        assert resolution.server == "food"
        assert resolution.prefix_stripped is True

    def test_duplicate_tool_name_across_servers_is_ambiguous(self, registry):
        resolution = registry.resolve("get_addresses")
        assert resolution.ambiguous is True
        assert resolution.server is None
        # Both candidates agree on stage and behaviour, so those are still safe
        # to emit even though the server is undecidable.
        assert resolution.stage == "Discover"
        assert resolution.behaviour == "read-only"

    def test_server_hint_disambiguates_duplicate_name(self, registry):
        resolution = registry.resolve("get_addresses", server_hint="instamart")
        assert resolution.server == "instamart"
        assert resolution.ambiguous is False

    def test_prefix_disambiguates_duplicate_name(self, registry):
        resolution = registry.resolve("food_get_addresses")
        assert resolution.server == "food"
        assert resolution.tool == "get_addresses"
        assert resolution.ambiguous is False

    def test_unknown_tool_resolves_to_itself(self, registry):
        resolution = registry.resolve("totally_unknown_tool")
        assert resolution.tool == "totally_unknown_tool"
        assert resolution.known is False
        assert resolution.stage is None

    def test_known_server_prefix_with_unknown_tool_still_strips(self, registry):
        resolution = registry.resolve("food_invented_tool")
        assert resolution.server == "food"
        assert resolution.tool == "invented_tool"
        assert resolution.prefix_stripped is True
        assert resolution.known is False

    def test_empty_name_is_handled(self, registry):
        resolution = registry.resolve("")
        assert resolution.tool == ""


class TestCapabilities:
    """The surface-gap test that separates a missing tool from a bad agent."""

    def test_surface_cannot_cancel_refund_or_replace(self, registry):
        assert registry.has_capability("cancel_order") is False
        assert registry.has_capability("refund") is False
        assert registry.has_capability("replacement") is False
        assert registry.has_capability("missing_item_claim") is False
        assert registry.has_capability("modify_placed_order") is False

    def test_report_error_does_not_satisfy_a_write_resolution(self, registry):
        # report_error exists on all three servers. It files a report; it does
        # not cancel, refund or escalate, and must not mask those gaps.
        assert registry.get("report_error", "food") is not None
        for capability in ("cancel_order", "refund", "replacement", "escalate_to_human"):
            satisfying = {m.name for m in registry.tools_for_capability(capability)}
            assert "report_error" not in satisfying

    def test_order_history_lookup_is_satisfied(self, registry):
        assert registry.has_capability("order_history_lookup") is True
        names = {m.name for m in registry.tools_for_capability("order_history_lookup")}
        assert names == {"get_food_orders", "get_orders"}

    def test_address_creation_does_not_satisfy_mid_cart_edit(self, registry):
        assert registry.has_capability("address_edit_mid_cart") is False

    def test_missing_capabilities_reports_nine_of_ten(self, registry):
        missing = registry.missing_capabilities()
        assert "order_history_lookup" not in missing
        assert len(missing) == len(KNOWN_CAPABILITIES) - 1

    def test_declared_capabilities_are_honoured(self):
        registry = MCPToolRegistry.from_dicts(
            [
                {
                    "server": "food",
                    "tools": [{"name": "abort_order", "capabilities": ["cancel_order"]}],
                }
            ]
        )
        assert registry.has_capability("cancel_order") is True

    def test_capability_patterns_can_be_overridden(self):
        registry = MCPToolRegistry.from_dicts(
            [FOOD_SCHEMA], capability_patterns={"refund": (r"report_error",)}
        )
        assert registry.has_capability("refund") is True


class TestSeparatorAssumptionCanary:
    """Fail loudly if the upstream composite-proxy separator stops being ``_``.

    Server attribution rests on an assumption this repo does not control: that a
    composite proxy joins namespace and tool name with an underscore. Two
    upstream behaviours make a silent change here expensive:

    * ``mcpadapt._sanitize_function_name`` runs ``name.replace("-", "_")`` and
      then strips remaining non-word characters, so a proxy that switched to
      ``-`` or ``.`` would arrive already rewritten - the registry would still
      resolve it, but nothing downstream could tell a rewritten name from a
      genuine one.
    * A separator that survives sanitisation but is not in
      :data:`DEFAULT_PREFIX_SEPARATORS` would stop being stripped, and every
      ground-truth comparison would quietly read 0%.

    Neither failure raises on its own. These tests are the alarm.
    """

    def test_fastmcp_still_joins_namespace_and_tool_with_underscore(self):
        """The live check: mount a server and read back the wire name."""
        fastmcp = pytest.importorskip("fastmcp", reason="fastmcp not installed")

        parent = fastmcp.FastMCP(name="parent")
        child = fastmcp.FastMCP(name="food")

        @child.tool(name="search_restaurants")
        def search_restaurants() -> str:  # pragma: no cover - body never runs
            return "ok"

        parent.mount(child, namespace="food")

        import asyncio

        listed = asyncio.run(parent.list_tools())
        names = sorted(getattr(t, "name", str(t)) for t in listed)
        assert "food_search_restaurants" in names, (
            "fastmcp no longer joins namespace and tool name with '_'. Observed "
            "names: {0}. Server attribution in MCPToolRegistry.resolve() assumes "
            "the underscore form - update DEFAULT_PREFIX_SEPARATORS and the "
            "server-attribution docs before shipping.".format(names)
        )

    def test_underscore_is_the_first_separator_tried(self):
        """A cheap guard that runs even without fastmcp installed."""
        from genai_otel.mcp_semconv.tool_registry import DEFAULT_PREFIX_SEPARATORS

        assert (
            DEFAULT_PREFIX_SEPARATORS[0] == "_"
        ), "Underscore must stay the primary composite-proxy separator; " "observed {0}".format(
            DEFAULT_PREFIX_SEPARATORS
        )

    def test_mcpadapt_sanitiser_still_rewrites_hyphens_to_underscores(self):
        """If mcpadapt stops rewriting, a hyphen separator would reach us intact."""
        mcpadapt_core = pytest.importorskip("mcpadapt.core", reason="mcpadapt not installed")
        sanitise = getattr(mcpadapt_core, "_sanitize_function_name", None)
        if sanitise is None:
            pytest.skip("mcpadapt._sanitize_function_name not present")

        assert sanitise("food-search_restaurants") == "food_search_restaurants", (
            "mcpadapt._sanitize_function_name no longer normalises '-' to '_'. "
            "A hyphen-separated proxy would now reach the registry unrewritten."
        )
