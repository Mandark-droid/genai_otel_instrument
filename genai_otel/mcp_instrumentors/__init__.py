"""Module for OpenTelemetry instrumentors for Model Context Protocol (MCP) tools.

This package contains individual instrumentor classes for various MCP tools,
including databases, caching layers, message queues, vector databases, and
generic API clients, enabling automatic tracing and metric collection of their operations.
"""

# pylint: disable=R0801

from .manager import MCPInstrumentorManager
