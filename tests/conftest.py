"""Global test configuration.

Increases the Python recursion limit to handle deep import chains from heavy
SDKs (e.g., llama_index, vertexai, litellm) when running the full test suite.

Note: On Windows, the C thread stack size (default 1MB) may still cause stack
overflows even with a higher Python limit. Run tests in batches if needed:
    pytest tests/evaluation/ tests/mcp_instrumentors/ tests/test_*.py
    pytest tests/instrumentors/
"""

import sys

sys.setrecursionlimit(10000)
