"""Keep the documented provider list in step with the instrumentor registry.

The provider count drifted three ways at once - ``docs/index.md`` claimed 19+,
``docs/guides/llm-providers.md`` claimed 21+, ``README.md`` claimed 21+, and the
table itself listed 17 - because every number is hand-maintained and nothing
failed when a provider was added in code but not in docs.

These tests bind the docs to ``INSTRUMENTORS``. Adding a provider instrumentor
without a table row (or vice versa) now fails, and the headline counts have to
agree with the table's actual length.

When you add a provider: add it to ``INSTRUMENTORS``, add a table row, and add
the key -> display-name pair to ``PROVIDER_DISPLAY_NAMES`` below. When you add a
framework or a piece of tooling instead, list its key in ``NON_PROVIDER_KEYS``.
"""

import re
from pathlib import Path

import pytest

from genai_otel.auto_instrument import INSTRUMENTORS

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_DOC = REPO_ROOT / "docs" / "guides" / "llm-providers.md"
INDEX_DOC = REPO_ROOT / "docs" / "index.md"
README = REPO_ROOT / "README.md"

# Registry entries that are NOT LLM providers: agent/orchestration frameworks,
# structured-output wrappers, and cross-cutting tooling. These are documented in
# docs/guides/multi-agent-frameworks.md and docs/guides/mcp-tools.md instead.
NON_PROVIDER_KEYS = {
    "agents",
    "autogen",
    "autogen_agentchat",
    "bedrock_agents",
    "crewai",
    "dspy",
    "google_adk",
    "guardrails",
    "haystack",
    "instructor",
    "langchain",
    "langgraph",
    "litellm_latency",
    "llama_index",
    "mcp_client",
    "openai_agents",
    "pydantic_ai",
}

# Registry key -> the Provider cell in the llm-providers.md table.
PROVIDER_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "anyscale": "Anyscale",
    "azure.ai.inference": "Azure AI Inference",
    "azure.ai.openai": "Azure OpenAI",
    "boto3": "AWS Bedrock",
    "cohere": "Cohere",
    "cometapi": "CometAPI",
    "elevenlabs": "ElevenLabs",
    "google.generativeai": "Google AI",
    "groq": "Groq",
    "hyperbolic": "Hyperbolic",
    "liquid_audio": "Liquid Audio",
    "mistralai": "Mistral AI",
    "ollama": "Ollama",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "replicate": "Replicate",
    "sambanova": "SambaNova",
    "sarvamai": "Sarvam AI",
    "sentence_transformers": "Sentence Transformers",
    "together": "Together AI",
    "transformers": "HuggingFace Transformers",
    "vertexai": "Vertex AI",
}


def _documented_providers():
    """Provider names from the first markdown table in llm-providers.md."""
    names = []
    in_table = False
    for line in PROVIDERS_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Provider |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            names.append(line.split("|")[1].strip())
    return names


def test_every_registry_key_is_classified():
    """No instrumentor may be silently neither a provider nor a framework."""
    unclassified = set(INSTRUMENTORS) - NON_PROVIDER_KEYS - set(PROVIDER_DISPLAY_NAMES)
    assert not unclassified, (
        f"instrumentors not classified as provider or framework: {sorted(unclassified)}. "
        f"Add each to PROVIDER_DISPLAY_NAMES (and the docs table) or NON_PROVIDER_KEYS."
    )


def test_classification_does_not_invent_instrumentors():
    """The classification lists may not name instrumentors that no longer exist."""
    stale = (NON_PROVIDER_KEYS | set(PROVIDER_DISPLAY_NAMES)) - set(INSTRUMENTORS)
    assert not stale, f"classified keys with no instrumentor: {sorted(stale)}"


def test_every_provider_has_a_docs_table_row():
    documented = set(_documented_providers())
    missing = {key: name for key, name in PROVIDER_DISPLAY_NAMES.items() if name not in documented}
    assert (
        not missing
    ), f"providers in INSTRUMENTORS with no row in {PROVIDERS_DOC.name}: {sorted(missing.values())}"


def test_docs_table_has_no_rows_for_unknown_providers():
    documented = _documented_providers()
    known = set(PROVIDER_DISPLAY_NAMES.values())
    unknown = [name for name in documented if name not in known]
    assert not unknown, f"{PROVIDERS_DOC.name} documents providers with no instrumentor: {unknown}"


def test_docs_table_has_no_duplicate_rows():
    documented = _documented_providers()
    dupes = {name for name in documented if documented.count(name) > 1}
    assert not dupes, f"duplicate provider rows in {PROVIDERS_DOC.name}: {sorted(dupes)}"


@pytest.mark.parametrize(
    "path,pattern",
    [
        (INDEX_DOC, r"across (\d+)\+? LLM providers"),
        (INDEX_DOC, r"\*\*(\d+)\+? LLM Providers\*\*"),
        (INDEX_DOC, r"llm-providers\.md\) - (\d+)\+? providers"),
        (PROVIDERS_DOC, r"auto-instruments (\d+)\+? LLM providers"),
        (README, r"\| LLM providers \| (\d+)\+? \|"),
    ],
)
def test_headline_counts_match_the_table(path, pattern):
    """Every advertised provider count equals the number of rows in the table."""
    expected = len(_documented_providers())
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    assert match, f"could not find the provider count in {path.name} via {pattern!r}"
    assert (
        int(match.group(1)) == expected
    ), f"{path.name} advertises {match.group(1)} providers but the table lists {expected}"
