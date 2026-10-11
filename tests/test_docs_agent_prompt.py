"""The coding-agent prompt is published twice and must say the same thing in both.

It lives on a docs page and, collapsed, in the README. People paste it as is, so a
copy that has drifted is a set of wrong instructions with nothing to flag it.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS_PAGE = ROOT / "docs" / "getting-started" / "coding-agents.md"
README = ROOT / "README.md"


def _prompt_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index("```text\n") + len("```text\n")
    return text[start : text.index("\n```\n", start)]


def test_readme_and_docs_carry_the_same_prompt():
    assert _prompt_block(README) == _prompt_block(DOCS_PAGE)


def test_prompt_names_the_package_and_the_two_lines_to_add():
    prompt = _prompt_block(DOCS_PAGE)

    assert "genai-otel-instrument" in prompt
    assert "import genai_otel" in prompt
    assert "genai_otel.instrument()" in prompt


def test_prompt_only_names_extras_the_package_defines():
    """An extra the package does not define makes the agent's install line fail."""
    try:
        import tomllib
    except ImportError:  # Python < 3.11 has no TOML parser in the standard library
        tomllib = pytest.importorskip("tomli")

    with open(ROOT / "pyproject.toml", "rb") as handle:
        extras = set(tomllib.load(handle)["project"]["optional-dependencies"])

    prompt = _prompt_block(DOCS_PAGE)
    line = next(line for line in prompt.splitlines() if "Extras are named after" in line)
    following = prompt[prompt.index(line) :].split("\n\n")[0]
    named = following.split("library:")[1].split("and others")[0]
    for extra in (name.strip(" ,\n") for name in named.replace("\n", " ").split(",")):
        if extra:
            assert extra in extras, extra
    assert {"openai", "langgraph"} <= extras
