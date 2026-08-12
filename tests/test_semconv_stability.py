"""Semantic-convention naming policy: opt-in parsing, defaults, provider aliasing.

Two renames reach consumers of this library:

    gen_ai.usage.prompt_tokens     -> gen_ai.usage.input_tokens    (semconv v1.27.0)
    gen_ai.usage.completion_tokens -> gen_ai.usage.output_tokens   (semconv v1.27.0)
    gen_ai.system                  -> gen_ai.provider.name

Both fail the same silent way when they are handled as a *replacement*. Nothing
raises — the attribute is simply absent, so a consumer reads zero tokens, or reads
no provider and drops the span as "not a GenAI call". A wrong number that looks
measured survives review in a way an exception never would.

1.9.0 learned this the hard way: it replaced the token names rather than emitting
both, and every consumer still reading the superseded spellings silently zeroed.
These tests pin the correction — dual emission is the default, and the opt-in
string is parsed as the comma-separated multi-area list it actually is.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from genai_otel.config import OTelConfig
from genai_otel.semconv import genai_semconv_modes


class TestOptInParsing:
    """OTEL_SEMCONV_STABILITY_OPT_IN is a comma-separated list shared by ALL areas."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("gen_ai/dup", (True, True)),
            ("gen_ai", (True, False)),
            (" gen_ai , http ", (True, False)),
            ("http,gen_ai/dup", (True, True)),
        ],
    )
    def test_explicit_requests_are_honoured(self, raw: str, expected: tuple) -> None:
        assert genai_semconv_modes(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "http/dup", "database/dup"])
    def test_absent_or_unrelated_opt_in_defaults_to_dual(self, raw: str | None) -> None:
        """Unset, or set only for another area, must not change GenAI emission.

        The transitional default is dual: with the conventions still at
        Development status there is no stable tier to fall back to, and dropping
        the superseded names by default is what caused the 1.9.0 regression.
        """
        assert genai_semconv_modes(raw) == (True, True)

    def test_another_areas_dup_does_not_override_an_explicit_gen_ai_request(self) -> None:
        """The regression a substring check caused.

        ``"gen_ai,http/dup"`` means *current GenAI names only* plus an unrelated
        HTTP opt-in. ``"dup" in raw`` is True, so the old check dual-emitted GenAI
        names against an explicit request not to.
        """
        raw = "gen_ai,http/dup"
        assert "dup" in raw  # what the old substring check saw
        assert genai_semconv_modes(raw) == (True, False)

    def test_current_names_are_never_disabled(self) -> None:
        """No opt-in value may switch off the current spec names."""
        for raw in [None, "", "gen_ai", "gen_ai/dup", "http/dup", "nonsense"]:
            assert genai_semconv_modes(raw)[0] is True


class TestConfigDefault:
    def test_default_config_emits_both_spellings(self, monkeypatch) -> None:
        monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)
        assert genai_semconv_modes(OTelConfig().semconv_stability_opt_in) == (True, True)

    def test_env_var_still_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai")
        assert genai_semconv_modes(OTelConfig().semconv_stability_opt_in) == (True, False)


class _Aliaser:
    """Minimal stand-in exposing only what _with_provider_aliases needs."""

    def __init__(self, opt_in: str | None = None) -> None:
        self.config = MagicMock()
        self.config.semconv_stability_opt_in = opt_in

    from genai_otel.instrumentors.base import BaseInstrumentor as _B

    _with_provider_aliases = _B._with_provider_aliases


class TestProviderAliasing:
    def test_superseded_spelling_is_mirrored_onto_the_current_one(self) -> None:
        """The ~29 instrumentors all write gen_ai.system as a raw literal."""
        out = _Aliaser()._with_provider_aliases({"gen_ai.system": "anthropic"})
        assert out["gen_ai.provider.name"] == "anthropic"
        assert out["gen_ai.system"] == "anthropic"

    def test_current_spelling_is_mirrored_back_under_dual_emission(self) -> None:
        out = _Aliaser()._with_provider_aliases({"gen_ai.provider.name": "openai"})
        assert out["gen_ai.system"] == "openai"

    def test_current_only_mode_does_not_add_the_superseded_spelling(self) -> None:
        out = _Aliaser("gen_ai")._with_provider_aliases({"gen_ai.provider.name": "openai"})
        assert "gen_ai.system" not in out

    def test_current_only_mode_still_mirrors_forward(self) -> None:
        """An instrumentor writing the superseded literal must still yield the
        current name, or current-only mode would emit no provider at all."""
        out = _Aliaser("gen_ai")._with_provider_aliases({"gen_ai.system": "cohere"})
        assert out["gen_ai.provider.name"] == "cohere"

    def test_an_explicitly_set_value_is_never_overwritten(self) -> None:
        attrs = {"gen_ai.provider.name": "azure_openai", "gen_ai.system": "openai"}
        assert _Aliaser()._with_provider_aliases(attrs) == attrs

    def test_non_genai_attrs_are_returned_untouched(self) -> None:
        attrs = {"http.method": "GET"}
        assert _Aliaser()._with_provider_aliases(attrs) is attrs

    def test_the_callers_dict_is_not_mutated(self) -> None:
        """Instrumentors may reuse or cache the dict they return."""
        original = {"gen_ai.system": "anthropic"}
        _Aliaser()._with_provider_aliases(original)
        assert original == {"gen_ai.system": "anthropic"}
