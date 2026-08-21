"""The carbon country code must fail loudly, not silently emit a global average.

codecarbon's offline dataset is keyed by THREE-letter ISO codes. Given "IN" it logs
an error and then quietly falls back to 475.0 gCO2e/kWh -- which is byte-identical
to this library's own manual fallback constant. Measured:

    IN  -> 475.0 gCO2e/kWh   (unsupported; codecarbon's global default)
    IND -> 713.4 gCO2e/kWh   (the real India grid factor)
    USA -> 369.5
    FRA ->  56.0

So an operator who sets the intuitive two-letter code gets numbers identical to
having codecarbon switched off, labelled `source: codecarbon`, and concludes the
integration does nothing. An error log plus a plausible number is the worst
possible combination: it is indistinguishable from success.
"""

from __future__ import annotations

import pytest

from genai_otel.gpu_metrics import normalize_country_iso_code


class TestTwoLetterCodesAreRepaired:
    """The common mistake is a 2-letter code. Repair it rather than punish it."""

    @pytest.mark.parametrize(
        "given,expected",
        [("IN", "IND"), ("in", "IND"), ("US", "USA"), ("FR", "FRA"), ("GB", "GBR"), ("DE", "DEU")],
    )
    def test_alpha2_is_converted_to_alpha3(self, given, expected) -> None:
        assert normalize_country_iso_code(given) == expected


class TestThreeLetterCodesPassThrough:
    @pytest.mark.parametrize("code", ["IND", "USA", "FRA", "GBR", "DEU"])
    def test_valid_alpha3_is_unchanged(self, code) -> None:
        assert normalize_country_iso_code(code) == code

    def test_lowercase_alpha3_is_upcased(self) -> None:
        assert normalize_country_iso_code("ind") == "IND"


class TestUnusableCodesRaiseRatherThanFallBack:
    """A code codecarbon cannot use must stop the integration, not silently
    produce the same number as having no integration at all."""

    @pytest.mark.parametrize("bad", ["XX", "ZZZ", "QQ", "NOTACOUNTRY", "1N"])
    def test_unknown_code_raises(self, bad) -> None:
        with pytest.raises(ValueError) as exc:
            normalize_country_iso_code(bad)
        assert bad.upper() in str(exc.value) or "ISO" in str(exc.value)

    def test_the_error_names_the_consequence(self) -> None:
        """An operator reading the log must learn WHY it matters."""
        with pytest.raises(ValueError) as exc:
            normalize_country_iso_code("XX")
        message = str(exc.value).lower()
        assert "3-letter" in message or "three-letter" in message

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_empty_is_not_an_error(self, empty) -> None:
        """Unset is a legitimate state -- the caller decides the default."""
        assert normalize_country_iso_code(empty) is None
