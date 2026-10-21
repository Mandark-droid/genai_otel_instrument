"""Decision-model settings: GENAI_DECISION_* environment variables."""

from genai_otel.config import OTelConfig


class TestConfig:
    def test_defaults(self, monkeypatch):
        for name in (
            "GENAI_DECISION_LEGACY_LLM_KIND",
            "GENAI_DECISION_HTTP_MATCHER",
            "GENAI_DECISION_PROVIDER",
        ):
            monkeypatch.delenv(name, raising=False)
        config = OTelConfig()
        assert config.decision_legacy_llm_kind is False
        assert config.decision_http_matcher is True
        assert config.decision_provider is None

    def test_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("GENAI_DECISION_LEGACY_LLM_KIND", "TRUE")
        monkeypatch.setenv("GENAI_DECISION_HTTP_MATCHER", "false")
        monkeypatch.setenv("GENAI_DECISION_PROVIDER", " nebius ")
        config = OTelConfig()
        assert config.decision_legacy_llm_kind is True
        assert config.decision_http_matcher is False
        assert config.decision_provider == "nebius"
