"""Tests for configuration module."""

import os
import pytest
from genai_otel.config import OTelConfig


class TestOTelConfig:
    """Test cases for OTelConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OTelConfig()

        assert config.service_name == os.getenv("OTEL_SERVICE_NAME", "genai-app")
        assert config.endpoint == os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        assert config.enable_gpu_metrics == (
            os.getenv("GENAI_ENABLE_GPU_METRICS", "true").lower() == "true"
        )
        assert config.enable_cost_tracking == (
            os.getenv("GENAI_ENABLE_COST_TRACKING", "true").lower() == "true"
        )
        assert config.enable_mcp_instrumentation == (
            os.getenv("GENAI_ENABLE_MCP_INSTRUMENTATION", "true").lower() == "true"
        )
        assert config.fail_on_error == (os.getenv("GENAI_FAIL_ON_ERROR", "false").lower() == "true")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OTelConfig(
            service_name="test-service",
            endpoint="http://test:4318",
            enable_gpu_metrics=False,
            enable_cost_tracking=False,
            enable_mcp_instrumentation=False,
            fail_on_error=True,
        )

        assert config.service_name == "test-service"
        assert config.endpoint == "http://test:4318"
        assert config.enable_gpu_metrics is False
        assert config.enable_cost_tracking is False
        assert config.enable_mcp_instrumentation is False
        assert config.fail_on_error is True

    def test_headers_parsing(self, monkeypatch):
        """Test OTLP headers parsing from environment."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "key1=value1,key2=value2")
        config = OTelConfig()

        assert config.headers is not None
        assert config.headers["key1"] == "value1"
        assert config.headers["key2"] == "value2"

    def test_invalid_headers_parsing(self, monkeypatch, caplog):
        """Test handling of invalid headers format."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "invalid_format")
        config = OTelConfig()

        # Should log an error but not crash
        assert "Failed to parse OTEL_EXPORTER_OTLP_HEADERS" in caplog.text

    def test_env_override(self, monkeypatch):
        """Test that environment variables are used as defaults."""
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env:4318")
        monkeypatch.setenv("GENAI_ENABLE_GPU_METRICS", "false")

        config = OTelConfig()

        assert config.service_name == "env-service"
        assert config.endpoint == "http://env:4318"
        assert config.enable_gpu_metrics is False

    def test_kwarg_override(self, monkeypatch):
        """Test that kwargs override environment variables."""
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")

        config = OTelConfig(service_name="kwarg-service")

        assert config.service_name == "kwarg-service"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
