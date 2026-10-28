"""Tests for the boto3 S3-compatible object storage instrumentor."""

import logging
import types
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors import s3_instrumentor
from genai_otel.mcp_instrumentors.s3_instrumentor import S3Instrumentor, _safe_server_address


@pytest.fixture(autouse=True)
def isolated_patch_registry():
    """Keep the module-level patch registry from leaking between tests."""
    saved = list(s3_instrumentor._PATCHED_TARGETS)
    s3_instrumentor._PATCHED_TARGETS.clear()
    yield
    s3_instrumentor._PATCHED_TARGETS[:] = saved


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def fake_botocore():
    """Install a stand-in botocore.client module, since boto3 is not a dev dependency."""
    original = MagicMock(return_value={"ResponseMetadata": {"HTTPStatusCode": 200}})

    class BaseClient:
        _make_api_call = original

    client_module = types.ModuleType("botocore.client")
    client_module.BaseClient = BaseClient
    botocore_module = types.ModuleType("botocore")
    botocore_module.client = client_module
    with patch.dict("sys.modules", {"botocore": botocore_module, "botocore.client": client_module}):
        yield BaseClient, original


def _instrumentor(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = S3Instrumentor(OTelConfig())
    instrumentor.tracer = provider.get_tracer("test.s3")
    return instrumentor


def _client(service_name="s3", endpoint_url="http://seaweedfs:8333"):
    client = MagicMock()
    client.meta.service_model.service_name = service_name
    client.meta.endpoint_url = endpoint_url
    return client


class TestSafeServerAddress:
    def test_returns_hostname_only(self):
        endpoint = "https://user:secret@seaweed.example.com:8333/bucket?X-Amz-Signature=abc"
        assert _safe_server_address(endpoint) == "seaweed.example.com"

    @pytest.mark.parametrize("endpoint", [None, 8333, b"http://seaweedfs:8333"])
    def test_non_string_endpoint(self, endpoint):
        assert _safe_server_address(endpoint) == ""

    def test_endpoint_without_hostname(self):
        assert _safe_server_address("not-a-url") == ""

    def test_malformed_endpoint(self):
        assert _safe_server_address("http://[::1") == ""


class TestS3Instrumentor:
    def test_init(self):
        config = OTelConfig()
        instrumentor = S3Instrumentor(config)
        assert instrumentor.config is config
        assert instrumentor.tracer is not None

    def test_instrument_missing_botocore(self, caplog):
        caplog.set_level(logging.DEBUG)
        with patch.dict("sys.modules", {"botocore": None, "botocore.client": None}):
            assert S3Instrumentor(OTelConfig()).instrument() is False
        assert "botocore not installed, skipping S3 instrumentation" in caplog.text

    def test_uninstrument_missing_botocore(self):
        with patch.dict("sys.modules", {"botocore": None, "botocore.client": None}):
            assert S3Instrumentor(OTelConfig()).uninstrument() is False

    def test_s3_call_creates_client_span(self, exporter, fake_botocore, caplog):
        base_client, original = fake_botocore
        caplog.set_level(logging.INFO)

        assert _instrumentor(exporter).instrument() is True
        assert "S3-compatible object storage instrumentation enabled" in caplog.text
        assert base_client._make_api_call is not original

        client = _client(endpoint_url="https://user:secret@seaweed.example.com:8333/path")
        params = {"Bucket": "media", "Key": "images/cat.png", "Body": b"secret-bytes"}
        result = base_client._make_api_call(client, "PutObject", params, "extra", flag=True)

        assert result == {"ResponseMetadata": {"HTTPStatusCode": 200}}
        original.assert_called_once_with(client, "PutObject", params, "extra", flag=True)

        (span,) = exporter.get_finished_spans()
        assert span.name == "s3.PutObject"
        assert span.kind == SpanKind.CLIENT
        assert span.status.status_code == StatusCode.OK
        assert dict(span.attributes) == {
            "db.system": "s3",
            "db.operation": "PutObject",
            "server.address": "seaweed.example.com",
            "s3.bucket": "media",
            "s3.object": "images/cat.png",
        }

    def test_bucket_without_key(self, exporter, fake_botocore):
        base_client, _ = fake_botocore
        _instrumentor(exporter).instrument()

        base_client._make_api_call(_client(), "ListObjectsV2", {"Bucket": "media"})

        (span,) = exporter.get_finished_spans()
        assert span.attributes["s3.bucket"] == "media"
        assert "s3.object" not in span.attributes

    def test_non_dict_params_and_missing_endpoint(self, exporter, fake_botocore):
        base_client, _ = fake_botocore
        _instrumentor(exporter).instrument()

        base_client._make_api_call(_client(endpoint_url=None), "ListBuckets", None)

        (span,) = exporter.get_finished_spans()
        assert dict(span.attributes) == {"db.system": "s3", "db.operation": "ListBuckets"}

    @pytest.mark.parametrize(
        "client",
        [
            _client(service_name="dynamodb"),
            types.SimpleNamespace(),  # no meta
            types.SimpleNamespace(meta=types.SimpleNamespace()),  # no service_model
        ],
    )
    def test_non_s3_calls_pass_through_untraced(self, exporter, fake_botocore, client):
        base_client, original = fake_botocore
        _instrumentor(exporter).instrument()

        result = base_client._make_api_call(client, "GetItem", {"TableName": "t"})

        assert result == original.return_value
        original.assert_called_once_with(client, "GetItem", {"TableName": "t"})
        assert exporter.get_finished_spans() == ()

    def test_error_records_type_without_message(self, exporter, fake_botocore):
        base_client, original = fake_botocore
        original.side_effect = PermissionError("AccessDenied for key=secret-token")
        _instrumentor(exporter).instrument()

        with pytest.raises(PermissionError, match="secret-token"):
            base_client._make_api_call(_client(), "GetObject", {"Bucket": "b", "Key": "k"})

        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "PermissionError"
        assert span.attributes["error.type"] == "PermissionError"
        assert "secret-token" not in str(dict(span.attributes))
        assert span.events == ()

    def test_instrument_twice_does_not_double_wrap(self, exporter, fake_botocore, caplog):
        base_client, _ = fake_botocore
        _instrumentor(exporter).instrument()
        wrapper = base_client._make_api_call
        caplog.set_level(logging.DEBUG)

        assert _instrumentor(exporter).instrument() is True

        assert base_client._make_api_call is wrapper
        assert len(s3_instrumentor._PATCHED_TARGETS) == 1
        assert "already instrumented" in caplog.text
        base_client._make_api_call(_client(), "HeadObject", {"Bucket": "b", "Key": "k"})
        assert len(exporter.get_finished_spans()) == 1

    def test_uninstrument_restores_original(self, exporter, fake_botocore, caplog):
        base_client, original = fake_botocore
        instrumentor = _instrumentor(exporter)
        instrumentor.instrument()
        caplog.set_level(logging.INFO)

        assert instrumentor.uninstrument() is True

        assert base_client._make_api_call is original
        assert s3_instrumentor._PATCHED_TARGETS == []
        assert "S3-compatible object storage instrumentation removed" in caplog.text
        assert instrumentor.uninstrument() is False

    def test_uninstrument_leaves_later_patch_in_place(self, exporter, fake_botocore):
        base_client, _ = fake_botocore
        instrumentor = _instrumentor(exporter)
        instrumentor.instrument()
        later_patch = MagicMock()
        base_client._make_api_call = later_patch

        assert instrumentor.uninstrument() is False

        assert base_client._make_api_call is later_patch
        assert s3_instrumentor._PATCHED_TARGETS == []

    def test_real_botocore_client(self, exporter):
        """Drive a real boto3-style S3 client end to end when botocore is installed."""
        botocore_session = pytest.importorskip("botocore.session")
        stub = pytest.importorskip("botocore.stub")

        instrumentor = _instrumentor(exporter)
        assert instrumentor.instrument() is True
        try:
            client = botocore_session.get_session().create_client(
                "s3",
                endpoint_url="http://seaweedfs:8333",
                region_name="us-east-1",
                aws_access_key_id="test",
                aws_secret_access_key="test",
            )
            params = {"Bucket": "media", "Key": "a.txt", "Body": b"secret-bytes"}
            with stub.Stubber(client) as stubber:
                stubber.add_response("put_object", {"ETag": '"abc"'}, params)
                assert client.put_object(**params)["ETag"] == '"abc"'
        finally:
            instrumentor.uninstrument()

        (span,) = exporter.get_finished_spans()
        assert span.name == "s3.PutObject"
        assert span.attributes["server.address"] == "seaweedfs"
        assert span.attributes["s3.bucket"] == "media"
        assert span.attributes["s3.object"] == "a.txt"
        assert "secret-bytes" not in str(dict(span.attributes))
