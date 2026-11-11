"""SeaweedFS: endpoint matching, S3-gateway labelling, Filer tracing, async S3, suppression."""

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors import s3_instrumentor, seaweedfs
from genai_otel.mcp_instrumentors.s3_instrumentor import S3Instrumentor
from genai_otel.mcp_instrumentors.seaweedfs import (
    SeaweedFilerInstrumentor,
    classify_operation,
    declared_endpoints,
    is_declared,
    parse_endpoints,
    path_prefix,
)
from genai_otel.media.suppress import is_suppressed


class _Cfg:
    def __init__(self, **kw):
        self.enable_seaweedfs_instrumentation = True
        self.seaweedfs_endpoints = None
        self.seaweedfs_path_prefix_segments = 2
        self.media_store = "none"
        self.media_store_endpoint = None
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def clean_registries():
    saved = (
        list(s3_instrumentor._PATCHED_TARGETS),
        list(s3_instrumentor._PATCHED_ASYNC_TARGETS),
        list(seaweedfs._PATCHED),
    )
    s3_instrumentor._PATCHED_TARGETS.clear()
    s3_instrumentor._PATCHED_ASYNC_TARGETS.clear()
    seaweedfs._PATCHED.clear()
    yield
    for owner, attr, original, wrapper in list(seaweedfs._PATCHED):
        if getattr(owner, attr, None) is wrapper:
            setattr(owner, attr, original)
    s3_instrumentor._PATCHED_TARGETS[:] = saved[0]
    s3_instrumentor._PATCHED_ASYNC_TARGETS[:] = saved[1]
    seaweedfs._PATCHED[:] = saved[2]


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


def _provider(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


# ---------------------------------------------------------------------------
# Endpoint parsing and matching
# ---------------------------------------------------------------------------
class TestEndpoints:
    def test_parse_forms(self):
        eps = parse_endpoints("Filer.Internal:8888, https://s3.internal, bare-host, *.corp:8333")
        assert ("filer.internal", 8888) in eps
        assert ("s3.internal", 443) in eps
        assert ("bare-host", None) in eps
        assert ("*.corp", 8333) in eps

    def test_parse_drops_garbage_and_duplicates(self):
        assert parse_endpoints("", None, " , ", "h:99999999") == []
        assert parse_endpoints("a:1,a:1") == [("a", 1)]

    def test_exact_match_requires_port(self):
        eps = parse_endpoints("filer.internal:8888")
        assert is_declared(eps, "filer.internal", 8888)
        assert not is_declared(eps, "filer.internal", 9999)
        assert not is_declared(eps, "other.internal", 8888)

    def test_bare_host_matches_any_port_and_wildcard_matches_suffix(self):
        eps = parse_endpoints("seaweed.internal,*.store.corp:8333")
        assert is_declared(eps, "seaweed.internal", 1234)
        assert is_declared(eps, "a.store.corp", 8333)
        assert not is_declared(eps, "store.corp", 8333)

    def test_no_port_heuristic(self):
        assert not is_declared([], "localhost", 8333)
        assert not is_declared(parse_endpoints("x:1"), "localhost", 8888)

    def test_media_store_seaweed_endpoint_is_auto_declared(self):
        cfg = _Cfg(media_store="seaweed", media_store_endpoint="https://sw.internal:8333")
        assert ("sw.internal", 8333) in declared_endpoints(cfg)

    def test_other_media_stores_are_not_declared(self):
        cfg = _Cfg(media_store="minio", media_store_endpoint="https://m.internal:9000")
        assert declared_endpoints(cfg) == []

    def test_config_reads_env(self, monkeypatch):
        monkeypatch.setenv("GENAI_OTEL_SEAWEEDFS_ENDPOINTS", "f.internal:8888")
        monkeypatch.setenv("GENAI_OTEL_MEDIA_STORE_TRACE_CALLS", "true")
        cfg = OTelConfig()
        assert cfg.seaweedfs_endpoints == "f.internal:8888"
        assert cfg.media_store_trace_calls is True
        assert cfg.enable_seaweedfs_instrumentation is True
        assert cfg.seaweedfs_path_prefix_segments == 2


# ---------------------------------------------------------------------------
# Operation classification and path privacy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method,path,query,expected",
    [
        ("POST", "/b/file.bin", "", "upload"),
        ("PUT", "/b/file.bin", "", "upload"),
        ("POST", "/b/file.bin", "op=append", "append"),
        ("POST", "/b/dir/", "", "mkdir"),
        ("GET", "/b/file.bin", "", "download"),
        ("HEAD", "/b/file.bin", "", "download"),
        ("GET", "/b/dir/", "limit=10", "list"),
        ("GET", "/b/file.bin", "metadata=true", "stat"),
        ("DELETE", "/b/file.bin", "", "delete"),
        ("DELETE", "/b/dir", "recursive=true", "delete_recursive"),
        ("POST", "/new", "mv.from=/old", "move"),
        ("POST", "/dst", "cp.from=/src", "copy"),
        ("PUT", "/b/file.bin", "tagging", "tag"),
        ("OPTIONS", "/b", "", "other"),
    ],
)
def test_classify_operation(method, path, query, expected):
    assert classify_operation(method, path, query) == expected


def test_path_prefix_truncates():
    assert path_prefix("/buckets/tenant/secret/file.pdf", 2) == "/buckets/tenant"
    assert path_prefix("/", 2) == "/"
    assert path_prefix("/a/b", 0) == ""


# ---------------------------------------------------------------------------
# S3 gateway labelling
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_botocore():
    original = MagicMock(
        return_value={
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {"x-amz-request-id": "req-123"},
            }
        }
    )

    class BaseClient:
        _make_api_call = original

    client_module = types.ModuleType("botocore.client")
    client_module.BaseClient = BaseClient
    botocore_module = types.ModuleType("botocore")
    botocore_module.client = client_module
    with patch.dict("sys.modules", {"botocore": botocore_module, "botocore.client": client_module}):
        yield BaseClient, original


def _client(endpoint_url):
    client = MagicMock()
    client.meta.service_model.service_name = "s3"
    client.meta.endpoint_url = endpoint_url
    return client


def _s3(exporter, cfg):
    inst = S3Instrumentor(cfg)
    inst.tracer = _provider(exporter).get_tracer("t")
    return inst


class TestS3Labelling:
    def test_declared_endpoint_gets_system_name_and_port(self, exporter, fake_botocore):
        BaseClient, _ = fake_botocore
        inst = _s3(exporter, _Cfg(seaweedfs_endpoints="sw.internal:8333"))
        assert inst.instrument()
        BaseClient._make_api_call(
            _client("http://sw.internal:8333"), "PutObject", {"Bucket": "b", "Key": "k"}
        )
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["db.system"] == "s3"
        assert attrs["db.system.name"] == "seaweedfs"
        assert attrs["server.port"] == 8333
        assert attrs["server.address"] == "sw.internal"
        assert attrs["aws.request_id"] == "req-123"

    def test_undeclared_endpoint_is_not_labelled(self, exporter, fake_botocore):
        BaseClient, _ = fake_botocore
        _s3(exporter, _Cfg(seaweedfs_endpoints="sw.internal:8333")).instrument()
        BaseClient._make_api_call(_client("https://s3.amazonaws.com"), "ListBuckets", {})
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert "db.system.name" not in attrs
        assert attrs["server.port"] == 443

    def test_standard_and_legacy_attribute_names_both_present(self, exporter, fake_botocore):
        BaseClient, _ = fake_botocore
        _s3(exporter, _Cfg()).instrument()
        BaseClient._make_api_call(_client("http://h:1"), "GetObject", {"Bucket": "b", "Key": "k"})
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["s3.bucket"] == attrs["aws.s3.bucket"] == "b"
        assert attrs["s3.object"] == attrs["aws.s3.key"] == "k"

    def test_suppressed_calls_are_not_traced(self, exporter, fake_botocore):
        BaseClient, original = fake_botocore
        _s3(exporter, _Cfg()).instrument()
        from genai_otel.media.suppress import suppress_instrumentation

        with suppress_instrumentation():
            BaseClient._make_api_call(_client("http://h:1"), "PutObject", {"Bucket": "b"})
        assert exporter.get_finished_spans() == ()
        assert original.call_count == 1


# ---------------------------------------------------------------------------
# Async S3 (aiobotocore) -- stand-in module, no network, no real dependency
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_aiobotocore():
    calls = []

    class AioBaseClient:
        async def _make_api_call(self, operation_name, api_params):
            calls.append(operation_name)
            if operation_name == "Boom":
                raise ConnectionError("secret endpoint detail")
            return {"ResponseMetadata": {"HTTPHeaders": {"x-amz-request-id": "async-1"}}}

    module = types.ModuleType("aiobotocore.client")
    module.AioBaseClient = AioBaseClient
    package = types.ModuleType("aiobotocore")
    package.client = module
    with patch.dict(sys.modules, {"aiobotocore": package, "aiobotocore.client": module}):
        yield AioBaseClient, calls


class TestAsyncS3:
    def test_async_calls_are_traced(self, exporter, fake_aiobotocore):
        AioBaseClient, calls = fake_aiobotocore
        inst = _s3(exporter, _Cfg(seaweedfs_endpoints="sw.internal:8333"))
        assert inst._instrument_async()
        client = AioBaseClient()
        client.meta = _client("http://sw.internal:8333").meta
        asyncio.run(client._make_api_call("PutObject", {"Bucket": "b", "Key": "k"}))
        span = exporter.get_finished_spans()[0]
        assert span.name == "s3.PutObject"
        assert span.attributes["db.system.name"] == "seaweedfs"
        assert span.attributes["aws.request_id"] == "async-1"
        assert calls == ["PutObject"]

    def test_async_error_records_type_only(self, exporter, fake_aiobotocore):
        AioBaseClient, _ = fake_aiobotocore
        _s3(exporter, _Cfg())._instrument_async()
        client = AioBaseClient()
        client.meta = _client("http://h:1").meta
        with pytest.raises(ConnectionError):
            asyncio.run(client._make_api_call("Boom", {}))
        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "ConnectionError"
        assert "secret" not in str(span.status.description)

    def test_async_suppressed_and_non_s3_pass_through(self, exporter, fake_aiobotocore):
        AioBaseClient, calls = fake_aiobotocore
        _s3(exporter, _Cfg())._instrument_async()
        client = AioBaseClient()
        client.meta = _client("http://h:1").meta
        from genai_otel.media.suppress import suppress_instrumentation

        async def run():
            with suppress_instrumentation():
                await client._make_api_call("PutObject", {})
            client.meta.service_model.service_name = "sqs"
            await client._make_api_call("SendMessage", {})

        asyncio.run(run())
        assert exporter.get_finished_spans() == ()
        assert calls == ["PutObject", "SendMessage"]

    def test_double_instrument_and_uninstrument(self, fake_aiobotocore):
        AioBaseClient, _ = fake_aiobotocore
        original = AioBaseClient._make_api_call
        inst = S3Instrumentor(_Cfg())
        assert inst._instrument_async()
        wrapper = AioBaseClient._make_api_call
        assert inst._instrument_async()
        assert AioBaseClient._make_api_call is wrapper
        assert inst.uninstrument()
        assert AioBaseClient._make_api_call is original

    def test_missing_aiobotocore_is_skipped(self):
        with patch.dict(sys.modules, {"aiobotocore": None, "aiobotocore.client": None}):
            assert S3Instrumentor(_Cfg())._instrument_async() is False


def test_real_aiobotocore_wrapper_installs_and_restores():
    aiobotocore_client = pytest.importorskip("aiobotocore.client")
    original = aiobotocore_client.AioBaseClient._make_api_call
    inst = S3Instrumentor(_Cfg())
    try:
        assert inst._instrument_async()
        assert aiobotocore_client.AioBaseClient._make_api_call is not original
    finally:
        inst.uninstrument()
    assert aiobotocore_client.AioBaseClient._make_api_call is original


# ---------------------------------------------------------------------------
# Media-store suppression
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_boto3_store():
    seen = {}
    client = MagicMock()
    client.create_bucket.side_effect = lambda **kw: seen.setdefault("create", is_suppressed())
    client.put_object.side_effect = lambda **kw: seen.setdefault("put", is_suppressed())
    boto3 = MagicMock()
    boto3.client.return_value = client
    with patch.dict(
        sys.modules,
        {"boto3": boto3, "botocore": MagicMock(), "botocore.client": MagicMock()},
    ):
        yield seen


def test_media_store_calls_are_suppressed_by_default(fake_boto3_store):
    from genai_otel.media.stores.s3_minio import S3MinioStore

    store = S3MinioStore(endpoint="https://sw.internal:8333", bucket="b")
    store.put(b"x", key="k", mime_type="image/png")
    assert fake_boto3_store == {"create": True, "put": True}
    assert not is_suppressed()


def test_media_store_trace_calls_opt_back_in(fake_boto3_store):
    from genai_otel.media.stores.s3_minio import S3MinioStore

    store = S3MinioStore(endpoint="https://sw.internal:8333", bucket="b", trace_calls=True)
    store.put(b"x", key="k", mime_type="image/png")
    assert fake_boto3_store == {"create": False, "put": False}


def test_media_upload_leaves_user_s3_calls_traced(exporter, fake_botocore):
    """Suppression is scoped to the SDK's upload; the user's own call is still a span."""
    BaseClient, _ = fake_botocore
    sys.modules["botocore.client"].Config = MagicMock()
    boto3 = MagicMock()
    _s3(exporter, _Cfg()).instrument()
    from genai_otel.media.stores.s3_minio import S3MinioStore

    with patch.dict(sys.modules, {"boto3": boto3}):
        S3MinioStore(endpoint="https://sw.internal:8333", bucket="b").put(
            b"x", key="k", mime_type="image/png"
        )
    BaseClient._make_api_call(_client("http://h:1"), "GetObject", {"Bucket": "b", "Key": "k"})
    assert [s.name for s in exporter.get_finished_spans()] == ["s3.GetObject"]


# ---------------------------------------------------------------------------
# Filer instrumentor
# ---------------------------------------------------------------------------
def _filer(exporter, **kw):
    cfg = _Cfg(seaweedfs_endpoints="filer.test:8888", **kw)
    inst = SeaweedFilerInstrumentor(cfg)
    inst.tracer = _provider(exporter).get_tracer("t")
    return inst


def test_filer_inert_without_endpoints_or_when_disabled():
    assert SeaweedFilerInstrumentor(_Cfg()).instrument() is False
    off = _Cfg(seaweedfs_endpoints="f:1", enable_seaweedfs_instrumentation=False)
    assert SeaweedFilerInstrumentor(off).instrument() is False
    assert seaweedfs._PATCHED == []


def test_requests_is_never_patched():
    requests = pytest.importorskip("requests")
    before = requests.Session.send
    SeaweedFilerInstrumentor(_Cfg(seaweedfs_endpoints="filer.test:8888")).instrument()
    assert requests.Session.send is before


class TestFilerHttpx:
    @pytest.fixture
    def httpx(self):
        return pytest.importorskip("httpx")

    def _client(self, httpx, status=200, body=b"SECRET-FILE-CONTENT"):
        def handler(request):
            return httpx.Response(status, content=body)

        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_upload_span_shape_and_privacy(self, httpx, exporter):
        inst = _filer(exporter)
        assert inst.instrument()
        with self._client(httpx, 201) as client:
            client.post(
                "http://filer.test:8888/buckets/tenant-a/private/report.pdf?Seaweed-Owner=alice",
                content=b"PAYLOAD-BYTES",
            )
        span = exporter.get_finished_spans()[0]
        assert span.name == "seaweedfs.filer.upload"
        a = dict(span.attributes)
        assert a["db.system.name"] == "seaweedfs"
        assert a["db.operation.name"] == "upload"
        assert a["server.address"] == "filer.test"
        assert a["server.port"] == 8888
        assert a["http.response.status_code"] == 201
        assert a["url.path.prefix"] == "/buckets/tenant-a"
        assert a["seaweedfs.request.size"] == len(b"PAYLOAD-BYTES")
        blob = repr(a) + repr(span.status.description)
        for secret in ("report.pdf", "private", "alice", "PAYLOAD", "SECRET-FILE"):
            assert secret not in blob

    def test_undeclared_host_is_not_traced(self, httpx, exporter):
        _filer(exporter).instrument()
        with self._client(httpx) as client:
            client.get("http://other.test:8888/x")
            client.get("http://filer.test:9999/x")
        assert exporter.get_finished_spans() == ()

    def test_404_on_download_is_not_an_error_but_500_is(self, httpx, exporter):
        _filer(exporter).instrument()
        with self._client(httpx, 404) as client:
            client.get("http://filer.test:8888/a/b.bin")
        with self._client(httpx, 500) as client:
            client.get("http://filer.test:8888/a/b.bin")
        missing, failed = exporter.get_finished_spans()
        assert missing.status.status_code == StatusCode.UNSET
        assert failed.status.status_code == StatusCode.ERROR
        assert failed.attributes["error.type"] == "500"

    def test_transport_error_records_type_only(self, httpx, exporter):
        _filer(exporter).instrument()

        def boom(request):
            raise httpx.ConnectError("secret internal detail")

        with httpx.Client(transport=httpx.MockTransport(boom)) as client:
            with pytest.raises(httpx.ConnectError):
                client.get("http://filer.test:8888/a")
        span = exporter.get_finished_spans()[0]
        assert span.attributes["error.type"] == "ConnectError"
        assert "secret" not in str(span.status.description)

    def test_async_client_is_traced(self, httpx, exporter):
        _filer(exporter).instrument()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x"))
            ) as client:
                await client.delete("http://filer.test:8888/dir?recursive=true")

        asyncio.run(run())
        assert exporter.get_finished_spans()[0].name == "seaweedfs.filer.delete_recursive"

    def test_uninstrument_restores_httpx(self, httpx, exporter):
        original = httpx.Client._send_single_request
        inst = _filer(exporter)
        inst.instrument()
        assert httpx.Client._send_single_request is not original
        assert inst.uninstrument()
        assert httpx.Client._send_single_request is original

    def test_instrument_twice_does_not_double_wrap(self, httpx, exporter):
        inst = _filer(exporter)
        inst.instrument()
        wrapped = httpx.Client._send_single_request
        assert inst.instrument()
        assert httpx.Client._send_single_request is wrapped


class TestFilerAiohttp:
    def test_aiohttp_filer_calls_are_traced(self, exporter):
        aiohttp = pytest.importorskip("aiohttp")
        web = pytest.importorskip("aiohttp.web")
        test_utils = pytest.importorskip("aiohttp.test_utils")

        async def handler(request):
            return web.Response(status=201, body=b"ok")

        async def run():
            app = web.Application()
            app.router.add_route("*", "/{tail:.*}", handler)
            server = test_utils.TestServer(app, host="127.0.0.1")
            await server.start_server()
            try:
                cfg = _Cfg(seaweedfs_endpoints="127.0.0.1")
                inst = SeaweedFilerInstrumentor(cfg)
                inst.tracer = _provider(exporter).get_tracer("t")
                assert inst.instrument()
                async with aiohttp.ClientSession() as session:
                    async with session.put(f"http://127.0.0.1:{server.port}/buckets/t/f.bin?x=1"):
                        pass
                inst.uninstrument()
                return server.port
            finally:
                await server.close()

        port = asyncio.run(run())
        span = exporter.get_finished_spans()[0]
        assert span.name == "seaweedfs.filer.upload"
        assert span.attributes["server.port"] == port
        assert span.attributes["http.response.status_code"] == 201
        assert span.attributes["url.path.prefix"] == "/buckets/t"
        assert "f.bin" not in repr(dict(span.attributes))
