"""The API instrumentor must RETURN THE RESPONSE, not the wrapper.

`create_span_wrapper` returns a `@wrapt.decorator`. Calling a wrapt decorator with
`(wrapped, instance, args, kwargs)` does not execute anything — it returns a
`FunctionWrapper`. So `httpx.Client.request` returned a wrapper object, and every caller
doing `resp.status_code` got:

    AttributeError: 'function' object has no attribute 'status_code'

Only `httpx.Client.request` is wrapped, which is why `send()` and `AsyncClient` were
unaffected and why this hid for so long: almost all callers use the async client. The
sync ones failed quietly wherever the exception was swallowed — in one case serving
stale on-disk prompts for months while a newer version sat in a central store.

These tests assert behaviour a type annotation cannot: that what comes back is the real
return value of the wrapped call.
"""

from unittest.mock import MagicMock

import pytest

wrapt = pytest.importorskip("wrapt")

from genai_otel.mcp_instrumentors.api_instrumentor import APIInstrumentor  # noqa: E402


class _Sentinel:
    """Stands in for an httpx.Response — the thing the caller must get back."""

    status_code = 200

    def json(self):
        return {"ok": True}


def _instrumentor():
    from genai_otel.config import OTelConfig

    inst = APIInstrumentor(OTelConfig(service_name="test-api-instrumentor"))
    # create_span_wrapper short-circuits straight to the wrapped call unless this is set,
    # and that short-circuit is not the path under test — the bug lived in the span path.
    inst._instrumented = True
    return inst


def test_wrap_api_call_returns_the_wrapped_calls_result_not_a_wrapper():
    inst = _instrumentor()
    sentinel = _Sentinel()

    def wrapped(method, url, **kwargs):
        return sentinel

    result = inst._wrap_api_call(wrapped, None, ("GET", "http://example.test/x"), {})

    assert result is sentinel, f"expected the wrapped call's return value, got {type(result).__name__}"
    # The regression in one line: this is the attribute access that used to blow up.
    assert result.status_code == 200


def test_wrapped_call_is_invoked_exactly_once():
    """A wrapper that re-invokes would double every outbound request — real money on
    LLM endpoints, and duplicated side effects everywhere else."""
    inst = _instrumentor()
    calls = []

    def wrapped(method, url, **kwargs):
        calls.append((method, url))
        return _Sentinel()

    inst._wrap_api_call(wrapped, None, ("POST", "http://example.test/y"), {})
    assert len(calls) == 1, f"wrapped call invoked {len(calls)} times, expected 1"


def test_arguments_reach_the_wrapped_call_unchanged():
    inst = _instrumentor()
    seen = {}

    def wrapped(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _Sentinel()

    inst._wrap_api_call(
        wrapped, None, ("GET", "http://example.test/z"), {"timeout": 5.0, "headers": {"a": "b"}}
    )
    assert seen["args"] == ("GET", "http://example.test/z")
    assert seen["kwargs"] == {"timeout": 5.0, "headers": {"a": "b"}}


def test_keyword_style_method_and_url_still_work():
    """httpx callers may pass method/url as keywords; span naming reads them either way."""
    inst = _instrumentor()
    sentinel = _Sentinel()
    result = inst._wrap_api_call(
        lambda **kw: sentinel, None, (), {"method": "get", "url": "http://example.test/kw"}
    )
    assert result is sentinel


def test_real_httpx_client_still_returns_a_response_after_instrumentation():
    """End-to-end guard against the exact production symptom.

    Instruments the real `httpx.Client.request` and asserts a plain `.get()` still yields
    a Response. Uses MockTransport so no network is touched. This is the test that would
    have caught it: every unit test above passes on a wrapper too, if you only assert the
    call happened.
    """
    httpx = pytest.importorskip("httpx")
    from genai_otel.config import OTelConfig
    from genai_otel.mcp_instrumentors import api_instrumentor as mod

    # The module keeps an identity-keyed registry so it will not double-wrap; clear it so
    # this test instruments a pristine httpx regardless of what ran before.
    mod._INSTRUMENTED_MODULES.clear()

    inst = APIInstrumentor(OTelConfig(service_name="test-api-instrumentor-e2e"))
    inst.instrument(OTelConfig(service_name="test-api-instrumentor-e2e"))

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with httpx.Client(transport=transport) as client:
        resp = client.get("http://example.test/ping")

    assert hasattr(resp, "status_code"), (
        f"got {type(resp).__name__}, not a Response — the wrapper was returned instead of called"
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
