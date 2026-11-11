"""The media store is built once, lazily, and an unreachable store never slows model calls."""

from __future__ import annotations

import base64
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.media.stores import clear_store_cache, get_store


@dataclass
class _Cfg:
    media_capture_mode: str = "full"
    media_store: str = "s3"
    media_store_endpoint: Optional[str] = "https://unreachable.invalid:9000"
    media_store_bucket: str = "genai-otel-media"
    media_store_prefix: str = "traces/{date}/{trace_id}/"
    media_store_access_key: Optional[str] = "k"
    media_store_secret_key: Optional[str] = "s"
    media_max_bytes: int = 10 * 1024 * 1024
    media_allowed_modalities: str = "image,audio,video,document"
    media_redactor: Optional[str] = None
    enable_content_capture: bool = False
    content_max_length: int = 200
    semconv_stability_opt_in: str = "gen_ai"


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_store_cache()
    yield
    clear_store_cache()


@pytest.fixture
def fake_boto():
    """Fake boto3/botocore; the client's calls fail like an unreachable endpoint."""
    client = MagicMock()
    client.create_bucket.side_effect = ConnectionError("endpoint unreachable")
    client.put_object.side_effect = ConnectionError("endpoint unreachable")
    boto3 = MagicMock()
    boto3.client.return_value = client
    botocore_client = MagicMock()
    with patch.dict(
        sys.modules,
        {"boto3": boto3, "botocore": MagicMock(), "botocore.client": botocore_client},
    ):
        yield boto3, client, botocore_client


def _span():
    span = MagicMock()
    span.get_span_context.return_value = MagicMock(trace_id=0xDEADBEEF)
    return span


def _image_request():
    data = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": data},
                    },
                ],
            }
        ]
    }


def _inst(cfg):
    from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor

    inst = AnthropicInstrumentor()
    inst.config = cfg
    return inst


def test_text_only_never_builds_a_store(fake_boto):
    boto3, client, _ = fake_boto
    inst = _inst(_Cfg())
    for _ in range(5):
        inst._emit_media_attributes(
            _span(),
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
            result=None,
        )
    assert boto3.client.call_count == 0
    assert client.create_bucket.call_count == 0


def test_media_calls_build_one_store_and_one_create_bucket(fake_boto):
    boto3, client, _ = fake_boto
    inst = _inst(_Cfg())
    for _ in range(6):
        inst._emit_media_attributes(_span(), _image_request(), result=None)
    assert boto3.client.call_count == 1
    assert client.create_bucket.call_count == 1


def test_get_store_is_cached_per_effective_config(fake_boto):
    boto3, _, _ = fake_boto
    assert get_store(_Cfg()) is get_store(_Cfg())
    assert boto3.client.call_count == 1
    get_store(_Cfg(media_store_bucket="other"))
    assert boto3.client.call_count == 2


def test_s3_client_has_short_timeouts_and_no_retries(fake_boto):
    _, _, botocore_client = fake_boto
    get_store(_Cfg())
    kwargs = botocore_client.Config.call_args.kwargs
    assert kwargs["connect_timeout"] <= 2
    assert kwargs["read_timeout"] <= 10
    assert kwargs["retries"] == {"max_attempts": 1}


def test_unreachable_store_adds_negligible_time_and_warns_once(fake_boto, caplog):
    inst = _inst(_Cfg())
    caplog.set_level(logging.DEBUG)
    t0 = time.monotonic()
    inst._emit_media_attributes(_span(), _image_request(), result=None)
    first = time.monotonic() - t0
    t0 = time.monotonic()
    for _ in range(10):
        inst._emit_media_attributes(_span(), _image_request(), result=None)
    rest = (time.monotonic() - t0) / 10
    assert first < 1.0
    assert rest < 0.1
    store_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "Media store" in r.getMessage()
    ]
    assert len(store_warnings) == 1


def test_failed_upload_keeps_span_and_drops_bytes(fake_boto):
    span = _span()
    attrs = {}
    span.set_attribute.side_effect = lambda k, v: attrs.__setitem__(k, v)
    _inst(_Cfg())._emit_media_attributes(span, _image_request(), result=None)
    assert attrs["gen_ai.prompt.0.content.1.type"] == "image"
    assert attrs["gen_ai.prompt.0.content.1.media_source"] == "reference_only"
