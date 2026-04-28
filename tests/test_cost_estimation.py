"""Tests for the public genai_otel.cost_estimation module."""

import pytest

from genai_otel import cost_estimation as ce

# --- coerce_text / result_text ---------------------------------------------


def test_coerce_text_string():
    assert ce.coerce_text("hi") == "hi"


def test_coerce_text_dict_keys():
    assert ce.coerce_text({"prompt": "p"}) == "p"
    assert ce.coerce_text({"text": "t", "content": "c"}) == "t"


def test_coerce_text_list_concat():
    assert ce.coerce_text(["a", "b"]) == "a b"


def test_coerce_text_none_and_other():
    assert ce.coerce_text(None) == ""
    assert ce.coerce_text(42) == ""


def test_result_text_dict_keys():
    assert ce.result_text({"generated_text": "g"}) == "g"
    assert ce.result_text([{"summary_text": "s"}]) == "s"


def test_result_text_string_passthrough():
    assert ce.result_text("hello") == "hello"


# --- count_images -----------------------------------------------------------


class _FakePIL:
    size = (10, 10)
    mode = "RGB"


def test_count_images_pil_image():
    assert ce._count_image_value(_FakePIL()) == 1


def test_count_images_url_string():
    assert ce._count_image_value("https://x/y.png") == 1
    assert ce._count_image_value("https://x/y.txt") == 0


def test_count_images_chat_style_no_double_count():
    msg = [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
    # The {type:image_url} is one image — the nested image_url payload is
    # metadata, not a second image.
    assert ce._count_image_value(msg) == 1


def test_count_images_kwargs_image_url_string():
    n = ce.count_images((), {"image_url": "https://x/y.png", "images": [_FakePIL(), _FakePIL()]})
    assert n == 3


def test_count_images_inputs_kwarg():
    n = ce.count_images((), {"inputs": [_FakePIL()]})
    assert n == 1


def test_count_images_zero_for_text_only():
    assert ce.count_images(("hello",), {}) == 0


# --- audio_seconds ----------------------------------------------------------


def test_audio_seconds_dict_with_array():
    sec = ce._audio_seconds_of({"array": list(range(16000)), "sampling_rate": 16000}, None)
    assert sec == pytest.approx(1.0)


def test_audio_seconds_uses_explicit_rate_for_tensor():
    class _T:
        shape = (1, 32000)

    assert ce.audio_seconds((), {"audio": _T()}, sampling_rate=16000) == pytest.approx(2.0)


def test_audio_seconds_zero_when_no_audio():
    assert ce.audio_seconds(("text",), {}, sampling_rate=16000) == 0.0


# --- estimate_pipeline_usage ------------------------------------------------


def test_pipeline_usage_summarization_text_in_text_out():
    u = ce.estimate_pipeline_usage(
        "summarization",
        args=("a" * 400,),
        result=[{"summary_text": "b" * 40}],
    )
    assert u["prompt_tokens"] == 100
    assert u["completion_tokens"] == 10
    assert u["total_tokens"] == 110
    assert u["estimated"] is True


def test_pipeline_usage_image_text_to_text_counts_images():
    u = ce.estimate_pipeline_usage(
        "image-text-to-text",
        kwargs={"images": [_FakePIL()], "inputs": "describe"},
        result=[{"generated_text": "a cat"}],
    )
    # 8 chars / 4 = 2 (text) + 1*256 = 258 prompt tokens
    assert u["prompt_tokens"] == 258
    assert u["completion_tokens"] == 2
    assert u["image_count"] == 1
    assert u["audio_seconds"] == 0.0


def test_pipeline_usage_asr_uses_audio_seconds():
    class _Pipe:
        class _FE:
            sampling_rate = 16000

        feature_extractor = _FE()

    u = ce.estimate_pipeline_usage(
        "automatic-speech-recognition",
        args=({"array": list(range(16000 * 3)), "sampling_rate": 16000},),
        result={"text": "hello world"},
        pipe=_Pipe(),
    )
    assert u["audio_seconds"] == pytest.approx(3.0)
    assert u["prompt_tokens"] == 150  # 3 sec * 50 tok/sec
    assert u["completion_tokens"] == 3  # "hello world" 11 chars / 4 ceil


def test_pipeline_usage_text_to_image_floor_completion():
    """Text-to-image returns image bytes, not text. We allow a 1-token
    completion floor so the cost row isn't dropped."""
    u = ce.estimate_pipeline_usage(
        "text-to-image",
        args=("a sunset over mountains",),  # 23 chars / 4 = 6 tokens
        result=None,
    )
    assert u["prompt_tokens"] == 6
    assert u["completion_tokens"] == 1
    assert u["total_tokens"] == 7


def test_pipeline_usage_returns_empty_when_nothing_to_count():
    u = ce.estimate_pipeline_usage("image-classification", args=(), kwargs={}, result=None)
    assert u == {}


def test_pipeline_usage_overridable_constants():
    u = ce.estimate_pipeline_usage(
        "image-text-to-text",
        kwargs={"images": [_FakePIL()], "inputs": "x"},
        result={"generated_text": ""},
        image_token_estimate=512,  # higher rate per image
    )
    assert u["prompt_tokens"] == 1 + 512


# --- estimate_chat_usage ----------------------------------------------------


def test_chat_usage_text_only():
    u = ce.estimate_chat_usage(
        messages=[{"role": "user", "content": "hello world"}],
        response_text="hi",
    )
    assert u["prompt_tokens"] == 3  # 11/4 ceil
    assert u["completion_tokens"] == 1
    assert u["image_count"] == 0


def test_chat_usage_with_images_via_messages_field():
    u = ce.estimate_chat_usage(
        messages=[
            {
                "role": "user",
                "content": "describe these",  # 14 chars / 4 = 4
                "images": ["b64a", "b64b"],
            }
        ],
        response_text="a cat and a dog",  # 15 / 4 = 4
    )
    assert u["prompt_tokens"] == 4 + 2 * 256
    assert u["completion_tokens"] == 4
    assert u["image_count"] == 2


def test_chat_usage_with_audio_seconds():
    u = ce.estimate_chat_usage(
        messages=[{"role": "user", "content": "transcribe please"}],  # 17 chars / 4 = 5
        response_text="ok",
        audio_seconds_in=2.0,  # 2 sec * 50 = 100 tokens
    )
    assert u["prompt_tokens"] == 5 + 100


def test_chat_usage_chat_style_image_url_part():
    u = ce.estimate_chat_usage(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ],
        response_text="",
    )
    assert u["image_count"] == 1
    assert u["prompt_tokens"] >= 256


def test_chat_usage_empty_returns_empty():
    assert ce.estimate_chat_usage(messages=[], response_text="") == {}
