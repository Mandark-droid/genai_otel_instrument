"""The OpenAI Responses API must produce a span, not silence (issue #26).

``openai_instrumentor`` wrapped ``chat.completions.create`` and
``embeddings.create`` only, so a caller on ``client.responses.create`` -- the
default path for native GPT-5.6+ models, because Chat Completions rejects
function tools combined with reasoning -- got no LLM span at all: no model, no
tokens, no cost, no tool calls. The instrumentor loaded, reported success, and
captured nothing.

The Responses shape differs from Chat Completions in every place the
instrumentor reads:

    request   ``input`` / ``instructions``   not ``messages``
    response  ``output[]``                   not ``choices[]``
    usage     ``input_tokens``/``output_tokens``  not ``prompt_``/``completion_``

so each extraction hook has to detect the shape rather than assume one.
"""

from unittest.mock import MagicMock

from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor


class Obj:
    """Attribute bag standing in for an SDK model.

    A MagicMock answers ``hasattr`` for everything, which would hide exactly the
    shape-detection bugs these tests exist to catch.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _responses_usage():
    return Obj(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_tokens_details=Obj(cached_tokens=20),
        output_tokens_details=Obj(reasoning_tokens=30),
    )


def _responses_result():
    return Obj(
        id="resp_abc123",
        model="gpt-5.6",
        status="completed",
        usage=_responses_usage(),
        output=[
            Obj(
                type="reasoning",
                id="rs_1",
                summary=[],
            ),
            Obj(
                type="message",
                role="assistant",
                content=[Obj(type="output_text", text="It is sunny in SF.")],
            ),
            Obj(
                type="function_call",
                call_id="call_1",
                name="get_weather",
                arguments='{"city": "SF"}',
            ),
        ],
    )


def _chat_completions_result():
    """The existing shape -- must keep working unchanged."""
    return Obj(
        id="chatcmpl_1",
        model="gpt-4o",
        usage=Obj(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        choices=[
            Obj(
                finish_reason="stop",
                message=Obj(content="hi", tool_calls=None),
            )
        ],
    )


class TestResponsesCreateIsWrapped:
    def test_sync_client_responses_create_is_wrapped(self):
        instrumentor = OpenAIInstrumentor()
        instrumentor.config = MagicMock()

        original = MagicMock(name="responses.create")
        client = Obj(base_url=None, responses=Obj(create=original))

        instrumentor._instrument_client(client)

        assert client.responses.create is not original, "responses.create was left uninstrumented"

    def test_async_client_responses_create_is_wrapped(self):
        instrumentor = OpenAIInstrumentor()
        instrumentor.config = MagicMock()

        original = MagicMock(name="async responses.create")
        client = Obj(base_url=None, responses=Obj(create=original))

        instrumentor._instrument_async_client(client)

        assert client.responses.create is not original

    def test_client_without_responses_resource_is_untouched(self):
        """Older openai SDKs have no .responses -- that must not raise."""
        instrumentor = OpenAIInstrumentor()
        instrumentor.config = MagicMock()
        client = Obj(base_url=None)

        instrumentor._instrument_client(client)  # must not raise


class TestUsageMapping:
    def test_responses_usage_maps_onto_canonical_keys(self):
        usage = OpenAIInstrumentor()._extract_usage(_responses_result())

        assert usage is not None, "Responses usage went unread, so the span is priced at zero"
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_reasoning_tokens_are_attributed_as_output(self):
        """Reasoning tokens are billed as output, so they must be recorded."""
        usage = OpenAIInstrumentor()._extract_usage(_responses_result())

        assert usage["completion_tokens_details"]["reasoning_tokens"] == 30

    def test_cached_prompt_tokens_are_recorded(self):
        usage = OpenAIInstrumentor()._extract_usage(_responses_result())

        assert usage["cache_read_input_tokens"] == 20

    def test_chat_completions_usage_still_works(self):
        """Regression: the existing shape must be untouched."""
        usage = OpenAIInstrumentor()._extract_usage(_chat_completions_result())

        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_missing_usage_returns_none(self):
        assert OpenAIInstrumentor()._extract_usage(Obj(id="resp_1")) is None


class TestRequestAttributes:
    def test_string_input_is_one_message(self):
        attrs = OpenAIInstrumentor()._extract_responses_attributes(
            None, (), {"model": "gpt-5.6", "input": "what is the weather?"}
        )

        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-5.6"
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.request.message_count"] == 1

    def test_list_input_counts_items(self):
        attrs = OpenAIInstrumentor()._extract_responses_attributes(
            None,
            (),
            {
                "model": "gpt-5.6",
                "input": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                ],
            },
        )

        assert attrs["gen_ai.request.message_count"] == 3

    def test_instructions_are_captured(self):
        attrs = OpenAIInstrumentor()._extract_responses_attributes(
            None, (), {"model": "gpt-5.6", "input": "hi", "instructions": "Be terse."}
        )

        assert attrs["gen_ai.request.instructions"] == "Be terse."

    def test_max_output_tokens_maps_to_max_tokens(self):
        """Responses spells it max_output_tokens; the semconv attribute is max_tokens."""
        attrs = OpenAIInstrumentor()._extract_responses_attributes(
            None, (), {"model": "gpt-5.6", "input": "hi", "max_output_tokens": 256}
        )

        assert attrs["gen_ai.request.max_tokens"] == 256

    def test_tools_are_serialized(self):
        attrs = OpenAIInstrumentor()._extract_responses_attributes(
            None,
            (),
            {"model": "gpt-5.6", "input": "hi", "tools": [{"type": "function", "name": "f"}]},
        )

        assert "llm.tools" in attrs


class TestResponseAttributes:
    def test_response_id_and_model_recorded(self):
        """response.id keeps store=true responses joinable server-side."""
        attrs = OpenAIInstrumentor()._extract_response_attributes(_responses_result())

        assert attrs["gen_ai.response.id"] == "resp_abc123"
        assert attrs["gen_ai.response.model"] == "gpt-5.6"

    def test_tool_calls_extracted_from_output(self):
        attrs = OpenAIInstrumentor()._extract_response_attributes(_responses_result())

        names = [v for k, v in attrs.items() if k.endswith(".tool_call.function.name")]
        args = [v for k, v in attrs.items() if k.endswith(".tool_call.function.arguments")]
        ids = [v for k, v in attrs.items() if k.endswith(".tool_call.id")]

        assert names == ["get_weather"]
        assert args == ['{"city": "SF"}']
        assert ids == ["call_1"]

    def test_chat_completions_tool_calls_still_extracted(self):
        """Regression: choices[]-shaped tool calls keep working."""
        result = Obj(
            id="c1",
            model="gpt-4o",
            choices=[
                Obj(
                    finish_reason="tool_calls",
                    message=Obj(
                        content=None,
                        tool_calls=[Obj(id="tc1", function=Obj(name="g", arguments="{}"))],
                    ),
                )
            ],
        )
        attrs = OpenAIInstrumentor()._extract_response_attributes(result)

        assert attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "g"


class TestFinishReason:
    def test_status_is_the_finish_reason(self):
        assert OpenAIInstrumentor()._extract_finish_reason(_responses_result()) == "completed"

    def test_incomplete_reason_preferred_when_present(self):
        result = Obj(
            id="r",
            status="incomplete",
            incomplete_details=Obj(reason="max_output_tokens"),
        )

        assert OpenAIInstrumentor()._extract_finish_reason(result) == "max_output_tokens"

    def test_chat_completions_finish_reason_still_works(self):
        assert OpenAIInstrumentor()._extract_finish_reason(_chat_completions_result()) == "stop"


class TestContentCapture:
    def test_responses_input_is_not_mistaken_for_an_embeddings_call(self):
        """The trap: embeddings also key on `input`.

        _add_content_events routes to the embeddings path whenever `input` is
        present and `messages` is not. The Responses API uses `input` too, so
        without a shape check a Responses call is recorded as a retrieval span:
        the completion is dropped and `embedding.model_name` is set instead.
        """
        instrumentor = OpenAIInstrumentor()
        instrumentor.config = MagicMock(content_max_length=0, capture_embedding_vectors=False)
        span = MagicMock()

        instrumentor._add_content_events(
            span, _responses_result(), {"model": "gpt-5.6", "input": "what is the weather?"}
        )

        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        assert "embedding.model_name" not in set_keys, "Responses call traced as an embedding"

    def test_output_text_is_captured_for_evaluation(self):
        instrumentor = OpenAIInstrumentor()
        instrumentor.config = MagicMock(content_max_length=0)
        span = MagicMock()

        instrumentor._add_content_events(
            span, _responses_result(), {"model": "gpt-5.6", "input": "what is the weather?"}
        )

        set_attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert set_attrs.get("gen_ai.response") == "It is sunny in SF."

    def test_prompt_content_recorded_from_string_input(self):
        instrumentor = OpenAIInstrumentor()
        instrumentor.config = MagicMock(content_max_length=0)
        span = MagicMock()

        instrumentor._add_content_events(
            span, _responses_result(), {"model": "gpt-5.6", "input": "what is the weather?"}
        )

        events = {c.args[0]: c.kwargs.get("attributes", {}) for c in span.add_event.call_args_list}
        assert "gen_ai.prompt.0" in events
        assert events["gen_ai.prompt.0"]["gen_ai.prompt.content"] == "what is the weather?"

    def test_embeddings_still_route_to_the_embeddings_path(self):
        """Regression: a real embeddings call must keep its retrieval treatment."""
        instrumentor = OpenAIInstrumentor()
        instrumentor.config = MagicMock(content_max_length=0, capture_embedding_vectors=False)
        span = MagicMock()
        result = Obj(data=[Obj(embedding=[0.1, 0.2])], model="text-embedding-3-small")

        instrumentor._add_content_events(
            span, result, {"model": "text-embedding-3-small", "input": "find me"}
        )

        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        assert "embedding.model_name" in set_keys
