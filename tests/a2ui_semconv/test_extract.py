"""The A2UI reader, on plain dicts. It needs neither ``a2a`` nor ``a2ui``."""

import json

from genai_otel.a2ui_semconv.attributes import (
    CONTENT_KEYS,
    EXTENSION_URI,
    SIDE_AGENT,
    SIDE_RENDERER,
    A2UIAttributes,
)
from genai_otel.a2ui_semconv.extract import (
    A2uiObservation,
    apply_to_span,
    has_extension,
    normalise_version,
    scrub_message,
    snake,
)

CATALOG = "https://example.test/catalog.json"


def create(surface="s1", catalog=CATALOG, version="1.0"):
    return {"version": version, "createSurface": {"surfaceId": surface, "catalogId": catalog}}


def components(surface="s1", n=2, version="1.0", **extra):
    items = [{"id": "root", "component": "Column"}] + [
        {"id": f"c{i}", "component": "Text", "text": f"t{i}", **extra} for i in range(n - 1)
    ]
    return {"version": version, "updateComponents": {"surfaceId": surface, "components": items}}


def attrs(obs):
    return obs.attributes()


def test_names_and_versions():
    assert snake("updateComponents") == "update_components"
    assert snake("callAgentFunction") == "call_agent_function"
    assert snake("action") == "action"
    assert normalise_version("1.0") == "1.0"
    assert normalise_version("v0.9") == "0.9"
    assert normalise_version("v0.9.1") == "0.9"
    assert normalise_version("nonsense") is None
    assert normalise_version(None) is None
    assert normalise_version(1.0) is None


def test_extension_matches_the_header_value_with_other_extensions_and_older_versions():
    assert has_extension([EXTENSION_URI])
    assert has_extension(["https://other.test/x, " + EXTENSION_URI + " ,more"])
    assert has_extension(["https://a2ui.org/a2a-extension/a2ui/v0.8"])
    assert not has_extension(["https://other.test/x"])
    assert not has_extension([])
    assert not has_extension([None, 5])


def test_agent_messages_are_counted_typed_and_identified():
    obs = A2uiObservation()
    obs.add_payload([create(), components(n=3)], SIDE_AGENT)
    always, content = attrs(obs)
    assert always[A2UIAttributes.PROTOCOL_VERSION] == "1.0"
    assert always[A2UIAttributes.AGENT_MESSAGE_COUNT] == 2
    assert A2UIAttributes.RENDERER_MESSAGE_COUNT not in always
    assert always[A2UIAttributes.MESSAGE_TYPES] == ["create_surface", "update_components"]
    assert always[A2UIAttributes.SURFACE_ID] == "s1"
    assert always[A2UIAttributes.SURFACE_COUNT] == 1
    assert always[A2UIAttributes.CATALOG_ID] == CATALOG
    assert always[A2UIAttributes.CATALOG_IDS] == [CATALOG]
    assert always[A2UIAttributes.COMPONENT_COUNT] == 3
    assert content == {}  # capture is off: nothing of the payload is even held


def test_a_single_message_and_a_list_read_the_same():
    one = A2uiObservation()
    one.add_payload(create(), SIDE_AGENT)
    many = A2uiObservation()
    many.add_payload([create()], SIDE_AGENT)
    assert attrs(one) == attrs(many)


def test_renderer_messages_action_function_and_errors():
    obs = A2uiObservation()
    obs.add_payload(
        {"version": "1.0", "action": {"name": "buy", "surfaceId": "s1", "context": {"a": 1}}},
        SIDE_RENDERER,
    )
    obs.add_payload(
        {
            "version": "1.0",
            "callAgentFunction": {
                "surfaceId": "s1",
                "functionCallId": "f1",
                "callFunction": {"@call": "lookup", "catalogId": CATALOG},
            },
        },
        SIDE_RENDERER,
    )
    obs.add_payload(
        {
            "version": "1.0",
            "error": {
                "code": "VALIDATION_FAILED",
                "surfaceId": "s1",
                "path": "/components/0",
                "message": "bad value for jane@example.test",
            },
        },
        SIDE_RENDERER,
    )
    always, _ = attrs(obs)
    assert always[A2UIAttributes.RENDERER_MESSAGE_COUNT] == 3
    assert always[A2UIAttributes.MESSAGE_TYPES] == ["action", "call_agent_function", "error"]
    # "buy" and "lookup" are different kinds of name, so each is single in its own key.
    assert always[A2UIAttributes.ACTION_NAME] == "buy"
    assert always[A2UIAttributes.FUNCTION_NAME] == "lookup"
    assert always[A2UIAttributes.ERROR_CODES] == ["VALIDATION_FAILED"]
    assert always[A2UIAttributes.ERROR_COUNT] == 1
    assert always[A2UIAttributes.CATALOG_ID] == CATALOG


def test_function_response_errors_are_counted_as_errors():
    obs = A2uiObservation()
    obs.add_payload(
        {
            "version": "1.0",
            "agentFunctionResponse": {
                "functionCallId": "f1",
                "error": {"code": "EXECUTION_ERROR", "message": "boom"},
            },
        },
        SIDE_AGENT,
    )
    always, _ = attrs(obs)
    assert always[A2UIAttributes.ERROR_CODES] == ["EXECUTION_ERROR"]
    assert always[A2UIAttributes.MESSAGE_TYPES] == ["agent_function_response"]


def test_one_event_per_error_with_code_and_surface():
    obs = A2uiObservation()
    for code, surface in (("UNALLOWED_PARENT", "s1"), ("UNALLOWED_CHILD", "s2")):
        obs.add_payload(
            {"version": "1.0", "error": {"code": code, "surfaceId": surface, "path": "/x"}},
            SIDE_RENDERER,
        )
    events = obs.events()
    assert [e[0] for e in events] == ["a2ui.error", "a2ui.error"]
    assert events[0][1] == {"a2ui.error.code": "UNALLOWED_PARENT", "a2ui.surface.id": "s1"}
    assert events[1][1] == {"a2ui.error.code": "UNALLOWED_CHILD", "a2ui.surface.id": "s2"}


def test_a_single_valued_identifier_is_omitted_when_several_were_seen():
    obs = A2uiObservation()
    obs.add_payload(
        [create("s1", "https://a/1"), create("s2", "https://a/2", version="v0.9")], SIDE_AGENT
    )
    obs.add_payload({"version": "1.0", "action": {"name": "a"}}, SIDE_RENDERER)
    obs.add_payload({"version": "1.0", "action": {"name": "b"}}, SIDE_RENDERER)
    always, _ = attrs(obs)
    assert A2UIAttributes.SURFACE_ID not in always
    assert always[A2UIAttributes.SURFACE_COUNT] == 2
    assert A2UIAttributes.CATALOG_ID not in always
    assert always[A2UIAttributes.CATALOG_IDS] == ["https://a/1", "https://a/2"]
    assert A2UIAttributes.PROTOCOL_VERSION not in always
    assert A2UIAttributes.ACTION_NAME not in always


def test_lists_are_sorted_and_deduplicated():
    obs = A2uiObservation()
    obs.add_payload(
        [create("s1", "https://b"), create("s2", "https://a"), create("s3", "https://b")],
        SIDE_AGENT,
    )
    always, _ = attrs(obs)
    assert always[A2UIAttributes.CATALOG_IDS] == ["https://a", "https://b"]
    assert always[A2UIAttributes.MESSAGE_TYPES] == ["create_surface"]


def test_a_component_can_override_the_catalog():
    obs = A2uiObservation()
    obs.add_payload(create(), SIDE_AGENT)
    obs.add_payload(components(**{"catalogId": "https://other/catalog"}), SIDE_AGENT)
    always, _ = attrs(obs)
    assert always[A2UIAttributes.CATALOG_IDS] == [CATALOG, "https://other/catalog"]


def test_v08_messages_are_counted_but_not_parsed():
    obs = A2uiObservation(capture=True)
    obs.add_payload(
        [
            {"beginRendering": {"surfaceId": "s1", "root": "r"}},
            {"surfaceUpdate": {"surfaceId": "s1", "components": [{"id": "r"}]}},
        ],
        SIDE_AGENT,
    )
    always, content = attrs(obs)
    assert always == {
        A2UIAttributes.PROTOCOL_VERSION: "0.8",
        A2UIAttributes.AGENT_MESSAGE_COUNT: 2,
    }
    assert content == {}


def test_a_message_with_no_known_key_is_counted_and_nothing_more():
    obs = A2uiObservation()
    obs.add_payload({"version": "1.0", "somethingNew": {"surfaceId": "s9"}}, SIDE_AGENT)
    always, _ = attrs(obs)
    assert always == {
        A2UIAttributes.PROTOCOL_VERSION: "1.0",
        A2UIAttributes.AGENT_MESSAGE_COUNT: 1,
    }


def test_a_message_on_the_wrong_side_is_not_misread():
    obs = A2uiObservation()
    obs.add_payload(create(), SIDE_RENDERER)  # createSurface is not a renderer message
    always, _ = attrs(obs)
    assert A2UIAttributes.MESSAGE_TYPES not in always
    assert always[A2UIAttributes.RENDERER_MESSAGE_COUNT] == 1


def test_non_objects_do_not_raise():
    obs = A2uiObservation()
    obs.add_payload([1, "x", None, create()], SIDE_AGENT)
    assert attrs(obs)[0][A2UIAttributes.AGENT_MESSAGE_COUNT] == 1


def test_capabilities_from_either_key_and_a_version_map():
    obs = A2uiObservation()
    obs.add_capabilities(
        {
            "a2uiRendererCapabilities": {
                "1.0": {"supportedCatalogIds": ["https://c/2", "https://c/1"]}
            }
        }
    )
    obs.add_capabilities(
        {
            "a2uiClientCapabilities": {
                "v0.9": {"supportedCatalogIds": ["https://c/1", "https://c/0"]}
            }
        }
    )
    obs.add_capabilities({"unrelated": 1})
    always, _ = attrs(obs)
    assert always[A2UIAttributes.RENDERER_SUPPORTED_CATALOG_IDS] == [
        "https://c/0",
        "https://c/1",
        "https://c/2",
    ]
    assert obs.has_data


def test_nothing_seen_means_nothing_to_write():
    obs = A2uiObservation()
    assert not obs.has_data
    assert attrs(obs) == ({}, {})
    obs.mark_extension(["https://other.test/x"])
    assert not obs.has_data
    obs.mark_extension([EXTENSION_URI])
    assert obs.has_data
    assert attrs(obs)[0] == {A2UIAttributes.EXTENSION_ACTIVE: True}


def test_invalid_is_not_overwritten_by_valid():
    obs = A2uiObservation()
    obs.set_outcome("invalid")
    obs.set_outcome("valid")
    assert attrs(obs)[0][A2UIAttributes.VALIDATION_OUTCOME] == "invalid"


def test_content_under_capture():
    obs = A2uiObservation(capture=True)
    obs.add_payload([create(), components(n=2)], SIDE_AGENT)
    obs.add_payload(
        {
            "version": "1.0",
            "updateDataModel": {"surfaceId": "s1", "path": "/u", "value": {"n": "Jane"}},
        },
        SIDE_AGENT,
    )
    obs.add_payload(
        {
            "version": "1.0",
            "action": {"name": "go", "surfaceId": "s1", "context": {"card": "4111"}},
        },
        SIDE_RENDERER,
    )
    obs.add_payload(
        {
            "version": "1.0",
            "error": {
                "code": "VALIDATION_FAILED",
                "path": "/components/1",
                "message": "Jane is bad",
            },
        },
        SIDE_RENDERER,
    )
    _, content = attrs(obs)
    assert set(content) == set(CONTENT_KEYS)
    assert json.loads(content[A2UIAttributes.MESSAGES])[0]["createSurface"]["surfaceId"] == "s1"
    assert json.loads(content[A2UIAttributes.DATA_MODEL]) == [{"n": "Jane"}]
    assert json.loads(content[A2UIAttributes.ACTION_CONTEXT]) == [{"card": "4111"}]
    assert content[A2UIAttributes.ERROR_PATHS] == ["/components/1"]
    assert content[A2UIAttributes.ERROR_MESSAGES] == ["Jane is bad"]
    assert content[A2UIAttributes.COMPONENT_TYPES] == ["Column", "Text"]


def test_media_urls_never_reach_the_recorded_content():
    secret = "https://cdn.test/passport-scan.png?token=abc"
    message = {
        "version": "1.0",
        "updateComponents": {
            "surfaceId": "s1",
            "components": [
                {"id": "i", "component": "Image", "url": secret},
                {"id": "v", "component": "Video", "url": {"@path": "/video"}},
                {"id": "a", "component": "Avatar", "imageUrl": secret, "src": secret},
                {"id": "d", "component": "Image", "url": "data:image/png;base64,AAAA"},
                {"id": "t", "component": "Text", "text": "plain text stays"},
            ],
        },
    }
    scrubbed = scrub_message(message)
    text = json.dumps(scrubbed)
    assert secret not in text
    assert "base64" not in text
    assert "plain text stays" in text
    comps = scrubbed["updateComponents"]["components"]
    assert comps[0]["url"] == "[redacted]"
    # A data binding names where a value lives, not the value, so it is kept.
    assert comps[1]["url"] == {"@path": "/video"}
    # The original message is not edited.
    assert message["updateComponents"]["components"][0]["url"] == secret

    obs = A2uiObservation(capture=True)
    obs.add_payload(message, SIDE_AGENT)
    assert secret not in attrs(obs)[1][A2UIAttributes.MESSAGES]


def test_data_uris_are_redacted_outside_components_too():
    obs = A2uiObservation(capture=True)
    obs.add_payload(
        {
            "version": "1.0",
            "updateDataModel": {"surfaceId": "s", "value": "data:image/png;base64,QQ=="},
        },
        SIDE_AGENT,
    )
    assert "QQ==" not in attrs(obs)[1][A2UIAttributes.DATA_MODEL]


def test_apply_to_span_writes_attributes_events_and_routes_content_through_put_content():
    class Span:
        def __init__(self):
            self.attributes, self.events = {}, []

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def add_event(self, name, attributes=None):
            self.events.append((name, attributes))

    obs = A2uiObservation(capture=True)
    obs.add_payload(
        {"version": "1.0", "error": {"code": "VALIDATION_FAILED", "message": "m"}}, SIDE_RENDERER
    )
    span, written = Span(), []
    apply_to_span(span, obs, lambda s, k, v: written.append(k))
    assert span.attributes[A2UIAttributes.ERROR_CODES] == ["VALIDATION_FAILED"]
    assert span.events[0][0] == "a2ui.error"
    assert A2UIAttributes.ERROR_MESSAGES in written
    assert A2UIAttributes.ERROR_MESSAGES not in span.attributes  # content goes only via put_content


def test_apply_to_span_never_raises():
    class Broken:
        def set_attribute(self, *a):
            raise RuntimeError("no")

    obs = A2uiObservation()
    obs.add_payload(create(), SIDE_AGENT)
    apply_to_span(Broken(), obs, lambda *a: None)
