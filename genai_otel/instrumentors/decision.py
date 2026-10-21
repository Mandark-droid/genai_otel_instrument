"""Decision spans: one attribute shape for every decision-model call path.

A decision model takes a state plus a set of typed questions and returns, for
each question, a probability distribution over answers the caller supplied. It
writes no text. OpenInference gives these calls their own span kind,
``DECISION``, with ``decision.*`` attributes (Python semconv 0.1.40,
experimental).

Two wire protocols carry almost every decision model today:

* **OpenAI Decisions** (``POST /v1/decisions``): ``model``, ``input``, a list of
  ``questions`` (``predicate`` / ``choice`` / ``score``) and a list of
  ``answers`` in question order, any of which may be a ``refusal``.
* **SystemOne** (``POST /v1/systemone``): ``state`` plus a mapping of named
  ``questions`` (``noul`` / ``choice`` / ``score`` / ``box``) and a mapping of
  named ``answers``. TypeSafe defined it; LiquidAI, Cloudflare, vLLM, SGLang and
  the community servers answer it.

This module turns either request/response pair (and a Granite Switch decision
adapter call, and whatever a caller records by hand) into one shape and returns
the span attributes for it, so every path emits the same contract.

What is recorded with content capture off is deliberately narrow: which model
decided, through which provider, how many questions of which types, how many
were refused, tokens and cost. The chosen answer, its confidence, every score
derived from them and the caller's question names are content, and are
recorded only with content capture on.
"""

import contextlib
import dataclasses
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .base import BaseInstrumentor, _server_attributes, _unwrap_api_response

logger = logging.getLogger(__name__)

DECISION_SPAN_KIND = "DECISION"

PROTOCOL_OPENAI_DECISIONS = "openai_decisions"
PROTOCOL_SYSTEMONE = "systemone"
PROTOCOL_GRANITE_ADAPTER = "granite_adapter"
PROTOCOL_MANUAL = "manual"

SYSTEM_OPENAI = "openai"
SYSTEM_TYPESAFE = "typesafe"

# What every serialised image is replaced with, matching OpenInference's
# ``hide_input_images``. A data URL is the image itself, and a URL may be a
# signed link to it; neither belongs on a span.
IMAGE_OMITTED = "[image omitted]"

# Recognised hosts. Anything else is self-hosted or a gateway, and gets no
# provider unless the operator names one: a guess would misattribute spend.
_PROVIDER_HOSTS = {
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
    "api.liquid.ai": "liquid",
    "api.typesafe.ai": "typesafe",
    "api.cloudflare.com": "cloudflare",
}
_PROVIDER_HOST_SUFFIXES = ((".openai.azure.com", "azure"),)

# Answer types whose confidence counts towards ``min_confidence``. A box answer's
# confidence is about where something is, not how sure the model is of a choice.
_CONFIDENCE_TYPES = frozenset({"choice", "score"})


def provider_for_base_url(base_url: Any, configured: Optional[str] = None) -> Optional[str]:
    """The provider that hosts ``base_url``, or ``configured`` when the host is unknown.

    ``configured`` is ``GENAI_DECISION_PROVIDER``: it names the provider behind a
    self-hosted server or a gateway, which no host lookup can know. A recognised
    host wins over it, because the operator setting is a fallback, not an
    override for traffic that demonstrably went to a named vendor.
    """
    host = None
    if base_url:
        try:
            host = urlparse(str(base_url)).hostname
        except Exception:  # pragma: no cover - urlparse is near-total
            host = None
    if host:
        host = host.lower()
        if host in _PROVIDER_HOSTS:
            return _PROVIDER_HOSTS[host]
        for suffix, provider in _PROVIDER_HOST_SUFFIXES:
            if host.endswith(suffix):
                return provider
    configured = (configured or "").strip()
    return configured or None


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, whichever ``value`` is."""
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None. Booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


def _count(value: Any) -> Optional[int]:
    """A token count the provider reported, or None when it reported nothing."""
    number = _number(value)
    if number is None or number < 0:
        return None
    return int(number)


def _label(value: Any) -> Optional[str]:
    """A choice value as text. Choice values are typed: ``True`` is not ``"True"``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _items(value: Any) -> List[Any]:
    """A request or response list as a list.

    Only real sequences are read. The SDKs accept any iterable, and a generator
    read here would reach the provider empty.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _size(value: Any) -> Optional[int]:
    """How many options a mapping or sequence of them holds, or None if it cannot be told."""
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return None


@dataclasses.dataclass
class DecisionQuestion:
    """One question as asked: its caller label, its type and how many options it offered."""

    name: Optional[str]
    type: str
    option_count: Optional[int] = None


@dataclasses.dataclass
class DecisionAnswer:
    """One answer as returned, reduced to what the span contract reads."""

    name: Optional[str]
    type: str
    choice: Optional[str] = None
    confidence: Optional[float] = None
    probability: Optional[float] = None
    score: Optional[float] = None
    probabilities: List[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DecisionUsage:
    """Token counts exactly as reported. None means the provider said nothing."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_write_input_tokens: Optional[int] = None

    @property
    def reported(self) -> bool:
        return self.input_tokens is not None or self.output_tokens is not None


@dataclasses.dataclass
class DecisionRequest:
    """A decision request in the one shape every call path is normalised to."""

    protocol: str
    system: str
    operation: str
    model: Optional[str]
    provider: Optional[str]
    questions: List[DecisionQuestion]
    # The request as it was sent, for ``input.value`` under content capture.
    payload: Any = None
    legacy_llm_kind: bool = False
    # Call-path specific request details (TypeSafe's ``extra_body``).
    extras: Dict[str, Any] = dataclasses.field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #


def _question_type(raw: Any) -> str:
    """The question type on the contract's vocabulary. SystemOne's ``noul`` is a predicate."""
    value = _get(raw, "type")
    value = str(getattr(value, "value", value) or "").strip().lower()
    if value == "noul":
        return "predicate"
    return value or "unknown"


def questions_of(questions: Any) -> List[DecisionQuestion]:
    """Questions in either protocol's shape: a SystemOne mapping or an OpenAI list."""
    if isinstance(questions, Mapping):
        return systemone_questions(questions)
    return openai_questions(questions)


def openai_questions(questions: Any) -> List[DecisionQuestion]:
    """Questions of an OpenAI Decisions request, in order."""
    out = []
    for raw in _items(questions):
        qtype = _question_type(raw)
        options = None
        if qtype == "choice":
            options = _size(_get(raw, "choices"))
        elif qtype == "score":
            options = _size(_get(raw, "levels"))
        name = _get(raw, "name")
        out.append(
            DecisionQuestion(name=str(name) if name else None, type=qtype, option_count=options)
        )
    return out


def systemone_questions(questions: Any) -> List[DecisionQuestion]:
    """Questions of a SystemOne request: a mapping of caller labels to questions."""
    out = []
    if not isinstance(questions, Mapping):
        return out
    for name, raw in questions.items():
        qtype = _question_type(raw)
        options = None
        if qtype in ("choice", "score"):
            options = _size(_get(raw, "criteria"))
        out.append(DecisionQuestion(name=str(name), type=qtype, option_count=options))
    return out


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


def _probability_values(raw: Any) -> List[float]:
    """Option probabilities, from a list of ``{probability}`` items or a label->p mapping."""
    values: List[float] = []
    if isinstance(raw, Mapping):
        candidates = list(raw.values())
    else:
        candidates = _items(raw)
    for item in candidates:
        number = _number(item)
        if number is None:
            number = _number(_get(item, "probability"))
        if number is not None:
            values.append(number)
    return values


def _answer_from(
    name: Optional[str], raw: Any, fallback_type: Optional[str] = None
) -> DecisionAnswer:
    """Reduce one answer object, in either protocol's shape, to a DecisionAnswer."""
    atype = _get(raw, "type")
    atype = str(getattr(atype, "value", atype) or "").strip().lower()
    if not atype:
        # Older SDK structs and hand-built payloads carry no tag; the field
        # that holds the answer says what it is.
        for field_name in ("noul", "choice", "score", "box"):
            if _get(raw, field_name) is not None:
                atype = field_name
                break
    if not atype and _number(raw) is not None:
        # A per-type map of bare numbers (``nouls: {"urgent": 0.91}``).
        return DecisionAnswer(name=name, type="predicate", probability=_number(raw))
    if not atype:
        atype = fallback_type or "unknown"
    if atype == "noul":
        atype = "predicate"

    answer = DecisionAnswer(name=name, type=atype)
    if atype == "refusal":
        return answer
    if atype == "predicate":
        answer.probability = _number(_get(raw, "probability"))
        if answer.probability is None:
            answer.probability = _number(_get(raw, "noul"))
        return answer
    answer.confidence = _number(_get(raw, "confidence"))
    if atype == "choice":
        answer.choice = _label(_get(raw, "choice"))
    elif atype == "score":
        answer.score = _number(_get(raw, "score"))
    answer.probabilities = _probability_values(_get(raw, "probabilities"))
    return answer


def openai_answers(answers: Any, questions: List[DecisionQuestion]) -> List[DecisionAnswer]:
    """Answers of an OpenAI Decisions response. They come back in question order."""
    out = []
    for index, raw in enumerate(_items(answers)):
        name = _get(raw, "name")
        if not name and index < len(questions):
            name = questions[index].name
        fallback = questions[index].type if index < len(questions) else None
        out.append(_answer_from(str(name) if name else None, raw, fallback))
    return out


_SYSTEMONE_TYPE_MAPS = (
    ("nouls", "noul"),
    ("choices", "choice"),
    ("scores", "score"),
    ("boxes", "box"),
)


def systemone_answers(response: Any, questions: List[DecisionQuestion]) -> List[DecisionAnswer]:
    """Answers of a SystemOne response, in question order where the questions are known.

    Read from the ``answers`` mapping, and from the per-type maps (``nouls``,
    ``choices``, ...) for SDK versions and servers that only return those.
    """
    found: Dict[str, Any] = {}
    answers = _get(response, "answers")
    if isinstance(answers, Mapping):
        found.update(answers)
    for map_name, answer_type in _SYSTEMONE_TYPE_MAPS:
        per_type = _get(response, map_name)
        if isinstance(per_type, Mapping):
            for name, raw in per_type.items():
                found.setdefault(name, (answer_type, raw))
    typed = {}
    for name, raw in found.items():
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str):
            typed[str(name)] = _answer_from(str(name), raw[1], raw[0])
        else:
            typed[str(name)] = _answer_from(str(name), raw)
    ordered = []
    for question in questions:
        if question.name in typed:
            ordered.append(typed.pop(question.name))
    ordered.extend(typed.values())
    return ordered


def granite_answers(result: Any, label_count: Optional[int]) -> List[DecisionAnswer]:
    """The one answer of a Granite Switch ``decision`` adapter call.

    The adapter answers with the softmax over the option labels' logits, which
    the caller reads from the first token's ``top_logprobs`` when it asked for
    them (``logprob_token_ids``). Without them the call has no distribution to
    report, and the answer carries its type only.
    """
    answer = DecisionAnswer(name=None, type="choice")
    choices = _get(result, "choices") or []
    first = _items(choices)[0] if _items(choices) else None
    content = _items(_get(_get(first, "logprobs"), "content"))
    top = _items(_get(content[0], "top_logprobs")) if content else []
    scored: List[Tuple[str, float]] = []
    for item in top:
        token = _get(item, "token")
        logprob = _number(_get(item, "logprob"))
        if token is not None and logprob is not None:
            scored.append((str(token), logprob))
    if label_count and len(scored) > label_count:
        scored = sorted(scored, key=lambda pair: pair[1], reverse=True)[:label_count]
    if scored:
        weights = [math.exp(lp) for _, lp in scored]
        total = sum(weights)
        if total > 0:
            answer.probabilities = [w / total for w in weights]
            best = max(range(len(scored)), key=lambda i: answer.probabilities[i])
            answer.choice = scored[best][0]
    return [answer]


def usage_of(raw: Any) -> DecisionUsage:
    """Token counts from a response's ``usage``, under either protocol's spelling."""
    if raw is None:
        return DecisionUsage()
    input_tokens = _count(_get(raw, "input_tokens"))
    if input_tokens is None:
        input_tokens = _count(_get(raw, "prompt_tokens"))
    output_tokens = _count(_get(raw, "output_tokens"))
    if output_tokens is None:
        output_tokens = _count(_get(raw, "completion_tokens"))
    details = _get(raw, "input_tokens_details") or _get(raw, "prompt_tokens_details")
    return DecisionUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=_count(_get(details, "cached_tokens")),
        cache_write_input_tokens=_count(_get(details, "cache_write_tokens")),
    )


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #

_IMAGE_PART_TYPES = frozenset({"input_image", "image", "image_url"})
_IMAGE_KEEP_KEYS = frozenset({"type", "detail"})


def to_jsonable(value: Any) -> Any:
    """Convert SDK models, structs and mappings into plain JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        # Raw bytes in a decision request are an image or other asset.
        return IMAGE_OMITTED
    try:
        import msgspec

        if isinstance(value, msgspec.Struct):
            return to_jsonable(msgspec.to_builtins(value))
    except (ImportError, TypeError):
        pass
    if type(value).__name__ in ("Image", "Asset") and not isinstance(value, Mapping):
        # typesafe-sdk asset objects: a file or bytes, never text worth keeping.
        return IMAGE_OMITTED
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return to_jsonable(method())
            except Exception:  # noqa: BLE001 - fall through to the next spelling
                continue
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            str(key): to_jsonable(item) for key, item in attrs.items() if not key.startswith("_")
        }
    # Anything else, a generator included, is described rather than read: reading
    # it here would consume what the SDK is about to send.
    return str(value)


def omit_images(value: Any) -> Any:
    """Replace every image in a JSON value with ``IMAGE_OMITTED``."""
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:200]:
            return IMAGE_OMITTED
        return value
    if isinstance(value, list):
        return [omit_images(item) for item in value]
    if isinstance(value, dict):
        if str(value.get("type", "")).lower() in _IMAGE_PART_TYPES:
            return {
                key: (item if key in _IMAGE_KEEP_KEYS else IMAGE_OMITTED)
                for key, item in value.items()
            }
        return {key: omit_images(item) for key, item in value.items()}
    return value


def payload_json(payload: Any, max_length: int = 0) -> str:
    """``payload`` as JSON with images omitted, capped at ``max_length`` (0 = unlimited)."""
    text = json.dumps(omit_images(to_jsonable(payload)), default=str, sort_keys=True)
    if max_length and max_length > 0:
        return text[:max_length]
    return text


# --------------------------------------------------------------------------- #
# Attributes
# --------------------------------------------------------------------------- #


def _distinct(values: List[str]) -> List[str]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def request_attributes(request: DecisionRequest) -> Dict[str, Any]:
    """Scalar span attributes known before the call. Never carries content."""
    attrs: Dict[str, Any] = {
        "openinference.span.kind": "LLM" if request.legacy_llm_kind else DECISION_SPAN_KIND,
        "decision.system": request.system,
        "gen_ai.system": request.system,
        "gen_ai.operation.name": request.operation,
        "gen_ai.request.type": "decision",
        "genai.decision.protocol": request.protocol,
        "genai.decision.question_count": len(request.questions),
    }
    if request.provider:
        attrs["decision.provider"] = request.provider
    if request.model:
        attrs["decision.model_name"] = request.model
        attrs["gen_ai.request.model"] = request.model
    if len(request.questions) == 1 and request.questions[0].option_count is not None:
        attrs["genai.decision.option_count"] = request.questions[0].option_count
    return attrs


def request_list_attributes(request: DecisionRequest, capture: bool) -> Dict[str, Any]:
    """Array attributes known before the call, kept apart so they stay arrays."""
    attrs: Dict[str, Any] = {}
    types = _distinct([q.type for q in request.questions])
    if types:
        attrs["genai.decision.question.types"] = types
    if capture:
        names = [q.name for q in request.questions if q.name]
        if names:
            attrs["genai.decision.question.names"] = names
    return attrs


def request_content_attributes(request: DecisionRequest, max_length: int) -> Dict[str, Any]:
    """``input.value`` for content capture. The caller decides whether capture is on."""
    if request.payload is None:
        return {}
    return {
        "input.value": payload_json(request.payload, max_length),
        "input.mime_type": "application/json",
    }


def response_attributes(
    request: DecisionRequest,
    answers: List[DecisionAnswer],
    usage: DecisionUsage,
    response_model: Optional[str],
    capture: bool,
    payload: Any = None,
    max_length: int = 0,
) -> Dict[str, Any]:
    """Span attributes from the response, under the content-capture rule (Q4)."""
    attrs: Dict[str, Any] = {}
    request_model = request.model
    if response_model:
        attrs["gen_ai.response.model"] = response_model
        attrs["decision.model_name"] = response_model
        if request_model and request_model != response_model:
            attrs["decision.request.model_name"] = request_model
            attrs["decision.response.model_name"] = response_model

    if usage.input_tokens is not None:
        attrs["decision.token_count.input"] = usage.input_tokens
        attrs["gen_ai.usage.input_tokens"] = usage.input_tokens
    if usage.output_tokens is not None:
        attrs["decision.token_count.output"] = usage.output_tokens
        attrs["gen_ai.usage.output_tokens"] = usage.output_tokens

    questions = request.questions
    count = len(questions) or len(answers)
    attrs["genai.decision.question_count"] = count
    attrs["genai.decision.refusal_count"] = sum(1 for a in answers if a.type == "refusal")

    # The types as answered: a refused question is recorded as a refusal.
    types = []
    for index in range(max(len(questions), len(answers))):
        answer = answers[index] if index < len(answers) else None
        if answer is not None and answer.type == "refusal":
            types.append("refusal")
        elif index < len(questions):
            types.append(questions[index].type)
        elif answer is not None:
            types.append(answer.type)
    if types:
        attrs["genai.decision.question.types"] = _distinct(types)

    single = count == 1
    answer = answers[0] if single and answers else None
    if single:
        option_count = questions[0].option_count if questions else None
        if option_count is None and answer is not None and answer.probabilities:
            option_count = len(answer.probabilities)
        if option_count is not None:
            attrs["genai.decision.option_count"] = option_count

    if not capture:
        return attrs

    names = [q.name for q in questions if q.name] or [a.name for a in answers if a.name]
    if names:
        attrs["genai.decision.question.names"] = names
    confidences = [
        a.confidence for a in answers if a.type in _CONFIDENCE_TYPES and a.confidence is not None
    ]
    if confidences:
        attrs["genai.decision.min_confidence"] = min(confidences)
    if answer is not None and answer.type != "refusal":
        if answer.choice is not None:
            attrs["genai.decision.choice"] = answer.choice
        if answer.confidence is not None:
            attrs["genai.decision.confidence"] = answer.confidence
        if answer.probability is not None:
            attrs["genai.decision.probability"] = answer.probability
        if answer.score is not None:
            attrs["genai.decision.score"] = answer.score
        if len(answer.probabilities) >= 2:
            top = sorted(answer.probabilities, reverse=True)
            attrs["genai.decision.margin"] = top[0] - top[1]
    if payload is not None:
        attrs["output.value"] = payload_json(payload, max_length)
        attrs["output.mime_type"] = "application/json"
    return attrs


def _inside_decision_span() -> bool:
    """Whether the current span is a decision span this library already opened."""
    span = trace.get_current_span()
    attributes = getattr(span, "attributes", None)
    try:
        return bool(attributes) and "genai.decision.protocol" in attributes
    except Exception:  # noqa: BLE001 - an exotic span type is simply not ours
        return False


class DecisionRecorder:
    """What a decision call returned, recorded by the code that made it."""

    def __init__(self):
        self.payload: Any = None
        self.has_result = False
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None

    def record(self, answers: Any, usage: Any = None, model: Optional[str] = None) -> None:
        """Record the answers, in either protocol's shape, and the usage if reported.

        ``answers`` is an OpenAI-style list (``[{"type": "choice", "choice": ...,
        "confidence": ..., "probabilities": [...]}]``) or a SystemOne-style mapping
        of question names to answers. ``usage`` takes ``input_tokens`` /
        ``output_tokens``; leave it out when the model reports none, and the span
        records no tokens rather than zero.
        """
        payload: Dict[str, Any] = {"answers": answers}
        if usage is not None:
            payload["usage"] = usage
        if model:
            payload["model"] = model
        self.record_response(payload)

    def record_response(self, response: Any) -> None:
        """Record a whole response object or JSON body (``answers``, ``usage``, ``model``)."""
        self.payload = response
        self.has_result = True

    def fail(self, message: str, error_type: Optional[str] = None) -> None:
        """Mark the decision failed without an exception (an HTTP error status, say)."""
        self.error = message
        self.error_type = error_type


# --------------------------------------------------------------------------- #
# Instrumentor base
# --------------------------------------------------------------------------- #

# Where the request travels from the moment the span opens to the moment the
# response is read: on the span object itself, which the base wrapper hands to
# both ends. (The base class keeps its agent-budget bookkeeping there too.)
_REQUEST_ON_SPAN = "_genai_decision_request"


class BaseDecisionInstrumentor(BaseInstrumentor):
    """What every decision call path shares: the span shape, usage, cost and evaluation.

    Subclasses build a :class:`DecisionRequest` from the call and read the
    answers out of the response; this class writes both onto the span through
    the base wrapper, so decisions get the same metrics, cost, block policies
    and detectors as every other call.
    """

    def instrument(self, config):  # pragma: no cover - subclasses implement it
        raise NotImplementedError

    # -- configuration -------------------------------------------------------

    def _capture(self) -> bool:
        """Content capture, strictly: with no configuration nothing is captured (Q4)."""
        return bool(
            self.config is not None and getattr(self.config, "enable_content_capture", False)
        )

    def _max_length(self) -> int:
        value = getattr(self.config, "content_max_length", 0) if self.config else 0
        return value if isinstance(value, int) and value > 0 else 0

    def _configured_provider(self) -> Optional[str]:
        return getattr(self.config, "decision_provider", None) if self.config else None

    # -- hooks for subclasses ------------------------------------------------

    def _decision_answers(
        self, request: DecisionRequest, payload: Any
    ) -> Tuple[List[DecisionAnswer], DecisionUsage, Optional[str]]:
        """Answers, usage and response model out of ``payload``."""
        raw_answers = _get(payload, "answers")
        if request.protocol == PROTOCOL_OPENAI_DECISIONS or (
            request.protocol == PROTOCOL_MANUAL and not isinstance(raw_answers, Mapping)
        ):
            answers = openai_answers(raw_answers, request.questions)
        elif request.protocol == PROTOCOL_GRANITE_ADAPTER:
            count = request.questions[0].option_count if request.questions else None
            answers = granite_answers(payload, count)
        else:
            answers = systemone_answers(payload, request.questions)
        model = _get(payload, "model")
        return answers, usage_of(_get(payload, "usage")), (str(model) if model else None)

    def _extra_request_attributes(self, request: DecisionRequest, capture: bool) -> Dict[str, Any]:
        """Request attributes a call path keeps for its own older readers."""
        return {}

    def _decision_payload(self, result: Any) -> Any:
        """The parsed response a wrapped call returned."""
        return _unwrap_api_response(result)

    # -- span lifecycle ------------------------------------------------------

    def _traced_decision(
        self,
        span_name: str,
        request: DecisionRequest,
        wrapped: Callable,
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Call ``wrapped`` inside a decision span opened by the base wrapper."""
        capture = self._capture()

        def extract(_instance, _args, _kwargs) -> Dict[str, Any]:
            attrs = request_attributes(request)
            for key, value in _server_attributes(instance).items():
                attrs.setdefault(key, value)
            if capture:
                attrs.update(request_content_attributes(request, self._max_length()))
            for key, value in self._extra_request_attributes(request, capture).items():
                attrs.setdefault(key, value)
            return attrs

        def call(*call_args, **call_kwargs):
            # Runs inside the span's context. Arrays are set here rather than
            # through ``extract``, whose values the base wrapper stringifies.
            span = trace.get_current_span()
            try:
                setattr(span, _REQUEST_ON_SPAN, request)
                for key, value in request_list_attributes(request, capture).items():
                    span.set_attribute(key, value)
            except Exception as e:  # noqa: BLE001 - telemetry must not break the call
                logger.debug("Could not annotate decision span: %s", e)
            return wrapped(*call_args, **call_kwargs)

        return self.create_span_wrapper(span_name=span_name, extract_attributes=extract)(call)(
            *args, **kwargs
        )

    def _decision_response_attributes(
        self, request: DecisionRequest, payload: Any
    ) -> Dict[str, Any]:
        answers, usage, response_model = self._decision_answers(request, payload)
        capture = self._capture()
        return response_attributes(
            request,
            answers,
            usage,
            response_model,
            capture,
            payload=self._content_payload(payload) if capture else None,
            max_length=self._max_length(),
        )

    def _content_payload(self, payload: Any) -> Any:
        """What ``output.value`` records: the response as returned."""
        return payload

    def _record_result_metrics(self, span, result, start_time: float, request_kwargs: dict = None):
        payload = self._decision_payload(result)
        request = getattr(span, _REQUEST_ON_SPAN, None)
        if isinstance(request, DecisionRequest):
            try:
                for key, value in self._decision_response_attributes(request, payload).items():
                    span.set_attribute(key, value)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to read decision response for span: %s", e)
        super()._record_result_metrics(span, payload, start_time, request_kwargs)

    def _extract_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Response attributes are written by :meth:`_record_result_metrics`."""
        return {}

    def _extract_usage(self, result: Any) -> Optional[Dict[str, Any]]:
        """Usage on the base wrapper's canonical keys, or None when none was reported."""
        usage = usage_of(_get(result, "usage"))
        if not usage.reported:
            return None
        prompt = usage.input_tokens or 0
        completion = usage.output_tokens or 0
        out: Dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        if usage.cached_input_tokens:
            out["cache_read_input_tokens"] = usage.cached_input_tokens
        if usage.cache_write_input_tokens:
            out["cache_creation_input_tokens"] = usage.cache_write_input_tokens
        return out

    @contextlib.contextmanager
    def _run_decision_span(
        self, span_name: str, request: DecisionRequest, eval_kwargs: Dict[str, Any]
    ):
        """A decision span for a call this library does not wrap through an SDK method.

        Used by the HTTP path matcher and the manual helper. Yields a
        :class:`DecisionRecorder`; whatever is recorded on it is written to the
        span when the block exits. Unlike the base wrapper, a call that returned
        but failed (an HTTP 4xx/5xx) ends with status ERROR.
        """
        capture = self._capture()
        attrs = request_attributes(request)
        if capture:
            attrs.update(request_content_attributes(request, self._max_length()))
        for key, value in self._extra_request_attributes(request, capture).items():
            attrs.setdefault(key, value)
        span = self.tracer.start_span(span_name, attributes=self._with_provider_aliases(attrs))
        token = otel_context.attach(trace.set_span_in_context(span))
        recording = span.is_recording()
        if recording:
            try:
                setattr(span, _REQUEST_ON_SPAN, request)
                for key, value in request_list_attributes(request, capture).items():
                    span.set_attribute(key, value)
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not annotate decision span: %s", e)
        start_time = time.time()
        recorder = DecisionRecorder()
        evaluate_later = False
        try:
            if BaseInstrumentor._blocking_active:
                self._enforce_precall_policy(span, eval_kwargs, request.operation)
            yield recorder
        except Exception as e:
            try:
                if self.error_counter:
                    self.error_counter.add(
                        1, {"operation": span_name, "error_type": type(e).__name__}
                    )
            except Exception:  # noqa: BLE001 - metrics must never break the call
                pass
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        else:
            try:
                if self.request_counter:
                    self.request_counter.add(1, {"operation": span_name})
                if recorder.error:
                    span.set_status(Status(StatusCode.ERROR, recorder.error))
                    span.set_attribute("error.type", recorder.error_type or "error")
                if recording and recorder.has_result:
                    self._record_result_metrics(span, recorder.payload, start_time, eval_kwargs)
                elif self.latency_histogram:
                    self.latency_histogram.record(
                        time.time() - start_time, {"operation": span_name}
                    )
                if not recorder.error:
                    span.set_status(Status(StatusCode.OK))
                    if recording and recorder.has_result:
                        evaluate_later = self._evaluate_or_defer(
                            span, (), eval_kwargs, recorder.payload
                        )
            except Exception as e:  # noqa: BLE001 - telemetry must not break the call
                logger.warning("Failed to record decision span '%s': %s", span_name, e)
        finally:
            otel_context.detach(token)
            if evaluate_later:
                self._end_span_after_evaluation(span, (), eval_kwargs, recorder.payload)
            else:
                span.end()

    # -- evaluation ----------------------------------------------------------

    def _extract_prompt_for_eval(self, kwargs, operation: Optional[str] = None) -> Optional[str]:
        """The decision's input: SystemOne ``state`` and questions, or OpenAI ``input``."""
        if not isinstance(kwargs, dict):
            return None
        if kwargs.get("state") is not None:
            return json.dumps(
                to_jsonable({"state": kwargs.get("state"), "questions": kwargs.get("questions")}),
                default=str,
                sort_keys=True,
            )
        if kwargs.get("input") is not None:
            return BaseInstrumentor._responses_prompt_text(kwargs)
        return BaseInstrumentor._extract_prompt_for_eval(kwargs, operation)

    def _extract_response_for_eval(self, result) -> Optional[str]:
        """A decision has no generated text: nothing for a response detector to read."""
        return None

    def _evaluation_texts(self, span, kwargs, result) -> Tuple[Optional[str], Optional[str]]:
        """The prompt only. Hallucination and response checks have no text to judge.

        The span says why rather than recording nothing, so an absent
        hallucination score is not read as "checked and clean".
        """
        prompt, _ = super()._evaluation_texts(span, kwargs, result)
        try:
            span.set_attribute("evaluation.skipped_reason", "decision_span")
        except Exception:  # noqa: BLE001
            pass
        return prompt, None
