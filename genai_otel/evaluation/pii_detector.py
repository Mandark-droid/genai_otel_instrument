"""PII Detection using Microsoft Presidio.

This module provides PII (Personally Identifiable Information) detection and redaction
capabilities using Microsoft Presidio library.
"""

import ipaddress
import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .config import DEFAULT_CACHE_SIZE, PIIConfig, PIIEntityType, PIIMode
from .result_cache import ResultCache

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
# How far past the nominal cut the model's part may start, to begin on a word.
_BOUNDARY_SEARCH = 32
# How far into the model's part the patterns keep reading, so that a number or an
# address that begins before the cut and ends after it is matched whole.
_PATTERN_OVERLAP = 64

# What Presidio gives a card number or an e-mail address that passed its
# validation, and a phone number that no word near it supports.
_VALIDATED_SCORE = 1.0
_PHONE_SCORE_WITHOUT_CONTEXT = 0.4

# Process-wide "log once" guard for the air-gapped notice.
_AIRGAP_NOTICE_LOGGED = False


def _log_airgap_notice_once(message: str) -> None:
    global _AIRGAP_NOTICE_LOGGED
    if not _AIRGAP_NOTICE_LOGGED:
        _AIRGAP_NOTICE_LOGGED = True
        logger.info(message)


# The checks below are the ones Presidio's recognisers make on what their
# patterns find. The patterns in this module read the part of a long text that
# the model does not; without the same checks they reported order numbers,
# timestamps and file names that Presidio, reading the same text, would not.


def _is_card_number(candidate: str) -> bool:
    """Whether sixteen digits start as a payment card does and pass its checksum."""
    digits = [int(char) for char in candidate if char.isdecimal()]
    if len(digits) < 2:
        return False
    if digits[0] not in (1, 3, 4, 6) and not (digits[0] == 5 and digits[1] <= 5):
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _is_issued_ssn(candidate: str) -> bool:
    """Whether ``ddd-dd-dddd`` could be a social security number that was issued."""
    digits = "".join(char for char in candidate if char.isdecimal())
    if len(set(digits)) == 1:
        return False
    if digits[3:5] == "00" or digits[5:] == "0000":
        return False
    if digits[:3] in ("000", "666"):
        return False
    # Numbers published as examples.
    return digits not in ("123456789", "987654320", "078051120")


def _is_ip_address(candidate: str) -> bool:
    try:
        ipaddress.ip_interface(candidate)
    except ValueError:
        return False
    return True


# The public suffix list, built the first time an address has to be checked.
# None = not tried yet, False = there is none to check against.
_SUFFIX_LIST: Any = None


def _has_public_suffix(address: str) -> Optional[bool]:
    """Whether an e-mail address ends in a domain that can exist.

    ``icon@2x.png`` has the shape of an address and is a file name. None when
    there is no list to check against.
    """
    global _SUFFIX_LIST
    if _SUFFIX_LIST is None:
        try:
            import tldextract

            # The list that ships with the package. Given no address to fetch a
            # newer one from and no cache directory, a lookup sends no request
            # and writes no file.
            _SUFFIX_LIST = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())
        except Exception as e:  # noqa: BLE001
            logger.debug("No public suffix list to check e-mail addresses against: %s", e)
            _SUFFIX_LIST = False
    if _SUFFIX_LIST is False:
        return None
    try:
        return _SUFFIX_LIST(address).fqdn != ""
    except Exception:  # noqa: BLE001
        return None


def _as_presidio_reports(entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A pattern finding as Presidio's recogniser for its type reports it.

    None when that recogniser does not report it at all. The recognisers for
    national identifiers are this library's own patterns, so those findings
    already are what the model path reports.
    """
    kind, text = entity["type"], entity["text"]
    if kind == "CREDIT_CARD":
        return dict(entity, score=_VALIDATED_SCORE) if _is_card_number(text) else None
    if kind == "EMAIL_ADDRESS":
        exists = _has_public_suffix(text)
        if exists is None:
            return entity
        return dict(entity, score=_VALIDATED_SCORE) if exists else None
    if kind == "US_SSN":
        return entity if _is_issued_ssn(text) else None
    if kind == "IP_ADDRESS":
        return entity if _is_ip_address(text) else None
    if kind == "PHONE_NUMBER":
        # Presidio raises a phone number above this only when a word before it
        # ("phone", "mobile", ...) supports it, and finding those words takes
        # the model. Ten digits alone are a timestamp as often as a phone number.
        return dict(entity, score=_PHONE_SCORE_WITHOUT_CONTEXT)
    return entity


_WORD_BOUNDARY = re.compile(r"\b")


class _AddressSearch:
    """Finds what an address pattern matches, looking from each "@" outwards.

    A search for the whole pattern tries it from every place a word starts:
    it reads on from there for as long as an address may be, to see whether an
    "@" follows. In a run of punctuation a word starts at every character, and
    a long prompt made of one, with a single "@" in it, kept every other
    thread waiting while each of them was read on from.

    Whether what follows an "@" completes an address does not depend on where
    the address began. So the "@" that are followed by the rest of an address
    are found first, and each address then begins at the first place a word
    starts in the run of address characters before its "@". The matches are
    the ones the pattern gives.
    """

    def __init__(self, pattern: str):
        # The shape both address patterns have: where a word starts, a bounded
        # run of the characters an address may have before its "@", the "@",
        # and the rest.
        before, at, after = pattern.partition("@")
        characters, bounded, longest = before[2:-1].rpartition("{1,")
        if not (
            at
            and after
            and before.startswith("\\b[")
            and before.endswith("}")
            and bounded
            and characters.endswith("]")
            and longest.isdigit()
        ):
            raise ValueError("Not an address pattern: %r" % (pattern,))
        self._before = re.compile(characters + "*")
        self._longest = int(longest)
        self._at_and_after = re.compile("@(?:%s)" % after)

    def spans(self, text: str) -> List[Tuple[int, int]]:
        """Where the pattern matches in ``text``, in order, as (start, end)."""
        found = []
        # Where the last match ended: the next one cannot begin before it.
        end = 0
        for rest in self._at_and_after.finditer(text):
            at = rest.start()
            # The run of address characters that ends at the "@", counted from
            # the "@" backwards. It cannot reach past another "@": that is not
            # an address character.
            earliest = max(end, at - self._longest)
            run = self._before.match(text[earliest:at][::-1]).end()
            begins = _WORD_BOUNDARY.search(text, at - run, at) if run else None
            if begins is not None and begins.start() < at:
                end = rest.end()
                found.append((begins.start(), end))
        return found


_ADDRESS_SEARCHES: Dict[str, _AddressSearch] = {}


def _address_spans(pattern: str, text: str) -> List[Tuple[int, int]]:
    """Where the address pattern ``pattern`` matches in ``text``."""
    search = _ADDRESS_SEARCHES.get(pattern)
    if search is None:
        search = _ADDRESS_SEARCHES[pattern] = _AddressSearch(pattern)
    return search.spans(text)


@dataclass
class PIIDetectionResult:
    """Result of PII detection.

    Attributes:
        has_pii: Whether PII was detected
        entities: List of detected PII entities
        entity_counts: Count of entities by type
        redacted_text: Text with PII redacted (if redaction enabled)
        original_text: Original input text
        score: Overall PII detection confidence score
        blocked: Whether the text was blocked due to PII
        model_analyzed_length: Characters at the end of the text that the NER
            model read, when it was limited to part of it. None when it read
            all of the text. Pattern-based findings always cover the whole text.
    """

    has_pii: bool
    entities: List[Dict[str, Any]] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    redacted_text: Optional[str] = None
    original_text: Optional[str] = None
    score: float = 0.0
    blocked: bool = False
    model_analyzed_length: Optional[int] = None


class PIIDetector:
    """PII detector using Microsoft Presidio.

    This detector uses Presidio's analyzer and anonymizer to detect and redact
    personally identifiable information from text.

    Requirements:
        pip install presidio-analyzer presidio-anonymizer spacy
        python -m spacy download en_core_web_lg
    """

    # detect() takes ``model_chars``: callers that cap how much text is analysed
    # hand over the whole text and let the patterns cover what the model skips.
    accepts_model_chars = True

    def __init__(self, config: PIIConfig):
        """Initialize PII detector.

        Args:
            config: PII detection configuration
        """
        self.config = config
        self._analyzer = None
        self._anonymizer = None
        self._presidio_available = False
        self._custom_entities: list = []
        self._cache = ResultCache(getattr(config, "cache_size", DEFAULT_CACHE_SIZE))
        # Air-gap / content-cap posture (mirrors OTelConfig; may be plumbed from
        # it). Env-var fallback so a foreign/older config still defaults safely.
        self._air_gapped = getattr(
            config,
            "air_gapped",
            os.getenv("GENAI_AIR_GAPPED", "false").lower() == "true",
        )
        self._content_max_length = getattr(
            config,
            "content_max_length",
            int(os.getenv("GENAI_CONTENT_MAX_LENGTH", "200")),
        )
        self._check_availability()

    def _cap_text(self, text: Optional[str]) -> Optional[str]:
        """Bound exported redacted text so a near-full copy of the prompt /
        response cannot bloat the span. 0 (or negative) means unlimited."""
        if text is None:
            return None
        cap = self._content_max_length
        if cap and cap > 0 and len(text) > cap:
            return text[:cap]
        return text

    def _check_availability(self):
        """Check if Presidio is available."""
        # On air-gapped hosts, force the underlying spaCy / HuggingFace stack
        # offline so model resolution can never trigger a runtime network
        # download. The required models (e.g. en_core_web_lg) must be
        # pre-installed; if they are missing, Presidio init fails and we fall
        # back to local regex patterns instead of reaching out to the network.
        if self._air_gapped:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            _log_airgap_notice_once(
                "Air-gapped mode enabled: PII models must be pre-installed "
                "(no runtime spaCy/HuggingFace download will be attempted)."
            )
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._disable_dependency_parser()
            self._anonymizer = AnonymizerEngine()
            self._register_india_recognizers()
            self._register_custom_recognizers()
            self._presidio_available = True
            logger.info("Presidio PII detection initialized successfully")
        except ImportError as e:
            logger.warning("Presidio not available, PII detection will be limited: %s", e)
            logger.info(
                "Install with: pip install presidio-analyzer presidio-anonymizer spacy && python -m spacy download en_core_web_lg"
            )
            self._presidio_available = False
        except Exception as e:
            logger.error("Failed to initialize Presidio: %s", e)
            self._presidio_available = False

    def _disable_dependency_parser(self):
        """Stop spaCy running its dependency parser on every text.

        Presidio reads entities, tokens and lemmas from the pipeline. Nothing
        reads the dependency parse, yet it is computed for every text analysed.
        A pipeline without a parser, an NLP engine that is not spaCy, or a parser
        that refuses to be disabled is left as it is: detection works either way.
        """
        try:
            pipelines = getattr(self._analyzer.nlp_engine, "nlp", None)
            if not isinstance(pipelines, dict):
                return
            for nlp in pipelines.values():
                if "parser" in getattr(nlp, "pipe_names", ()):
                    nlp.disable_pipe("parser")
                    logger.debug("spaCy dependency parser disabled for PII detection")
        except Exception as e:  # noqa: BLE001
            logger.debug("spaCy dependency parser left enabled: %s", e)

    def _register_india_recognizers(self):
        """Register custom Presidio recognizers for India-specific PII types."""
        try:
            from presidio_analyzer import Pattern, PatternRecognizer

            # Aadhaar: 12 digits in groups of 4 (XXXX XXXX XXXX)
            aadhaar_recognizer = PatternRecognizer(
                supported_entity="IN_AADHAAR",
                name="Indian Aadhaar Recognizer",
                patterns=[
                    Pattern(
                        name="aadhaar",
                        regex=r"\b\d{4}\s?\d{4}\s?\d{4}\b",
                        score=0.85,
                    )
                ],
                supported_language="en",
            )

            # PAN: 5 uppercase letters + 4 digits + 1 uppercase letter (ABCDE1234F)
            pan_recognizer = PatternRecognizer(
                supported_entity="IN_PAN",
                name="Indian PAN Recognizer",
                patterns=[
                    Pattern(
                        name="pan",
                        regex=r"\b[A-Z]{5}\d{4}[A-Z]\b",
                        score=0.9,
                    )
                ],
                supported_language="en",
            )

            # UPI: user@bankname (e.g., name@oksbi, name@paytm, name@ybl). The
            # name is read up to a fixed length, as in _scan_patterns.
            upi_recognizer = PatternRecognizer(
                supported_entity="IN_UPI",
                name="Indian UPI Address Recognizer",
                patterns=[
                    Pattern(
                        name="upi",
                        regex=r"\b[\w.+-]{1,256}@(?:ok(?:hdfcbank|axis|icici|sbi|bizaxis)|paytm|ybl|upi|apl|ibl|axl|barodampay|mahb|cnrb|pnb|sib|federal|kotak|indus|citi|rbl|idbi|aubank|dlb|kvb|csbpay|idfcfirst|freecharge|slice|jupiter|fi|gpay|oksbi)\b",
                        score=0.9,
                    )
                ],
                supported_language="en",
            )

            # Indian phone: +91/91/0 prefix, starts with 6-9
            in_phone_recognizer = PatternRecognizer(
                supported_entity="IN_PHONE",
                name="Indian Phone Number Recognizer",
                patterns=[
                    Pattern(
                        name="in_phone",
                        regex=r"\b(?:\+91|91|0)?[6-9]\d{9}\b",
                        score=0.8,
                    )
                ],
                supported_language="en",
            )

            ifsc_recognizer = PatternRecognizer(
                supported_entity="IN_IFSC",
                name="Indian IFSC Recognizer",
                patterns=[Pattern(name="ifsc", regex=r"\b[A-Z]{4}0[A-Z0-9]{6}\b", score=0.9)],
                supported_language="en",
            )
            registry = self._analyzer.registry
            registry.add_recognizer(aadhaar_recognizer)
            registry.add_recognizer(ifsc_recognizer)
            registry.add_recognizer(pan_recognizer)
            registry.add_recognizer(upi_recognizer)
            registry.add_recognizer(in_phone_recognizer)
            logger.debug("Registered India-specific PII recognizers (Aadhaar, PAN, UPI, Phone)")

        except Exception as e:
            logger.warning("Failed to register India PII recognizers: %s", e)

    def _register_custom_recognizers(self):
        """Register custom Presidio recognizers from ``config.custom_patterns``.

        ``custom_patterns`` maps an entity label to either a regex string or a
        dict ``{"regex": ..., "score": 0.8}`` — letting customers add their own
        PII / BFSI classes (e.g. an internal customer-reference number) with no
        code changes. Registered entity labels are tracked so :meth:`detect`
        includes them in the analysis.
        """
        patterns = self.config.custom_patterns or {}
        if not patterns:
            return
        try:
            from presidio_analyzer import Pattern, PatternRecognizer

            registry = self._analyzer.registry
            for entity, spec in patterns.items():
                regex = spec["regex"] if isinstance(spec, dict) else spec
                score = spec.get("score", 0.8) if isinstance(spec, dict) else 0.8
                registry.add_recognizer(
                    PatternRecognizer(
                        supported_entity=entity,
                        name=f"Custom {entity} Recognizer",
                        patterns=[Pattern(name=entity.lower(), regex=regex, score=score)],
                        supported_language="en",
                    )
                )
                self._custom_entities.append(entity)
            logger.debug(
                "Registered %d custom PII recognizer(s): %s", len(patterns), list(patterns)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to register custom PII recognizers: %s", e)

    def is_available(self) -> bool:
        """Check if PII detector is available.

        Returns:
            bool: True if Presidio is available
        """
        return self._presidio_available

    @property
    def reads_whole_text(self) -> bool:
        """Whether the model reads all of a text whatever limit a caller asks for.

        In redact mode it does. A redacted copy is exported in place of the
        text; built from text the model did not read, it would carry every name
        in that text and still be called redacted.
        """
        return self.config.mode == PIIMode.REDACT

    def detect(
        self, text: str, language: str = "en", model_chars: Optional[int] = 0
    ) -> PIIDetectionResult:
        """Detect PII in text.

        Args:
            text: Text to analyze
            language: Language code (default: "en")
            model_chars: When positive and the text is longer, the NER model
                reads only that many characters at the end of it; the patterns
                that need no model read everything before. 0 means the model
                reads the whole text, and in redact mode it always does.

        Returns:
            PIIDetectionResult: Detection results
        """
        if not self.config.enabled:
            return PIIDetectionResult(has_pii=False, original_text=text)

        if self.reads_whole_text:
            model_chars = 0

        key = self._cache_key(text, language, model_chars)
        cached = self._cache.get(key)
        if cached is not None:
            return self._remembered(cached, text)

        if not self._presidio_available:
            logger.warning("Presidio not available, using pattern-based detection")
            result = self._fallback_detection(text)
        else:
            try:
                result = self._model_detection(text, language, model_chars)
            except Exception as e:
                # Not remembered: a detector that could not run answers "nothing
                # found", and it must be asked again once the fault has cleared.
                logger.error("Error detecting PII: %s", e, exc_info=True)
                return PIIDetectionResult(has_pii=False, original_text=text)

        self._cache.put(key, self._to_remember(result))
        return result

    @staticmethod
    def _to_remember(result: PIIDetectionResult) -> PIIDetectionResult:
        """``result`` as the cache keeps it: where things were found, not what they were.

        An entry outlives the request it came from. It holds entity types,
        positions, scores and counts. The text, the text of each finding and the
        redacted copy are left out, and put back from the caller's text when the
        entry is used.
        """
        return replace(
            result,
            entities=[dict(entity, text=None) for entity in result.entities],
            entity_counts=dict(result.entity_counts),
            redacted_text=None,
            original_text=None,
        )

    def _remembered(self, cached: PIIDetectionResult, text: str) -> PIIDetectionResult:
        """A cached entry made whole again for ``text``, the text it was computed from."""
        entities = [
            dict(entity, text=text[entity["start"] : entity["end"]]) for entity in cached.entities
        ]
        redacted_text = None
        if self.config.mode == PIIMode.REDACT and cached.has_pii:
            if self._presidio_available:
                redacted_text = self._cap_text(self._redact_pii(text, entities))
            else:
                redacted_text = self._cap_text(self._redact_each(text, entities))
        return replace(
            cached,
            entities=entities,
            entity_counts=dict(cached.entity_counts),
            redacted_text=redacted_text,
            original_text=self._text_to_report(cached.has_pii, text),
        )

    def _cache_key(self, text: str, language: str, model_chars: Optional[int]):
        """Key for a result: the text and every setting the result depends on."""
        if not self._cache.enabled:
            return None
        config = self.config
        try:
            return self._cache.key(
                text,
                language,
                model_chars if isinstance(model_chars, int) and model_chars > 0 else 0,
                self._presidio_available,
                config.mode,
                config.threshold,
                frozenset(config.entity_types),
                tuple(config.allow_list or ()),
                repr(config.custom_patterns) if config.custom_patterns else "",
                config.redaction_char,
                self._content_max_length,
            )
        except TypeError:
            return None

    def _text_to_report(self, has_pii: bool, text: str) -> Optional[str]:
        """The text a result carries: withheld once PII was found, unless only detecting."""
        return None if (has_pii and self.config.mode != PIIMode.DETECT) else text

    @staticmethod
    def _model_start(text: str, model_chars: Optional[int]) -> int:
        """Where the NER model starts reading ``text``. 0 = at the beginning."""
        if isinstance(model_chars, bool) or not isinstance(model_chars, int) or model_chars <= 0:
            return 0
        if len(text) <= model_chars:
            return 0
        start = len(text) - model_chars
        if text[start - 1].isspace():
            return start
        # Begin on a word rather than in the middle of one. Forward only, so the
        # model never reads more than it was allowed.
        boundary = _WHITESPACE.search(text, start, start + _BOUNDARY_SEARCH)
        if boundary and boundary.end() < len(text):
            return boundary.end()
        return start

    def _model_detection(
        self, text: str, language: str, model_chars: Optional[int]
    ) -> PIIDetectionResult:
        """Detect with Presidio. On long text the model reads the end, patterns the rest."""
        # Convert entity types to Presidio format
        entity_types = [entity.value for entity in self.config.entity_types] + self._custom_entities

        # Whoever calls this, a redacted copy is only ever built from text the
        # model has read all of.
        model_start = 0 if self.reads_whole_text else self._model_start(text, model_chars)

        # Analyze text
        results = self._analyzer.analyze(
            text=text[model_start:] if model_start else text,
            language=language,
            entities=entity_types,
            score_threshold=self.config.threshold,
            allow_list=self.config.allow_list,
        )

        # Process results
        entities: List[Dict[str, Any]] = []
        entity_counts: Dict[str, int] = {}

        # What the patterns found before the part the model read. A finding that
        # begins there and runs on into the model part is reported from here,
        # whole: the model was shown only the second half of it.
        covered_to = 0
        if model_start:
            entities = self._pattern_entities_before(text, model_start)
            covered_to = max((entity["end"] for entity in entities), default=0)
            for entity in entities:
                entity_counts[entity["type"]] = entity_counts.get(entity["type"], 0) + 1

        for result in results:
            start, end = result.start + model_start, result.end + model_start
            if start < covered_to:
                continue
            entity = {
                "type": result.entity_type,
                "start": start,
                "end": end,
                "score": result.score,
                "text": text[start:end],
            }
            entities.append(entity)

            # Count by type
            entity_type = result.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        has_pii = len(entities) > 0

        # Calculate overall score (max confidence)
        score = max([e["score"] for e in entities], default=0.0)

        # Redact if mode is REDACT
        redacted_text = None
        if self.config.mode == PIIMode.REDACT and has_pii:
            # Bound the redacted copy so it cannot export a near-full,
            # untruncated copy of the (still sensitive) original text.
            redacted_text = self._cap_text(self._redact_pii(text, entities))

        # Block if mode is BLOCK
        blocked = self.config.mode == PIIMode.BLOCK and has_pii

        return PIIDetectionResult(
            has_pii=has_pii,
            entities=entities,
            entity_counts=entity_counts,
            redacted_text=redacted_text,
            original_text=self._text_to_report(has_pii, text),
            score=score,
            blocked=blocked,
            model_analyzed_length=(len(text) - model_start) if model_start else None,
        )

    def _pattern_entities_before(self, text: str, model_start: int) -> List[Dict[str, Any]]:
        """Pattern findings that begin before ``model_start``, under the configured rules.

        Reported as Presidio would report them, since it is Presidio that would
        have read this part of the text: see :func:`_as_presidio_reports`.
        """
        allowed = set(self.config.allow_list or ())
        found, _ = self._scan_patterns(text[: model_start + _PATTERN_OVERLAP], as_presidio=True)
        return [
            entity
            for entity in found
            if entity["start"] < model_start
            and entity["score"] >= self.config.threshold
            and entity["text"] not in allowed
        ]

    def _redact_spans(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Replace each finding in ``text`` with the mark the anonymizer would use.

        Overlapping findings are covered once, over their whole extent.
        """
        configured = {entity_type.value for entity_type in self.config.entity_types}
        spans: List[List[Any]] = []
        for entity in sorted(entities, key=lambda e: (e["start"], e["end"])):
            if spans and entity["start"] < spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], entity["end"])
            else:
                spans.append([entity["start"], entity["end"], entity["type"]])
        pieces = []
        cursor = 0
        for start, end, entity_type in spans:
            pieces.append(text[cursor:start])
            if entity_type in configured:
                pieces.append(self.config.redaction_char * 8)
            else:
                pieces.append("<%s>" % entity_type)
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def _redact_pii(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Redact PII from text.

        Args:
            text: Original text
            entities: What was found in it: type, start, end and score of each

        Returns:
            str: Text with PII redacted
        """
        try:
            from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

            # Create anonymization config
            operators = {
                entity_type.value: OperatorConfig(
                    "replace", {"new_value": self.config.redaction_char * 8}
                )
                for entity_type in self.config.entity_types
            }

            # Built from the findings rather than passed on from the analyzer,
            # so that a result taken from the cache, which keeps only these four
            # things, is redacted exactly as it was the first time.
            analyzer_results = [
                RecognizerResult(
                    entity_type=entity["type"],
                    start=entity["start"],
                    end=entity["end"],
                    score=entity["score"],
                )
                for entity in entities
            ]

            # Anonymize
            anonymized = self._anonymizer.anonymize(
                text=text, analyzer_results=analyzer_results, operators=operators
            )

            return anonymized.text

        except Exception as e:
            # Never the text as it came: what this returns is exported in its
            # place, labelled as redacted.
            logger.error("Error redacting PII: %s", e)
            return self._redact_spans(text, entities)

    def _pattern_entities(self, text: str) -> List[Dict[str, Any]]:
        """Findings of the patterns that need no model."""
        return self._scan_patterns(text)[0]

    def _scan_patterns(self, text: str, as_presidio: bool = False):
        """Run the patterns that need no model over ``text``.

        With ``as_presidio`` the findings of the generic patterns are checked
        and scored the way Presidio's recognisers do it, and those it would not
        report are left out. Returns the findings and a count per entity type.
        """
        # The part of an address before the "@" is read up to a fixed length:
        # 64 characters is the longest a mail system accepts (RFC 5321), 256
        # more than a payment address gets. Unbounded, each word of a long run
        # without spaces (a dotted package list, a base64 blob) was read to the
        # end of the run and found to have no "@" after it. That took time in
        # proportion to the square of the run's length.
        email_pattern = r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        # US phone number
        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        cc_pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        # Aadhaar: 12 digits in groups of 4
        aadhaar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
        # PAN: ABCDE1234F
        pan_pattern = r"\b[A-Z]{5}\d{4}[A-Z]\b"
        # UPI address: user@bankhandle
        upi_pattern = r"\b[\w.+-]{1,256}@(?:ok(?:hdfcbank|axis|icici|sbi|bizaxis)|paytm|ybl|upi|apl|ibl|axl|barodampay|mahb|cnrb|pnb|sib|federal|kotak|indus|citi|rbl|idbi|aubank|dlb|kvb|csbpay|idfcfirst|freecharge|slice|jupiter|fi|gpay|oksbi)\b"
        # Indian phone number: +91/91/0 prefix, starts with 6-9
        in_phone_pattern = r"\b(?:\+91|91|0)?[6-9]\d{9}\b"
        # IFSC: bank branch code, e.g. HDFC0001234
        ifsc_pattern = r"\b[A-Z]{4}0[A-Z0-9]{6}\b"

        generic = (
            (PIIEntityType.EMAIL_ADDRESS, email_pattern, 0.9),
            (PIIEntityType.PHONE_NUMBER, phone_pattern, 0.8),
            (PIIEntityType.CREDIT_CARD, cc_pattern, 0.85),
            (PIIEntityType.US_SSN, ssn_pattern, 0.95),
            (PIIEntityType.IP_ADDRESS, ip_pattern, 0.9),
            (PIIEntityType.IN_AADHAAR, aadhaar_pattern, 0.85),
            (PIIEntityType.IN_PAN, pan_pattern, 0.9),
            (PIIEntityType.IN_UPI, upi_pattern, 0.9),
            (PIIEntityType.IN_PHONE, in_phone_pattern, 0.8),
            (PIIEntityType.IN_IFSC, ifsc_pattern, 0.9),
        )

        entities = []
        entity_counts: Dict[str, int] = {}
        addresses = (PIIEntityType.EMAIL_ADDRESS, PIIEntityType.IN_UPI)

        for entity_type, pattern, score in generic:
            if entity_type not in self.config.entity_types:
                continue
            if entity_type in addresses:
                # The same matches, found without trying the pattern from
                # every place a word starts: see _AddressSearch.
                spans = _address_spans(pattern, text)
            else:
                spans = [match.span() for match in re.finditer(pattern, text)]
            for start, end in spans:
                entity: Optional[Dict[str, Any]] = {
                    "type": entity_type.value,
                    "start": start,
                    "end": end,
                    "score": score,
                    "text": text[start:end],
                }
                if as_presidio:
                    entity = _as_presidio_reports(entity)
                    if entity is None:
                        continue
                entities.append(entity)
                entity_counts[entity["type"]] = entity_counts.get(entity["type"], 0) + 1

        # Custom recognizers (config.custom_patterns) — mirror the Presidio path
        for _entity, _spec in (self.config.custom_patterns or {}).items():
            _regex = _spec["regex"] if isinstance(_spec, dict) else _spec
            _score = _spec.get("score", 0.8) if isinstance(_spec, dict) else 0.8
            for match in re.finditer(_regex, text):
                entities.append(
                    {
                        "type": _entity,
                        "start": match.start(),
                        "end": match.end(),
                        "score": _score,
                        "text": match.group(),
                    }
                )
                entity_counts[_entity] = entity_counts.get(_entity, 0) + 1

        return entities, entity_counts

    def _redact_each(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Star out each finding, character for character. For use without Presidio."""
        redacted_text = text
        for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
            start, end = entity["start"], entity["end"]
            replacement = self.config.redaction_char * (end - start)
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
        return redacted_text

    def _fallback_detection(self, text: str) -> PIIDetectionResult:
        """Fallback pattern-based PII detection.

        This is used when Presidio is not available. It uses simple regex patterns
        for common PII types.

        Args:
            text: Text to analyze

        Returns:
            PIIDetectionResult: Detection results
        """
        entities, entity_counts = self._scan_patterns(text)

        has_pii = len(entities) > 0
        score = max([e["score"] for e in entities], default=0.0)

        # Simple redaction for fallback
        redacted_text = None
        if self.config.mode == PIIMode.REDACT and has_pii:
            # Bound the redacted copy so it cannot export a near-full copy of the
            # (still sensitive) original text.
            redacted_text = self._cap_text(self._redact_each(text, entities))

        blocked = self.config.mode == PIIMode.BLOCK and has_pii

        return PIIDetectionResult(
            has_pii=has_pii,
            entities=entities,
            entity_counts=entity_counts,
            redacted_text=redacted_text,
            original_text=None if (has_pii and self.config.mode != PIIMode.DETECT) else text,
            score=score,
            blocked=blocked,
        )

    def analyze_batch(self, texts: List[str], language: str = "en") -> List[PIIDetectionResult]:
        """Analyze multiple texts for PII.

        Args:
            texts: List of texts to analyze
            language: Language code (default: "en")

        Returns:
            List[PIIDetectionResult]: Detection results for each text
        """
        results = []
        for text in texts:
            results.append(self.detect(text, language))
        return results

    def get_statistics(self, results: List[PIIDetectionResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including total PII count, entity type distribution
        """
        total_detections = sum(1 for r in results if r.has_pii)
        total_entities = sum(len(r.entities) for r in results)

        # Aggregate entity counts
        entity_type_counts: Dict[str, int] = {}
        for result in results:
            for entity_type, count in result.entity_counts.items():
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + count

        # Calculate average score
        scores = [r.score for r in results if r.has_pii]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_texts_analyzed": len(results),
            "texts_with_pii": total_detections,
            "total_entities_detected": total_entities,
            "entity_type_distribution": entity_type_counts,
            "average_confidence_score": avg_score,
            "detection_rate": total_detections / len(results) if results else 0.0,
        }
