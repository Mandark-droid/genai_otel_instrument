"""Toxicity Detection using Google Perspective API and Detoxify.

This module provides toxicity detection capabilities using:
1. Google Perspective API (optional, requires API key)
2. Detoxify local model (fallback/alternative)
"""

import bisect
import logging
import os
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .config import DEFAULT_CACHE_SIZE, ToxicityConfig
from .result_cache import ResultCache

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
# Word pieces in each piece that a long text is read in. Much less than the
# model can read in one pass, on purpose: the model scores a piece as a whole,
# and one toxic sentence among a few hundred word pieces of ordinary text is
# scored as ordinary text unless it happens to come first. In a piece this size
# it is scored as what it is wherever it stands.
_PIECE_WORD_PIECES = 64
# Words a piece shares with the piece before it, so that a phrase lying across
# the place where the text was cut is read whole by one of the two.
_PIECE_OVERLAP_WORDS = 8
# Pieces scored in one call to the model. Several at once take about half the
# time of one call each; the number is bounded so that memory is, too.
_PIECES_PER_CALL = 16

# Process-wide "log once" guards so a blocked egress / air-gapped download is
# surfaced exactly once instead of on every request.
_EGRESS_BLOCK_LOGGED = False
_AIRGAP_BLOCK_LOGGED = False
_QUANTIZE_FAILURE_LOGGED = False


def _log_egress_block_once(message: str) -> None:
    global _EGRESS_BLOCK_LOGGED
    if not _EGRESS_BLOCK_LOGGED:
        _EGRESS_BLOCK_LOGGED = True
        logger.warning(message)


def _log_airgap_block_once(message: str) -> None:
    global _AIRGAP_BLOCK_LOGGED
    if not _AIRGAP_BLOCK_LOGGED:
        _AIRGAP_BLOCK_LOGGED = True
        logger.warning(message)


def _log_quantize_failure_once(message: str, error: Exception) -> None:
    global _QUANTIZE_FAILURE_LOGGED
    if not _QUANTIZE_FAILURE_LOGGED:
        _QUANTIZE_FAILURE_LOGGED = True
        logger.warning(message, error)


# The two optional backends are imported when a detector first needs one, not
# with this module. Detoxify brings torch and transformers with it, and every
# process that enables any detector imports this module: one that only matches
# patterns has no use for a deep-learning runtime.
#
# The names stay at module level so that tests can put a stand-in there. Anything
# other than the marker is used as it is.
_NOT_IMPORTED = object()
Detoxify: Any = _NOT_IMPORTED
discovery: Any = _NOT_IMPORTED


def _detoxify_class():
    """The Detoxify class, imported on first use. None when it is not installed."""
    global Detoxify
    if Detoxify is _NOT_IMPORTED:
        try:
            from detoxify import Detoxify as imported
        except ImportError:
            imported = None
        Detoxify = imported
    return Detoxify


def _perspective_discovery():
    """Google's API discovery module, imported on first use. None when not installed."""
    global discovery
    if discovery is _NOT_IMPORTED:
        try:
            from googleapiclient import discovery as imported
        except ImportError:
            imported = None
        discovery = imported
    return discovery


@dataclass
class ToxicityDetectionResult:
    """Result of toxicity detection.

    Attributes:
        is_toxic: Whether toxicity was detected above threshold
        scores: Toxicity scores by category
        max_score: Maximum toxicity score across all categories
        toxic_categories: List of categories that exceeded threshold
        original_text: Original input text
        blocked: Whether the text was blocked due to toxicity
    """

    is_toxic: bool
    scores: Dict[str, float] = field(default_factory=dict)
    max_score: float = 0.0
    toxic_categories: List[str] = field(default_factory=list)
    original_text: Optional[str] = None
    blocked: bool = False


class ToxicityDetector:
    """Toxicity detector using Perspective API and/or Detoxify.

    This detector can use:
    - Google Perspective API for production-grade detection
    - Detoxify local model for offline/fallback detection

    Requirements:
        Perspective API: pip install google-api-python-client
        Detoxify: pip install detoxify
    """

    def __init__(self, config: ToxicityConfig):
        """Initialize toxicity detector.

        Args:
            config: Toxicity detection configuration
        """
        self.config = config
        self._perspective_client = None
        self._detoxify_model = None
        self._perspective_available = False
        self._detoxify_available = False
        self._quantized = False
        self._cache = ResultCache(getattr(config, "cache_size", DEFAULT_CACHE_SIZE))
        # Egress / air-gap posture. Read from the config (which mirrors
        # OTelConfig) with an env-var fallback so an older/foreign config object
        # that lacks the field still defaults to the hardened env behaviour.
        self._egress_allowed = getattr(
            config,
            "allow_external_egress",
            os.getenv("GENAI_ALLOW_EXTERNAL_EGRESS", "true").lower() == "true",
        )
        self._air_gapped = getattr(
            config,
            "air_gapped",
            os.getenv("GENAI_AIR_GAPPED", "false").lower() == "true",
        )
        self._check_availability()

    def _sanitize(self, text: str) -> str:
        """Strip the Perspective API key (and any ``key=`` URL param) from a
        string before it is logged, so the developer key never reaches logs."""
        if not text:
            return text
        key = getattr(self.config, "perspective_api_key", None)
        if key:
            text = text.replace(key, "***REDACTED***")
        # googleapiclient appends the key as a ?key=... URL parameter.
        text = re.sub(r"(?i)(key=)[^&\s\"']+", r"\1***REDACTED***", text)
        return text

    def _check_availability(self):
        """Check which toxicity detection methods are available."""
        # Check Perspective API
        if self.config.use_perspective_api and self.config.perspective_api_key:
            self._init_perspective()

        # Check Detoxify
        if self.config.use_local_model:
            self._init_detoxify()

        if not self._perspective_available and not self._detoxify_available:
            logger.warning(
                "No toxicity detection method available. Install either:\n"
                "  - Perspective API: pip install google-api-python-client\n"
                "  - Detoxify: pip install detoxify"
            )

    def _init_perspective(self):
        """Initialize the Google Perspective API client (external egress)."""
        # HARD GATE: never initialize / call the external Google Perspective API
        # when third-party egress is disallowed. This keeps customer text inside
        # the application boundary. Fall back to local Detoxify instead.
        if not self._egress_allowed:
            _log_egress_block_once(
                "Perspective API requested but external egress is disabled "
                "(allow_external_egress=False / bank profile). Skipping "
                "Perspective API; will use local Detoxify if available."
            )
            self._perspective_available = False
            return
        try:
            discovery = _perspective_discovery()
            if discovery is None:
                raise ImportError("googleapiclient not installed")

            # Use httplib2 with timeout to prevent hanging API calls
            build_kwargs = {
                "serviceName": "commentanalyzer",
                "version": "v1alpha1",
                "developerKey": self.config.perspective_api_key,
                "discoveryServiceUrl": "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                "static_discovery": False,
            }
            try:
                import httplib2

                build_kwargs["http"] = httplib2.Http(timeout=self.config.api_timeout)
            except ImportError:
                logger.debug("httplib2 not available, Perspective API calls will have no timeout")

            self._perspective_client = discovery.build(**build_kwargs)
            self._perspective_available = True
            logger.info("Google Perspective API initialized successfully")
        except ImportError as e:
            logger.warning(
                "Perspective API not available: %s. Install with: pip install google-api-python-client",
                e,
            )
            self._perspective_available = False
        except Exception as e:
            # Never log with exc_info here: a traceback can include the request
            # URI carrying the developer key as a ?key=... parameter.
            logger.error("Failed to initialize Perspective API: %s", self._sanitize(str(e)))
            self._perspective_available = False

    def _init_detoxify(self):
        """Load the local Detoxify model, honoring the air-gapped posture."""
        # Optional local checkpoint so air-gapped sites can point at pre-staged
        # weights instead of triggering a torch-hub / huggingface.co download.
        local_path = os.getenv("GENAI_TOXICITY_DETOXIFY_PATH") or os.getenv(
            "GENAI_DETOXIFY_MODEL_PATH"
        )
        try:
            if self._air_gapped and not local_path:
                # Do NOT trigger a runtime network download on an air-gapped
                # host. Skip Detoxify with a single clear log line. Decided
                # before detoxify is imported: with nothing to load there is no
                # reason to bring torch into the process.
                _log_airgap_block_once(
                    "Air-gapped mode is enabled but no local Detoxify model "
                    "path is configured (set GENAI_TOXICITY_DETOXIFY_PATH to "
                    "pre-staged weights). Skipping Detoxify to avoid a runtime "
                    "model download."
                )
                self._detoxify_available = False
                return

            detoxify_class = _detoxify_class()
            if detoxify_class is None:
                raise ImportError("detoxify not installed")

            if self._air_gapped:
                # Belt-and-suspenders: force the underlying HF/torch stack offline
                # so no metadata/weight fetch can escape to the network.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self._detoxify_model = detoxify_class(checkpoint=local_path)
            elif local_path:
                self._detoxify_model = detoxify_class(checkpoint=local_path)
            else:
                # Load the model (using "original" model by default)
                self._detoxify_model = detoxify_class("original")

            self._set_torch_threads()
            self._quantize_model()

            self._detoxify_available = True
            logger.info("Detoxify model loaded successfully")
        except ImportError as e:
            logger.warning(
                "Detoxify not available: %s. Install with: pip install detoxify",
                e,
            )
            self._detoxify_available = False
        except Exception as e:
            logger.error("Failed to load Detoxify model: %s", e)
            self._detoxify_available = False

    def _set_torch_threads(self) -> None:
        """Apply the configured torch thread count. Does nothing unless one is set.

        The count is process-wide: it changes how every torch model in the
        application runs, not only this one. That is why it is never set unasked.
        """
        threads = getattr(self.config, "torch_threads", None)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            return
        try:
            import torch

            torch.set_num_threads(threads)
            logger.info("torch thread count set to %d for toxicity detection", threads)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not set the torch thread count to %s: %s", threads, e)

    def _quantize_model(self) -> None:
        """Replace the model's linear layers with 8-bit ones, when asked to.

        Scores from the quantized model differ from the original's, in places
        by enough to change a verdict, so this is never done unasked. A torch
        without the API, or one that quantizes but cannot run the result, leaves
        the original model in place.
        """
        if not getattr(self.config, "quantize", False):
            return
        original = getattr(self._detoxify_model, "model", None)
        try:
            import torch

            with warnings.catch_warnings():
                # torch announces the planned removal of this API on every call.
                warnings.simplefilter("ignore")
                self._detoxify_model.model = torch.quantization.quantize_dynamic(
                    original, {torch.nn.Linear}, dtype=torch.qint8
                )
                # Some builds quantize without complaint and fail at the first
                # prediction. Found here, not on a request: a detector that
                # raises on every text reports all of them as not toxic.
                self._detoxify_model.predict("ok")
            self._quantized = True
            logger.info("Detoxify model quantized to 8-bit weights")
        except Exception as e:  # noqa: BLE001
            self._detoxify_model.model = original
            self._quantized = False
            _log_quantize_failure_once(
                "GENAI_TOXICITY_QUANTIZE is set but this torch cannot quantize the "
                "toxicity model (%s). Using the unquantized model.",
                e,
            )

    def is_available(self) -> bool:
        """Check if toxicity detector is available.

        Returns:
            bool: True if at least one detection method is available
        """
        return self._perspective_available or self._detoxify_available

    def detect(self, text: str) -> ToxicityDetectionResult:
        """Detect toxicity in text.

        Args:
            text: Text to analyze

        Returns:
            ToxicityDetectionResult: Detection results
        """
        if not self.config.enabled:
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

        if not self.is_available():
            logger.warning("No toxicity detection method available")
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

        try:
            # Try Perspective API first if configured, available, AND egress is
            # permitted. The extra egress check is defence-in-depth: the client
            # is not built when egress is disallowed, so this should already be
            # False, but we never route customer text to Google without it.
            use_perspective = (
                self.config.use_perspective_api
                and self._perspective_available
                and self._egress_allowed
            )
            key = self._cache_key(text, use_perspective)
            cached = self._cache.get(key)
            if cached is not None:
                return replace(cached, original_text=text)

            result = self._ask_perspective(text) if use_perspective else None
            if use_perspective and result is None:
                # Perspective could not answer and the local model will. Its
                # answer is remembered as the local model's: under
                # Perspective's key it would still be given once Perspective
                # was back, as if Perspective had said it.
                key = self._cache_key(text, False)
                cached = self._cache.get(key)
                if cached is not None:
                    return replace(cached, original_text=text)
                if self._detoxify_available:
                    logger.info("Falling back to Detoxify")

            if result is None:
                if not self._detoxify_available:
                    return ToxicityDetectionResult(is_toxic=False, original_text=text)
                result = self._detect_with_detoxify(text)

            # A backend that failed answers with no scores at all. That is not
            # remembered: it must be asked again once the fault has cleared.
            # Stored without the text, which the entry would otherwise keep alive.
            if isinstance(result, ToxicityDetectionResult) and result.scores:
                self._cache.put(key, replace(result, original_text=None))
            return result

        except Exception as e:
            logger.error("Error detecting toxicity: %s", e, exc_info=True)
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

    def _cache_key(self, text: str, use_perspective: bool):
        """Key for a result: the text and every setting the result depends on."""
        if not self._cache.enabled:
            return None
        config = self.config
        try:
            return self._cache.key(
                text,
                bool(use_perspective),
                self._quantized,
                config.threshold,
                frozenset(config.categories),
                bool(config.block_on_detection),
            )
        except TypeError:
            return None

    def _detect_with_perspective(self, text: str) -> ToxicityDetectionResult:
        """Detect toxicity using Google Perspective API.

        Args:
            text: Text to analyze

        Returns:
            ToxicityDetectionResult: Detection results, from the local model
            when the API could not answer
        """
        result = self._ask_perspective(text)
        if result is not None:
            return result
        # Fall back to Detoxify if available
        if self._detoxify_available:
            logger.info("Falling back to Detoxify")
            return self._detect_with_detoxify(text)
        return ToxicityDetectionResult(is_toxic=False, original_text=text)

    def _ask_perspective(self, text: str) -> Optional[ToxicityDetectionResult]:
        """What the Perspective API says about ``text``. None when it could not answer."""
        try:
            # Build the request
            analyze_request = {
                "comment": {"text": text},
                "requestedAttributes": {},
            }

            # Map our categories to Perspective API attributes
            category_mapping = {
                "toxicity": "TOXICITY",
                "severe_toxicity": "SEVERE_TOXICITY",
                "identity_attack": "IDENTITY_ATTACK",
                "insult": "INSULT",
                "profanity": "PROFANITY",
                "threat": "THREAT",
            }

            # Add requested categories
            for category in self.config.categories:
                if category in category_mapping:
                    analyze_request["requestedAttributes"][category_mapping[category]] = {}

            # Make the API call
            response = self._perspective_client.comments().analyze(body=analyze_request).execute()

            # Parse scores
            scores = {}
            for category in self.config.categories:
                if category in category_mapping:
                    api_attr = category_mapping[category]
                    if api_attr in response.get("attributeScores", {}):
                        score_data = response["attributeScores"][api_attr]
                        scores[category] = score_data["summaryScore"]["value"]

            # Determine if toxic
            max_score = max(scores.values(), default=0.0)
            toxic_categories = [
                cat for cat, score in scores.items() if score >= self.config.threshold
            ]
            is_toxic = len(toxic_categories) > 0

            # Check if should block
            blocked = self.config.block_on_detection and is_toxic

            return ToxicityDetectionResult(
                is_toxic=is_toxic,
                scores=scores,
                max_score=max_score,
                toxic_categories=toxic_categories,
                original_text=text,
                blocked=blocked,
            )

        except Exception as e:
            # Never log with exc_info: a traceback can include the request URI
            # carrying the developer key. Sanitize the message as well.
            logger.error("Perspective API error: %s", self._sanitize(str(e)))
            return None

    def _detect_with_detoxify(self, text: str) -> ToxicityDetectionResult:
        """Detect toxicity using Detoxify local model.

        Args:
            text: Text to analyze

        Returns:
            ToxicityDetectionResult: Detection results
        """
        try:
            # Get predictions: in one pass, or for a text longer than the
            # model reads in one, the highest score of each category over the
            # pieces it was read in.
            predictions = self._predict_whole(text)

            # Map Detoxify outputs to our categories
            # Detoxify outputs: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
            category_mapping = {
                "toxicity": "toxicity",
                "severe_toxicity": "severe_toxicity",
                "identity_attack": "identity_attack",
                "insult": "insult",
                "profanity": "obscene",  # Map obscene to profanity
                "threat": "threat",
            }

            scores = {}
            for our_cat, detoxify_cat in category_mapping.items():
                if our_cat in self.config.categories and detoxify_cat in predictions:
                    scores[our_cat] = float(predictions[detoxify_cat])

            # Determine if toxic
            max_score = max(scores.values(), default=0.0)
            toxic_categories = [
                cat for cat, score in scores.items() if score >= self.config.threshold
            ]
            is_toxic = len(toxic_categories) > 0

            # Check if should block
            blocked = self.config.block_on_detection and is_toxic

            return ToxicityDetectionResult(
                is_toxic=is_toxic,
                scores=scores,
                max_score=max_score,
                toxic_categories=toxic_categories,
                original_text=text,
                blocked=blocked,
            )

        except Exception as e:
            logger.error("Detoxify error: %s", e, exc_info=True)
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

    # --- text longer than the model reads in one pass ----------------------------
    #
    # The model reads a fixed number of word pieces and its tokenizer drops the
    # rest without a word. Given a long conversation it scored the oldest part
    # and reported the result as the conversation's. So a long text is read in
    # pieces that each fit, and every piece is scored.

    def _one_pass(self) -> Optional[int]:
        """Word pieces of text the model reads in one pass. None when that cannot be told."""
        tokenizer = getattr(self._detoxify_model, "tokenizer", None)
        limit = getattr(tokenizer, "model_max_length", None)
        # A tokenizer with no limit of its own reports an absurdly large one.
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit < 100000:
            return None
        try:
            markers = int(tokenizer.num_special_tokens_to_add(False))
        except Exception:  # noqa: BLE001
            markers = 2
        return limit - markers if limit > markers else None

    def _word_pieces(self, text: str) -> int:
        """How many word pieces the model's tokenizer makes of ``text``."""
        tokenizer = self._detoxify_model.tokenizer
        if callable(tokenizer):
            # Asked quietly. Left to itself the tokenizer warns that a text
            # this long cannot be given to the model, which is the very thing
            # being seen to here.
            try:
                counted = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)
                return len(counted["input_ids"])
            except Exception:  # noqa: BLE001 - a tokenizer that is not called this way
                pass
        return len(tokenizer.tokenize(text))

    def _pieces(self, text: str) -> List[str]:
        """``text`` as the model is to be given it: whole, or in pieces that each fit one pass."""
        one_pass = self._one_pass()
        # A word piece is at least one character, so a text this short fits
        # without being counted. With the default limits every text does.
        if one_pass is None or not isinstance(text, str) or len(text) <= one_pass:
            return [text]
        try:
            total = self._word_pieces(text)
            if total <= one_pass:
                return [text]
            return self._cut(text, one_pass, total)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not divide a long text for the toxicity model (%s). "
                "It is given whole and the model reads the start of it only.",
                type(e).__name__,
            )
            return [text]

    def _cut(self, text: str, one_pass: int, total: int) -> List[str]:
        """Pieces of ``text`` that together cover all of it, none longer than a pass.

        ``total`` is how many word pieces the text is. Cut at word boundaries.
        Each piece after the first starts a few words back, inside the piece
        before it.
        """
        # Where each word starts.
        starts = [match.end() for match in _WHITESPACE.finditer(text)]
        size = min(one_pass, _PIECE_WORD_PIECES)
        # A quarter of each piece is kept free for the words it shares with
        # the piece before it.
        spans = self._spans_that_fit(text, max(1, size - size // 4), total, starts)
        pieces = []
        for number, (start, end, used) in enumerate(spans):
            if number:
                start = self._a_few_words_back(
                    text, start, spans[number - 1][0], starts, size - used
                )
            pieces.append(text[start:end])
        return pieces

    def _spans_that_fit(
        self, text: str, budget: int, total: int, starts: List[int]
    ) -> List[Tuple[int, int, int]]:
        """Divide ``text`` into (start, end, word pieces) of at most ``budget`` word pieces each.

        Each span is as much text as holds a budget's worth of word pieces on
        average, then counted, and shortened if it turns out to hold more.
        """
        # Characters that hold a little under one budget of word pieces.
        stride = max(1, int(len(text) * budget * 0.92 / max(total, 1)))
        spans: List[Tuple[int, int, int]] = []
        start = 0
        while start < len(text):
            end = len(text)
            if end - start > stride:
                end = self._word_start_near(start + stride, start, len(text), starts)
            used = self._word_pieces(text[start:end])
            while used > budget and end - start > 1:
                # Too much: keep the share that fits, and not quite all of it.
                keep = start + max(1, int((end - start) * budget * 0.95 / used))
                end = self._word_start_before(keep, start, starts)
                used = self._word_pieces(text[start:end])
            spans.append((start, end, used))
            start = end
        return spans

    @staticmethod
    def _word_start_near(target: int, after: int, before: int, starts: List[int]) -> int:
        """The start of a word nearest ``target``, strictly between ``after`` and ``before``.

        ``target`` itself when there is none: a single word longer than a pass
        has to be cut somewhere.
        """
        at = bisect.bisect_left(starts, target)
        candidates = [starts[i] for i in (at - 1, at) if 0 <= i < len(starts)]
        candidates = [c for c in candidates if after < c < before]
        if not candidates:
            return target
        return min(candidates, key=lambda c: abs(c - target))

    @staticmethod
    def _word_start_before(target: int, after: int, starts: List[int]) -> int:
        """The last start of a word at or before ``target`` and after ``after``; else ``target``."""
        at = bisect.bisect_right(starts, target) - 1
        if at >= 0 and starts[at] > after:
            return starts[at]
        return target

    def _a_few_words_back(
        self, text: str, start: int, floor: int, starts: List[int], room: int
    ) -> int:
        """Where a piece that would begin at ``start`` begins instead, to overlap the one before.

        Up to ``_PIECE_OVERLAP_WORDS`` words back, as many as ``room`` word
        pieces allow, and never before ``floor``.
        """
        at = bisect.bisect_left(starts, start)
        # The furthest back first: it nearly always fits, and then it is the
        # only one that has to be counted.
        for back in range(min(_PIECE_OVERLAP_WORDS, at), 0, -1):
            candidate = starts[at - back]
            if candidate < floor:
                continue
            # One spare: word pieces of two texts side by side can come to
            # one more than the two counted apart.
            if self._word_pieces(text[candidate:start]) + 1 <= room:
                return candidate
        return start

    def _predict_whole(self, text: str) -> Dict[str, Any]:
        """The model's score for each of its categories over all of ``text``.

        For a text read in pieces, the highest score any piece got.
        """
        pieces = self._pieces(text)
        if len(pieces) == 1:
            return self._detoxify_model.predict(pieces[0])
        highest: Dict[str, float] = {}
        for first in range(0, len(pieces), _PIECES_PER_CALL):
            scored = self._detoxify_model.predict(pieces[first : first + _PIECES_PER_CALL])
            for category, scores in scored.items():
                top = max(float(score) for score in scores)
                if category not in highest or top > highest[category]:
                    highest[category] = top
        return highest

    def analyze_batch(self, texts: List[str]) -> List[ToxicityDetectionResult]:
        """Analyze multiple texts for toxicity.

        Args:
            texts: List of texts to analyze

        Returns:
            List[ToxicityDetectionResult]: Detection results for each text
        """
        # Detoxify supports batch processing
        if self._detoxify_available and not self.config.use_perspective_api:
            try:
                return self._batch_detect_with_detoxify(texts)
            except Exception as e:
                logger.error("Batch detection error: %s", e)

        # Fall back to sequential processing
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

    def _batch_detect_with_detoxify(self, texts: List[str]) -> List[ToxicityDetectionResult]:
        """Batch detect toxicity using Detoxify.

        Args:
            texts: List of texts to analyze

        Returns:
            List[ToxicityDetectionResult]: Detection results
        """
        try:
            if any(len(self._pieces(text)) > 1 for text in texts):
                # A batch is one pass per text. A text longer than a pass is
                # read in pieces, so these are scored one text at a time.
                return [self._detect_with_detoxify(text) for text in texts]

            # Get batch predictions
            predictions = self._detoxify_model.predict(texts)

            results = []
            for i, text in enumerate(texts):
                # Extract scores for this text
                category_mapping = {
                    "toxicity": "toxicity",
                    "severe_toxicity": "severe_toxicity",
                    "identity_attack": "identity_attack",
                    "insult": "insult",
                    "profanity": "obscene",
                    "threat": "threat",
                }

                scores = {}
                for our_cat, detoxify_cat in category_mapping.items():
                    if our_cat in self.config.categories and detoxify_cat in predictions:
                        scores[our_cat] = float(predictions[detoxify_cat][i])

                # Determine if toxic
                max_score = max(scores.values(), default=0.0)
                toxic_categories = [
                    cat for cat, score in scores.items() if score >= self.config.threshold
                ]
                is_toxic = len(toxic_categories) > 0
                blocked = self.config.block_on_detection and is_toxic

                results.append(
                    ToxicityDetectionResult(
                        is_toxic=is_toxic,
                        scores=scores,
                        max_score=max_score,
                        toxic_categories=toxic_categories,
                        original_text=text,
                        blocked=blocked,
                    )
                )

            return results

        except Exception as e:
            logger.error("Batch Detoxify error: %s", e, exc_info=True)
            return [ToxicityDetectionResult(is_toxic=False, original_text=text) for text in texts]

    def get_statistics(self, results: List[ToxicityDetectionResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including toxicity rate, category distribution
        """
        total_toxic = sum(1 for r in results if r.is_toxic)

        # Aggregate category counts
        category_counts: Dict[str, int] = {}
        for result in results:
            for category in result.toxic_categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Calculate average scores
        avg_scores: Dict[str, float] = {}
        for category in self.config.categories:
            scores = [r.scores.get(category, 0.0) for r in results if r.scores.get(category)]
            avg_scores[category] = sum(scores) / len(scores) if scores else 0.0

        # Calculate max scores seen
        max_scores: Dict[str, float] = {}
        for category in self.config.categories:
            scores = [r.scores.get(category, 0.0) for r in results if r.scores.get(category)]
            max_scores[category] = max(scores, default=0.0)

        return {
            "total_texts_analyzed": len(results),
            "toxic_texts_count": total_toxic,
            "toxicity_rate": total_toxic / len(results) if results else 0.0,
            "category_counts": category_counts,
            "average_scores": avg_scores,
            "max_scores": max_scores,
            "detection_method": (
                "perspective_api"
                if self._perspective_available and self.config.use_perspective_api
                else "detoxify"
            ),
        }
