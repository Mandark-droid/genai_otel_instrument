"""A small bounded cache of detector results.

A detector is asked about the same text more often than it looks. With a block
policy the prompt is analysed before the request is sent and again when the span
is annotated; a retry or an agent loop sends a request word for word; a span that
the instrumentation does not evaluate itself is looked at by the span processor
and then by the exporter. Each detector keeps its last results so that the
analysis runs once.

What an entry holds, for as long as it stays in the cache:

* The key: a digest of the analysed text, its length, and the settings the
  result depends on. Never the text.
* PII: for each finding its entity type, where it starts and ends, and its
  score; the count per type and the verdict. Not the text of any finding and
  not the redacted copy: both are put back from the caller's text when the entry
  is used.
* Toxicity: the score per category and the verdict.
* Bias, prompt injection and restricted topics: the scores, the verdict, and
  the phrases that matched a pattern. Those phrases are pieces of the analysed
  text, a few words each ("ignore all previous instructions", "women are").
* Hallucination: the counts, the indicators, and up to three sentences of the
  response reported as unsupported claims. Those sentences are pieces of the
  analysed text too.

``GENAI_EVALUATION_CACHE_SIZE=0`` keeps nothing at all.
"""

import copy
import hashlib
import os
import threading
import weakref
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple

# Every cache in use, so that a forked child can be given locks of its own.
_CACHES: "weakref.WeakSet[ResultCache]" = weakref.WeakSet()


class ResultCache:
    """Least-recently-used cache of results, keyed by a digest of the analysed text.

    The key is a digest rather than the text: an entry lives for as long as the
    process keeps using it, and a prompt should not be held in memory that long
    only to be compared against.

    A result is copied when it is stored and again each time it is handed out.
    Results hold lists and dicts, and a caller that cleared its own would
    otherwise empty the answer given to every caller after it.
    """

    def __init__(self, max_size: int):
        try:
            self._max_size = max(0, int(max_size))
        except (TypeError, ValueError):
            self._max_size = 0
        self._entries: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()
        _CACHES.add(self)

    @property
    def enabled(self) -> bool:
        return self._max_size > 0

    def key(self, text: Any, *settings: Hashable) -> Optional[Tuple]:
        """The key for ``text`` analysed under ``settings``, or None when there is none.

        ``settings`` is everything besides the text that the result depends on.
        None means "do not cache this call": the cache is off, the input is not
        text, or a setting cannot be compared.
        """
        if self._max_size <= 0 or not isinstance(text, str):
            return None
        digest = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()
        key = (digest, len(text)) + settings
        try:
            hash(key)
        except TypeError:
            return None
        return key

    def get(self, key: Optional[Hashable]) -> Optional[Any]:
        if key is None:
            return None
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
        # Copied outside the lock: the entry itself is never changed once stored.
        return copy.deepcopy(value)

    def put(self, key: Optional[Hashable], value: Any) -> None:
        if key is None or self._max_size <= 0:
            return
        value = copy.deepcopy(value)
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def _new_locks_after_fork() -> None:
    """Give every cache a lock no thread holds.

    A forked child has the parent's memory and none of its threads. A lock that
    another thread held at the moment of the fork stays held there for good,
    and the first detector to look in its cache would wait on it for ever. The
    entries are kept: each one is whole, whatever the other thread was doing.
    """
    for cache in list(_CACHES):
        cache._lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_locks_after_fork)
