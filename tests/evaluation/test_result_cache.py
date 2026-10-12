"""The bounded cache each detector keeps of its own results."""

import gc
import os
import threading
import weakref

import pytest

from genai_otel.evaluation import result_cache
from genai_otel.evaluation.result_cache import ResultCache

SECRET = "My card number is 4111 1111 1111 1111."


def test_a_stored_result_is_returned_for_the_same_text():
    cache = ResultCache(4)
    key = cache.key(SECRET)
    assert cache.get(key) is None
    cache.put(key, "result")
    assert cache.get(cache.key(SECRET)) == "result"


def test_different_text_is_a_different_entry():
    cache = ResultCache(4)
    cache.put(cache.key("one"), 1)
    assert cache.get(cache.key("two")) is None


def test_settings_that_change_the_result_are_part_of_the_key():
    cache = ResultCache(4)
    cache.put(cache.key(SECRET, 0.5, "detect"), "lenient")
    assert cache.get(cache.key(SECRET, 0.9, "detect")) is None
    assert cache.get(cache.key(SECRET, 0.5, "block")) is None
    assert cache.get(cache.key(SECRET, 0.5, "detect")) == "lenient"


def test_the_text_itself_is_not_kept_as_the_key():
    """A prompt must not sit in memory as a dictionary key for as long as the entry lives."""
    cache = ResultCache(4)
    key = cache.key(SECRET, "en")
    cache.put(key, "result")
    for part in key:
        assert part != SECRET
        if isinstance(part, (str, bytes)):
            assert "4111" not in str(part)
    assert SECRET not in repr(cache._entries)


def test_text_that_differs_only_past_any_prefix_is_told_apart():
    cache = ResultCache(4)
    long_text = "a" * 5000
    cache.put(cache.key(long_text + "x"), "x")
    assert cache.get(cache.key(long_text + "y")) is None


def test_it_holds_no_more_than_its_size_and_drops_the_least_recently_used():
    cache = ResultCache(2)
    cache.put(cache.key("a"), "a")
    cache.put(cache.key("b"), "b")
    assert cache.get(cache.key("a")) == "a"  # "b" is now the least recently used
    cache.put(cache.key("c"), "c")
    assert len(cache) == 2
    assert cache.get(cache.key("b")) is None
    assert cache.get(cache.key("a")) == "a"
    assert cache.get(cache.key("c")) == "c"


def test_size_zero_turns_it_off():
    cache = ResultCache(0)
    assert cache.enabled is False
    assert cache.key(SECRET) is None
    cache.put(cache.key(SECRET), "result")
    assert len(cache) == 0
    assert cache.get(cache.key(SECRET)) is None


def test_something_that_is_not_text_is_not_cached():
    cache = ResultCache(4)
    assert cache.key(None) is None
    assert cache.key(["a list"]) is None


def test_a_setting_that_cannot_be_a_key_turns_caching_off_for_that_call():
    cache = ResultCache(4)
    assert cache.key(SECRET, {"unhashable": "setting"}) is None


def test_text_with_lone_surrogates_can_be_keyed():
    cache = ResultCache(4)
    key = cache.key("broken \ud83d emoji")
    cache.put(key, "result")
    assert cache.get(key) == "result"


def test_it_can_be_used_from_several_threads_at_once():
    cache = ResultCache(8)
    errors = []

    def worker(offset):
        try:
            for i in range(500):
                key = cache.key("text %d" % ((i + offset) % 20))
                cache.put(key, i)
                cache.get(key)
                assert len(cache) <= 8
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(cache) <= 8


# --- what one caller does to a result stays with that caller ----------------------


def test_what_is_stored_is_a_copy():
    cache = ResultCache(4)
    result = {"types": ["PERSON"], "counts": {"PERSON": 1}}
    cache.put(cache.key("text"), result)

    result["types"].append("LOCATION")
    result["counts"].clear()

    assert cache.get(cache.key("text")) == {"types": ["PERSON"], "counts": {"PERSON": 1}}


def test_what_is_handed_out_is_a_copy():
    cache = ResultCache(4)
    cache.put(cache.key("text"), {"types": ["PERSON"], "counts": {"PERSON": 1}})

    first = cache.get(cache.key("text"))
    first["types"].clear()
    first["counts"]["PERSON"] = 99

    assert cache.get(cache.key("text")) == {"types": ["PERSON"], "counts": {"PERSON": 1}}


# --- a forked child ---------------------------------------------------------------


def test_a_lock_held_when_the_process_forks_is_replaced_in_the_child():
    """The child has the parent's memory and none of its threads. A lock some
    other thread held at that moment would never be released there."""
    cache = ResultCache(4)
    key = cache.key("text")
    cache.put(key, "result")
    held = cache._lock
    held.acquire()
    try:
        result_cache._new_locks_after_fork()

        assert cache._lock is not held
        assert cache.get(key) == "result"  # would wait for ever on the old lock
    finally:
        held.release()


def test_a_cache_no_longer_in_use_is_not_kept_alive_for_the_fork_hook():
    cache = ResultCache(4)
    reference = weakref.ref(cache)
    del cache
    gc.collect()
    assert reference() is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork()")
def test_a_forked_child_can_use_a_cache_whose_lock_another_thread_held():
    cache = ResultCache(4)
    key = cache.key("text")
    cache.put(key, "result")
    holding, release = threading.Event(), threading.Event()

    def hold():
        with cache._lock:
            holding.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holding.wait(30)
    try:
        pid = os.fork()
        if pid == 0:
            answered = []
            reader = threading.Thread(target=lambda: answered.append(cache.get(key)), daemon=True)
            reader.start()
            reader.join(10)
            os._exit(0 if answered == ["result"] else 1)
        _, status = os.waitpid(pid, 0)
    finally:
        release.set()
        holder.join(30)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
