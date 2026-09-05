"""Tests for the post-release verification script.

Each case here corresponds to a way the release check has produced, or could
produce, a confident wrong answer. The network-facing parts of
``scripts/verify_release.py`` are not exercised; the decision logic is, because
that is where a false pass or a false failure actually comes from.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

import verify_release as vr  # noqa: E402


def probe(**overrides):
    """A probe result from a healthy release, before overrides."""
    base = {
        "installed_version": "1.27.0",
        "package_path": "/tmp/verify/venv/lib/python3.12/site-packages/genai_otel",
        "cwd": "/tmp/verify",
        "chat_entries": 1495,
        "models": {
            "gpt-5.6-luna": {
                "key": "gpt-5.6-luna",
                "promptPrice": 0.0002,
                "completionPrice": 0.0012,
                "cacheReadPrice": 2e-05,
                "cacheWritePrice": 0.00025,
            }
        },
        "notes": {"gpt-5-6-luna": "Above 272000 tokens of context the vendor charges"},
    }
    base.update(overrides)
    return base


# --- the simple index vs the JSON API ----------------------------------------
# PyPI's JSON API and simple index propagate separately. For about a minute the
# JSON API reports a version pip cannot yet install, which makes a healthy
# release look broken.


def test_index_lists_the_exact_version():
    html = '<a href="...">genai_otel_instrument-1.27.0-py3-none-any.whl</a>'
    assert vr.index_has_version(html, "1.27.0")


def test_index_accepts_an_sdist_only_listing():
    html = '<a href="...">genai_otel_instrument-1.27.0.tar.gz</a>'
    assert vr.index_has_version(html, "1.27.0")


def test_index_does_not_match_a_version_that_is_merely_a_prefix():
    """1.2 must not be satisfied by 1.27.0, or the wait returns immediately."""
    html = '<a href="...">genai_otel_instrument-1.27.0-py3-none-any.whl</a>'
    assert not vr.index_has_version(html, "1.2")


def test_index_reports_absent_version():
    html = '<a href="...">genai_otel_instrument-1.26.0-py3-none-any.whl</a>'
    assert not vr.index_has_version(html, "1.27.0")


def test_wheel_name_uses_the_underscored_distribution_name():
    assert vr.wheel_name("1.27.0") == "genai_otel_instrument-1.27.0-py3-none-any.whl"


# --- the checks that a stale environment defeats ------------------------------


def test_healthy_release_produces_no_failures():
    assert evaluate_ok(probe()) == []


def evaluate_ok(result):
    return vr.evaluate(
        result,
        "1.27.0",
        require_model=[("gpt-5.6-luna", 0.0002, 0.0012)],
        require_note=[("gpt-5-6-luna", "Above 272000 tokens of context")],
        min_chat_entries=1495,
    )


def test_stale_venv_is_caught_by_asserting_the_installed_version():
    """The failure that actually happened: a leftover venv kept 1.25.0 while
    pip's install of 1.27.0 had failed, and every content check then described
    the OLD release."""
    failures = evaluate_ok(probe(installed_version="1.25.0"))
    assert any("installed version is '1.25.0'" in f for f in failures)


def test_missing_package_metadata_is_a_failure_not_a_pass():
    failures = evaluate_ok(probe(installed_version=None))
    assert any("installed version is None" in f for f in failures)


def test_import_from_the_working_tree_is_caught():
    """Running from the repo root shadows site-packages, so the probe would
    verify uncommitted local code instead of the published wheel."""
    failures = evaluate_ok(probe(package_path="/checkout/genai_otel_instrument/genai_otel"))
    assert any("not inside the venv" in f for f in failures)


def test_windows_style_site_packages_path_is_accepted():
    ok = probe(package_path=r"C:\tmp\venv\Lib\site-packages\genai_otel")
    assert evaluate_ok(ok) == []


# --- content checks -----------------------------------------------------------


def test_model_resolving_to_another_entry_is_caught():
    """gpt-5.6-luna falling through to gpt-5 is the exact mis-billing this
    release fixed, so a regression must fail the release check."""
    failures = evaluate_ok(
        probe(
            models={
                "gpt-5.6-luna": {"key": "gpt-5", "promptPrice": 0.00125, "completionPrice": 0.01}
            }
        )
    )
    assert any("inheriting another model's rate" in f for f in failures)


def test_unpriced_model_is_caught():
    failures = evaluate_ok(probe(models={"gpt-5.6-luna": {"key": None}}))
    assert any("resolves to nothing" in f for f in failures)


def test_wrong_price_is_caught():
    failures = evaluate_ok(
        probe(
            models={
                "gpt-5.6-luna": {
                    "key": "gpt-5.6-luna",
                    "promptPrice": 0.02,
                    "completionPrice": 0.0012,
                }
            }
        )
    )
    assert any("promptPrice is 0.02" in f for f in failures)


def test_shrunken_table_is_caught():
    failures = evaluate_ok(probe(chat_entries=1400))
    assert any("1400 entries" in f for f in failures)


def test_wrong_note_text_is_caught():
    """The 200K threshold shipped in prose and nearly caused a downstream
    mis-billing, so the note text is worth asserting at release time."""
    failures = evaluate_ok(probe(notes={"gpt-5-6-luna": "Above 200K context"}))
    assert any("note does not contain" in f for f in failures)


def test_several_failures_are_all_reported():
    failures = evaluate_ok(probe(installed_version="1.25.0", chat_entries=1))
    assert len(failures) >= 2


def test_optional_expectations_are_skipped_when_not_requested():
    assert vr.evaluate(probe(chat_entries=0), "1.27.0") == []


# --- argument parsing ---------------------------------------------------------


def test_model_spec_with_prices():
    assert vr.parse_model_spec("gpt-5.6-luna=0.0002/0.0012") == ("gpt-5.6-luna", 0.0002, 0.0012)


def test_model_spec_without_prices_checks_resolution_only():
    assert vr.parse_model_spec("gpt-6-astra") == ("gpt-6-astra", None, None)


def test_model_spec_rejects_a_malformed_price_pair():
    with pytest.raises(Exception):
        vr.parse_model_spec("gpt-5.6-luna=0.0002")


def test_note_spec_keeps_text_containing_equals():
    key, text = vr.parse_note_spec("gpt-5-6-luna=Above 272000 tokens")
    assert key == "gpt-5-6-luna"
    assert text == "Above 272000 tokens"


def test_note_spec_rejects_missing_text():
    with pytest.raises(Exception):
        vr.parse_note_spec("gpt-5-6-luna")
