"""Settings for where the evaluation detectors run and how much text they read."""

import dataclasses
import logging

import pytest

from genai_otel.auto_instrument import _evaluation_configs
from genai_otel.config import MAX_EVALUATION_SHUTDOWN_TIMEOUT, OTelConfig
from genai_otel.evaluation.config import (
    BiasConfig,
    HallucinationConfig,
    PIIConfig,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    ToxicityConfig,
)

EVALUATION_ENV = (
    "GENAI_EVALUATION_MODE",
    "GENAI_EVALUATION_QUEUE_SIZE",
    "GENAI_EVALUATION_MAX_CHARS",
    "GENAI_EVALUATION_SHUTDOWN_TIMEOUT",
    "GENAI_EVALUATION_CACHE_SIZE",
    "GENAI_TOXICITY_QUANTIZE",
    "GENAI_TOXICITY_TORCH_THREADS",
    "GENAI_TOXICITY_MAX_CHARS",
    # A hardened profile changes the defaults below; none may leak in from the shell.
    "GENAI_PROFILE",
)

DETECTOR_CONFIGS = (
    PIIConfig,
    ToxicityConfig,
    BiasConfig,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    HallucinationConfig,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in EVALUATION_ENV:
        monkeypatch.delenv(name, raising=False)


class TestWhereDetectorsRun:
    def test_defaults(self):
        config = OTelConfig()
        assert config.evaluation_mode == "background"
        assert config.evaluation_queue_size == 1000
        assert config.evaluation_max_chars == 1500
        assert config.evaluation_shutdown_timeout == 5.0

    def test_are_declared_fields(self):
        names = {f.name for f in dataclasses.fields(OTelConfig)}
        assert {
            "evaluation_mode",
            "evaluation_queue_size",
            "evaluation_max_chars",
            "evaluation_shutdown_timeout",
        } <= names

    def test_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("GENAI_EVALUATION_MODE", "Inline ")
        monkeypatch.setenv("GENAI_EVALUATION_QUEUE_SIZE", "25")
        monkeypatch.setenv("GENAI_EVALUATION_MAX_CHARS", "0")
        monkeypatch.setenv("GENAI_EVALUATION_SHUTDOWN_TIMEOUT", "1.5")
        config = OTelConfig()
        assert config.evaluation_mode == "inline"
        assert config.evaluation_queue_size == 25
        assert config.evaluation_max_chars == 0
        assert config.evaluation_shutdown_timeout == 1.5

    def test_keyword_arguments_win(self, monkeypatch):
        monkeypatch.setenv("GENAI_EVALUATION_MODE", "inline")
        config = OTelConfig(evaluation_mode="background", evaluation_max_chars=300)
        assert config.evaluation_mode == "background"
        assert config.evaluation_max_chars == 300

    @pytest.mark.parametrize("value", ["sync", "off", "1"])
    def test_unknown_mode_falls_back_to_the_default_and_says_so(self, monkeypatch, caplog, value):
        monkeypatch.setenv("GENAI_EVALUATION_MODE", value)
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig()
        assert config.evaluation_mode == "background"
        assert "GENAI_EVALUATION_MODE" in caplog.text

    def test_unknown_mode_passed_in_code_falls_back_too(self, caplog):
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig(evaluation_mode="deferred")
        assert config.evaluation_mode == "background"
        assert "deferred" in caplog.text

    @pytest.mark.parametrize(
        "name, field, default",
        [
            ("GENAI_EVALUATION_QUEUE_SIZE", "evaluation_queue_size", 1000),
            ("GENAI_EVALUATION_MAX_CHARS", "evaluation_max_chars", 1500),
            ("GENAI_EVALUATION_SHUTDOWN_TIMEOUT", "evaluation_shutdown_timeout", 5.0),
        ],
    )
    def test_a_value_that_is_not_a_number_falls_back(
        self, monkeypatch, caplog, name, field, default
    ):
        monkeypatch.setenv(name, "lots")
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig()
        assert getattr(config, field) == default
        assert name in caplog.text

    def test_a_negative_limit_on_analysed_text_means_no_limit(self, monkeypatch):
        monkeypatch.setenv("GENAI_EVALUATION_MAX_CHARS", "-5")
        assert OTelConfig().evaluation_max_chars == 0

    def test_the_queue_always_has_room_for_one_span(self, monkeypatch):
        """A limit of zero would mark every span as overloaded."""
        monkeypatch.setenv("GENAI_EVALUATION_QUEUE_SIZE", "0")
        assert OTelConfig().evaluation_queue_size == 1
        assert OTelConfig(evaluation_queue_size=-3).evaluation_queue_size == 1


class TestToxicityLimit:
    """The local toxicity model is the slowest detector, so it reads less by default."""

    def test_default(self):
        assert OTelConfig().toxicity_max_chars == 500

    def test_is_a_declared_field(self):
        assert "toxicity_max_chars" in {f.name for f in dataclasses.fields(OTelConfig)}

    def test_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", "300")
        assert OTelConfig().toxicity_max_chars == 300

    def test_zero_means_no_limit_of_its_own(self, monkeypatch):
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", "0")
        assert OTelConfig().toxicity_max_chars == 0
        assert OTelConfig(toxicity_max_chars=0).toxicity_max_chars == 0

    def test_keyword_arguments_win(self, monkeypatch):
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", "300")
        assert OTelConfig(toxicity_max_chars=800).toxicity_max_chars == 800

    def test_a_negative_limit_means_no_limit(self, monkeypatch):
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", "-5")
        assert OTelConfig().toxicity_max_chars == 0

    def test_a_value_that_is_not_a_number_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", "few")
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig()
        assert config.toxicity_max_chars == 500
        assert "GENAI_TOXICITY_MAX_CHARS" in caplog.text


HARDENED = ["strict", "bfsi", "bank"]


class TestAHardenedProfileReadsEverything:
    """Under a hardened profile nothing is left unread unless the operator says so.

    A limit the operator set, in the environment or in code, is the operator's
    decision and stays, also when it happens to equal the ordinary default.
    """

    @pytest.mark.parametrize("profile", HARDENED)
    def test_with_nothing_set_there_is_no_limit(self, monkeypatch, profile):
        monkeypatch.setenv("GENAI_PROFILE", profile)
        config = OTelConfig()
        assert config.evaluation_max_chars == 0
        assert config.toxicity_max_chars == 0

    @pytest.mark.parametrize("profile", HARDENED)
    def test_the_profile_can_be_given_in_code(self, profile):
        config = OTelConfig(profile=profile)
        assert config.evaluation_max_chars == 0
        assert config.toxicity_max_chars == 0

    @pytest.mark.parametrize("profile", HARDENED)
    @pytest.mark.parametrize("value", ["1500", "200", "0"])
    def test_the_general_limit_set_in_the_environment_stays(self, monkeypatch, profile, value):
        monkeypatch.setenv("GENAI_PROFILE", profile)
        monkeypatch.setenv("GENAI_EVALUATION_MAX_CHARS", value)
        config = OTelConfig()
        assert config.evaluation_max_chars == int(value)
        assert config.toxicity_max_chars == 0  # not set, so still the profile's

    @pytest.mark.parametrize("profile", HARDENED)
    @pytest.mark.parametrize("value", ["500", "200", "0"])
    def test_the_toxicity_limit_set_in_the_environment_stays(self, monkeypatch, profile, value):
        monkeypatch.setenv("GENAI_PROFILE", profile)
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", value)
        config = OTelConfig()
        assert config.toxicity_max_chars == int(value)
        assert config.evaluation_max_chars == 0

    @pytest.mark.parametrize("profile", HARDENED)
    @pytest.mark.parametrize("value", [1500, 200, 0])
    def test_the_general_limit_passed_in_code_stays(self, monkeypatch, profile, value):
        monkeypatch.setenv("GENAI_PROFILE", profile)
        config = OTelConfig(evaluation_max_chars=value)
        assert config.evaluation_max_chars == value
        assert config.toxicity_max_chars == 0

    @pytest.mark.parametrize("profile", HARDENED)
    @pytest.mark.parametrize("value", [500, 200, 0])
    def test_the_toxicity_limit_passed_in_code_stays(self, monkeypatch, profile, value):
        monkeypatch.setenv("GENAI_PROFILE", profile)
        config = OTelConfig(toxicity_max_chars=value)
        assert config.toxicity_max_chars == value
        assert config.evaluation_max_chars == 0

    def test_code_wins_over_the_environment_under_the_profile_too(self, monkeypatch):
        monkeypatch.setenv("GENAI_PROFILE", "bank")
        monkeypatch.setenv("GENAI_EVALUATION_MAX_CHARS", "200")
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", "100")
        config = OTelConfig(evaluation_max_chars=1500, toxicity_max_chars=500)
        assert config.evaluation_max_chars == 1500
        assert config.toxicity_max_chars == 500

    def test_a_value_that_is_not_a_number_is_not_a_setting(self, monkeypatch):
        monkeypatch.setenv("GENAI_PROFILE", "bank")
        monkeypatch.setenv("GENAI_EVALUATION_MAX_CHARS", "lots")
        monkeypatch.setenv("GENAI_TOXICITY_MAX_CHARS", "few")
        config = OTelConfig()
        assert config.evaluation_max_chars == 0
        assert config.toxicity_max_chars == 0

    @pytest.mark.parametrize("profile", ["", "default", "dev"])
    def test_without_the_profile_the_ordinary_defaults_apply(self, monkeypatch, profile):
        monkeypatch.setenv("GENAI_PROFILE", profile)
        config = OTelConfig()
        assert config.evaluation_max_chars == 1500
        assert config.toxicity_max_chars == 500

    def test_both_limits_are_whole_numbers_once_resolved(self, monkeypatch):
        for profile in ("", "bank"):
            monkeypatch.setenv("GENAI_PROFILE", profile)
            config = OTelConfig()
            assert type(config.evaluation_max_chars) is int
            assert type(config.toxicity_max_chars) is int


class TestShutdownTimeout:
    """A wait that cannot be waited for is not a setting: the default applies."""

    @staticmethod
    def _warnings(caplog):
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
            and "GENAI_EVALUATION_SHUTDOWN_TIMEOUT" in record.getMessage()
        ]

    @pytest.mark.parametrize("value", ["inf", "-inf", "nan", "-1", "-0.5"])
    def test_a_value_nobody_can_wait_for_falls_back_with_one_warning(
        self, monkeypatch, caplog, value
    ):
        monkeypatch.setenv("GENAI_EVALUATION_SHUTDOWN_TIMEOUT", value)
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig()
        assert config.evaluation_shutdown_timeout == 5.0
        assert len(self._warnings(caplog)) == 1

    @pytest.mark.parametrize("value", [float("inf"), float("nan"), -1, -0.5, "soon", None])
    def test_the_same_passed_in_code(self, caplog, value):
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig(evaluation_shutdown_timeout=value)
        assert config.evaluation_shutdown_timeout == 5.0
        assert len(self._warnings(caplog)) == 1

    def test_zero_is_a_setting(self, monkeypatch, caplog):
        monkeypatch.setenv("GENAI_EVALUATION_SHUTDOWN_TIMEOUT", "0")
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig()
        assert config.evaluation_shutdown_timeout == 0.0
        assert self._warnings(caplog) == []

    def test_an_absurd_wait_is_capped_and_says_so(self, monkeypatch, caplog):
        monkeypatch.setenv("GENAI_EVALUATION_SHUTDOWN_TIMEOUT", "1e12")
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig()
        assert config.evaluation_shutdown_timeout == MAX_EVALUATION_SHUTDOWN_TIMEOUT
        assert len(self._warnings(caplog)) == 1

    def test_the_cap_itself_is_accepted_quietly(self, caplog):
        with caplog.at_level(logging.WARNING, logger="genai_otel.config"):
            config = OTelConfig(evaluation_shutdown_timeout=MAX_EVALUATION_SHUTDOWN_TIMEOUT)
        assert config.evaluation_shutdown_timeout == MAX_EVALUATION_SHUTDOWN_TIMEOUT
        assert self._warnings(caplog) == []

    def test_a_usable_value_is_a_float(self):
        value = OTelConfig(evaluation_shutdown_timeout=3).evaluation_shutdown_timeout
        assert value == 3.0 and isinstance(value, float)


class TestDetectorSettingsOnTheMainConfig:
    """The result cache and the toxicity model's options are OTelConfig settings
    like the rest, so ``genai_otel.instrument(evaluation_cache_size=0)`` works."""

    def test_defaults(self):
        config = OTelConfig()
        assert config.evaluation_cache_size == 256
        assert config.toxicity_quantize is False
        assert config.toxicity_torch_threads is None

    def test_are_declared_fields(self):
        names = {f.name for f in dataclasses.fields(OTelConfig)}
        assert {"evaluation_cache_size", "toxicity_quantize", "toxicity_torch_threads"} <= names

    def test_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("GENAI_EVALUATION_CACHE_SIZE", "32")
        monkeypatch.setenv("GENAI_TOXICITY_QUANTIZE", "TRUE")
        monkeypatch.setenv("GENAI_TOXICITY_TORCH_THREADS", "4")
        config = OTelConfig()
        assert config.evaluation_cache_size == 32
        assert config.toxicity_quantize is True
        assert config.toxicity_torch_threads == 4

    def test_keyword_arguments_win(self, monkeypatch):
        monkeypatch.setenv("GENAI_EVALUATION_CACHE_SIZE", "32")
        monkeypatch.setenv("GENAI_TOXICITY_QUANTIZE", "true")
        monkeypatch.setenv("GENAI_TOXICITY_TORCH_THREADS", "4")
        config = OTelConfig(
            evaluation_cache_size=0, toxicity_quantize=False, toxicity_torch_threads=2
        )
        assert config.evaluation_cache_size == 0
        assert config.toxicity_quantize is False
        assert config.toxicity_torch_threads == 2

    @pytest.mark.parametrize("value, expected", [("big", 256), ("-4", 0), ("", 256)])
    def test_cache_size_from_the_environment_that_is_not_a_count(
        self, monkeypatch, value, expected
    ):
        monkeypatch.setenv("GENAI_EVALUATION_CACHE_SIZE", value)
        assert OTelConfig().evaluation_cache_size == expected

    @pytest.mark.parametrize("value, expected", [(-4, 0), ("12", 12), ("big", 256), (None, 256)])
    def test_cache_size_passed_in_code_is_made_safe(self, value, expected):
        assert OTelConfig(evaluation_cache_size=value).evaluation_cache_size == expected

    @pytest.mark.parametrize("value", ["", "0", "-2", "many"])
    def test_torch_threads_from_the_environment_that_are_not_a_positive_count(
        self, monkeypatch, value
    ):
        monkeypatch.setenv("GENAI_TOXICITY_TORCH_THREADS", value)
        assert OTelConfig().toxicity_torch_threads is None

    @pytest.mark.parametrize("value", [0, -2, "many", True, 2.5])
    def test_torch_threads_passed_in_code_that_are_not_a_positive_count(self, value):
        assert OTelConfig(toxicity_torch_threads=value).toxicity_torch_threads is None

    def test_torch_threads_passed_as_text_are_read_as_a_count(self):
        assert OTelConfig(toxicity_torch_threads="3").toxicity_torch_threads == 3


class TestDetectorSettingsReachTheDetectors:
    """What instrument() is told is what the detectors are built with."""

    @staticmethod
    def _configs(**settings):
        config = OTelConfig(
            enable_pii_detection=True,
            enable_toxicity_detection=True,
            enable_bias_detection=True,
            enable_prompt_injection_detection=True,
            enable_restricted_topics=True,
            enable_hallucination_detection=True,
            # Pinned, not inherited from the shell: with it on, the toxicity
            # config asks for an API key.
            toxicity_use_perspective_api=False,
            **settings,
        )
        return _evaluation_configs(config)

    def test_every_detector_gets_the_cache_size(self):
        configs = self._configs(evaluation_cache_size=7)
        assert len(configs) == 6
        assert {name: c.cache_size for name, c in configs.items()} == dict.fromkeys(configs, 7)

    def test_the_cache_can_be_turned_off_for_all_of_them(self):
        configs = self._configs(evaluation_cache_size=0)
        assert [c.cache_size for c in configs.values()] == [0] * 6

    def test_the_toxicity_model_options_reach_its_config(self):
        toxicity = self._configs(toxicity_quantize=True, toxicity_torch_threads=2)[
            "toxicity_config"
        ]
        assert toxicity.quantize is True
        assert toxicity.torch_threads == 2

    def test_keyword_arguments_beat_the_environment_all_the_way_down(self, monkeypatch):
        monkeypatch.setenv("GENAI_EVALUATION_CACHE_SIZE", "99")
        monkeypatch.setenv("GENAI_TOXICITY_QUANTIZE", "true")
        monkeypatch.setenv("GENAI_TOXICITY_TORCH_THREADS", "8")
        configs = self._configs(
            evaluation_cache_size=3, toxicity_quantize=False, toxicity_torch_threads=2
        )
        assert [c.cache_size for c in configs.values()] == [3] * 6
        assert configs["toxicity_config"].quantize is False
        assert configs["toxicity_config"].torch_threads == 2

    def test_the_environment_still_works_on_its_own(self, monkeypatch):
        monkeypatch.setenv("GENAI_EVALUATION_CACHE_SIZE", "11")
        monkeypatch.setenv("GENAI_TOXICITY_QUANTIZE", "true")
        monkeypatch.setenv("GENAI_TOXICITY_TORCH_THREADS", "8")
        configs = self._configs()
        assert [c.cache_size for c in configs.values()] == [11] * 6
        assert configs["toxicity_config"].quantize is True
        assert configs["toxicity_config"].torch_threads == 8

    def test_a_detector_that_is_off_gets_no_config(self):
        configs = _evaluation_configs(
            OTelConfig(
                enable_pii_detection=False,
                enable_toxicity_detection=False,
                enable_bias_detection=True,
                enable_prompt_injection_detection=False,
                enable_restricted_topics=False,
                enable_hallucination_detection=False,
            )
        )
        assert list(configs) == ["bias_config"]


class TestDetectorSettings:
    @pytest.mark.parametrize("config_class", DETECTOR_CONFIGS, ids=lambda c: c.__name__)
    def test_result_cache_size_default(self, config_class):
        assert config_class().cache_size == 256

    @pytest.mark.parametrize("config_class", DETECTOR_CONFIGS, ids=lambda c: c.__name__)
    def test_result_cache_size_from_the_environment(self, monkeypatch, config_class):
        monkeypatch.setenv("GENAI_EVALUATION_CACHE_SIZE", "0")
        assert config_class().cache_size == 0

    @pytest.mark.parametrize("config_class", DETECTOR_CONFIGS, ids=lambda c: c.__name__)
    def test_result_cache_size_that_is_not_a_number_falls_back(self, monkeypatch, config_class):
        monkeypatch.setenv("GENAI_EVALUATION_CACHE_SIZE", "big")
        assert config_class().cache_size == 256

    def test_toxicity_quantisation_is_off_unless_asked_for(self, monkeypatch):
        assert ToxicityConfig().quantize is False
        monkeypatch.setenv("GENAI_TOXICITY_QUANTIZE", "true")
        assert ToxicityConfig().quantize is True

    def test_torch_threads_are_left_alone_unless_set(self, monkeypatch):
        """The thread count is process-wide and belongs to the application."""
        assert ToxicityConfig().torch_threads is None
        monkeypatch.setenv("GENAI_TOXICITY_TORCH_THREADS", "4")
        assert ToxicityConfig().torch_threads == 4

    @pytest.mark.parametrize("value", ["", "0", "-2", "many"])
    def test_torch_threads_ignores_values_that_are_not_a_positive_count(self, monkeypatch, value):
        monkeypatch.setenv("GENAI_TOXICITY_TORCH_THREADS", value)
        assert ToxicityConfig().torch_threads is None
