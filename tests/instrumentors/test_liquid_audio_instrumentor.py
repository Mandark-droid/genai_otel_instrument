import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.liquid_audio_instrumentor import LiquidAudioInstrumentor


def _mock_lfm2_module(model_cls):
    mock_module = MagicMock()
    mock_module.LFM2AudioModel = model_cls
    return mock_module


class TestLiquidAudioInstrumentor(unittest.TestCase):
    """Tests for LiquidAudioInstrumentor.instrument() and its guards."""

    def test_init_available(self):
        with patch.dict("sys.modules", {"liquid_audio": MagicMock()}):
            instrumentor = LiquidAudioInstrumentor()
            self.assertTrue(instrumentor._available)

    def test_init_not_available(self):
        with patch.dict("sys.modules", {"liquid_audio": None}):
            instrumentor = LiquidAudioInstrumentor()
            self.assertFalse(instrumentor._available)

    def test_instrument_skips_when_not_available(self):
        with patch.dict("sys.modules", {"liquid_audio": None}):
            instrumentor = LiquidAudioInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            self.assertFalse(instrumentor._instrumented)

    def test_instrument_wraps_generation_methods_when_available(self):
        class MockLFM2AudioModel:
            def generate_sequential(self, *args, **kwargs):
                return "sequential"

            def generate_interleaved(self, *args, **kwargs):
                return "interleaved"

        with patch.dict("sys.modules", {"liquid_audio": _mock_lfm2_module(MockLFM2AudioModel)}):
            instrumentor = LiquidAudioInstrumentor()
            config = OTelConfig()
            original_sequential = MockLFM2AudioModel.generate_sequential
            original_interleaved = MockLFM2AudioModel.generate_interleaved

            instrumentor.instrument(config)

            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            self.assertIsNot(MockLFM2AudioModel.generate_sequential, original_sequential)
            self.assertIsNot(MockLFM2AudioModel.generate_interleaved, original_interleaved)
            self.assertTrue(
                getattr(MockLFM2AudioModel, "_genai_otel_liquid_audio_instrumented", False)
            )

    def test_instrument_is_idempotent(self):
        class MockLFM2AudioModel:
            def generate_sequential(self, *args, **kwargs):
                return "sequential"

            def generate_interleaved(self, *args, **kwargs):
                return "interleaved"

        with patch.dict("sys.modules", {"liquid_audio": _mock_lfm2_module(MockLFM2AudioModel)}):
            instrumentor = LiquidAudioInstrumentor()
            instrumentor.instrument(OTelConfig())
            wrapped_once = MockLFM2AudioModel.generate_sequential

            second_instrumentor = LiquidAudioInstrumentor()
            second_instrumentor.instrument(OTelConfig())

            # A second instrument() call must not re-wrap an already-wrapped method.
            self.assertEqual(MockLFM2AudioModel.generate_sequential, wrapped_once)

    def test_instrument_logs_and_swallows_error_by_default(self):
        class MockLFM2AudioModel:
            def generate_sequential(self, *args, **kwargs):
                return "sequential"

        with patch.dict("sys.modules", {"liquid_audio": _mock_lfm2_module(MockLFM2AudioModel)}):
            instrumentor = LiquidAudioInstrumentor()
            config = OTelConfig()
            config.fail_on_error = False

            import wrapt as _wrapt

            with patch.object(_wrapt, "wrap_function_wrapper", side_effect=RuntimeError("boom")):
                instrumentor.instrument(config)  # should not raise

    def test_instrument_reraises_when_fail_on_error(self):
        class MockLFM2AudioModel:
            def generate_sequential(self, *args, **kwargs):
                return "sequential"

        with patch.dict("sys.modules", {"liquid_audio": _mock_lfm2_module(MockLFM2AudioModel)}):
            instrumentor = LiquidAudioInstrumentor()
            config = OTelConfig()
            config.fail_on_error = True

            import wrapt as _wrapt

            with patch.object(_wrapt, "wrap_function_wrapper", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_has_audio_input(self):
        self.assertFalse(LiquidAudioInstrumentor._has_audio_input({}))
        self.assertTrue(
            LiquidAudioInstrumentor._has_audio_input(
                {"audio_in": SimpleNamespace(shape=(1, 16000))}
            )
        )
        self.assertFalse(
            LiquidAudioInstrumentor._has_audio_input({"audio_in": SimpleNamespace(shape=(1, 0))})
        )

    def test_audio_seconds(self):
        self.assertEqual(LiquidAudioInstrumentor._audio_seconds({}), 0.0)
        self.assertEqual(
            LiquidAudioInstrumentor._audio_seconds({"audio_in": SimpleNamespace(shape=(1, 32000))}),
            2.0,
        )

    def test_extract_usage_returns_none(self):
        instrumentor = LiquidAudioInstrumentor()
        self.assertIsNone(instrumentor._extract_usage(None))
        self.assertIsNone(instrumentor._extract_usage("anything"))


if __name__ == "__main__":
    unittest.main()
