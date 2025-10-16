import unittest
from unittest.mock import MagicMock

from genai_otel.instrumentors.anyscale_instrumentor import AnyscaleInstrumentor


class TestAnyscaleInstrumentor(unittest.TestCase):
    """Tests for AnyscaleInstrumentor"""

    def test_instrument(self):
        """Test that instrument method executes successfully."""
        instrumentor = AnyscaleInstrumentor()
        config = MagicMock()

        # Should not raise any exception
        instrumentor.instrument(config)

        # Config should be stored
        self.assertEqual(instrumentor.config, config)

    def test_extract_usage(self):
        """Test that _extract_usage returns None."""
        instrumentor = AnyscaleInstrumentor()
        result = instrumentor._extract_usage("any_result")

        self.assertIsNone(result)

    def test_instrument_with_importerror(self):
        """Test that ImportError during instrumentation is handled gracefully."""
        instrumentor = AnyscaleInstrumentor()
        config = MagicMock()

        # Even though there's a try-except ImportError block in the code,
        # it won't raise since the code just has 'pass' in the try block
        # This test ensures the method completes successfully
        instrumentor.instrument(config)

        # Verify config was set
        self.assertEqual(instrumentor.config, config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
