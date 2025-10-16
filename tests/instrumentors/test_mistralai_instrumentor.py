import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from opentelemetry.trace import Span
from genai_otel.instrumentors.mistralai_instrumentor import MistralAIInstrumentor
from genai_otel.config import OTelConfig

def original_chat_success(*args, **kwargs):
    mock_response = MagicMock()
    type(mock_response).usage = PropertyMock(
        return_value=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )
    return mock_response

def original_embeddings_success(*args, **kwargs):
    mock_response = MagicMock()
    type(mock_response).usage = PropertyMock(
        return_value=MagicMock(prompt_tokens=15, total_tokens=15)
    )
    return mock_response

def original_chat_error(*args, **kwargs):
    raise ValueError("Test error")

def original_embeddings_error(*args, **kwargs):
    raise ValueError("Test error")

class Wrapper:
    def __init__(self):
        self.wrapped_function = None

    def capture(self, module, name, wrapper):
        self.wrapped_function = wrapper

@patch("wrapt.patch_function_wrapper")
class TestMistralAIInstrumentor(unittest.TestCase):
    
    def setUp(self):
        self.instrumentor = MistralAIInstrumentor()
        self.config = OTelConfig()
        self.tracer = MagicMock()
        self.instrumentor.tracer = self.tracer
        self.instrumentor.request_counter = MagicMock()
        self.instrumentor.token_counter = MagicMock()
        self.instrumentor.cost_counter = MagicMock()

    def test_instrument_methods_are_called(self, mock_patch):
        self.instrumentor.instrument(self.config)
        self.assertEqual(mock_patch.call_count, 2)

    def test_chat_success(self, mock_patch):
        wrapper = Wrapper()
        mock_patch.side_effect = wrapper.capture
        
        self.instrumentor._instrument_chat()
        
        mock_span = MagicMock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        wrapper.wrapped_function(original_chat_success, None, (), {"model": "mistral-tiny"})
        
        self.tracer.start_as_current_span.assert_called_with("mistralai.chat")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "mistral-tiny")
        self.instrumentor.token_counter.add.assert_any_call(10, {'type': 'input', 'provider': 'mistralai'})

    def test_chat_error(self, mock_patch):
        wrapper = Wrapper()
        mock_patch.side_effect = wrapper.capture

        self.instrumentor._instrument_chat()
        
        mock_span = MagicMock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        with self.assertRaises(ValueError):
            wrapper.wrapped_function(original_chat_error, None, (), {"model": "mistral-tiny"})
        
        mock_span.set_attribute.assert_any_call("error", True)

    def test_embeddings_success(self, mock_patch):
        wrapper = Wrapper()
        mock_patch.side_effect = wrapper.capture

        self.instrumentor._instrument_embeddings()
        
        mock_span = MagicMock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        wrapper.wrapped_function(original_embeddings_success, None, (), {"model": "mistral-embed"})
        
        self.tracer.start_as_current_span.assert_called_with("mistralai.embeddings")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "mistral-embed")
        self.instrumentor.token_counter.add.assert_any_call(15, {'type': 'input', 'provider': 'mistralai', 'operation': 'embeddings'})

    def test_embeddings_error(self, mock_patch):
        wrapper = Wrapper()
        mock_patch.side_effect = wrapper.capture

        self.instrumentor._instrument_embeddings()
        
        mock_span = MagicMock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        with self.assertRaises(ValueError):
            wrapper.wrapped_function(original_embeddings_error, None, (), {"model": "mistral-embed"})
        
        mock_span.set_attribute.assert_any_call("error", True)

if __name__ == "__main__":
    unittest.main()