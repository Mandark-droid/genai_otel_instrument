"""Module for OpenTelemetry instrumentors for various LLM providers and frameworks.

This package contains individual instrumentor classes for different Generative AI
libraries and frameworks, allowing for automatic tracing and metric collection
of their operations.

All imports are done lazily to avoid ImportError when optional dependencies
are not installed.
"""

# Import instrumentors only - they handle their own dependency checking
from .openai_instrumentor import OpenAIInstrumentor
from .anthropic_instrumentor import AnthropicInstrumentor
from .google_ai_instrumentor import GoogleAIInstrumentor
from .aws_bedrock_instrumentor import AWSBedrockInstrumentor
from .azure_openai_instrumentor import AzureOpenAIInstrumentor
from .cohere_instrumentor import CohereInstrumentor
from .mistralai_instrumentor import MistralAIInstrumentor
from .togetherai_instrumentor import TogetherAIInstrumentor
from .groq_instrumentor import GroqInstrumentor
from .ollama_instrumentor import OllamaInstrumentor
from .vertexai_instrumentor import VertexAIInstrumentor
from .replicate_instrumentor import ReplicateInstrumentor
from .anyscale_instrumentor import AnyscaleInstrumentor
from .langchain_instrumentor import LangChainInstrumentor
from .llamaindex_instrumentor import LlamaIndexInstrumentor
from .huggingface_instrumentor import HuggingFaceInstrumentor

__all__ = [
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",
    "GoogleAIInstrumentor",
    "AWSBedrockInstrumentor",
    "AzureOpenAIInstrumentor",
    "CohereInstrumentor",
    "MistralAIInstrumentor",
    "TogetherAIInstrumentor",
    "GroqInstrumentor",
    "OllamaInstrumentor",
    "VertexAIInstrumentor",
    "ReplicateInstrumentor",
    "AnyscaleInstrumentor",
    "LangChainInstrumentor",
    "LlamaIndexInstrumentor",
    "HuggingFaceInstrumentor",
]
