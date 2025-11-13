"""Module for OpenTelemetry instrumentors for various LLM providers and frameworks.

This package contains individual instrumentor classes for different Generative AI
libraries and frameworks, allowing for automatic tracing and metric collection
of their operations.

All imports are done lazily to avoid ImportError when optional dependencies
are not installed.
"""

from .anthropic_instrumentor import AnthropicInstrumentor
from .anyscale_instrumentor import AnyscaleInstrumentor
from .aws_bedrock_instrumentor import AWSBedrockInstrumentor
from .azure_openai_instrumentor import AzureOpenAIInstrumentor
from .cohere_instrumentor import CohereInstrumentor
from .crewai_instrumentor import CrewAIInstrumentor
from .google_ai_instrumentor import GoogleAIInstrumentor
from .groq_instrumentor import GroqInstrumentor
from .huggingface_instrumentor import HuggingFaceInstrumentor
from .hyperbolic_instrumentor import HyperbolicInstrumentor
from .langchain_instrumentor import LangChainInstrumentor
from .langgraph_instrumentor import LangGraphInstrumentor
from .llamaindex_instrumentor import LlamaIndexInstrumentor
from .mistralai_instrumentor import MistralAIInstrumentor
from .ollama_instrumentor import OllamaInstrumentor

# Import instrumentors only - they handle their own dependency checking
from .openai_instrumentor import OpenAIInstrumentor
from .openai_agents_instrumentor import OpenAIAgentsInstrumentor
from .replicate_instrumentor import ReplicateInstrumentor
from .sambanova_instrumentor import SambaNovaInstrumentor
from .togetherai_instrumentor import TogetherAIInstrumentor
from .vertexai_instrumentor import VertexAIInstrumentor

__all__ = [
    "OpenAIInstrumentor",
    "OpenAIAgentsInstrumentor",
    "AnthropicInstrumentor",
    "GoogleAIInstrumentor",
    "AWSBedrockInstrumentor",
    "AzureOpenAIInstrumentor",
    "CohereInstrumentor",
    "CrewAIInstrumentor",
    "MistralAIInstrumentor",
    "TogetherAIInstrumentor",
    "GroqInstrumentor",
    "OllamaInstrumentor",
    "VertexAIInstrumentor",
    "ReplicateInstrumentor",
    "AnyscaleInstrumentor",
    "SambaNovaInstrumentor",
    "HyperbolicInstrumentor",
    "LangChainInstrumentor",
    "LangGraphInstrumentor",
    "LlamaIndexInstrumentor",
    "HuggingFaceInstrumentor",
]
