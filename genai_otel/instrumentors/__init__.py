"""Module for OpenTelemetry instrumentors for various LLM providers and frameworks.

This package contains individual instrumentor classes for different Generative AI
libraries and frameworks, allowing for automatic tracing and metric collection
of their operations.

All imports are done lazily to avoid ImportError when optional dependencies
are not installed.
"""

from .anthropic_instrumentor import AnthropicInstrumentor
from .anyscale_instrumentor import AnyscaleInstrumentor
from .autogen_instrumentor import AutoGenInstrumentor
from .aws_bedrock_instrumentor import AWSBedrockInstrumentor
from .azure_openai_instrumentor import AzureOpenAIInstrumentor
from .bedrock_agents_instrumentor import BedrockAgentsInstrumentor
from .cohere_instrumentor import CohereInstrumentor
from .crewai_instrumentor import CrewAIInstrumentor
from .dspy_instrumentor import DSPyInstrumentor
from .google_ai_instrumentor import GoogleAIInstrumentor
from .groq_instrumentor import GroqInstrumentor
from .guardrails_ai_instrumentor import GuardrailsAIInstrumentor
from .haystack_instrumentor import HaystackInstrumentor
from .huggingface_instrumentor import HuggingFaceInstrumentor
from .hyperbolic_instrumentor import HyperbolicInstrumentor
from .instructor_instrumentor import InstructorInstrumentor
from .langchain_instrumentor import LangChainInstrumentor
from .langgraph_instrumentor import LangGraphInstrumentor
from .llamaindex_instrumentor import LlamaIndexInstrumentor
from .mistralai_instrumentor import MistralAIInstrumentor
from .ollama_instrumentor import OllamaInstrumentor
from .openai_agents_instrumentor import OpenAIAgentsInstrumentor

# Import instrumentors only - they handle their own dependency checking
from .openai_instrumentor import OpenAIInstrumentor
from .pydantic_ai_instrumentor import PydanticAIInstrumentor
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
    "AutoGenInstrumentor",
    "BedrockAgentsInstrumentor",
    "CohereInstrumentor",
    "CrewAIInstrumentor",
    "DSPyInstrumentor",
    "MistralAIInstrumentor",
    "TogetherAIInstrumentor",
    "GroqInstrumentor",
    "GuardrailsAIInstrumentor",
    "HaystackInstrumentor",
    "InstructorInstrumentor",
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
    "PydanticAIInstrumentor",
]
