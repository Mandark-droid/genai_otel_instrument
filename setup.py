from setuptools import setup, find_packages
import os

# Read the README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read version from __version__.py
version = {}
with open(os.path.join('genai_otel', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), version)

setup(
    name="genai-otel-instrument",
    version=version['__version__'],
    packages=find_packages(exclude=['tests', 'tests.*', 'docs']),
    include_package_data=True,
    package_data={
        'genai_otel': ['llm_pricing.json', 'py.typed'],
    },

    # Core dependencies only
    install_requires=[
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-instrumentation>=0.41b0",
        "opentelemetry-exporter-otlp>=1.20.0",
        "opentelemetry-semantic-conventions>=0.41b0",
        "opentelemetry-instrumentation-requests>=0.41b0",
        "opentelemetry-instrumentation-httpx>=0.41b0",
        "opentelemetry-instrumentation-mysql>=0.41b0",
        "wrapt>=1.14.0",
        "httpx>=0.23.0",
        "mysql-connector-python",
        "mysql==0.0.3",
        "opentelemetry-instrumentation-psycopg2>=0.41b0",
        "psycopg2-binary",
        "opentelemetry-instrumentation-redis>=0.41b0",
        "redis",
        "opentelemetry-instrumentation-pymongo>=0.41b0",
        "pymongo",
        "opentelemetry-instrumentation-sqlalchemy>=0.41b0",
        "sqlalchemy",
        "opentelemetry-instrumentation-kafka-python>=0.41b0",
        "kafka-python",

    ],

    # Optional dependencies
    extras_require={
        # GPU metrics support
        "gpu": [
            "pynvml>=11.5.0",
        ],

        # Individual LLM providers
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "google": ["google-generativeai>=0.3.0"],
        "aws": ["boto3>=1.28.0"],
        "azure": ["azure-ai-openai>=1.0.0"],
        "cohere": ["cohere>=4.0.0"],
        "mistral": ["mistralai>=0.0.7"],
        "together": ["together>=0.2.0"],
        "groq": ["groq>=0.4.0"],
        "ollama": ["ollama>=0.1.0"],
        "replicate": ["replicate>=0.15.0"],

        # LLM Frameworks
        "langchain": ["langchain>=0.1.0"],
        "llamaindex": ["llama-index>=0.9.0"],
        "huggingface": ["transformers>=4.30.0"],

        # Database instrumentations
        "databases": [
            "opentelemetry-instrumentation-sqlalchemy>=0.41b0",
            "sqlalchemy"
            "opentelemetry-instrumentation-redis>=0.41b0",
            "redis",
            "opentelemetry-instrumentation-pymongo>=0.41b0",
            "pymongo",
            "opentelemetry-instrumentation-psycopg2>=0.41b0",
            "psycopg2-binary",
            "opentelemetry-instrumentation-mysql>=0.41b0",
            "mysql-connector-python",
        ],

        # Message queue instrumentation
        "messaging": [
            "opentelemetry-instrumentation-kafka-python>=0.41b0",
            "kafka-python",
        ],

        # Vector databases
        "vector-dbs": [
            "pinecone-client>=2.0.0",
            "weaviate-client>=3.0.0",
            "qdrant-client>=1.0.0",
            "chromadb>=0.4.0",
            "pymilvus>=2.3.0",
            "faiss-cpu>=1.7.0",
        ],

        # All LLM providers
        "all-providers": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "google-generativeai>=0.3.0",
            "boto3>=1.28.0",
            "azure-ai-openai>=1.0.0",
            "cohere>=4.0.0",
            "mistralai>=0.0.7",
            "together>=0.2.0",
            "groq>=0.4.0",
            "ollama>=0.1.0",
            "replicate>=0.15.0",
            "langchain>=0.1.0",
            "llama-index>=0.9.0",
            "transformers>=4.30.0",
        ],

        # All MCP tools
        "all-mcp": [
            "opentelemetry-instrumentation-sqlalchemy>=0.41b0",
            "opentelemetry-instrumentation-redis>=0.41b0",
            "opentelemetry-instrumentation-pymongo>=0.41b0",
            "opentelemetry-instrumentation-psycopg2>=0.41b0",
            "opentelemetry-instrumentation-mysql>=0.41b0",
            "opentelemetry-instrumentation-kafka-python>=0.41b0",
            "pinecone-client>=2.0.0",
            "weaviate-client>=3.0.0",
            "qdrant-client>=1.0.0",
            "chromadb>=0.4.0",
            "pymilvus>=2.3.0",
            "faiss-cpu>=1.7.0",
        ],

        # Everything
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "google-generativeai>=0.3.0",
            "boto3>=1.28.0",
            "azure-ai-openai>=1.0.0",
            "cohere>=4.0.0",
            "mistralai>=0.0.7",
            "together>=0.2.0",
            "groq>=0.4.0",
            "ollama>=0.1.0",
            "replicate>=0.15.0",
            "langchain>=0.1.0",
            "llama-index>=0.9.0",
            "transformers>=4.30.0",
            "pynvml>=11.5.0",
            "opentelemetry-instrumentation-sqlalchemy>=0.41b0",
            "opentelemetry-instrumentation-redis>=0.41b0",
            "opentelemetry-instrumentation-pymongo>=0.41b0",
            "opentelemetry-instrumentation-psycopg2>=0.41b0",
            "opentelemetry-instrumentation-mysql>=0.41b0",
            "opentelemetry-instrumentation-kafka-python>=0.41b0",
            "pinecone-client>=2.0.0",
            "weaviate-client>=3.0.0",
            "qdrant-client>=1.0.0",
            "chromadb>=0.4.0",
            "pymilvus>=2.3.0",
            "faiss-cpu>=1.7.0",
            "pymongo",
            "nvidia-ml-py",
            "mysql",
            "mysql-connector-python",
            "httpx"
        ],

        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "genai-instrument=genai_otel.cli:main",
        ],
    },

    # Metadata
    author="Kshitij Thakkar",
    author_email="kshitijthakkar@rocketmail.com",
    description="Comprehensive OpenTelemetry auto-instrumentation for LLM/GenAI applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mandark-droid/genai_otel_instrument",
    project_urls={
        "Bug Reports": "https://github.com/Mandark-droid/genai_otel_instrument/issues",
        "Source": "https://github.com/Mandark-droid/genai_otel_instrument",
        "Documentation": "https://github.com/Mandark-droid/genai_otel_instrument#readme",
    },
    license="Apache-2.0",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="opentelemetry observability llm genai instrumentation tracing metrics monitoring",
    zip_safe=False,
    scripts=['scripts/test_release.sh'],
)