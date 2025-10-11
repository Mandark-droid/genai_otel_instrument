from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="genai-otel-instrument",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-instrumentation>=0.41b0",
        "opentelemetry-exporter-otlp>=1.20.0",
        "opentelemetry-semantic-conventions>=0.41b0",
        # Existing OpenTelemetry instrumentations
        "opentelemetry-instrumentation-requests>=0.41b0",
        "opentelemetry-instrumentation-httpx>=0.41b0",
        "opentelemetry-instrumentation-sqlalchemy>=0.41b0",
        "opentelemetry-instrumentation-redis>=0.41b0",
        "opentelemetry-instrumentation-kafka-python>=0.41b0",
        "opentelemetry-instrumentation-pymongo>=0.41b0",
        "opentelemetry-instrumentation-psycopg2>=0.41b0",
        "opentelemetry-instrumentation-mysql>=0.41b0",
        "wrapt>=1.14.0",
        "pynvml>=11.5.0",
        "tiktoken>=0.5.0",
    ],
    entry_points={
        "console_scripts": [
            "genai-instrument=genai_otel.cli:main",
        ],
    },
    # New additions
    author="Kshitij Thakkar",  # Replace with actual author
    author_email="kshitijthakkar@rocketmail.com",  # Replace with actual email
    description="A comprehensive wrapper for automatic instrumentation of LLM/GenAI applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mandark-droid/genai_otel_instrument",  # Replace with repo URL if applicable
    license="Apache-2.0 license",  # Matches the README
    python_requires=">=3.8",  # Assuming based on dependencies; adjust if needed
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache-2.0 license",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta",  # Adjust based on maturity
    ],
    include_package_data=True,  # If you add non-code files like templates,
    extras_require={
    "gpu": ["pynvml>=11.5.0"],
    # Add others if needed, e.g., for specific providers
   }
)