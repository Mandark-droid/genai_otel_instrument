FROM python:3.11-slim

# Install git (required by setuptools-scm for version detection)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project (build context is project root: ../..)
COPY . /tmp/genai-otel-instrument

# Install the genai-otel-instrument package from source with core dependencies
WORKDIR /tmp/genai-otel-instrument
RUN pip install --no-cache-dir -e ".[openai,anthropic,langchain]"

# Install demo-specific dependencies
RUN pip install --no-cache-dir \
    langchain-openai

# Set working directory for the app
WORKDIR /app

# Copy demo application
COPY examples/demo/app.py .

# Run the application
CMD ["python", "app.py"]
