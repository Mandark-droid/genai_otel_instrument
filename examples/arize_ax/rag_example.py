"""RAG example with genai-otel-instrument, exported to Arize AX.

Index, retrieve, generate - as one trace. The embedding calls are traced as
their own spans, so the retrieval leg is visible rather than implied by a chat
span that mysteriously contains context.

As with example.py, there is no Arize or OpenInference package involved and no
Arize-specific code. The entire integration is standard OpenTelemetry
environment variables.
"""

import math
import os
import sys

# Content capture is off by default because retrieval inputs routinely carry
# user data. It is enabled here so the embedded text is visible in the trace,
# which is the point of the example. Set it deliberately in production.
os.environ.setdefault("GENAI_ENABLE_CONTENT_CAPTURE", "true")

import genai_otel  # noqa: E402

REQUIRED = ("ARIZE_SPACE_ID", "ARIZE_API_KEY", "OPENAI_API_KEY")
missing = [name for name in REQUIRED if not os.getenv(name)]
if missing:
    sys.exit(f"Missing environment variables: {', '.join(missing)}")

PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME", "traceverde-example")

# --- The entire Arize configuration: standard OTel variables only. ---
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otlp.arize.com")
os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
    f"space_id={os.environ['ARIZE_SPACE_ID']},api_key={os.environ['ARIZE_API_KEY']}"
)
os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"openinference.project.name={PROJECT_NAME}"
os.environ.setdefault("OTEL_SERVICE_NAME", "traceverde-arize-example")
# ---------------------------------------------------------------------

genai_otel.instrument()

from openai import OpenAI  # noqa: E402
from opentelemetry import trace  # noqa: E402

client = OpenAI()
tracer = trace.get_tracer("rag.example")

DOCUMENTS = [
    "OpenTelemetry semantic conventions v1.27.0 renamed gen_ai.usage.prompt_tokens "
    "to gen_ai.usage.input_tokens.",
    "Arize AX ingests the OpenTelemetry GenAI semantic conventions natively and "
    "normalises them onto its OpenInference model.",
    "The OTLP HTTP exporter appends /v1/traces to the configured base endpoint.",
]
QUESTION = "Which OTel version renamed the token usage attributes?"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# The parent span groups the three calls into one trace. Without it they arrive
# as three unrelated traces and the pipeline has to be reassembled by eye.
with tracer.start_as_current_span("rag.pipeline") as pipeline_span:
    pipeline_span.set_attribute("rag.document_count", len(DOCUMENTS))

    # 1. Index. One span for the batch; gen_ai.request.input_count is 3.
    indexed = client.embeddings.create(model="text-embedding-3-small", input=DOCUMENTS)
    document_vectors = [item.embedding for item in indexed.data]
    print(f"Indexed {len(document_vectors)} documents, dimension {len(document_vectors[0])}")

    # 2. Retrieve. One span for the query embedding.
    queried = client.embeddings.create(model="text-embedding-3-small", input=QUESTION)
    query_vector = queried.data[0].embedding

    ranked = sorted(
        (
            (cosine_similarity(query_vector, vector), document)
            for vector, document in zip(document_vectors, DOCUMENTS)
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    top_score, top_document = ranked[0]
    pipeline_span.set_attribute("rag.top_score", top_score)
    print(f"Retrieved (score {top_score:.4f}): {top_document}")

    # 3. Generate with the retrieved context. The usual chat span.
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using only this context:\n{top_document}"},
            {"role": "user", "content": QUESTION},
        ],
        max_tokens=80,
    )
    print(f"Answer: {answer.choices[0].message.content}")

# Short-lived scripts must flush before exit, or the batch span processor is
# torn down with spans still queued and nothing reaches Arize.
provider = trace.get_tracer_provider()
if hasattr(provider, "force_flush"):
    provider.force_flush(30000)

print(f"\nRAG trace exported to Arize AX project '{PROJECT_NAME}':")
print("  rag.pipeline")
print("    openai.embeddings      (index, input_count=3)")
print("    openai.embeddings      (query, input_count=1)")
print("    openai.chat.completion (generate)")
