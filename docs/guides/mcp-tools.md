# MCP Tools (Model Context Protocol)

TraceVerde auto-instruments databases, caches, message queues, vector databases, and APIs used as MCP tools in your GenAI applications.

## Quick Setup

```bash
# Install with specific tool categories
pip install genai-otel-instrument[databases]     # PostgreSQL, MySQL, MongoDB, Redis, etc.
pip install genai-otel-instrument[messaging]     # Kafka, RabbitMQ
pip install genai-otel-instrument[vector-dbs]    # Pinecone, Weaviate, Qdrant, ChromaDB, etc.
pip install genai-otel-instrument[all-mcp]       # Everything
```

MCP instrumentation is enabled by default. To disable:

```bash
export GENAI_ENABLE_MCP_INSTRUMENTATION=false
```

## Databases

| Tool | Operations Traced | Install |
|------|-------------------|---------|
| PostgreSQL | Queries via psycopg2 | `[databases]` |
| MySQL | Queries via mysql-connector | `[databases]` |
| MongoDB | All CRUD operations | `[databases]` |
| SQLAlchemy | ORM queries and sessions | `[databases]` |
| TimescaleDB | Hypertable operations | `[databases]` |
| OpenSearch | Search, index, bulk | `[databases]` |
| Elasticsearch | Search, index, bulk | `[databases]` |
| FalkorDB | Cypher queries, graph management | `[falkordb]` |

### Example: PostgreSQL + LLM

```python
import genai_otel
genai_otel.instrument(service_name="rag-app")

import psycopg2
from openai import OpenAI

# Database query - automatically traced
conn = psycopg2.connect("dbname=mydb")
cursor = conn.cursor()
cursor.execute("SELECT content FROM documents WHERE topic = %s", ("AI",))
context = cursor.fetchall()

# LLM call with context - automatically traced with cost
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"Context: {context}"},
        {"role": "user", "content": "Summarize the key points"},
    ],
)

# Trace shows: DB query -> LLM call with full context propagation
```

## Caching

| Tool | Operations Traced | Install |
|------|-------------------|---------|
| Redis | GET, SET, DEL, pipeline operations | `[databases]` |

### Example: Redis Cache + LLM

```python
import genai_otel
genai_otel.instrument()

import redis
from openai import OpenAI

r = redis.Redis()
client = OpenAI()

# Check cache - traced
cached = r.get("query:capital-of-france")
if cached:
    print(f"Cache hit: {cached}")
else:
    # LLM call - traced with cost
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    answer = response.choices[0].message.content
    r.set("query:capital-of-france", answer, ex=3600)  # Cache for 1 hour - traced

# Trace shows: Redis GET -> OpenAI call -> Redis SET
```

## Message Queues

| Tool | Operations Traced | Install |
|------|-------------------|---------|
| Apache Kafka | Produce, consume | `[messaging]` |
| RabbitMQ (pika) | Publish, consume | `[messaging]` |

## Vector Databases

| Tool | Operations Traced | Install |
|------|-------------------|---------|
| Pinecone | Query, upsert, delete | `[vector-dbs]` |
| Weaviate | Search, CRUD | `[vector-dbs]` |
| Qdrant | Search, upsert, delete | `[vector-dbs]` |
| ChromaDB | Query, add, delete | `[vector-dbs]` |
| Milvus | Search, insert, delete | `[vector-dbs]` |
| FAISS | Search, add | `[vector-dbs]` |
| LanceDB | Search, add | `[vector-dbs]` |

### Example: RAG Pipeline (Vector DB + LLM)

```python
import genai_otel
genai_otel.instrument(service_name="rag-pipeline")

import pinecone
from openai import OpenAI

client = OpenAI()

# Generate embedding - traced
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What are the benefits of OpenTelemetry?"
)
query_vector = embedding_response.data[0].embedding

# Vector search - traced
index = pinecone.Index("knowledge-base")
results = index.query(vector=query_vector, top_k=5, include_metadata=True)

# LLM call with retrieved context - traced with cost
context = "\n".join([r.metadata["text"] for r in results.matches])
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"Answer based on: {context}"},
        {"role": "user", "content": "What are the benefits of OpenTelemetry?"},
    ],
)

# Complete RAG trace: Embedding -> Pinecone search -> LLM completion
```

## Object Storage

| Tool | Operations Traced | Install |
|------|-------------------|---------|
| MinIO (S3-compatible) | Put, get, list objects | `[object-storage]` |

## HTTP/REST APIs

HTTPx instrumentation is enabled by default for HTTP calls.

!!! warning "Requests Library Limitation"
    The `requests` library cannot be auto-instrumented when using OTLP HTTP exporters (default). Use `httpx` instead, or switch to OTLP gRPC exporters. See [Known Limitations](https://github.com/Mandark-droid/genai_otel_instrument#known-limitations) for details.

## Full-Stack Example

```python
import genai_otel
genai_otel.instrument(service_name="full-stack-genai")

import openai
import pinecone
import redis
import psycopg2

# All of these are automatically instrumented:
cache = redis.Redis().get('key')                           # Cache check
index = pinecone.Index("embeddings")
results = index.query(vector=[...], top_k=5)               # Vector search
conn = psycopg2.connect("dbname=mydb")
cursor = conn.cursor()
cursor.execute("SELECT * FROM context")                     # Database query
client = openai.OpenAI()
response = client.chat.completions.create(model="gpt-4", messages=[...])  # LLM call

# You get:
# - Distributed traces across all services
# - Cost tracking for LLM calls
# - Performance metrics for DB, cache, vector DB
# - Complete observability with zero manual instrumentation
```

## Span Attributes

MCP tool spans include:

| Attribute | Description |
|-----------|-------------|
| `db.system` | Database type (e.g., "postgresql", "redis") |
| `db.operation` | Operation type (e.g., "SELECT", "GET") |
| `db.name` | Database/index name |
| `db.statement` | Query/command (truncated for safety) |
| `net.peer.name` | Server hostname |
| `net.peer.port` | Server port |
