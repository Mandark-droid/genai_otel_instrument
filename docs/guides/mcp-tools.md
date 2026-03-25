# MCP Tools (Model Context Protocol)

TraceVerde auto-instruments databases, caches, message queues, vector databases, and APIs used as MCP tools.

## Databases

| Tool | Install Extra | Instrumented Operations |
|------|---------------|------------------------|
| PostgreSQL | `[databases]` | Queries via psycopg2 |
| MySQL | `[databases]` | Queries via mysql-connector |
| MongoDB | `[databases]` | All CRUD operations |
| SQLAlchemy | `[databases]` | ORM queries and sessions |
| TimescaleDB | `[databases]` | Hypertable operations |
| OpenSearch | `[databases]` | Search, index, bulk operations |
| Elasticsearch | `[databases]` | Search, index, bulk operations |
| FalkorDB | `[falkordb]` | Cypher queries, graph management |

## Caching

| Tool | Install Extra | Instrumented Operations |
|------|---------------|------------------------|
| Redis | `[databases]` | GET, SET, DEL, pipeline operations |

## Message Queues

| Tool | Install Extra | Instrumented Operations |
|------|---------------|------------------------|
| Apache Kafka | `[messaging]` | Produce, consume |
| RabbitMQ (pika) | `[messaging]` | Publish, consume |

## Vector Databases

| Tool | Install Extra | Instrumented Operations |
|------|---------------|------------------------|
| Pinecone | `[vector-dbs]` | Query, upsert, delete |
| Weaviate | `[vector-dbs]` | Search, CRUD |
| Qdrant | `[vector-dbs]` | Search, upsert, delete |
| ChromaDB | `[vector-dbs]` | Query, add, delete |
| Milvus | `[vector-dbs]` | Search, insert, delete |
| FAISS | `[vector-dbs]` | Search, add |
| LanceDB | `[vector-dbs]` | Search, add |

## Object Storage

| Tool | Install Extra | Instrumented Operations |
|------|---------------|------------------------|
| MinIO (S3) | `[object-storage]` | Put, get, list objects |

## HTTP/REST APIs

HTTPx instrumentation is enabled by default. Requests library instrumentation is disabled due to conflicts with OTLP HTTP exporters (see Known Limitations in README).

## Configuration

MCP instrumentation is enabled by default. To disable:

```bash
export GENAI_ENABLE_MCP_INSTRUMENTATION=false
```

Or selectively install only the MCP tools you need:

```bash
pip install genai-otel-instrument[databases,vector-dbs]
```
