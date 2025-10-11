"""
===========================================
GenAI OpenTelemetry Instrumentation Examples
===========================================

METHOD 1: Environment Variables Only (Zero Code Changes)
---------------------------------------------------------
export OTEL_SERVICE_NAME=my-llm-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export GENAI_ENABLE_GPU_METRICS=true
export GENAI_ENABLE_COST_TRACKING=true
export GENAI_ENABLE_MCP_INSTRUMENTATION=true

python your_app.py


METHOD 2: Single Line of Code
------------------------------
"""
import genai_otel
genai_otel.instrument()

# Your existing code works unchanged!

# Example 1: OpenAI
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Example 2: Anthropic
import anthropic
client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Example 3: Google AI
import google.generativeai as genai
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Hello!")

# Example 4: LangChain with MCP Tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

# Database calls are auto-instrumented
import psycopg2
conn = psycopg2.connect("dbname=mydb user=user")
cursor = conn.cursor()
cursor.execute("SELECT * FROM embeddings")

# Redis calls are auto-instrumented
import redis
r = redis.Redis(host='localhost', port=6379)
r.set('key', 'value')

# Vector DB calls are auto-instrumented
import pinecone
pinecone.init(api_key="YOUR_KEY")
index = pinecone.Index("my-index")
results = index.query(vector=[0.1]*1536, top_k=10)

# Kafka calls are auto-instrumented
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers='localhost:9092')
producer.send('my-topic', b'message')

"""
METHOD 3: CLI Wrapper
---------------------
genai-instrument python your_app.py


COLLECTED TELEMETRY
-------------------

Traces:
- genai.requests: All LLM API calls
- genai.chain.*: LangChain chains and agents
- db.*: Database queries (SQL, MongoDB, Redis)
- http.*: HTTP/API calls
- kafka.*: Kafka producer/consumer operations
- vector.*: Vector database operations

Metrics:
- genai.requests: Request counts by model/provider
- genai.tokens: Token usage (prompt/completion)
- genai.latency: Request latency histogram
- genai.cost: Estimated costs in USD
- genai.gpu.utilization: GPU utilization %
- genai.gpu.memory.used: GPU memory usage
- genai.gpu.temperature: GPU temperature
- genai.gpu.power.usage: GPU power consumption

Attributes on Spans:
- gen_ai.system: Provider (openai, anthropic, google, etc.)
- gen_ai.request.model: Model name
- gen_ai.usage.prompt_tokens: Input tokens
- gen_ai.usage.completion_tokens: Output tokens
- gen_ai.cost.amount: Estimated cost
- db.system: Database type
- db.operation: Operation type
- vector.collection: Vector DB collection
- vector.top_k: Number of results


SUPPORTED PROVIDERS
-------------------

LLM Providers (15+):
✓ OpenAI (GPT-4, GPT-3.5, etc.)
✓ Anthropic (Claude 3.x)
✓ Google AI (Gemini)
✓ AWS Bedrock
✓ Azure OpenAI
✓ Cohere
✓ Mistral AI
✓ Together AI
✓ Groq
✓ Ollama (local)
✓ Vertex AI
✓ Replicate
✓ Anyscale
✓ HuggingFace Transformers

Frameworks:
✓ LangChain (chains, agents, tools)
✓ LlamaIndex (query engines, indices)

MCP Tools - Databases:
✓ PostgreSQL (psycopg2)
✓ MySQL
✓ MongoDB
✓ SQLAlchemy (all supported databases)

MCP Tools - Caching:
✓ Redis

MCP Tools - Message Queues:
✓ Apache Kafka

MCP Tools - Vector Databases:
✓ Pinecone
✓ Weaviate
✓ Qdrant
✓ ChromaDB
✓ Milvus
✓ FAISS

MCP Tools - APIs:
✓ HTTP/REST (via requests, httpx)
✓ Custom API clients


ADVANCED CONFIGURATION
----------------------
"""

# Custom configuration
import genai_otel

config = genai_otel.instrument(
    service_name="my-advanced-app",
    endpoint="https://my-otel-collector:4318",
    enable_gpu_metrics=True,
    enable_cost_tracking=True,
    enable_mcp_instrumentation=True
)

# Environment variable configuration
"""
OTEL_SERVICE_NAME=my-llm-app
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_HEADERS=x-api-key=secret,x-tenant=acme

# GenAI specific
GENAI_ENABLE_GPU_METRICS=true
GENAI_ENABLE_COST_TRACKING=true
GENAI_ENABLE_MCP_INSTRUMENTATION=true

# Standard OpenTelemetry
OTEL_TRACES_EXPORTER=otlp
OTEL_METRICS_EXPORTER=otlp
OTEL_LOGS_EXPORTER=otlp
"""


# Example: Complete GenAI Application with All Features
"""
import genai_otel
genai_otel.instrument()

import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pinecone
import redis
import psycopg2

# Initialize services
openai_client = openai.OpenAI()
redis_client = redis.Redis(host='localhost')
db_conn = psycopg2.connect("dbname=vectors")
pinecone.init(api_key="key")
vector_index = pinecone.Index("embeddings")

# This entire flow is automatically instrumented:

# 1. Check cache
cached = redis_client.get('user:123:context')

# 2. If not cached, query vector DB
if not cached:
    results = vector_index.query(
        vector=get_embedding("user context"),
        top_k=5
    )
    context = process_results(results)
    redis_client.setex('user:123:context', 3600, context)

# 3. Query SQL database for user history
cursor = db_conn.cursor()
cursor.execute("SELECT * FROM conversations WHERE user_id = %s", (123,))
history = cursor.fetchall()

# 4. Make LLM call with context
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": "What's my order status?"}
    ]
)

# All of the above generates:
# - Traces showing the complete request flow
# - Metrics for each operation (LLM, DB, cache, vector DB)
# - Cost tracking for the LLM call
# - GPU metrics if using local models
# - Span attributes for debugging

print(response.choices[0].message.content)
"""
