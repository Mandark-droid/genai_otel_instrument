"""LlamaIndex Example"""

import genai_otel

genai_otel.instrument()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load documents and create index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(f"Response: {response}")
print("✅ LlamaIndex operations instrumented!")
