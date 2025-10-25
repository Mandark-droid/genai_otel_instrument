"""LangChain Example with genai-otel-instrument

This example demonstrates how to use genai-otel-instrument with LangChain.
"""

import genai_otel

# Auto-instrument LangChain (and all other supported libraries)
genai_otel.instrument()

# Now use LangChain normally
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Create a simple chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

prompt = PromptTemplate(input_variables=["topic"], template="Explain {topic} in simple terms.")

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run(topic="OpenTelemetry")

print(f"Response: {result}")
print("✅ Traces and metrics have been automatically sent to your OTLP endpoint!")
