"""GenAI OTel Instrumentation Demo Application

This demo showcases automatic instrumentation of multiple LLM providers
and frameworks with cost tracking, metrics, and distributed tracing.
"""

import time

import genai_otel

# Enable auto-instrumentation for all supported libraries
print("🚀 Starting GenAI OTel Demo...")
print("📊 Auto-instrumenting all LLM libraries...")
genai_otel.instrument()
print("✅ Instrumentation enabled!\n")

# Demo 1: OpenAI
print("=" * 60)
print("DEMO 1: OpenAI GPT-3.5 Turbo")
print("=" * 60)
try:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is OpenTelemetry in one sentence?"},
        ],
        max_tokens=100,
    )

    print(f"✅ OpenAI Response: {response.choices[0].message.content}")
    print(f"📊 Tokens: {response.usage.total_tokens}")
    print(f"💰 Estimated cost captured in metrics\n")
except Exception as e:
    print(f"⚠️  OpenAI demo skipped: {e}\n")

time.sleep(1)

# Demo 2: Anthropic
print("=" * 60)
print("DEMO 2: Anthropic Claude")
print("=" * 60)
try:
    from anthropic import Anthropic

    anthropic_client = Anthropic()
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "What is distributed tracing?"}],
    )

    print(f"✅ Claude Response: {message.content[0].text}")
    print(f"📊 Tokens: {message.usage.input_tokens + message.usage.output_tokens}")
    print(f"💰 Estimated cost captured in metrics\n")
except Exception as e:
    print(f"⚠️  Anthropic demo skipped: {e}\n")

time.sleep(1)

# Demo 3: LangChain
print("=" * 60)
print("DEMO 3: LangChain with OpenAI")
print("=" * 60)
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["topic"], template="Explain {topic} in simple terms in one sentence."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(topic="observability")

    print(f"✅ LangChain Response: {result}")
    print(f"📊 Chain execution traced automatically\n")
except Exception as e:
    print(f"⚠️  LangChain demo skipped: {e}\n")

print("=" * 60)
print("🎉 Demo Complete!")
print("=" * 60)
print("\n📈 View traces and metrics:")
print("   - Jaeger UI: http://localhost:16686")
print("   - Prometheus: http://localhost:9091")
print("   - Grafana: http://localhost:3000")
print("\n🔍 What was captured:")
print("   ✅ Distributed traces with parent-child relationships")
print("   ✅ Token usage metrics (prompt, completion, total)")
print("   ✅ Cost calculations for each request")
print("   ✅ Model and provider metadata")
print("   ✅ Request/response timing and latency")
print("   ✅ Error tracking (if any)")
print("\n💡 Check the Jaeger UI to see all the telemetry data!")
print("\nKeeping container alive for 5 minutes to explore traces...")
time.sleep(300)
