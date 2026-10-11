# Instrument with a Coding Agent

If you use an AI coding agent or assistant, you can hand it the prompt below and let it
add TraceVerde to your application. The prompt is self-contained: paste it as is into the
agent, in the repository you want instrumented.

It tells the agent to add two lines at the top of your entry point, install the right
extras, configure through environment variables, and then prove the result by printing
spans to the console. It does not ask the agent to change any of your LLM calls.

## The prompt

```text
Add OpenTelemetry tracing for LLM calls to this Python application using TraceVerde.

Package on PyPI: genai-otel-instrument      Import name: genai_otel
Docs: https://mandark-droid.github.io/genai_otel_instrument/

Steps

1. Find the entry point of every process this application starts: the module that
   runs first (CLI main, ASGI/WSGI app module, worker start-up script, the first cell
   of a notebook).

2. Find which LLM SDKs and agent frameworks the code imports, for example openai,
   anthropic, langchain, langgraph, crewai, pydantic_ai, autogen_agentchat. Install
   the package with the matching extras and add it to the project's dependency file
   (requirements.txt, pyproject.toml, or whatever this project uses):

       pip install "genai-otel-instrument[openai,langgraph]"

   Extras are named after the library: openai, anthropic, google, langchain,
   langgraph, crewai, pydantic-ai, autogen-agentchat, llamaindex, and others listed
   in the docs. Plain "pip install genai-otel-instrument" is enough when the SDK is
   already a dependency of this project.

3. At the very top of each entry point, add:

       import genai_otel
       genai_otel.instrument()

   This must run before any LLM SDK or agent framework is imported, and before any
   client, LLM or agent object is created. A client created earlier is not traced.
   Call it once per process. Do not wrap, decorate or otherwise change existing LLM
   calls: instrumentation is automatic.

4. Configure through environment variables, not code. Add them wherever this project
   keeps its configuration (.env.example, deployment manifests, the README):

       OTEL_SERVICE_NAME=<a name for this service>
       OTEL_EXPORTER_OTLP_ENDPOINT=<OTLP/HTTP endpoint, for example http://localhost:4318>
       OTEL_EXPORTER_OTLP_HEADERS=<key1=val1,key2=val2, only if the backend needs auth>

   Never hard-code endpoints, API keys or headers. Leave content capture off, which
   is the default: GENAI_ENABLE_CONTENT_CAPTURE=true records prompts and completions
   on spans, so only turn it on if I ask for it.

5. Only if I ask for it:
   - limit what is instrumented: GENAI_ENABLED_INSTRUMENTORS=openai,langgraph
   - sample traces: GENAI_SAMPLING_RATE=0.5

6. Verify with a real run. Run the smallest path of the application that makes an
   LLM call, with OTEL_EXPORTER_OTLP_ENDPOINT set to an empty string for that run.
   Spans are then printed to the console instead of exported. Check that:
   - there is one span per LLM call, carrying gen_ai.request.model,
     gen_ai.usage.input_tokens and gen_ai.usage.output_tokens, and
     gen_ai.usage.cost.total when the model has a known price;
   - if an agent framework is used, its spans are the parents of the LLM spans, in
     the same trace.
   Then restore the real endpoint.

7. Report back: the files you changed, the install line, the environment variables to
   set, and the span names you saw in step 6. If you could not verify something, say
   so instead of assuming it works.

Rules

- Make the smallest change that works. Do not refactor unrelated code.
- Do not add a second LLM tracing library next to this one. If one is already
  present, or the application already sets up its own OpenTelemetry TracerProvider,
  stop and ask me how to proceed.
- If an agent framework ships its own tracing, leave it off unless I ask; running
  both reports every model request twice.
- Do not commit secrets, and do not print or log API keys.
```

## What you get

After the agent is done, every LLM call your application makes is a span with the model,
token counts and cost, and agent framework runs appear as parent spans around them. Point
`OTEL_EXPORTER_OTLP_ENDPOINT` at any OpenTelemetry backend to see them.

## Doing it by hand

The same steps, without an agent, are in the [Quick Start](quickstart.md). Framework
specifics, including the order rule for CrewAI and what each framework's spans look like,
are in [Multi-Agent Frameworks](../guides/multi-agent-frameworks.md).
