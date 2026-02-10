"""Sarvam Arya Agent Orchestration Example with genai-otel-instrument

Demonstrates how TraceVerde captures OpenTelemetry traces for agentic
workflows built on Sarvam AI's APIs - the same primitives that power
Sarvam Arya, their production-grade agent orchestration stack.

Sarvam Arya (Drop 7, Feb 2026) is built around four guarantees:
  1. Composable Primitives - small reusable building blocks
  2. Immutable State - strict rules on data read/write
  3. Controlled Dynamism - LLM decisions bounded by guardrails
  4. Declarative Authoring - config-driven workflow definitions

This example simulates a Sarvam Arya-style agent that:
- Accepts voice/text input in any Indian language
- Routes through a multi-step reasoning pipeline
- Uses tools (translate, detect language, search, synthesize)
- Maintains immutable conversation state
- Produces localized voice+text responses

Since Arya is an enterprise platform (available on AWS Marketplace),
this example uses the underlying sarvamai SDK APIs that Arya agents
orchestrate. TraceVerde instruments every API call, giving you full
observability into agent execution.

Prerequisites:
    pip install sarvamai genai-otel-instrument

Environment variables:
    SARVAM_API_KEY: Your Sarvam AI API subscription key
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4318)

References:
    - Sarvam Arya: https://www.sarvam.ai/blogs/introducing-sarvam-arya
    - Sarvam AI Docs: https://docs.sarvam.ai/
    - AWS Marketplace: https://aws.amazon.com/marketplace/pp/prodview-cqx47vs5o6h4e
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from genai_otel import instrument

# Initialize instrumentation BEFORE importing the Sarvam AI client
instrument(
    service_name="sarvam-arya-agent",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_cost_tracking=True,
    enable_content_capture=True,
)

from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))


# ---------------------------------------------------------------------------
# 1. Immutable State - Arya Guarantee #2
# Each step produces a new state snapshot, never mutating the previous one.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentState:
    """Immutable state container for a single agent turn.

    Arya enforces immutable state to ensure reproducibility and
    auditability of agent decisions. Each step produces a new state.
    """

    turn_id: int
    user_input: str
    detected_language: Optional[str] = None
    english_query: Optional[str] = None
    intent: Optional[str] = None
    tool_results: tuple = field(default_factory=tuple)
    agent_response_en: Optional[str] = None
    agent_response_local: Optional[str] = None
    audio_generated: bool = False
    error: Optional[str] = None

    def with_update(self, **kwargs):
        """Create new state with updates (immutability pattern)."""
        current = asdict(self)
        current.update(kwargs)
        return AgentState(**current)


# ---------------------------------------------------------------------------
# 2. Composable Primitives - Arya Guarantee #1
# Each function is a self-contained primitive that can be composed freely.
# ---------------------------------------------------------------------------


def detect_language_primitive(state: AgentState) -> AgentState:
    """Primitive: Detect input language using Sarvam's language identification."""
    print(f"    [primitive] detect_language")
    try:
        result = client.text.identify_language(input=state.user_input)
        lang_code = getattr(result, "language_code", "en-IN")
        print(f"    -> detected: {lang_code}")
        return state.with_update(detected_language=lang_code)
    except Exception as e:
        print(f"    -> error: {e}")
        return state.with_update(detected_language="en-IN", error=str(e))


def translate_to_english_primitive(state: AgentState) -> AgentState:
    """Primitive: Translate user input to English for reasoning."""
    if state.detected_language == "en-IN":
        print(f"    [primitive] translate_to_en (skipped, already English)")
        return state.with_update(english_query=state.user_input)

    print(f"    [primitive] translate_to_en ({state.detected_language} -> en-IN)")
    try:
        result = client.text.translate(
            input=state.user_input,
            source_language_code=state.detected_language,
            target_language_code="en-IN",
        )
        en_text = getattr(result, "translated_text", state.user_input)
        print(f"    -> english: {en_text[:100]}")
        return state.with_update(english_query=en_text)
    except Exception as e:
        print(f"    -> error: {e}")
        return state.with_update(english_query=state.user_input, error=str(e))


def classify_intent_primitive(state: AgentState) -> AgentState:
    """Primitive: Use LLM to classify user intent (controlled dynamism)."""
    print(f"    [primitive] classify_intent")
    try:
        result = client.chat.completions(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. Classify the user message into "
                        "exactly one of these intents: ORDER_STATUS, REFUND, "
                        "ACCOUNT_HELP, PRODUCT_INFO, GENERAL_QUERY. "
                        "Respond with only the intent label, nothing else."
                    ),
                },
                {"role": "user", "content": state.english_query or state.user_input},
            ],
        )
        if hasattr(result, "choices") and result.choices:
            intent = result.choices[0].message.content.strip()
        else:
            intent = "GENERAL_QUERY"
        print(f"    -> intent: {intent}")
        return state.with_update(intent=intent)
    except Exception as e:
        print(f"    -> error: {e}")
        return state.with_update(intent="GENERAL_QUERY", error=str(e))


def execute_tool_primitive(state: AgentState) -> AgentState:
    """Primitive: Execute the appropriate tool based on classified intent.

    This demonstrates Controlled Dynamism (Arya Guarantee #3):
    The LLM chose the intent, but the tool execution is deterministic.
    """
    print(f"    [primitive] execute_tool (intent={state.intent})")

    # Simulated tool responses (in production, these would call real backends)
    tool_responses = {
        "ORDER_STATUS": "Order #12345 is out for delivery. Expected by 6 PM today.",
        "REFUND": "Your refund of Rs 1,299 has been initiated. It will reflect in 3-5 business days.",
        "ACCOUNT_HELP": "Please verify your registered email. A password reset link has been sent.",
        "PRODUCT_INFO": "The product is available in 3 colors. Free delivery on orders above Rs 499.",
        "GENERAL_QUERY": "Thank you for reaching out. How can I assist you further?",
    }

    tool_result = tool_responses.get(state.intent, tool_responses["GENERAL_QUERY"])
    print(f"    -> tool_result: {tool_result}")
    return state.with_update(tool_results=(tool_result,))


def generate_response_primitive(state: AgentState) -> AgentState:
    """Primitive: Generate a natural language response using the LLM."""
    print(f"    [primitive] generate_response")
    try:
        tool_context = "\n".join(state.tool_results) if state.tool_results else "No data"
        result = client.chat.completions(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support agent for an Indian e-commerce "
                        "company. Use the tool results below to craft a brief, empathetic "
                        "response. Keep it under 3 sentences.\n\n"
                        f"Tool Results:\n{tool_context}"
                    ),
                },
                {"role": "user", "content": state.english_query or state.user_input},
            ],
        )
        if hasattr(result, "choices") and result.choices:
            response_en = result.choices[0].message.content
        else:
            response_en = str(result)
        print(f"    -> response_en: {response_en[:150]}")
        return state.with_update(agent_response_en=response_en)
    except Exception as e:
        print(f"    -> error: {e}")
        return state.with_update(
            agent_response_en="We apologize for the inconvenience.", error=str(e)
        )


def translate_response_primitive(state: AgentState) -> AgentState:
    """Primitive: Translate the agent response back to the user's language."""
    if state.detected_language == "en-IN":
        print(f"    [primitive] translate_response (skipped, English user)")
        return state.with_update(agent_response_local=state.agent_response_en)

    print(f"    [primitive] translate_response (en-IN -> {state.detected_language})")
    try:
        result = client.text.translate(
            input=state.agent_response_en,
            source_language_code="en-IN",
            target_language_code=state.detected_language,
        )
        local_text = getattr(result, "translated_text", state.agent_response_en)
        print(f"    -> localized: {local_text[:150]}")
        return state.with_update(agent_response_local=local_text)
    except Exception as e:
        print(f"    -> error: {e}")
        return state.with_update(agent_response_local=state.agent_response_en, error=str(e))


def synthesize_speech_primitive(state: AgentState) -> AgentState:
    """Primitive: Convert the localized response to speech using Bulbul v3."""
    print(f"    [primitive] synthesize_speech (Bulbul v3, lang={state.detected_language})")
    try:
        client.text_to_speech.convert(
            text=state.agent_response_local or state.agent_response_en,
            target_language_code=state.detected_language or "en-IN",
            speaker="meera",
        )
        print(f"    -> audio: generated")
        return state.with_update(audio_generated=True)
    except Exception as e:
        print(f"    -> error: {e}")
        return state.with_update(audio_generated=False, error=str(e))


# ---------------------------------------------------------------------------
# 3. Declarative Authoring - Arya Guarantee #4
# The agent pipeline is defined as a list of primitives, not imperative code.
# In Arya, this would be a YAML/JSON config. Here we use a Python list.
# ---------------------------------------------------------------------------

# This is the "declarative" pipeline definition
# In Arya's production system, this would be a YAML configuration file
AGENT_PIPELINE = [
    detect_language_primitive,
    translate_to_english_primitive,
    classify_intent_primitive,
    execute_tool_primitive,
    generate_response_primitive,
    translate_response_primitive,
    synthesize_speech_primitive,
]


def run_agent(user_input: str, turn_id: int) -> AgentState:
    """Execute the agent pipeline over a user input.

    Each primitive in the pipeline receives the current immutable state
    and returns a new state. This mirrors how Arya orchestrates agents.
    """
    state = AgentState(turn_id=turn_id, user_input=user_input)

    for primitive in AGENT_PIPELINE:
        state = primitive(state)

    return state


# ---------------------------------------------------------------------------
# 4. Multi-turn agent conversation simulation
# ---------------------------------------------------------------------------

CONVERSATIONS = [
    {
        "user": "Rahul (Hindi)",
        "messages": [
            "Mera order kab aayega? Order number 54321.",
            "Aur refund ka kya hua?",
        ],
    },
    {
        "user": "Ananya (Tamil)",
        "messages": [
            "Ennoda account la login panna mudiyala.",
            "Product details theriyuma?",
        ],
    },
    {
        "user": "David (English)",
        "messages": [
            "What is the status of my return request?",
        ],
    },
    {
        "user": "Meera (Bengali)",
        "messages": [
            "Amar product bhenge gechhe. Exchange hobe?",
        ],
    },
]


if __name__ == "__main__":
    print("=" * 70)
    print("Sarvam Arya Agent Orchestration Example")
    print("with TraceVerde OpenTelemetry Instrumentation")
    print("=" * 70)
    print()
    print("Arya's Four Guarantees in action:")
    print("  1. Composable Primitives - 7 independent, reusable steps")
    print("  2. Immutable State - frozen dataclass, new snapshot each step")
    print("  3. Controlled Dynamism - LLM classifies intent, tools are deterministic")
    print("  4. Declarative Authoring - pipeline defined as ordered list of primitives")
    print()

    total_api_calls = 0
    turn_id = 0
    start_time = time.time()

    for conversation in CONVERSATIONS:
        user = conversation["user"]
        print(f"\n{'='*70}")
        print(f"Conversation: {user}")
        print(f"{'='*70}")

        for msg in conversation["messages"]:
            turn_id += 1
            print(f'\n  Turn {turn_id}: "{msg}"')
            print(f"  {'-'*60}")

            state = run_agent(msg, turn_id)

            print(f"\n  Final State:")
            print(f"    language:  {state.detected_language}")
            print(f"    intent:    {state.intent}")
            print(f"    response:  {(state.agent_response_local or '')[:120]}")
            print(f"    audio:     {'yes' if state.audio_generated else 'no'}")
            if state.error:
                print(f"    error:     {state.error}")

            # Count API calls (varies by language - English skips translations)
            calls = 7 if state.detected_language != "en-IN" else 5
            total_api_calls += calls

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print("Agent Execution Summary")
    print(f"{'='*70}")
    print(f"  Conversations:   {len(CONVERSATIONS)}")
    print(f"  Total turns:     {turn_id}")
    print(f"  Total API calls: ~{total_api_calls}")
    print(f"  Elapsed time:    {elapsed:.2f}s")
    print(f"  Languages:       Hindi, Tamil, English, Bengali")

    print(f"\n{'='*70}")
    print("OpenTelemetry Trace Structure (per agent turn)")
    print(f"{'='*70}")
    print(
        """
  Each agent turn generates a sequence of instrumented spans:

  [sarvam.text.identify_language]     -- detect user language
    -> sarvam.detected_language = 'hi-IN'

  [sarvam.text.translate]             -- translate to English
    -> sarvam.source_language = 'hi-IN'
    -> sarvam.target_language = 'en-IN'

  [sarvam.chat.completions]           -- classify intent (controlled dynamism)
    -> gen_ai.request.model = 'sarvam-m'
    -> gen_ai.operation.name = 'chat'

  [sarvam.chat.completions]           -- generate response
    -> gen_ai.request.model = 'sarvam-m'
    -> gen_ai.usage.prompt_tokens / completion_tokens

  [sarvam.text.translate]             -- translate response back
    -> sarvam.source_language = 'en-IN'
    -> sarvam.target_language = 'hi-IN'

  [sarvam.text_to_speech.convert]     -- synthesize voice
    -> gen_ai.request.model = 'bulbul'
    -> sarvam.speaker = 'meera'

  All spans are automatically captured by TraceVerde with:
  - Cost tracking for each API call
  - Latency measurements
  - Token usage metrics
  - Request/response content (when enabled)

  View traces at your OTLP endpoint (Jaeger, Grafana Tempo, etc.)
  to see the full agent execution flow.
"""
    )
