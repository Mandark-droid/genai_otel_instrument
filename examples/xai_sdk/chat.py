"""Trace one chat request through the official xAI Python SDK."""

from xai_sdk import Client
from xai_sdk.chat import user

import genai_otel


def main():
    genai_otel.instrument()
    client = Client()
    chat = client.chat.create(model="grok-4.7")
    chat.append(user("Say hello in one sentence."))
    response = chat.sample()
    print(response.content)
    print("xAI-reported request cost: $%s" % response.cost_usd)


if __name__ == "__main__":
    main()
