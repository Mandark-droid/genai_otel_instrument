"""Voice-agent turn telemetry: who spoke when (issue #48).

``shape`` holds the ``voice.*`` attribute shape and its interval arithmetic,
``manual`` the ``voice_conversation()`` helper for applications with their own
audio loop. The Pipecat and LiveKit Agents integrations emit the same shape.
"""

from .manual import VoiceConversation, voice_conversation
from .shape import Turn, conversation_attributes, turn_attributes

__all__ = [
    "Turn",
    "VoiceConversation",
    "conversation_attributes",
    "turn_attributes",
    "voice_conversation",
]
