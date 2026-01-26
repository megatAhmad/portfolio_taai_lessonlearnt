"""Configuration module for the Maintenance RAG system."""

from .settings import settings, Settings, load_settings, get_settings
from .prompts import (
    ENRICHMENT_SYSTEM_PROMPT,
    ENRICHMENT_USER_PROMPT_TEMPLATE,
    RELEVANCE_SYSTEM_PROMPT,
    RELEVANCE_USER_PROMPT_TEMPLATE,
)
from .llm_client import (
    create_chat_client,
    create_embedding_client,
    get_model_name,
    call_chat_completion,
    call_embedding,
)

__all__ = [
    "settings",
    "Settings",
    "load_settings",
    "get_settings",
    "ENRICHMENT_SYSTEM_PROMPT",
    "ENRICHMENT_USER_PROMPT_TEMPLATE",
    "RELEVANCE_SYSTEM_PROMPT",
    "RELEVANCE_USER_PROMPT_TEMPLATE",
    "create_chat_client",
    "create_embedding_client",
    "get_model_name",
    "call_chat_completion",
    "call_embedding",
]
