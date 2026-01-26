"""Configuration module for the Maintenance RAG system."""

from .settings import settings, Settings, load_settings, get_settings
from .prompts import (
    ENRICHMENT_SYSTEM_PROMPT,
    ENRICHMENT_USER_PROMPT_TEMPLATE,
    RELEVANCE_SYSTEM_PROMPT,
    RELEVANCE_USER_PROMPT_TEMPLATE,
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
]
