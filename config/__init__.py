"""Configuration module for the Maintenance RAG system."""

from .settings import settings
from .prompts import (
    ENRICHMENT_SYSTEM_PROMPT,
    ENRICHMENT_USER_PROMPT_TEMPLATE,
    RELEVANCE_SYSTEM_PROMPT,
    RELEVANCE_USER_PROMPT_TEMPLATE,
)

__all__ = [
    "settings",
    "ENRICHMENT_SYSTEM_PROMPT",
    "ENRICHMENT_USER_PROMPT_TEMPLATE",
    "RELEVANCE_SYSTEM_PROMPT",
    "RELEVANCE_USER_PROMPT_TEMPLATE",
]
