"""LLM client factory for Azure OpenAI and OpenRouter."""

from typing import Dict, Any, Optional
import logging

from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)


def create_chat_client(settings) -> OpenAI:
    """
    Create a chat client based on the configured LLM provider.

    Args:
        settings: Application settings

    Returns:
        OpenAI-compatible client (AzureOpenAI or OpenAI for OpenRouter)
    """
    if settings.is_azure:
        logger.info("Creating Azure OpenAI chat client")
        return AzureOpenAI(
            azure_endpoint=settings.azure_openai.endpoint,
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
        )
    elif settings.is_openrouter:
        logger.info("Creating OpenRouter chat client")
        return OpenAI(
            base_url=settings.openrouter.base_url,
            api_key=settings.openrouter.api_key,
            default_headers=_get_openrouter_headers(settings),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def create_embedding_client(settings) -> OpenAI:
    """
    Create an embedding client based on the configured LLM provider.

    Args:
        settings: Application settings

    Returns:
        OpenAI-compatible client for embeddings
    """
    if settings.is_azure:
        logger.info("Creating Azure OpenAI embedding client")
        return AzureOpenAI(
            azure_endpoint=settings.azure_openai.endpoint,
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
        )
    elif settings.is_openrouter:
        logger.info("Creating OpenRouter embedding client")
        return OpenAI(
            base_url=settings.openrouter.base_url,
            api_key=settings.openrouter.api_key,
            default_headers=_get_openrouter_headers(settings),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def _get_openrouter_headers(settings) -> Dict[str, str]:
    """
    Get optional headers for OpenRouter requests.

    Args:
        settings: Application settings

    Returns:
        Dictionary of headers
    """
    headers = {}

    if settings.openrouter.site_url:
        headers["HTTP-Referer"] = settings.openrouter.site_url

    if settings.openrouter.app_name:
        headers["X-Title"] = settings.openrouter.app_name

    return headers


def get_model_name(settings, model_type: str = "chat") -> str:
    """
    Get the model name based on provider and type.

    Args:
        settings: Application settings
        model_type: Type of model ("chat", "embedding", "enrichment")

    Returns:
        Model name string
    """
    if model_type == "chat":
        return settings.get_chat_model()
    elif model_type == "embedding":
        return settings.get_embedding_model()
    elif model_type == "enrichment":
        return settings.get_enrichment_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def call_chat_completion(
    client: OpenAI,
    model: str,
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 500,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call chat completion API with unified interface.

    Args:
        client: OpenAI-compatible client
        model: Model name/deployment
        messages: List of message dicts
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response
        response_format: Optional response format (e.g., {"type": "json_object"})

    Returns:
        Response content string
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_embedding(
    client: OpenAI,
    model: str,
    texts: list,
) -> list:
    """
    Call embedding API with unified interface.

    Args:
        client: OpenAI-compatible client
        model: Model name/deployment
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    # Sort by index to maintain order
    embeddings = [None] * len(texts)
    for item in response.data:
        embeddings[item.index] = item.embedding

    return embeddings
