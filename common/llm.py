"""LLM factory. Returns an OpenAI-compatible chat model.

Supports both OpenAI directly and OpenRouter:
- Set OPENAI_API_KEY  + LLM_BASE_URL=https://api.openai.com/v1  (default)
- Or OPENROUTER_API_KEY + LLM_BASE_URL=https://openrouter.ai/api/v1
"""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set — copy .env.example to .env and fill it in")
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        temperature=temperature,
    )
