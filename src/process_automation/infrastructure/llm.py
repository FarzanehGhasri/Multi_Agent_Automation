"""Chat-model adapters and the factory that decides whether to use one.

The system is fully functional **without** any language model: the retrieval,
the gap analysis and the remediation catalogue are deterministic. A chat model
is an optional enhancement that improves the narrative of the Action Plan.

This file is the only place in the project that knows a vendor SDK exists.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import Settings
from ..interfaces import LanguageModel


class ChatModelAdapter:
    """Adapts a LangChain ``BaseChatModel`` to the :class:`LanguageModel` port."""

    def __init__(self, chat_model: Any, name: str | None = None) -> None:
        self._chat_model = chat_model
        self._name = name or type(chat_model).__name__

    @property
    def name(self) -> str:
        return self._name

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Send one prompt and return the text of the answer."""
        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))
        response = self._chat_model.invoke(messages)
        # ``content`` may be a string or a list of content blocks depending on
        # the provider; normalise both shapes here so callers never care.
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)


class EchoLanguageModel:
    """Test double: returns a deterministic echo instead of calling an API."""

    @property
    def name(self) -> str:
        return "echo"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""


def build_language_model(settings: Settings) -> LanguageModel | None:
    """Return a chat model according to ``settings.reasoning_mode``.

    * ``offline`` - always ``None``; the deterministic components are used.
    * ``llm``     - a model is required; a missing key or package raises.
    * ``auto``    - use a model when one can be built, otherwise fall back
                    silently. This is the default and is what makes the
                    notebook runnable on a machine with no API key.
    """
    mode = settings.reasoning_mode.lower()
    if mode == "offline":
        return None

    try:
        model = _instantiate(settings)
    except Exception as error:  # noqa: BLE001 - provider/SDK errors vary widely
        if mode == "llm":
            raise RuntimeError(
                "reasoning_mode='llm' was requested but no chat model could be "
                f"created: {error}"
            ) from error
        return None
    return model


def _instantiate(settings: Settings) -> LanguageModel:
    """Pick a provider from the environment. Add new ones here only."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI  # imported lazily: optional dep

        return ChatModelAdapter(
            ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature),
            name=f"openai:{settings.llm_model}",
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic  # optional dep

        model_name = (
            settings.llm_model
            if settings.llm_model.startswith("claude")
            else "claude-sonnet-4-5"
        )
        return ChatModelAdapter(
            ChatAnthropic(model=model_name, temperature=settings.llm_temperature),
            name=f"anthropic:{model_name}",
        )
    raise RuntimeError("no provider API key found in the environment")


__all__ = ["ChatModelAdapter", "EchoLanguageModel", "build_language_model"]
