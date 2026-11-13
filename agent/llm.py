"""Pick the model from environment variables.

ANTHROPIC_API_KEY  -> Claude   (model from ANTHROPIC_MODEL)
DEEPSEEK_API_KEY   -> DeepSeek (model from DEEPSEEK_MODEL)
GROQ_API_KEY       -> Groq     (model from GROQ_MODEL)
none               -> rule-based demo model (no key, no cost)

Every provider with a key is used, in that order: the first is primary and the
rest are automatic fallbacks if a call fails (for example a rejected key).

Tracing: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to send every run
to LangSmith (project from LANGSMITH_PROJECT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from .demo_model import ScriptedAgentModel

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class ModelChoice:
    llm: BaseChatModel
    label: str
    live: bool
    fallbacks: list[BaseChatModel] = field(default_factory=list)


def available_models() -> list[tuple[BaseChatModel, str]]:
    """Every live provider that has a key, in priority order."""
    models: list[tuple[BaseChatModel, str]] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        name = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        models.append((ChatAnthropic(model=name, temperature=0, max_tokens=2048), f"Claude · {name}"))
    if os.getenv("DEEPSEEK_API_KEY"):
        from langchain_deepseek import ChatDeepSeek

        name = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        models.append((ChatDeepSeek(model=name, temperature=0), f"DeepSeek · {name}"))
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        models.append((ChatGroq(model=name, temperature=0), f"Groq · {name}"))
    return models


def get_model() -> ModelChoice:
    models = available_models()
    if not models:
        return ModelChoice(ScriptedAgentModel(), "Demo mode (rule-based, no API key)", False)
    (primary, label), rest = models[0], models[1:]
    if rest:
        label += " (fallback: " + ", ".join(lbl for _, lbl in rest) + ")"
    return ModelChoice(primary, label, True, [m for m, _ in rest])
