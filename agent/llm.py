"""Pick the model from environment variables (first match wins).

ANTHROPIC_API_KEY  -> Claude   (model from ANTHROPIC_MODEL)
DEEPSEEK_API_KEY   -> DeepSeek (model from DEEPSEEK_MODEL)
GROQ_API_KEY       -> Groq     (model from GROQ_MODEL)
none               -> rule-based demo model (no key, no cost)

Tracing: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to send every run
to LangSmith (project from LANGSMITH_PROJECT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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


def get_model() -> ModelChoice:
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        name = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return ModelChoice(ChatAnthropic(model=name, temperature=0, max_tokens=2048), f"Claude · {name}", True)
    if os.getenv("DEEPSEEK_API_KEY"):
        from langchain_deepseek import ChatDeepSeek

        name = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        return ModelChoice(ChatDeepSeek(model=name, temperature=0), f"DeepSeek · {name}", True)
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        return ModelChoice(ChatGroq(model=name, temperature=0), f"Groq · {name}", True)
    return ModelChoice(ScriptedAgentModel(), "Demo mode (rule-based, no API key)", False)
