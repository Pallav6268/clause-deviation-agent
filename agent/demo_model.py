"""A rule-based stand-in for the language model, used when no API key is set.

It drives the real graph and the real tools, deciding each step from the tool
results it has seen. It only catches deviations visible to the value check
(numbers and missing clauses). Wording changes need the live model, which is
exactly the point of tier 2.
"""

from __future__ import annotations

import json
import re
from itertools import count
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .graph import COVERAGE_PREFIX

HIGH_IMPACT = {"consideration", "possession", "delay_compensation", "cancellation_by_allottee", "car_parking"}


class ScriptedAgentModel(BaseChatModel):
    """Deterministic agent policy that emits native tool calls.

    ``skip_first_pass`` and ``ignore_coverage`` simulate a careless model in tests: it skips the last
    clauses on its first pass and, unless told to ignore it, obeys the graph's coverage check."""

    skip_first_pass: int = 0
    ignore_coverage: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted-demo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedAgentModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- policy -----------------------------------------------------------
    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        agreement_id = _agreement_id(messages)
        results = _tool_results(messages)
        ids = count(len(results) + 1)

        def call(name: str, **args) -> dict:
            return {"name": name, "args": args, "id": f"call_{next(ids)}", "type": "tool_call"}

        listed = [r for r in results if r["name"] == "list_clauses" and r["args"]["document_id"] == "template"]
        if not listed:
            return AIMessage(
                content="First I need the template's clause list so I know what to check.",
                tool_calls=[call("list_clauses", document_id="template")],
            )
        clauses = [c["key"] for c in listed[0]["output"]]

        compared = {r["args"]["clause_key"]: r["output"] for r in results if r["name"] == "review_clause"}
        if not compared:
            first = clauses[: len(clauses) - self.skip_first_pass]
            calls = [call("review_clause", agreement_id=agreement_id, clause_key=k) for k in first]
            return AIMessage(
                content=f"Reviewing {len(first)} clauses in parallel: both texts plus an exact value check.",
                tool_calls=calls,
            )

        missing = [k for k in clauses if k not in compared]
        last = messages[-1]
        nudged = isinstance(last, HumanMessage) and str(last.content).startswith(COVERAGE_PREFIX)
        if missing and nudged and not self.ignore_coverage:
            calls = [call("review_clause", agreement_id=agreement_id, clause_key=k) for k in missing]
            return AIMessage(content=f"Reviewing the {len(missing)} clause(s) I skipped.", tool_calls=calls)

        recorded = {r["args"]["clause_key"] for r in results if r["name"] == "record_finding"}
        deviating = [k for k in clauses if _deviates(compared.get(k, {}))]
        to_record = [k for k in deviating if k not in recorded]
        if to_record:
            calls = [call("record_finding", **_finding(k, compared[k])) for k in to_record]
            return AIMessage(
                content=f"{len(to_record)} clause(s) show a value change or are missing. Recording them.",
                tool_calls=calls,
            )

        top = deviating[0].replace("_", " ") if deviating else "none"
        summary = (
            f"Checked {len(compared)} of {len(clauses)} clauses of {agreement_id}. "
            f"{len(deviating)} {'clause deviates' if len(deviating) == 1 else 'clauses deviate'} "
            "on values or are missing. "
            f"Most important: {top}.\n\n"
            "Demo mode is rule-based: it catches changed numbers and missing clauses only. "
            "Wording changes (for example 'shall pay' to 'may pay', or a different arbitration seat) "
            "need the live LLM."
        )
        return AIMessage(content=summary)


def _agreement_id(messages: list[BaseMessage]) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            m = re.search(r"agreement ID:\s*([\w-]+)", str(msg.content))
            if m:
                return m.group(1)
    raise ValueError("No agreement ID in the task message.")


def _tool_results(messages: list[BaseMessage]) -> list[dict]:
    calls: dict[str, dict] = {}
    out: list[dict] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for c in msg.tool_calls:
                calls[c["id"]] = c
        elif isinstance(msg, ToolMessage) and msg.tool_call_id in calls:
            c = calls[msg.tool_call_id]
            try:
                output = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except (json.JSONDecodeError, TypeError):
                output = msg.content
            out.append({"name": c["name"], "args": c["args"], "output": output})
    return out


def _deviates(result: dict) -> bool:
    return bool(result.get("clause_missing_in_agreement") or result.get("value_check", {}).get("values_deviate"))


def _finding(key: str, result: dict) -> dict:
    severity = "high" if key in HIGH_IMPACT else "medium"
    if result.get("clause_missing_in_agreement"):
        return {
            "clause_key": key,
            "severity": severity,
            "template_says": "Clause present in the template.",
            "agreement_says": "Clause missing.",
            "explanation": "The agreement drops a standard clause, so the allottee loses that protection.",
        }
    return {
        "clause_key": key,
        "severity": severity,
        "template_says": ", ".join(result["value_check"]["only_in_template"]) or "-",
        "agreement_says": ", ".join(result["value_check"]["only_in_agreement"]) or "-",
        "explanation": "Values differ from the standard template; review before signing.",
    }
