"""The agent: a reason-act-observe loop built as a LangGraph StateGraph.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

The ``agent`` node asks the model what to do next. If the model requests tools,
the ``tools`` node runs them and feeds the results back. When the model answers
without tool calls, the loop ends.

Review mode adds a ``coverage`` node: if the model tries to finish before it has
reviewed every template clause, the graph sends it back with the list of clauses
it skipped (at most MAX_COVERAGE_NUDGES times). Chat mode answers free-form
questions with the read-only tools.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .documents import STORE, TEMPLATE_ID
from .tools import CHAT_TOOLS, REVIEW_TOOLS

SYSTEM_PROMPT = """You are a legal-document review agent for a real-estate developer.
Your job: find every clause where an Agreement for Sale deviates from the standard template.

Procedure (put several tool calls in one message wherever you can):
1. Call list_clauses for 'template'.
2. Call review_clause for every template clause. It returns both texts and an exact value check.
3. A clause deviates if: it is missing from the agreement; value_check reports
   values_deviate = true; or the wording changes an obligation, right, venue or condition
   (for example 'shall pay' becoming 'may at its discretion pay', or a different
   arbitration seat), even when all numbers match. Filled-in placeholders such as unit,
   area and price are NOT deviations.
4. Call record_finding exactly once for each deviating clause. Quote the template and the
   agreement briefly, and explain the practical impact on the allottee in one sentence.
5. Finish with a short summary: how many clauses were checked, how many deviate, and the
   single most important one. Do not call tools in your final message."""

CHAT_PROMPT = """You answer questions from a real-estate developer's legal team about the standard
Agreement for Sale template and the agreements stored in this app.
- Call list_agreements to see which agreement IDs exist, list_clauses for a document's clauses, and
  review_clause to read one clause of the template and an agreement side by side with an exact value check.
- To compare two agreements, review the same clause in each.
- Answer only from what the tools return and quote the words that matter. If the documents do not
  answer the question, say so instead of guessing.
- Keep answers short and plain. This is a document comparison, not legal advice."""

MAX_STEPS = 100  # graph steps; one model turn plus its tool run is 2 steps
CHAT_MAX_STEPS = 40
MAX_COVERAGE_NUDGES = 2
COVERAGE_PREFIX = "Coverage check:"


def build_graph(
    llm: BaseChatModel,
    fallbacks: Sequence[BaseChatModel] = (),
    mode: Literal["review", "chat"] = "review",
):
    """Compile the agent. If the primary model fails (bad key, outage), each fallback is tried in order."""
    tools = REVIEW_TOOLS if mode == "review" else CHAT_TOOLS
    prompt = SYSTEM_PROMPT if mode == "review" else CHAT_PROMPT
    models = [(m.__class__.__name__, m.bind_tools(tools)) for m in (llm, *fallbacks)]

    def agent(state: MessagesState) -> dict:
        messages = [SystemMessage(prompt), *state["messages"]]
        errors = []
        for name, model in models:
            try:
                return {"messages": [model.invoke(messages)]}
            except Exception as exc:  # try the next provider, but keep every error
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Every model provider failed:\n- " + "\n- ".join(errors))

    def coverage(state: MessagesState) -> dict:
        missing = ", ".join(missing_clauses(state["messages"]))
        return {
            "messages": [
                HumanMessage(
                    f"{COVERAGE_PREFIX} you have not reviewed these template clauses yet: {missing}. "
                    "Call review_clause for each of them, record any deviation, then give the final summary again."
                )
            ]
        }

    def route_review(state: MessagesState) -> str:
        messages = state["messages"]
        if getattr(messages[-1], "tool_calls", None):
            return "tools"
        nudges = sum(isinstance(m, HumanMessage) and str(m.content).startswith(COVERAGE_PREFIX) for m in messages)
        if missing_clauses(messages) and nudges < MAX_COVERAGE_NUDGES:
            return "coverage"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    if mode == "review":
        graph.add_node("coverage", coverage)
        graph.add_conditional_edges("agent", route_review, ["tools", "coverage", END])
        graph.add_edge("coverage", "agent")
    else:
        graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def agreement_in_task(messages) -> str | None:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            m = re.search(r"agreement ID:\s*([\w-]+)", str(msg.content))
            if m:
                return m.group(1)
    return None


def reviewed_clauses(messages) -> set[str]:
    """Template clauses the agent actually ran review_clause on (for the task's agreement)."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    agreement_id = agreement_in_task(messages)
    return {
        call["args"].get("clause_key")
        for msg in messages
        if isinstance(msg, AIMessage)
        for call in msg.tool_calls
        if call["name"] == "review_clause"
        and call["id"] in answered
        and call["args"].get("agreement_id") == agreement_id
    }


def missing_clauses(messages) -> list[str]:
    if agreement_in_task(messages) is None:
        return []
    done = reviewed_clauses(messages)
    return [key for key in STORE.get(TEMPLATE_ID).clauses if key not in done]


def answered_by(messages) -> str:
    """Model name reported by the provider on the last model reply, if any."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            meta = msg.response_metadata or {}
            return meta.get("model_name") or meta.get("model") or ""
    return ""


def task_message(agreement_id: str) -> HumanMessage:
    return HumanMessage(
        f"Review agreement ID: {agreement_id}. Compare it with the template and record every deviation."
    )


def run_config() -> dict:
    return {"recursion_limit": MAX_STEPS, "run_name": "clause-deviation-review"}


def chat_config() -> dict:
    return {"recursion_limit": CHAT_MAX_STEPS, "run_name": "clause-deviation-chat"}


def extract_findings(messages) -> list[dict]:
    """Collect record_finding calls from the conversation (last call per clause wins)."""
    findings: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                if call["name"] == "record_finding":
                    args = dict(call["args"])
                    findings[args.get("clause_key", "?")] = args
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings.values(), key=lambda f: (order.get(f.get("severity"), 3), f.get("clause_key", "")))


def message_text(msg) -> str:
    """Plain text of a message whose content may be a string or content blocks."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(p for p in parts if p)
