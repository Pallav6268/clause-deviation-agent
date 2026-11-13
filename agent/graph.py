"""The agent: a reason-act-observe loop built as a LangGraph StateGraph.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

The ``agent`` node asks the model what to do next. If the model requests tools,
the ``tools`` node runs them and feeds the results back. When the model answers
without tool calls, the loop ends.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import TOOLS

SYSTEM_PROMPT = """You are a legal-document review agent for a real-estate developer.
Your job: find every clause where an Agreement for Sale deviates from the standard template.

Procedure (follow it, and work in parallel where you can):
1. Call list_clauses for 'template' and for the agreement.
2. For every template clause, call compare_values (exact numeric check) and get_clause
   for both documents so you can judge the wording.
3. A clause deviates if: it is missing from the agreement; compare_values reports
   values_deviate = true; or the wording changes an obligation, right, venue or condition
   (for example 'shall pay' becoming 'may at its discretion pay', or a different
   arbitration seat), even when all numbers match. Filled-in placeholders such as unit,
   area and price are NOT deviations.
4. Call record_finding exactly once for each deviating clause. Quote the template and the
   agreement briefly, and explain the practical impact on the allottee in one sentence.
5. Finish with a short summary: how many clauses were checked, how many deviate, and the
   single most important one. Do not call tools in your final message."""

MAX_STEPS = 40


def build_graph(llm: BaseChatModel):
    model = llm.bind_tools(TOOLS)

    def agent(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def task_message(agreement_id: str) -> HumanMessage:
    return HumanMessage(
        f"Review agreement ID: {agreement_id}. Compare it with the template and record every deviation."
    )


def run_config() -> dict:
    return {"recursion_limit": MAX_STEPS, "run_name": "clause-deviation-review"}


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
