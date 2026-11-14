"""End-to-end runs of the real graph and tools, driven by the scripted model."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.demo_model import ScriptedAgentModel
from agent.documents import STORE, TEMPLATE_ID
from agent.graph import (
    COVERAGE_PREFIX,
    MAX_COVERAGE_NUDGES,
    build_graph,
    chat_config,
    extract_findings,
    reviewed_clauses,
    run_config,
    task_message,
)
from agent.tools import CHAT_TOOLS

EXPECTED = {
    "A-101": {"possession": "high", "maintenance": "medium"},
    "B-204": {"cancellation_by_allottee": "high", "car_parking": "high"},
    "C-307": {"payment_schedule": "medium"},
}


@pytest.fixture(scope="module")
def graph():
    return build_graph(ScriptedAgentModel())


@pytest.mark.parametrize("agreement_id", sorted(EXPECTED))
def test_demo_run_finds_value_deviations(graph, agreement_id):
    state = graph.invoke({"messages": [task_message(agreement_id)]}, config=run_config())
    findings = {f["clause_key"]: f["severity"] for f in extract_findings(state["messages"])}
    assert findings == EXPECTED[agreement_id]
    assert not state["messages"][-1].tool_calls  # loop ended with a final answer


def test_loop_uses_tools_node(graph):
    nodes = [
        node
        for update in graph.stream({"messages": [task_message("A-101")]}, config=run_config(), stream_mode="updates")
        for node in update
    ]
    assert nodes[0] == "agent" and "tools" in nodes and nodes[-1] == "agent"


class FailingModel(ScriptedAgentModel):
    """Stands in for a provider whose key is rejected."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("401 - Authentication Fails")


def test_falls_back_when_primary_model_fails():
    graph = build_graph(FailingModel(), [ScriptedAgentModel()])
    state = graph.invoke({"messages": [task_message("A-101")]}, config=run_config())
    findings = {f["clause_key"]: f["severity"] for f in extract_findings(state["messages"])}
    assert findings == EXPECTED["A-101"]


def test_reports_every_provider_error_when_all_fail():
    graph = build_graph(FailingModel(), [FailingModel()])
    with pytest.raises(RuntimeError, match="Every model provider failed") as info:
        graph.invoke({"messages": [task_message("A-101")]}, config=run_config())
    assert str(info.value).count("401 - Authentication Fails") == 2


def test_coverage_check_sends_a_lazy_model_back_to_skipped_clauses():
    graph = build_graph(ScriptedAgentModel(skip_first_pass=2))
    state = graph.invoke({"messages": [task_message("B-204")]}, config=run_config())
    assert reviewed_clauses(state["messages"]) == set(STORE.get(TEMPLATE_ID).clauses)
    nudges = [m for m in state["messages"] if isinstance(m, HumanMessage) and m.content.startswith(COVERAGE_PREFIX)]
    assert len(nudges) == 1 and "dispute_resolution, governing_law" in nudges[0].content
    findings = {f["clause_key"]: f["severity"] for f in extract_findings(state["messages"])}
    assert findings == EXPECTED["B-204"]


def test_coverage_check_gives_up_after_the_limit():
    graph = build_graph(ScriptedAgentModel(skip_first_pass=2, ignore_coverage=True))
    state = graph.invoke({"messages": [task_message("A-101")]}, config=run_config())
    nudges = [m for m in state["messages"] if isinstance(m, HumanMessage) and m.content.startswith(COVERAGE_PREFIX)]
    assert len(nudges) == MAX_COVERAGE_NUDGES
    assert len(reviewed_clauses(state["messages"])) == 9


class ScriptedChatModel(GenericFakeChatModel):
    """Replays fixed replies; stands in for a live model in chat mode."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_chat_mode_answers_with_read_only_tools():
    replies = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "list_agreements", "args": {}, "id": "c1", "type": "tool_call"}],
            ),
            AIMessage(content="Three agreements are stored: A-101, B-204 and C-307."),
        ]
    )
    graph = build_graph(ScriptedChatModel(messages=replies), mode="chat")
    state = graph.invoke({"messages": [HumanMessage("Which agreements do you have?")]}, config=chat_config())
    tool_output = next(m.content for m in state["messages"] if isinstance(m, ToolMessage))
    assert all(a in tool_output for a in ("A-101", "B-204", "C-307")) and "template" not in tool_output
    assert state["messages"][-1].content.startswith("Three agreements")
    assert "record_finding" not in {t.name for t in CHAT_TOOLS}
