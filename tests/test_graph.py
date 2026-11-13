"""End-to-end runs of the real graph and tools, driven by the scripted model."""

import pytest

from agent.demo_model import ScriptedAgentModel
from agent.graph import build_graph, extract_findings, run_config, task_message

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
