"""Streamlit UI: run the agent and watch every reasoning step and tool call."""

from __future__ import annotations

import json
import os
import uuid

import pandas as pd
import streamlit as st

# Streamlit Cloud secrets -> environment, before the agent reads them.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # no secrets file locally
    pass

from agent.documents import STORE, TEMPLATE_ID
from agent.graph import build_graph, extract_findings, message_text, run_config, task_message
from agent.llm import get_model

st.set_page_config(page_title="Clause Deviation Agent", page_icon="📄", layout="wide")

GITHUB = "https://github.com/Pallav6268/clause-deviation-agent"
LINKEDIN = "https://www.linkedin.com/in/pallav-kumar-ai"


@st.cache_resource
def load_agent():
    choice = get_model()
    return build_graph(choice.llm), choice


graph, choice = load_agent()
template = STORE.get(TEMPLATE_ID)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("How it works")
    st.graphviz_chart(
        """digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
        START [shape=circle, label=""]; END [shape=doublecircle, label=""];
        START -> agent; agent -> tools [label="tool calls"]; tools -> agent [label="results"];
        agent -> END [label="final answer"]; }"""
    )
    st.markdown(
        "A single **LangGraph** agent runs a reason → act → observe loop with native "
        "tool calling.\n\n"
        "**Tier 1:** `compare_values` checks amounts, percentages, durations and dates exactly.\n\n"
        "**Tier 2:** the LLM reads the wording to catch changed obligations, rights or venue.\n\n"
        "Each run can be traced in **LangSmith**."
    )
    st.caption(f"Model: {choice.label}")
    st.markdown(f"[Source on GitHub]({GITHUB}) · [Pallav Kumar]({LINKEDIN})")
    st.caption("All agreements in this demo are fictional.")

# ---------------------------------------------------------------- header
st.title("Clause Deviation Agent")
st.write(
    "Compares an Agreement for Sale against the developer's standard template and shows its reasoning step by step."
)
if not choice.live:
    st.info(
        "Running in **demo mode**: a rule-based stand-in drives the same graph and tools. It catches "
        "changed numbers and missing clauses. Wording changes need the live LLM.",
        icon="ℹ️",
    )

# ---------------------------------------------------------------- input
samples = STORE.agreement_ids()
source = st.radio("Agreement", [*samples, "Paste your own"], horizontal=True)

if source == "Paste your own":
    st.caption("Use the same Markdown headings as the template, e.g. `## 5. Possession`.")
    pasted = st.text_area("Agreement text", height=260, placeholder="## 1. Parties\n...")
    agreement_id = None
    if pasted.strip():
        agreement_id = f"custom-{uuid.uuid5(uuid.NAMESPACE_OID, pasted).hex[:8]}"
        STORE.add(agreement_id, pasted)
else:
    agreement_id = source

col1, col2 = st.columns(2)
with col1, st.expander("Standard template"):
    for c in template.clauses.values():
        st.markdown(f"**{c.title}** — {c.text}")
with col2, st.expander("Agreement"):
    if agreement_id:
        doc = STORE.get(agreement_id)
        for c in doc.clauses.values():
            st.markdown(f"**{c.title}** — {c.text}")

run = st.button("Run review", type="primary", disabled=agreement_id is None)


# ---------------------------------------------------------------- run
def short(value, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " …"


if run and agreement_id:
    messages = [task_message(agreement_id)]
    step = 0
    with st.status("Agent is working…", expanded=True) as status:
        try:
            for update in graph.stream({"messages": messages}, config=run_config(), stream_mode="updates"):
                for node, payload in update.items():
                    new = payload.get("messages", [])
                    messages.extend(new)
                    if node == "agent":
                        for msg in new:
                            step += 1
                            thought = message_text(msg).strip()
                            st.markdown(f"**Step {step} · think**  \n{thought or '_(no text, tools only)_'}")
                            calls = [
                                f"{c['name']}(" + ", ".join(f"{k}={short(v, 60)!r}" for k, v in c["args"].items()) + ")"
                                for c in getattr(msg, "tool_calls", [])
                            ]
                            if calls:
                                st.code("\n".join(calls), language="python")
                    elif new:
                        st.markdown(f"*Observe · {len(new)} tool result{'s' if len(new) > 1 else ''}*")
                        with st.container(height=min(60 + 28 * len(new), 180)):
                            for msg in new:
                                st.caption(f"↳ {msg.name} → {short(msg.content)}")
            status.update(label=f"Done in {step} steps · reasoning trace", state="complete", expanded=True)
        except Exception as exc:  # surface model/API errors in the UI
            status.update(label="Run failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
    st.session_state["result"] = {
        "agreement_id": agreement_id,
        "findings": extract_findings(messages),
        "summary": message_text(messages[-1]) if len(messages) > 1 else "",
        "steps": step,
    }

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result:
    findings = result["findings"]
    st.subheader(f"Findings · {result['agreement_id']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clauses checked", len(template.clauses))
    m2.metric("Deviations", len(findings))
    m3.metric("High severity", sum(f.get("severity") == "high" for f in findings))
    if findings:
        df = pd.DataFrame(findings)[["clause_key", "severity", "template_says", "agreement_says", "explanation"]]
        df.columns = ["Clause", "Severity", "Template says", "Agreement says", "Impact"]
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button(
            "Download findings (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"deviations_{result['agreement_id']}.csv",
            mime="text/csv",
        )
    else:
        st.success("No deviations found.")
    if result["summary"]:
        st.markdown("**Agent summary**")
        st.write(result["summary"])
