"""Streamlit UI: review an agreement step by step, or chat with the agent about the documents."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.documents import STORE, TEMPLATE_ID
from agent.graph import (
    COVERAGE_PREFIX,
    answered_by,
    build_graph,
    chat_config,
    extract_findings,
    message_text,
    reviewed_clauses,
    run_config,
    task_message,
)
from agent.llm import get_model

st.set_page_config(page_title="Clause Deviation Agent", page_icon="📄", layout="wide")

GITHUB = "https://github.com/Pallav6268/clause-deviation-agent"
LINKEDIN = "https://www.linkedin.com/in/pallav-kumar-ai"
MODEL_SETTINGS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
)


def load_secrets_into_env() -> None:
    """Copy Streamlit secrets (top level or inside any [section]) into the environment.

    Runs on every rerun, so secrets saved after the app started are picked up."""
    try:
        secrets = st.secrets.to_dict()
    except Exception:  # no secrets file (local run)
        return

    def put(mapping: dict) -> None:
        for key, value in mapping.items():
            if isinstance(value, dict):
                put(value)
            elif isinstance(value, str) and value.strip():
                os.environ[key.strip().upper()] = value.strip()

    put(secrets)


@st.cache_resource
def load_agent(settings: tuple):
    # `settings` is only the cache key: the agent is rebuilt when the keys or models change.
    choice = get_model()
    review = build_graph(choice.llm, choice.fallbacks)
    chat = build_graph(choice.llm, choice.fallbacks, mode="chat")
    return review, chat, choice


def settings_fingerprint() -> tuple:
    """Changes when a model setting or the agent code changes, so a redeploy never serves a stale agent."""
    code = b"".join(p.read_bytes() for p in sorted((Path(__file__).parent / "agent").glob("*.py")))
    settings = (hashlib.sha256(os.getenv(name, "").encode()).hexdigest()[:12] for name in MODEL_SETTINGS)
    return (hashlib.sha256(code).hexdigest()[:12], *settings)


def short(value, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " …"


def call_lines(msg) -> str:
    return "\n".join(
        f"{c['name']}(" + ", ".join(f"{k}={short(v, 60)!r}" for k, v in c["args"].items()) + ")"
        for c in getattr(msg, "tool_calls", [])
    )


load_secrets_into_env()
review_graph, chat_graph, choice = load_agent(settings_fingerprint())
template = STORE.get(TEMPLATE_ID)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("How it works")
    st.graphviz_chart(
        """digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
        START [shape=circle, label=""]; END [shape=doublecircle, label=""];
        START -> agent; agent -> tools [label="tool calls"]; tools -> agent [label="results"];
        agent -> coverage [label="skipped a clause?", style=dashed]; coverage -> agent [style=dashed];
        agent -> END [label="final answer"]; }"""
    )
    st.markdown(
        "A single **LangGraph** agent runs a reason → act → observe loop with native "
        "tool calling.\n\n"
        "**Tier 1:** `review_clause` checks amounts, percentages, durations and dates exactly.\n\n"
        "**Tier 2:** the LLM reads the wording to catch changed obligations, rights or venue.\n\n"
        "**Coverage check:** if the agent tries to finish early, the graph sends it back to the clauses it skipped."
    )
    st.caption(f"Model: {choice.label}")
    if not choice.live:
        st.caption("Live mode reads one of these secrets: ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY.")
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
        "changed numbers and missing clauses. Wording changes and chat need the live LLM.",
        icon="ℹ️",
    )

review_tab, chat_tab = st.tabs(["Review an agreement", "Chat with the agent"])

# ================================================================ review tab
with review_tab:
    samples = STORE.agreement_ids()
    source = st.radio("Agreement", [*samples, "Paste your own"], horizontal=True)

    if source == "Paste your own":
        st.caption("Use the same Markdown headings as the template, e.g. `## 5. Possession`.")
        pasted = st.text_area("Agreement text", height=260, placeholder="## 1. Parties\n...")
        agreement_id = None
        if pasted.strip():
            agreement_id = f"custom-{uuid.uuid5(uuid.NAMESPACE_OID, pasted).hex[:8]}"
            STORE.add(agreement_id, pasted)
            st.caption(f"Saved as `{agreement_id}`. You can ask about it in the chat tab too.")
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

    if run and agreement_id:
        messages = [task_message(agreement_id)]
        step = 0
        error = None
        with st.status("Agent is working…", expanded=True) as status:
            try:
                for update in review_graph.stream({"messages": messages}, config=run_config(), stream_mode="updates"):
                    for node, payload in update.items():
                        new = payload.get("messages", [])
                        messages.extend(new)
                        if node == "agent":
                            for msg in new:
                                step += 1
                                thought = message_text(msg).strip()
                                st.markdown(f"**Step {step} · think**  \n{thought or '_(no text, tools only)_'}")
                                if calls := call_lines(msg):
                                    st.code(calls, language="python")
                        elif node == "coverage":
                            for msg in new:
                                st.warning(msg.content.removeprefix(COVERAGE_PREFIX).strip(), icon="↩️")
                        elif new:
                            st.markdown(f"*Observe · {len(new)} tool result{'s' if len(new) > 1 else ''}*")
                            with st.container(height=min(60 + 28 * len(new), 180)):
                                for msg in new:
                                    st.caption(f"↳ {msg.name} → {short(msg.content)}")
                status.update(label=f"Done in {step} steps · reasoning trace", state="complete", expanded=True)
            except Exception as exc:  # surface model/API errors in the UI
                status.update(label="Run failed", state="error")
                error = f"{type(exc).__name__}: {exc}"
        st.session_state["result"] = {
            "error": error,
            "agreement_id": agreement_id,
            "findings": extract_findings(messages),
            "summary": message_text(messages[-1]) if len(messages) > 1 else "",
            "steps": step,
            "reviewed": len(reviewed_clauses(messages)),
            "model": answered_by(messages),
        }

    result = st.session_state.get("result")
    if result and result.get("error"):
        st.error(f"The model call failed, so no review was produced.\n\n{result['error']}")
        if "401" in result["error"] or "Authentication" in result["error"]:
            st.caption("The API key in the app's secrets was rejected by the provider. Check or replace it.")
    elif result:
        findings = result["findings"]
        st.subheader(f"Findings · {result['agreement_id']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Clauses reviewed", f"{result['reviewed']} / {len(template.clauses)}")
        m2.metric("Deviations", len(findings))
        m3.metric("High severity", sum(f.get("severity") == "high" for f in findings))
        if result["reviewed"] < len(template.clauses):
            st.warning("The agent did not review every clause, so this list may be incomplete.")
        if result["model"]:
            st.caption(f"Answered by: {result['model']}")
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

# ================================================================ chat tab
with chat_tab:
    if not choice.live:
        st.info("Chat needs a live model. Add an API key to the app's secrets to turn it on.", icon="💬")
    else:
        st.caption(
            "Ask in plain words, e.g. *What changed in B-204's cancellation clause?*, "
            "*Compare the possession clause in A-101 and B-204*, or *Which agreement has the worst delay terms?*"
        )
        history = st.session_state.setdefault("chat", [])
        if history and st.button("Clear chat"):
            history.clear()
            st.rerun()
        box = st.container()
        with box:
            for msg in history:
                with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
                    st.markdown(message_text(msg))
        question = st.chat_input("Ask about the template or any agreement")
        if question:
            history.append(HumanMessage(question))
            with box:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    steps = st.status("Looking it up…", expanded=False)
                    messages = list(history)
                    used = 0
                    try:
                        for update in chat_graph.stream(
                            {"messages": messages}, config=chat_config(), stream_mode="updates"
                        ):
                            for node, payload in update.items():
                                new = payload.get("messages", [])
                                messages.extend(new)
                                for msg in new:
                                    if node == "agent" and (calls := call_lines(msg)):
                                        used += len(msg.tool_calls)
                                        steps.code(calls, language="python")
                                    elif node == "tools":
                                        steps.caption(f"↳ {msg.name} → {short(msg.content, 200)}")
                        model = answered_by(messages)
                        label = f"Used {used} tool call{'s' if used != 1 else ''}" + (f" · {model}" if model else "")
                        steps.update(label=label, state="complete")
                        answer = message_text(messages[-1]).strip() or "_(no answer)_"
                        st.markdown(answer)
                        history.append(AIMessage(answer))
                    except Exception as exc:  # surface model/API errors in the UI
                        steps.update(label="Failed", state="error")
                        st.error(f"The model call failed.\n\n{type(exc).__name__}: {exc}")
                        history.pop()
