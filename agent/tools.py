"""Tools the agent can call. Each docstring is what the model reads."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool

from .documents import STORE, TEMPLATE_ID
from .values import diff_values

MISSING = "MISSING: this clause is not present in the document."


@tool
def list_clauses(document_id: str) -> list[dict]:
    """List the clauses of a document as {key, title}. Use document_id='template'
    for the standard template, or the agreement ID (for example 'B-204')."""
    doc = STORE.get(document_id)
    return [{"key": c.key, "title": c.title} for c in doc.clauses.values()]


@tool
def review_clause(agreement_id: str, clause_key: str) -> dict:
    """Everything needed to judge one clause, in a single call: the template text,
    the agreement text (or a MISSING notice) and an exact value check of amounts,
    percentages, durations, areas and dates. Placeholder clauses (unit, price) may
    carry filled-in values. The value check does not judge wording; you do."""
    template = STORE.get(TEMPLATE_ID).clauses.get(clause_key)
    agreement = STORE.get(agreement_id).clauses.get(clause_key)
    if template is None:
        return {"error": f"'{clause_key}' is not a template clause."}
    if agreement is None:
        return {"template_text": template.text, "agreement_text": MISSING, "clause_missing_in_agreement": True}
    return {
        "template_text": template.text,
        "agreement_text": agreement.text,
        "clause_missing_in_agreement": False,
        "value_check": diff_values(template.text, agreement.text),
    }


@tool
def record_finding(
    clause_key: str,
    severity: Literal["high", "medium", "low"],
    template_says: str,
    agreement_says: str,
    explanation: str,
) -> str:
    """Record one deviation between the agreement and the template. Call once
    per deviating clause, including clauses missing from the agreement.
    severity: high = shifts money, risk or rights against the allottee;
    medium = changes terms that should be reviewed; low = cosmetic."""
    return f"Recorded {severity} finding for '{clause_key}'."


TOOLS = [list_clauses, review_clause, record_finding]
