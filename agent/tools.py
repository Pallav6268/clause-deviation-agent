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
def get_clause(document_id: str, clause_key: str) -> str:
    """Return the full text of one clause from a document. Returns a MISSING
    notice if the document does not contain that clause."""
    clause = STORE.get(document_id).clauses.get(clause_key)
    return clause.text if clause else MISSING


@tool
def compare_values(agreement_id: str, clause_key: str) -> dict:
    """Exactly compare the amounts, percentages, durations, areas and dates of a
    clause in the agreement against the same clause in the template. Placeholder
    clauses (unit, price) are allowed to carry extra filled-in values. Use this
    before judging any clause; it does not judge wording."""
    template = STORE.get(TEMPLATE_ID).clauses.get(clause_key)
    agreement = STORE.get(agreement_id).clauses.get(clause_key)
    if template is None:
        return {"error": f"'{clause_key}' is not a template clause."}
    if agreement is None:
        return {"clause_missing_in_agreement": True}
    result = diff_values(template.text, agreement.text)
    result["clause_missing_in_agreement"] = False
    return result


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


TOOLS = [list_clauses, get_clause, compare_values, record_finding]
