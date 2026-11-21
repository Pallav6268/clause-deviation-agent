"""LangExtract grounding, run end to end with the keyless rule-based provider."""

import pytest

from agent.documents import DATA_DIR
from agent.grounding import RuleBasedExtractor, diff_terms, extract_terms, grounding_rate, highlight

# Every deviation planted in the sample agreements that a key term can show (see README).
PLANTED = {
    "A-101": {("grace_period", ""), ("maintenance_advance", "")},
    "B-204": {("forfeiture", ""), ("refund_period", ""), ("car_parking", ""), ("arbitration_seat", "")},
    "C-307": {("payment_milestone", "booking"), ("payment_milestone", "structure"), ("delay_compensation", "")},
}


def _text(doc_id: str) -> str:
    path = DATA_DIR / ("template.md" if doc_id == "template" else f"agreements/{doc_id}.md")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def template_terms():
    return extract_terms(_text("template"), RuleBasedExtractor())


@pytest.mark.parametrize("doc_id", ["template", *PLANTED])
def test_every_term_is_grounded_to_its_exact_source_span(doc_id):
    text = _text(doc_id)
    terms = extract_terms(text, RuleBasedExtractor())
    assert terms
    assert grounding_rate(text, terms) == 1.0
    assert all(t.clause_key for t in terms)


@pytest.mark.parametrize("doc_id", PLANTED)
def test_diff_finds_planted_deviations_and_nothing_else(template_terms, doc_id):
    diffs = diff_terms(template_terms, extract_terms(_text(doc_id), RuleBasedExtractor()))
    flagged = {(d.term, d.identity) for d in diffs if d.status != "same"}
    assert flagged == PLANTED[doc_id]


def test_discretionary_delay_compensation_is_flagged(template_terms):
    diffs = diff_terms(template_terms, extract_terms(_text("C-307"), RuleBasedExtractor()))
    (d,) = [d for d in diffs if d.term == "delay_compensation"]
    assert d.notes == ["obligation: mandatory → discretionary"]
    assert "may, at its sole discretion" in d.agreement.text


def test_highlight_escapes_and_marks():
    text = _text("B-204")
    terms = extract_terms(text, RuleBasedExtractor())
    out = highlight(text, [(t, "changed") for t in terms])
    assert out.count("<mark") == len(terms)
    assert "arbitration seated in Mumbai</mark>" in out
