from agent.documents import STORE, TEMPLATE_ID, parse_markdown
from agent.tools import MISSING, list_clauses, review_clause
from agent.values import diff_values, extract_values


def test_template_has_eleven_clauses():
    assert len(STORE.get(TEMPLATE_ID).clauses) == 11


def test_parse_markdown_keys():
    doc = parse_markdown("x", "# Title\n\n## 5. Possession\nText A\n\n## 9. Car Parking\nText B\n")
    assert list(doc.clauses) == ["possession", "car_parking"]
    assert doc.clauses["car_parking"].text == "Text B"


def test_extract_values_kinds():
    text = "Pay ₹ 62,40,000 and 10% within 45 days; 1,040 sq ft; by 31 March 2028; 12 months."
    assert extract_values(text) == sorted(["₹6240000", "10%", "45 days", "1040 sq ft", "31 March 2028", "12 months"])


def test_placeholder_clause_allows_filled_values():
    result = diff_values("Price is {{total_price}}, booking 10%.", "Price is ₹ 50,00,000, booking 10%.")
    assert result["variable_clause"] is True
    assert result["values_deviate"] is False


def test_review_clause_detects_grace_period_change():
    result = review_clause.invoke({"agreement_id": "A-101", "clause_key": "possession"})
    check = result["value_check"]
    assert check["only_in_template"] == ["6 months"]
    assert check["only_in_agreement"] == ["12 months"]
    assert check["values_deviate"] is True
    assert "12 months" in result["agreement_text"] and "6 months" in result["template_text"]


def test_missing_clause():
    result = review_clause.invoke({"agreement_id": "B-204", "clause_key": "car_parking"})
    assert result["clause_missing_in_agreement"] is True
    assert result["agreement_text"] == MISSING


def test_list_clauses_returns_keys():
    keys = [c["key"] for c in list_clauses.invoke({"document_id": "C-307"})]
    assert "payment_schedule" in keys
