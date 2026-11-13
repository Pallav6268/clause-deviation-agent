"""Deterministic extraction of the values that matter in a clause.

Tier 1 of the review: amounts, percentages, durations, areas and dates are
pulled out with regular expressions and compared exactly. The language model
only has to judge what the numbers cannot show (wording, obligations, venue).
"""

from __future__ import annotations

import re
from collections import Counter

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("date", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\s+(\d{{4}})\b")),
    ("money", re.compile(r"(?:₹|Rs\.?|INR)\s?(\d[\d,]*(?:\.\d+)?)")),
    ("percent", re.compile(r"(\d+(?:\.\d+)?)\s?%")),
    ("area", re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?(?:sq\.?\s?ft|square feet)", re.IGNORECASE)),
    ("duration", re.compile(r"\b(\d+)\s+(day|month|year)s?\b", re.IGNORECASE)),
]

PLACEHOLDER = re.compile(r"\{\{\s*[a-z_]+\s*\}\}")


def extract_values(text: str) -> list[str]:
    """Return normalised values found in ``text``, for example ``'12 months'``."""
    text = PLACEHOLDER.sub(" ", text)
    found: list[str] = []
    consumed: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= s or span[0] >= e for s, e in consumed)

    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if not free(m.span()):
                continue
            consumed.append(m.span())
            if kind == "date":
                found.append(f"{int(m.group(1))} {m.group(2)} {m.group(3)}")
            elif kind == "money":
                found.append("₹" + m.group(1).replace(",", ""))
            elif kind == "percent":
                found.append(f"{m.group(1)}%")
            elif kind == "area":
                found.append(m.group(1).replace(",", "") + " sq ft")
            else:
                found.append(f"{m.group(1)} {m.group(2).lower()}s")
    return sorted(found)


def has_placeholders(text: str) -> bool:
    return bool(PLACEHOLDER.search(text))


def diff_values(template_text: str, agreement_text: str) -> dict:
    """Compare the values of a template clause with the same agreement clause."""
    t = Counter(extract_values(template_text))
    a = Counter(extract_values(agreement_text))
    variable = has_placeholders(template_text)
    only_t = sorted((t - a).elements())
    only_a = sorted((a - t).elements())
    # In a clause with placeholders, extra agreement values are the filled-in
    # fields (unit, price, area) and are expected.
    deviates = bool(only_t) or (bool(only_a) and not variable)
    return {
        "template_values": sorted(t.elements()),
        "agreement_values": sorted(a.elements()),
        "only_in_template": only_t,
        "only_in_agreement": only_a,
        "variable_clause": variable,
        "values_deviate": deviates,
    }
