"""Grounded key-term extraction with LangExtract.

The agent's findings quote clause text, but a reviewer still has to find the
words in the agreement. This module pulls the key commercial terms (grace
period, forfeiture, arbitration seat, ...) out of a document with Google's
LangExtract library. Every term is mapped back to its exact character span in
the source, so each difference can be highlighted and checked by eye.

The language model is the same one the agent uses (Claude, Gemini, DeepSeek or
Groq through LangChain). With no API key, a rule-based provider returns the
same JSON a model would, so LangExtract's chunking, parsing and source
alignment still run end to end.
"""

from __future__ import annotations

import html
import re
import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import langextract as lx
from langextract.core import base_model, types

from .documents import _HEADING, slugify

# ---------------------------------------------------------------- what to extract

# term -> attributes that are compared between template and agreement
VALUE_ATTRS: dict[str, tuple[str, ...]] = {
    "payment_milestone": ("percent",),
    "possession_date": ("date",),
    "grace_period": ("months",),
    "delay_compensation": ("obligation", "rate_percent"),
    "maintenance_advance": ("months",),
    "forfeiture": ("percent",),
    "refund_period": ("days",),
    "car_parking": ("allotted",),
    "arbitration_seat": ("city",),
}
# Terms that occur more than once are told apart by this attribute.
IDENTITY_ATTR = {"payment_milestone": "stage"}

PROMPT = textwrap.dedent(
    """\
    Extract the key commercial terms of this real-estate Agreement for Sale.
    Use exact text from the document for every extraction; do not paraphrase
    and do not overlap extractions. List them in order of appearance.
    Terms: payment_milestone (one per instalment), possession_date, grace_period,
    delay_compensation, maintenance_advance, forfeiture, refund_period,
    car_parking, arbitration_seat.
    Attributes: numbers as digits only. payment_milestone.stage is the event the
    instalment is due on, in a few words without "completion of".
    delay_compensation.obligation is "mandatory" when the developer shall/must pay
    and "discretionary" when it may pay or pays at its discretion.
    car_parking.allotted is "yes" or "no".
    Skip template placeholders such as {{unit_no}}."""
)

EXAMPLE_TEXT = (
    "The Buyer shall pay 15% on booking, 35% on completion of the plinth and 50% on possession. "
    "The Developer shall deliver possession on or before 30 June 2027, with a grace period of 3 months "
    "for force majeure. If possession is delayed beyond the grace period, the Developer shall pay the Buyer "
    "interest at 8% per annum. Maintenance is charged monthly, with 6 months payable in advance. "
    "On cancellation by the Buyer, the Developer may forfeit 5% of the price and shall refund the balance "
    "within 30 days. Two open car parking spaces are allotted with the flat. "
    "Disputes shall be referred to arbitration seated in Pune."
)


def _x(term: str, text: str, **attrs: str) -> lx.data.Extraction:
    return lx.data.Extraction(term, text, attributes=attrs)


EXAMPLES = [
    lx.data.ExampleData(
        text=EXAMPLE_TEXT,
        extractions=[
            _x("payment_milestone", "15% on booking", stage="booking", percent="15"),
            _x("payment_milestone", "35% on completion of the plinth", stage="plinth", percent="35"),
            _x("payment_milestone", "50% on possession", stage="possession", percent="50"),
            _x("possession_date", "on or before 30 June 2027", date="30 June 2027"),
            _x("grace_period", "grace period of 3 months", months="3"),
            _x(
                "delay_compensation",
                "the Developer shall pay the Buyer interest at 8% per annum",
                obligation="mandatory",
                rate_percent="8",
            ),
            _x("maintenance_advance", "6 months payable in advance", months="6"),
            _x("forfeiture", "forfeit 5% of the price", percent="5"),
            _x("refund_period", "refund the balance within 30 days", days="30"),
            _x("car_parking", "Two open car parking spaces are allotted", allotted="yes"),
            _x("arbitration_seat", "arbitration seated in Pune", city="Pune"),
        ],
    )
]

# ---------------------------------------------------------------- providers


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


class LangChainExtractor(base_model.BaseLanguageModel):
    """LangExtract provider backed by any LangChain chat model (with its fallbacks)."""

    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    def infer(self, batch_prompts: Sequence[str], **kwargs) -> Iterator[Sequence[types.ScoredOutput]]:
        from .graph import message_text

        for reply in self.llm.batch(list(batch_prompts)):
            yield [types.ScoredOutput(score=1.0, output=_strip_fences(message_text(reply)))]


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("payment_milestone", re.compile(r"(\d+)% on (?:completion of )?(?:the )?([a-z ]+?)(?=,| and |\.)")),
    ("possession_date", re.compile(rf"on or before (\d{{1,2}} (?:{_MONTHS}) \d{{4}})")),
    ("grace_period", re.compile(r"grace period of (\d+) months")),
    (
        "delay_compensation",
        re.compile(
            r"the Promoter (shall|may(?:, at its sole discretion,)?) pay the Allottee "
            r"interest at (\d+(?:\.\d+)?)% per annum"
        ),
    ),
    ("maintenance_advance", re.compile(r"(\d+) months payable in advance")),
    ("forfeiture", re.compile(r"forfeit (\d+)% of the total consideration")),
    ("refund_period", re.compile(r"refund the balance within (\d+) days")),
    ("car_parking", re.compile(r"car parking space is (allotted)")),
    ("arbitration_seat", re.compile(r"arbitration seated in ([A-Z][a-z]+)")),
]


def _rule_attrs(term: str, m: re.Match[str]) -> dict[str, str]:
    g = m.groups()
    if term == "payment_milestone":
        return {"stage": g[1].strip(), "percent": g[0]}
    if term == "possession_date":
        return {"date": g[0]}
    if term == "delay_compensation":
        return {"obligation": "mandatory" if g[0] == "shall" else "discretionary", "rate_percent": g[1]}
    if term == "car_parking":
        return {"allotted": "yes"}
    key = {
        "grace_period": "months",
        "maintenance_advance": "months",
        "forfeiture": "percent",
        "refund_period": "days",
        "arbitration_seat": "city",
    }[term]
    return {key: g[0]}


class RuleBasedExtractor(base_model.BaseLanguageModel):
    """Keyless stand-in: answers LangExtract's prompt with regex matches in the same JSON format."""

    def infer(self, batch_prompts: Sequence[str], **kwargs) -> Iterator[Sequence[types.ScoredOutput]]:
        import json

        for prompt in batch_prompts:
            chunk = prompt.rsplit("\nQ: ", 1)[-1].rsplit("\nA:", 1)[0]
            found = []
            for term, pattern in _RULES:
                for m in pattern.finditer(chunk):
                    found.append((m.start(), {term: m.group(0), f"{term}_attributes": _rule_attrs(term, m)}))
            items = [item for _, item in sorted(found, key=lambda p: p[0])]
            yield [types.ScoredOutput(score=1.0, output=json.dumps({"extractions": items}))]


def extraction_model() -> tuple[base_model.BaseLanguageModel, str, bool]:
    """The agent's live model wrapped for LangExtract, or the rule-based stand-in."""
    from .llm import get_model

    choice = get_model()
    if not choice.live:
        return RuleBasedExtractor(), "Demo mode (rule-based provider)", False
    llm = choice.llm.with_fallbacks(choice.fallbacks) if choice.fallbacks else choice.llm
    return LangChainExtractor(llm), choice.label, True


# ---------------------------------------------------------------- extraction and diff


@dataclass
class Term:
    term: str
    text: str
    start: int | None
    end: int | None
    attrs: dict[str, str]
    clause_key: str | None
    alignment: str

    @property
    def grounded(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def key(self) -> tuple[str, str]:
        ident = IDENTITY_ATTR.get(self.term)
        return self.term, _norm_stage(self.attrs.get(ident, "")) if ident else ""

    def value(self) -> tuple[str, ...]:
        return tuple(_norm(self.attrs.get(a, "")) for a in VALUE_ATTRS.get(self.term, ()))


@dataclass
class TermDiff:
    term: str
    identity: str
    status: str  # same | changed | missing | extra
    template: Term | None = None
    agreement: Term | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.term + (f" ({self.identity})" if self.identity else "")


def _norm(value) -> str:
    v = str(value).strip().lower().rstrip("%").replace(",", "")
    v = re.sub(r"\s*(months?|days?|per cent|percent)$", "", v)
    try:
        return f"{float(v):g}"
    except ValueError:
        return v


def _norm_stage(value: str) -> str:
    v = re.sub(r"\b(completion of|the|works?)\b", " ", str(value).lower())
    return slugify(v)


def _clause_at(text: str, pos: int | None) -> str | None:
    if pos is None:
        return None
    key = None
    for m in _HEADING.finditer(text):
        if m.start() > pos:
            break
        key = slugify(m.group(1))
    return key


def extract_terms(text: str, model: base_model.BaseLanguageModel | None = None) -> list[Term]:
    """Run LangExtract over a document and return its key terms with source spans."""
    model = model or extraction_model()[0]
    result = lx.extract(
        text,
        prompt_description=PROMPT,
        examples=EXAMPLES,
        model=model,
        fence_output=False,
        use_schema_constraints=False,
        max_char_buffer=1500,
        show_progress=False,
    )
    terms = []
    for e in result.extractions or []:
        ci = e.char_interval
        start = ci.start_pos if ci else None
        end = ci.end_pos if ci else None
        terms.append(
            Term(
                term=e.extraction_class,
                text=e.extraction_text,
                start=start,
                end=end,
                attrs={k: str(v) for k, v in (e.attributes or {}).items()},
                clause_key=_clause_at(text, start),
                alignment=e.alignment_status.name if e.alignment_status else "UNGROUNDED",
            )
        )
    return terms


def diff_terms(template: list[Term], agreement: list[Term]) -> list[TermDiff]:
    """Match terms by type (and stage for milestones) and compare their values."""
    t_by = {t.key: t for t in template if t.term in VALUE_ATTRS}
    a_by = {a.key: a for a in agreement if a.term in VALUE_ATTRS}
    diffs = []
    for key in list(t_by) + [k for k in a_by if k not in t_by]:
        t, a = t_by.get(key), a_by.get(key)
        if t and not a:
            status = "missing"
        elif a and not t:
            status = "extra"
        elif t.value() != a.value():
            status = "changed"
        else:
            status = "same"
        notes = []
        if status == "changed":
            for attr, tv, av in zip(VALUE_ATTRS[key[0]], t.value(), a.value(), strict=True):
                if tv != av:
                    notes.append(f"{attr}: {t.attrs.get(attr)} → {a.attrs.get(attr)}")
        diffs.append(TermDiff(key[0], key[1], status, t, a, notes))
    return diffs


def grounding_rate(text: str, terms: list[Term]) -> float:
    """Share of terms whose span reproduces the quoted text exactly (1.0 = every quote verifiable)."""
    if not terms:
        return 0.0
    ok = sum(t.grounded and text[t.start : t.end] == t.text for t in terms)
    return ok / len(terms)


# ---------------------------------------------------------------- rendering

_COLOURS = {"same": "#d9f2e3", "changed": "#ffd9a8", "missing": "#ffc2c2", "extra": "#c9dcff"}


def highlight(text: str, marks: list[tuple[Term, str]]) -> str:
    """HTML of ``text`` with each grounded term wrapped in a <mark> coloured by its diff status."""
    spans = sorted(((t.start, t.end, t, s) for t, s in marks if t and t.grounded), key=lambda x: x[0])
    out, pos = [], 0
    for start, end, term, status in spans:
        if start < pos:  # overlapping span: skip rather than break the HTML
            continue
        out.append(html.escape(text[pos:start]))
        tip = html.escape(f"{term.term} · {status} · chars {start}–{end}")
        out.append(
            f'<mark title="{tip}" style="background:{_COLOURS[status]};padding:0 2px;border-radius:3px">'
            f"{html.escape(text[start:end])}</mark>"
        )
        pos = end
    out.append(html.escape(text[pos:]))
    body = "".join(out)
    return f'<div style="white-space:pre-wrap;font-size:0.9rem;line-height:1.5">{body}</div>'
