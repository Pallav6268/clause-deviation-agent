"""Load agreements written as Markdown and split them into clauses.

Each clause is a level-2 heading such as ``## 5. Possession``. The clause key is
the heading title in snake_case (``possession``), so the same clause can be
looked up in the template and in any agreement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEMPLATE_ID = "template"

_HEADING = re.compile(r"^##\s+(?:\d+\.\s*)?(.+?)\s*$", re.MULTILINE)


@dataclass
class Clause:
    key: str
    title: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    clauses: dict[str, Clause] = field(default_factory=dict)


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def parse_markdown(doc_id: str, text: str) -> Document:
    """Split Markdown text into clauses keyed by their heading."""
    first_line = text.strip().splitlines()[0] if text.strip() else doc_id
    title = first_line.lstrip("# ").strip() or doc_id
    doc = Document(doc_id=doc_id, title=title)
    matches = list(_HEADING.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        clause_title = match.group(1).strip()
        key = slugify(clause_title)
        doc.clauses[key] = Clause(key=key, title=clause_title, text=body)
    return doc


class DocumentStore:
    """In-memory registry of the template and the agreements."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def add(self, doc_id: str, text: str) -> Document:
        doc = parse_markdown(doc_id, text)
        self._docs[doc_id] = doc
        return doc

    def get(self, doc_id: str) -> Document:
        if doc_id not in self._docs:
            raise KeyError(f"Unknown document '{doc_id}'. Known: {sorted(self._docs)}")
        return self._docs[doc_id]

    def ids(self) -> list[str]:
        return sorted(self._docs)

    def agreement_ids(self) -> list[str]:
        return [d for d in self.ids() if d != TEMPLATE_ID and not d.startswith("custom-")]


def load_default_store(data_dir: Path = DATA_DIR) -> DocumentStore:
    store = DocumentStore()
    store.add(TEMPLATE_ID, (data_dir / "template.md").read_text(encoding="utf-8"))
    for path in sorted((data_dir / "agreements").glob("*.md")):
        store.add(path.stem, path.read_text(encoding="utf-8"))
    return store


STORE = load_default_store()
