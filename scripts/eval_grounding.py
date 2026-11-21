"""Score grounded extraction against the planted deviations, with whichever model is configured.

    python scripts/eval_grounding.py            # live model if a key is set, else demo provider
    python scripts/eval_grounding.py --demo     # force the rule-based provider

Prints, per agreement: terms extracted, grounding rate (share of quotes that match the
source span character for character), and deviations found vs planted. Totals give
precision = true flags / all flags and recall = true flags / planted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.documents import DATA_DIR  # noqa: E402
from agent.grounding import RuleBasedExtractor, diff_terms, extract_terms, extraction_model, grounding_rate  # noqa: E402
from tests.test_grounding import PLANTED  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use the rule-based provider")
    args = parser.parse_args()
    model, label = (RuleBasedExtractor(), "Demo (rule-based)") if args.demo else extraction_model()[:2]
    print(f"Model: {label}\n")

    template_text = (DATA_DIR / "template.md").read_text(encoding="utf-8")
    template_terms = extract_terms(template_text, model)
    print(f"template  terms={len(template_terms):2d}  grounding={grounding_rate(template_text, template_terms):.0%}")

    tp = fp = planted = 0
    for doc_id, expected in PLANTED.items():
        text = (DATA_DIR / "agreements" / f"{doc_id}.md").read_text(encoding="utf-8")
        terms = extract_terms(text, model)
        flagged = {(d.term, d.identity) for d in diff_terms(template_terms, terms) if d.status != "same"}
        hit, extra, miss = flagged & expected, flagged - expected, expected - flagged
        tp, fp, planted = tp + len(hit), fp + len(extra), planted + len(expected)
        print(
            f"{doc_id}     terms={len(terms):2d}  grounding={grounding_rate(text, terms):.0%}  "
            f"found {len(hit)}/{len(expected)}"
            + (f"  missed={sorted(miss)}" if miss else "")
            + (f"  false={sorted(extra)}" if extra else "")
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    print(f"\nprecision = {tp}/{tp + fp} = {precision:.0%}   recall = {tp}/{planted} = {tp / planted:.0%}")


if __name__ == "__main__":
    main()
