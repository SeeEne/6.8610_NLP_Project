"""Shared constants and data loading for the perturbation pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.anchor_selection import AnchorResult


# ── Ambiguity type definitions ──────────────────────────────────────────────

AMBIGUITY_TYPE_DEFS = {
    "coreferential": (
        "COREFERENTIAL ambiguity: A pronoun or definite noun phrase ('it', 'them', "
        "'the result') whose referent is grammatically ambiguous across two plausible "
        "antecedents. The same pronoun could bind to either of two previously mentioned "
        "entities, yielding logically distinct operations."
    ),
    "syntactic": (
        "SYNTACTIC ambiguity: A modifier phrase — typically a prepositional phrase, "
        "relative clause, or adjunct — that can syntactically attach to two different "
        "constituents, yielding distinct logical scopes. The attachment site determines "
        "which noun or verb phrase the modifier constrains."
    ),
    "scopal": (
        "SCOPAL ambiguity: A quantifier, distributive operator, or numerical expression "
        "whose scope ordering relative to another operator is underdetermined. Wide-scope "
        "and narrow-scope readings produce distinct execution semantics."
    ),
    "collective_distributive": (
        "COLLECTIVE/DISTRIBUTIVE ambiguity: An operation described over a set where it is "
        "unclear whether it applies jointly to the set as a whole (collective) or "
        "individually to each member (distributive). The two readings produce different "
        "data flow and partitioning semantics."
    ),
    "elliptical": (
        "ELLIPTICAL ambiguity: A verb phrase or predicate is omitted (elided) and must be "
        "recovered by the reader. The elided content is inferable under one reading but "
        "absent under another, producing two valid completions with different operational "
        "semantics."
    ),
}


# ── Anchor loading & selection ──────────────────────────────────────────────

def load_anchor_results(path: str | Path) -> list[AnchorResult]:
    """Load anchor results from JSONL."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(AnchorResult.from_dict(json.loads(line)))
    return results


def select_anchors(
    anchor_results: list[AnchorResult],
    min_feasibility: float = 2.0,
    risk_level: str = "low",
    max_tasks: int = 100,
) -> list[AnchorResult]:
    """Select and sort anchors for perturbation.

    Filters by risk level and minimum feasibility, then sorts by
    feasibility_score descending (best candidates first).
    """
    filtered = [
        r for r in anchor_results
        if r.risk_level == risk_level
        and r.feasibility_score >= min_feasibility
        and r.best_ambiguity_type
    ]
    filtered.sort(key=lambda r: r.feasibility_score, reverse=True)
    return filtered[:max_tasks]
