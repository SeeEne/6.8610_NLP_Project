"""Unified data models for benchmark tasks and perturbed items."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class BenchmarkTask:
    """A single benchmark task, normalised across HumanEval / MBPP / DS-1000.

    Attributes:
        task_id:            Unique identifier, e.g. "HumanEval/0", "MBPP/601", "DS1000/Pandas/42".
        source:             Origin benchmark — "humaneval", "mbpp", or "ds1000".
        prompt:             The natural-language (+ optional code context) problem statement
                            that gets sent to the model.
        canonical_solution: Reference solution code.
        test_code:          Executable test code. Running `exec(canonical_solution + test_code)`
                            should pass. For MBPP this is assembled from test_list.
        entry_point:        Function name the tests call (HumanEval). None for others.
        library:            DS-1000 only — e.g. "Pandas", "Numpy".
        metadata:           Any extra source-specific fields.
    """

    task_id: str
    source: str
    prompt: str
    canonical_solution: str
    test_code: str
    entry_point: Optional[str] = None
    library: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkTask:
        return cls(**d)


# ── Ambiguity types ──────────────────────────────────────────────────────────

AMBIGUITY_TYPES = [
    "coreferential",
    "syntactic",
    "scopal",
    "collective_distributive",
    "elliptical",
]

RISK_LEVELS = ["high", "low"]


@dataclass
class BenchmarkItem:
    """A fully constructed benchmark item: anchor + perturbation + quality gate.

    This is the final deliverable of Phase 1, consumed by Phase 2 (inference).
    """

    # ── Layer 1: Anchor (from raw data) ──────────────────────────────────
    task_id: str                    # benchmark-specific ID, e.g. "AMBI/001"
    anchor_task_id: str             # original source ID, e.g. "HumanEval/15"
    source: str                     # "humaneval" | "mbpp" | "ds1000"
    prompt: str                     # clean anchor prompt (sent to model for baseline)
    canonical_solution: str         # original reference solution
    test_code: str                  # original test suite
    entry_point: Optional[str] = None   # HumanEval only
    library: Optional[str] = None       # DS-1000 only

    # ── Layer 2: Perturbation (Steps 1.2–1.4) ───────────────────────────
    perturbed_prompt: str = ""      # ambiguous version of the prompt
    ambiguity_type: str = ""        # one of AMBIGUITY_TYPES
    risk_level: str = ""            # "high" | "low"
    interpretation_a: str = ""      # plain-text description of interpretation A
    interpretation_b: str = ""      # plain-text description of interpretation B
    ref_solution_a: str = ""        # code correct under interpretation A
    ref_solution_b: str = ""        # code correct under interpretation B
    test_a: str = ""                # test suite for interpretation A
    test_b: str = ""                # test suite for interpretation B

    # ── Layer 3: Quality Gate (Step 1.5) ─────────────────────────────────
    quality_gate_a: Optional[bool] = None   # Stage A: sandbox exclusivity check
    quality_gate_b: Optional[bool] = None   # Stage B: entropy annotation filter
    quality_gate_b_votes: dict = field(default_factory=dict)  # model -> "A"|"B"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkItem:
        return cls(**d)

    @classmethod
    def from_task(cls, task: BenchmarkTask, task_id: str) -> BenchmarkItem:
        """Create a BenchmarkItem from a BenchmarkTask, pre-filling anchor fields."""
        return cls(
            task_id=task_id,
            anchor_task_id=task.task_id,
            source=task.source,
            prompt=task.prompt,
            canonical_solution=task.canonical_solution,
            test_code=task.test_code,
            entry_point=task.entry_point,
            library=task.library,
        )
