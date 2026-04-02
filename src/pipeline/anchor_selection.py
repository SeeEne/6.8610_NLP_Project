"""Phase 1 Pipeline: Anchor Selection & Ambiguity Injection Scoring.

For each anchor task, N judges (randomly sampled from a model pool) each make
ONE combined API call that evaluates ambiguity potential, risk level, and
feasibility in a single response.

All scoring uses structured boolean rubrics for cross-model consistency.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.data.model import BenchmarkTask
from src.pipeline.prompts import get_prompt, load_pipeline_config, render_prompt
from src.util.llm import LLMClient, ModelConfig


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class JudgeVote:
    """One judge's combined evaluation (ambiguity + risk + feasibility)."""
    model: str

    # Ambiguity — per-type rubric results
    coreferential: dict = field(default_factory=dict)
    syntactic: dict = field(default_factory=dict)
    scopal: dict = field(default_factory=dict)
    collective_distributive: dict = field(default_factory=dict)
    elliptical: dict = field(default_factory=dict)

    # Risk — 4 boolean questions
    q1_irreversibility: Optional[bool] = None
    q2_external_state: Optional[bool] = None
    q3_security_sensitivity: Optional[bool] = None
    q4_data_integrity: Optional[bool] = None
    risk_level: str = ""
    risk_rationale: str = ""

    # Feasibility — 5 boolean dimensions
    d1_multi_entity: Optional[bool] = None
    d2_structural_complexity: Optional[bool] = None
    d3_testability: Optional[bool] = None
    d4_natural_perturbation: Optional[bool] = None
    d5_interpretation_divergence: Optional[bool] = None
    feasibility_score: int = 0
    feasibility_rationale: str = ""

    raw_response: str = ""
    error: Optional[str] = None


@dataclass
class AnchorResult:
    """Aggregated result for one anchor task.

    Only stores task_id and source as identifiers — prompt, canonical_solution,
    and test_code can be looked up from the raw data via task_id.
    """
    task_id: str
    source: str
    entry_point: Optional[str] = None
    library: Optional[str] = None

    # Raw votes (raw_response stripped for size)
    votes: list[dict] = field(default_factory=list)

    # Computed aggregates — ambiguity
    ambiguity_scores: dict = field(default_factory=dict)    # type -> avg score (0–3)
    best_ambiguity_type: str = ""
    perturbation_sketches: dict = field(default_factory=dict)  # type -> list of sketches

    # Computed aggregates — risk
    risk_level: str = ""              # majority vote
    risk_agreement: float = 0.0       # fraction of judges agreeing

    # Computed aggregates — feasibility
    feasibility_score: float = 0.0    # avg across judges (0–5)
    feasibility_dimensions: dict = field(default_factory=dict)  # dimension -> agreement ratio

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AnchorResult:
        return cls(**d)


# ── Constants ────────────────────────────────────────────────────────────────

AMBIGUITY_TYPES = [
    "coreferential", "syntactic", "scopal",
    "collective_distributive", "elliptical",
]

AMBIGUITY_QUESTIONS = ["q1_structural_fit", "q2_natural_phrasing", "q3_code_divergence"]

FEASIBILITY_DIMENSIONS = [
    "d1_multi_entity", "d2_structural_complexity", "d3_testability",
    "d4_natural_perturbation", "d5_interpretation_divergence",
]


# ── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, handling common issues."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    # Strip markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Find the outermost { ... }
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:100]}")

    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    text = text[start:end]

    # Try as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: single quotes → double quotes
    return json.loads(text.replace("'", '"'))


def _sample_judges(config: dict) -> list[str]:
    """Randomly sample judge models from the pool."""
    pool = config["anchor_selection"]["judge_models"]
    k = config["anchor_selection"]["judges_per_task"]
    return random.sample(pool, min(k, len(pool)))


def _score_ambiguity_type(type_data: dict) -> int:
    """Count true answers across the 3 boolean questions for one ambiguity type."""
    return sum(1 for q in AMBIGUITY_QUESTIONS if type_data.get(q) is True)


# ── Single combined LLM call ────────────────────────────────────────────────

def evaluate_single_judge(
    client: LLMClient,
    task: BenchmarkTask,
    model_alias: str,
    config: dict,
) -> JudgeVote:
    """One judge evaluates ambiguity + risk + feasibility in a single API call."""
    ambiguity_defs = get_prompt("anchor_selection.ambiguity_definitions")
    system = get_prompt("anchor_selection.combined_evaluation.system")
    prompt = render_prompt(
        "anchor_selection.combined_evaluation.task",
        prompt=task.prompt,
        canonical_solution=task.canonical_solution,
        test_code=task.test_code,
        ambiguity_definitions=ambiguity_defs,
    )

    mc = ModelConfig(
        model=model_alias,
        temperature=config["anchor_selection"]["temperature"],
        max_tokens=config["anchor_selection"]["max_tokens"],
    )

    raw = ""
    try:
        resp = client.call(mc, prompt=prompt, system=system)
        raw = resp.choices[0]
        parsed = _parse_json_response(raw)

        # Extract the 3 sections
        amb = parsed.get("ambiguity", {})
        risk = parsed.get("risk", {})
        feas = parsed.get("feasibility", {})

        # Compute feasibility score from dimensions (don't trust LLM's sum)
        computed_feas = sum(
            1 for d in FEASIBILITY_DIMENSIONS if feas.get(d) is True
        )

        return JudgeVote(
            model=model_alias,
            # Ambiguity
            coreferential=amb.get("coreferential", {}),
            syntactic=amb.get("syntactic", {}),
            scopal=amb.get("scopal", {}),
            collective_distributive=amb.get("collective_distributive", {}),
            elliptical=amb.get("elliptical", {}),
            # Risk
            q1_irreversibility=risk.get("q1_irreversibility"),
            q2_external_state=risk.get("q2_external_state"),
            q3_security_sensitivity=risk.get("q3_security_sensitivity"),
            q4_data_integrity=risk.get("q4_data_integrity"),
            risk_level=risk.get("risk_level", ""),
            risk_rationale=risk.get("rationale", ""),
            # Feasibility
            d1_multi_entity=feas.get("d1_multi_entity"),
            d2_structural_complexity=feas.get("d2_structural_complexity"),
            d3_testability=feas.get("d3_testability"),
            d4_natural_perturbation=feas.get("d4_natural_perturbation"),
            d5_interpretation_divergence=feas.get("d5_interpretation_divergence"),
            feasibility_score=computed_feas,
            feasibility_rationale=feas.get("rationale", ""),
            raw_response=raw,
        )
    except Exception as e:
        error_msg = str(e)
        if raw:
            error_msg += f"\n    RAW_RESPONSE: {raw[:300]}"
        return JudgeVote(model=model_alias, raw_response=raw, error=error_msg)


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_result(
    task: BenchmarkTask,
    votes: list[JudgeVote],
) -> AnchorResult:
    """Aggregate judge votes into a single AnchorResult."""
    valid = [v for v in votes if not v.error]

    # ── Ambiguity: avg score per type (0–3)
    ambiguity_scores = {}
    perturbation_sketches = {}
    for atype in AMBIGUITY_TYPES:
        type_scores = []
        sketches = []
        for v in valid:
            type_data = getattr(v, atype, {})
            if isinstance(type_data, dict):
                type_scores.append(_score_ambiguity_type(type_data))
                sketch = type_data.get("perturbation_sketch", "")
                if sketch:
                    sketches.append(sketch)
        ambiguity_scores[atype] = round(
            sum(type_scores) / len(type_scores), 2
        ) if type_scores else 0.0
        if sketches:
            perturbation_sketches[atype] = sketches

    best_type = max(ambiguity_scores, key=ambiguity_scores.get) if ambiguity_scores else ""

    # ── Risk: majority vote
    risk_labels = [v.risk_level for v in valid if v.risk_level]
    if risk_labels:
        risk_level = max(set(risk_labels), key=risk_labels.count)
        risk_agreement = risk_labels.count(risk_level) / len(risk_labels)
    else:
        risk_level = "low"
        risk_agreement = 0.0

    # ── Feasibility: avg score + per-dimension agreement
    fscores = [v.feasibility_score for v in valid]
    feasibility_avg = round(
        sum(fscores) / len(fscores), 2
    ) if fscores else 0.0

    feasibility_dimensions = {}
    for dim in FEASIBILITY_DIMENSIONS:
        vals = [getattr(v, dim) for v in valid if getattr(v, dim) is not None]
        feasibility_dimensions[dim] = round(
            sum(1 for x in vals if x is True) / len(vals), 2
        ) if vals else 0.0

    # Strip raw_response from votes for output
    clean_votes = [
        {k: v for k, v in asdict(vote).items() if k != "raw_response"}
        for vote in votes
    ]

    return AnchorResult(
        task_id=task.task_id,
        source=task.source,
        entry_point=task.entry_point,
        library=task.library,
        votes=clean_votes,
        ambiguity_scores=ambiguity_scores,
        best_ambiguity_type=best_type,
        perturbation_sketches=perturbation_sketches,
        risk_level=risk_level,
        risk_agreement=risk_agreement,
        feasibility_score=feasibility_avg,
        feasibility_dimensions=feasibility_dimensions,
    )


# ── Pipeline runner ──────────────────────────────────────────────────────────

def evaluate_anchor(
    client: LLMClient,
    task: BenchmarkTask,
    config: dict,
    judges: list[str] | None = None,
) -> AnchorResult:
    """Run combined evaluation on a single anchor task.

    Each judge makes ONE API call covering ambiguity + risk + feasibility.
    All judge calls fire concurrently.
    """
    if judges is None:
        judges = _sample_judges(config)

    vote_results: dict[str, JudgeVote] = {}

    def _call_judge(model):
        vote_results[model] = evaluate_single_judge(client, task, model, config)

    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        for model_alias in judges:
            pool.submit(_call_judge, model_alias)

    votes = [vote_results[m] for m in judges]
    return aggregate_result(task, votes)


def run_anchor_selection(
    tasks: list[BenchmarkTask],
    config: dict | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
    debug: bool = False,
) -> list[AnchorResult]:
    """Run the full anchor selection pipeline on a list of tasks.

    Each task is evaluated in a separate thread (question-level parallelism).
    Results are written to JSONL incrementally in completion order.
    """
    random.seed(seed)

    if config is None:
        config = load_pipeline_config()

    if output_dir is None:
        output_dir = Path(config["anchor_selection"]["output_dir"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_workers = config["anchor_selection"].get("max_workers", 6)
    judges_per_task = config["anchor_selection"]["judges_per_task"]
    client = LLMClient()

    # Pre-assign judges for each task (deterministic with seed)
    task_judges = [_sample_judges(config) for _ in tasks]

    # Thread-safe write and progress tracking
    write_lock = threading.Lock()
    completed_count = [0]
    total_errors = [0]
    error_counts_by_model: dict[str, int] = {}
    results_by_index: dict[int, AnchorResult] = {}
    pipeline_start = time.time()

    output_path = output_dir / "anchor_results.jsonl"
    debug_path = output_dir / "errors.log"
    f = open(output_path, "w", encoding="utf-8")
    debug_f = open(debug_path, "w", encoding="utf-8") if debug else None

    def _log_errors(task_id: str, result: AnchorResult) -> int:
        """Log errors and return error count."""
        error_entries = []
        for v in result.votes:
            if v.get("error"):
                model = v["model"]
                error_msg = v["error"]
                error_entries.append((model, error_msg))
                error_counts_by_model[model] = error_counts_by_model.get(model, 0) + 1

        if error_entries and debug_f:
            debug_f.write(f"\n{'='*60}\n")
            debug_f.write(f"TASK: {task_id}\n")
            debug_f.write(f"{'='*60}\n")
            for model, error_msg in error_entries:
                debug_f.write(f"\n  {model}:\n")
                debug_f.write(f"    {error_msg}\n")
            debug_f.flush()

        return len(error_entries)

    def _process_task(index: int, task: BenchmarkTask) -> None:
        judges = task_judges[index]
        task_start = time.time()
        result = evaluate_anchor(client, task, config, judges=judges)
        task_elapsed = time.time() - task_start

        with write_lock:
            completed_count[0] += 1
            results_by_index[index] = result
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            f.flush()

            errors = _log_errors(task.task_id, result)
            total_errors[0] += errors

            total_elapsed = time.time() - pipeline_start
            avg_per_task = total_elapsed / completed_count[0]
            remaining = len(tasks) - completed_count[0]
            eta_s = avg_per_task * remaining

            error_str = f"  errors={errors}" if errors else ""

            print(
                f"[{completed_count[0]}/{len(tasks)}] {task.task_id:25s}  "
                f"best={result.best_ambiguity_type:22s}  "
                f"risk={result.risk_level:4s}  "
                f"feas={result.feasibility_score}/5  "
                f"({task_elapsed:.1f}s){error_str}  "
                f"ETA {eta_s/60:.1f}min"
            )

            if debug and errors:
                for v in result.votes:
                    if v.get("error"):
                        print(f"    └─ {v['model']}: {v['error'][:120]}")

    print(f"Evaluating {len(tasks)} tasks with {max_workers} workers "
          f"({judges_per_task} judges per task, 1 API call each)...")
    if debug:
        print(f"Debug mode ON — errors logged to {debug_path}")
    print()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_task, i, task): i
            for i, task in enumerate(tasks)
        }
        for future in futures:
            future.result()

    f.close()
    if debug_f:
        debug_f.close()

    total_time = time.time() - pipeline_start
    total_calls = len(tasks) * judges_per_task
    print(f"\nCompleted in {total_time:.1f}s ({total_time/len(tasks):.1f}s/task avg)")
    print(f"Total API calls: {total_calls}  Errors: {total_errors[0]} ({total_errors[0]/total_calls*100:.1f}%)")

    if total_errors[0] > 0:
        print(f"\nErrors by model:")
        for model, count in sorted(error_counts_by_model.items(), key=lambda x: -x[1]):
            print(f"  {model:25s}: {count}")
        if debug:
            print(f"\nFull error details: {debug_path}")

    # Return results in original task order
    results = [results_by_index[i] for i in range(len(tasks))]

    # Rewrite file in original order (completion order may differ)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_path}")
    return results
