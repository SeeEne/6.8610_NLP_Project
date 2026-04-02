# Data Pipeline Guide

This guide covers how to work with the benchmark data for AmbiCode-Eval.

---

## Part A: Raw Data Collection (Step 1.1)

### Setup

```bash
pip install -e ".[dev]"
```

### Download Raw Anchors

```bash
python scripts/download_data.py                    # download all
python scripts/download_data.py humaneval mbpp      # specific sources only
python scripts/download_data.py --output data/raw   # custom output directory
```

Creates JSONL files in `data/raw/`:

| File | Tasks | Description |
|---|---|---|
| `humaneval.jsonl` | 164 | Function-level Python tasks with docstrings |
| `mbpp.jsonl` | 257 | Short NL descriptions + solutions (sanitized subset) |
| `ds1000.jsonl` | 1000 | Data science tasks across 7 libraries |

### Working with Raw Data

```python
from src.data import BenchmarkStore

store = BenchmarkStore.load_local("data/raw")

humaneval = store.filter(source="humaneval")
mbpp = store.filter(source="mbpp")
pandas = store.filter(library="Pandas")

task = store.get("HumanEval/15")
print(task.prompt)               # the question
print(task.canonical_solution)   # the answer
print(task.test_code)            # unit tests
```

---

## Part B: Anchor Selection Pipeline (Step 1.1b)

Before manually building benchmark items, run the automated anchor selection pipeline to score and rank all candidate tasks. This helps the team select the best 50 anchors by evaluating **ambiguity potential**, **risk level**, and **feasibility**.

### How It Works

For each task, 3 randomly selected judge models (from an 8-model pool) independently evaluate:

| Evaluation | Method | Output |
|---|---|---|
| **Ambiguity Classification** | 3 boolean questions per ambiguity type (structural fit, natural phrasing, code divergence) | Score 0–3 per type |
| **Risk Assessment** | 4 boolean questions (irreversibility, external state, security, data integrity). Any true → high risk | `"high"` or `"low"` |
| **Feasibility Scoring** | 5 boolean dimensions (multi-entity, structural complexity, testability, natural perturbation, interpretation divergence) | Score 0–5 |

All scoring uses **structured boolean rubrics** — judges answer yes/no questions rather than subjective scales, ensuring consistency across different LLMs.

### Configuration

Judge models and parameters are configured in `config/pipeline.yaml`:

```yaml
anchor_selection:
  judge_models:          # 3 randomly sampled per task
    - gpt-5.4
    - claude-sonnet
    - gemini-3.1-pro
    - deepseek-v3.2
    - qwen-3.5
    - minimax-m2.7
    - grok-4.20
    - glm-5-turbo
  judges_per_task: 3
  temperature: 0.0
  max_tokens: 1024
  max_workers: 6         # concurrent tasks
```

All prompts (system + task) are in `config/prompts.yaml`.
Model aliases resolve to OpenRouter IDs via `config/models.yaml`.

### Running

```bash
# Run on all configured sources (humaneval + mbpp + ds1000)
python scripts/run_anchor_selection.py

# Run on one source
python scripts/run_anchor_selection.py --source humaneval

# Run on first N tasks
python scripts/run_anchor_selection.py --source humaneval --limit 20

# Run on specific tasks
python scripts/run_anchor_selection.py --task-ids HumanEval/74 MBPP/11
```

Each task makes 9 concurrent API calls (3 judges × 3 evaluations). Tasks are processed in parallel (default 6 workers). Progress output shows per-task timing and ETA:

```
[12/164] HumanEval/74           best=coreferential        risk=low   feas=4.3/5  (4.2s)  ETA 2.1min
```

### Output

Results are saved incrementally to `data/intermediate/anchor_selection/anchor_results.jsonl` — one JSON line per task with:

```jsonc
{
  "task_id": "HumanEval/74",
  "source": "humaneval",
  "prompt": "...",
  "canonical_solution": "...",
  "test_code": "...",

  // Aggregated scores
  "ambiguity_scores": {          // avg boolean score (0–3) per type
    "coreferential": 3.0,
    "syntactic": 1.5,
    "scopal": 1.5,
    "collective_distributive": 3.0,
    "elliptical": 1.5
  },
  "best_ambiguity_type": "coreferential",
  "perturbation_sketches": {     // model-generated perturbation ideas
    "coreferential": ["Reword to '...returns it...' where 'it' could refer to..."]
  },

  "risk_level": "low",
  "risk_agreement": 1.0,        // fraction of judges agreeing

  "feasibility_score": 4.33,    // avg across judges (0–5)
  "feasibility_dimensions": {   // per-dimension agreement ratio
    "d1_multi_entity": 1.0,
    "d2_structural_complexity": 0.5,
    "d3_testability": 1.0,
    "d4_natural_perturbation": 1.0,
    "d5_interpretation_divergence": 0.5
  },

  // Raw votes from each judge (for auditing)
  "ambiguity_votes": [...],
  "risk_votes": [...],
  "feasibility_votes": [...]
}
```

### Selecting Anchors

After running the pipeline, analyze the results to select 50 anchors:

1. **Filter** by `feasibility_score >= 3` (good candidates)
2. **Balance** across `best_ambiguity_type` (target 10 per type)
3. **Mix** `risk_level` (high and low within each type)
4. **Review** `perturbation_sketches` — these are LLM-suggested starting points for the manual perturbation step

The summary printed at the end of the script shows the top candidates, ambiguity type distribution, and risk distribution.

---

## Part C: Benchmark Construction (Steps 1.2–1.5)

Team members select 50 anchors from the raw data and build perturbed benchmark items. The final output is `data/benchmark/benchmark.jsonl`.

### BenchmarkItem Schema

Each line in `benchmark.jsonl` is a JSON object with 20 fields in 3 layers:

#### Layer 1 — Anchor (from raw data)

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Benchmark ID: `"AMBI/001"` |
| `anchor_task_id` | str | Original source ID: `"HumanEval/15"` |
| `source` | str | `"humaneval"` \| `"mbpp"` \| `"ds1000"` |
| `prompt` | str | Clean anchor prompt — **baseline** for pass@k |
| `canonical_solution` | str | Original reference solution |
| `test_code` | str | Original test suite |
| `entry_point` | str? | Function name (HumanEval only) |
| `library` | str? | Python library (DS-1000 only) |

#### Layer 2 — Perturbation (Steps 1.2–1.4)

| Field | Type | Description |
|---|---|---|
| `perturbed_prompt` | str | Ambiguous version of the prompt |
| `ambiguity_type` | str | `"coreferential"` \| `"syntactic"` \| `"scopal"` \| `"collective_distributive"` \| `"elliptical"` |
| `risk_level` | str | `"high"` \| `"low"` |
| `interpretation_a` | str | Plain-text description of interpretation A |
| `interpretation_b` | str | Plain-text description of interpretation B |
| `ref_solution_a` | str | Code correct under interpretation A |
| `ref_solution_b` | str | Code correct under interpretation B |
| `test_a` | str | Test suite for interpretation A (exclusive with test_b) |
| `test_b` | str | Test suite for interpretation B (exclusive with test_a) |

#### Layer 3 — Quality Gate (Step 1.5)

| Field | Type | Description |
|---|---|---|
| `quality_gate_a` | bool | Stage A passed: sandbox exclusivity verified |
| `quality_gate_b` | bool | Stage B passed: entropy annotation filter |
| `quality_gate_b_votes` | dict | Model forced-choice votes: `{"gpt-5.4": "A", ...}` |

### Creating a BenchmarkItem from Code

```python
from src.data.model import BenchmarkItem
from src.data import BenchmarkStore

store = BenchmarkStore.load_local("data/raw")
task = store.get("HumanEval/15")

# Pre-fills all Layer 1 fields from the anchor
item = BenchmarkItem.from_task(task, task_id="AMBI/001")

# Fill in Layer 2 (your perturbation work)
item.perturbed_prompt = "..."
item.ambiguity_type = "elliptical"
item.risk_level = "low"
item.interpretation_a = "..."
item.interpretation_b = "..."
item.ref_solution_a = "..."
item.ref_solution_b = "..."
item.test_a = "..."
item.test_b = "..."

# Fill in Layer 3 (after running quality gate)
item.quality_gate_a = True
item.quality_gate_b = True
item.quality_gate_b_votes = {"gpt-5.4": "A", "claude-sonnet": "B", ...}
```

### Running Quality Gate Stage A

```python
from src.util import Sandbox

sandbox = Sandbox()
passed = sandbox.validate_quality_gate_a(
    ref_solution_a=item.ref_solution_a,
    ref_solution_b=item.ref_solution_b,
    test_a=item.test_a,
    test_b=item.test_b,
)
# True = strict exclusivity: A passes A, fails B; B passes B, fails A
```

### Saving / Loading Benchmark Items

```python
import json
from src.data.model import BenchmarkItem

# Save
with open("data/benchmark/benchmark.jsonl", "a") as f:
    f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

# Load all items
items = []
with open("data/benchmark/benchmark.jsonl") as f:
    for line in f:
        items.append(BenchmarkItem.from_dict(json.loads(line)))
```

---

## Part C: How Phase 2 Consumes the Benchmark

Phase 2 (inference + evaluation) reads `data/benchmark/benchmark.jsonl` and for each item:

### 1. Baseline (clean prompt)

Send `item.prompt` to each model → execute responses against `item.test_code` → compute **pass@k(Baseline)**.

### 2. Perturbed (ambiguous prompt)

Send `item.perturbed_prompt` to each model → classify response as **SA / EA / AC** → for SA/EA responses, run dual-blind execution:

```python
from src.util import Sandbox

sandbox = Sandbox()
result_a, result_b = sandbox.run_dual_blind(
    code=model_generated_code,
    test_a=item.test_a,
    test_b=item.test_b,
)
# result_a.passed → pass@k(A)
# result_b.passed → pass@k(B)
```

### 3. Metrics

- **Ambiguity Tax** = pass@k(Baseline) − pass@k(A)
- **Inductive bias** = asymmetry between pass@k(A) and pass@k(B)
- **Behavioral distribution** = SA% / EA% / AC% per model, stratified by ambiguity type and risk level

---

## Target Distribution

The final benchmark should contain **50 items**:

| Ambiguity Type | Count | Risk Levels |
|---|---|---|
| Coreferential | 10 | Mix of high/low |
| Syntactic (Attachment) | 10 | Mix of high/low |
| Scopal | 10 | Mix of high/low |
| Collective/Distributive | 10 | Mix of high/low |
| Elliptical | 10 | Mix of high/low |

All 50 must pass both Quality Gate Stage A and Stage B.
