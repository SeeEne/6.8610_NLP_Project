# Data Pipeline Guide

This guide covers how to work with the benchmark data for AmbiCode-Eval.

## Setup

```bash
pip install -e ".[dev]"
```

## Step 1.1 — Download Raw Anchors

Download all benchmark datasets (HumanEval, MBPP, DS-1000) into `data/raw/`:

```bash
python scripts/download_data.py
```

Or download specific sources:

```bash
python scripts/download_data.py humaneval mbpp
```

This creates JSONL files in `data/raw/` — one JSON object per line, all sharing the same schema.

## Working with the Data

```python
from src.data import BenchmarkStore, BenchmarkTask

# Load from local files (no network needed after download)
store = BenchmarkStore.load_local("data/raw")

# Browse by source
humaneval_tasks = store.filter(source="humaneval")   # 164 tasks
mbpp_tasks = store.filter(source="mbpp")             # 257 tasks
ds1000_tasks = store.filter(source="ds1000")         # 1000 tasks

# Filter DS-1000 by library
pandas_tasks = store.filter(library="Pandas")

# Get a specific task by ID
task = store.get("HumanEval/0")
```

## BenchmarkTask Schema

Every task, regardless of source, has these fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Unique ID: `HumanEval/0`, `MBPP/11`, `DS1000/Pandas/0` |
| `source` | str | `"humaneval"`, `"mbpp"`, or `"ds1000"` |
| `prompt` | str | The problem statement — this is the **clean anchor** |
| `canonical_solution` | str | Reference solution code |
| `test_code` | str | Executable unit tests for the canonical solution |
| `entry_point` | str? | Function name (HumanEval only) |
| `library` | str? | Python library (DS-1000 only, e.g. `"Pandas"`) |
| `metadata` | dict | Source-specific extras |

## Steps 1.2–1.5: Building Benchmark Items

These steps are done per-task by the team. For each of the 50 selected anchors:

### Step 1.2 — Ambiguity Injection

Take the `prompt` (clean anchor) and manually perturb it to inject one of the 5 ambiguity types:
- Coreferential, Syntactic (Attachment), Scopal, Collective/Distributive, Elliptical

The perturbed prompt should have exactly **two valid interpretations** (A and B).

### Step 1.3 — Reference Solution Authoring

Write two reference solutions:
- **Reference A** — correct under interpretation A
- **Reference B** — correct under interpretation B

### Step 1.4 — Test Authoring (Test-A & Test-B)

Write two mutually exclusive test suites:
- **Test-A** — passes for interpretation A, fails for interpretation B
- **Test-B** — passes for interpretation B, fails for interpretation A

Both test suites should be **incremental** — they test beyond what the original `test_code` covers.

### Step 1.5 — Quality Gate

**Stage A (Mechanical Sandbox Check):**

```python
from src.util import Sandbox

sandbox = Sandbox()
passed = sandbox.validate_quality_gate_a(
    ref_solution_a=ref_a_code,
    ref_solution_b=ref_b_code,
    test_a=test_a_code,
    test_b=test_b_code,
)
# Must be True — strict exclusivity
```

**Stage B (LLM Forced-Choice Annotation):**
Five models from distinct families are given the perturbed prompt and asked to choose between interpretation A and B. Items where models converge too strongly (low entropy) are rejected as insufficiently ambiguous.

## Saving Perturbed Items

Store your perturbed benchmark items in `data/benchmark/` as JSONL. Suggested schema (extends BenchmarkTask):

```json
{
  "task_id": "HumanEval/0",
  "source": "humaneval",
  "prompt": "... original clean prompt ...",
  "canonical_solution": "... original solution ...",
  "test_code": "... original tests ...",
  "entry_point": "has_close_elements",
  "perturbed_prompt": "... ambiguous version ...",
  "ambiguity_type": "scopal",
  "risk_level": "low",
  "ref_solution_a": "... code for interpretation A ...",
  "ref_solution_b": "... code for interpretation B ...",
  "test_a": "... tests for interpretation A ...",
  "test_b": "... tests for interpretation B ...",
  "quality_gate_a_passed": true,
  "quality_gate_b_passed": true
}
```
