# Data Pipeline Guide

This guide covers the full Phase 1 data pipeline for AmbiCode-Eval.

---

## Prerequisites

```bash
# Use the base Anaconda environment
conda activate base

# Install project
pip install -e ".[dev]"

# Ensure .env has OPENROUTER_API_KEY
cp .env.example .env  # then edit with your key

# Docker Desktop must be running for sandbox execution (Step 1.5)
```

---

## Step 1.1 — Download Raw Anchors

Download benchmark datasets from HuggingFace into `data/raw/`:

```bash
python scripts/download_data.py                    # download all
python scripts/download_data.py humaneval mbpp      # specific sources only
```

Creates:

| File | Tasks | Description |
|---|---|---|
| `humaneval.jsonl` | 164 | Function-level Python tasks with docstrings |
| `mbpp.jsonl` | 257 | Short NL descriptions + solutions (sanitized subset) |
| `ds1000.jsonl` | 1000 | Data science tasks across 7 libraries |

### Working with Raw Data

```python
from src.data import BenchmarkStore

store = BenchmarkStore.load_local("data/raw")
task = store.get("HumanEval/74")
print(task.prompt)               # the question
print(task.canonical_solution)   # the answer
print(task.test_code)            # unit tests
```

---

## Step 1.1a — DS-1000 Normalization

DS-1000 tasks use a harness format (`exec_context` with `[insert]` placeholder) incompatible with simple `exec(code + test)`. The normalization step converts them to a concatenation-friendly format.

```bash
python scripts/normalize_ds1000.py
python scripts/normalize_ds1000.py --verify 10   # verify in Docker sandbox
```

Creates `data/raw/ds1000_normalized.jsonl` (845 tasks, Matplotlib excluded).

**What changes:**
- `canonical_solution`: original code fragment wrapped as `__SOLUTION__ = r"""..."""`
- `test_code`: original harness preserved verbatim + `test_execution(__SOLUTION__)` appended
- `metadata`: adds `original_solution` and `exec_context_imports`

**What stays the same:**
- `prompt`: unchanged (downstream sees original wording)
- `task_id`, `source`, `library`: unchanged

When both `ds1000.jsonl` and `ds1000_normalized.jsonl` exist in `data/raw/`, `BenchmarkStore.load_local()` automatically prefers the normalized version.

---

## Step 1.1b — Anchor Selection Pipeline

The automated anchor selection pipeline scores and ranks all candidate tasks to help the team select the best 50 anchors. Each task is evaluated by multiple LLM judges.

### How It Works

For each task, judges (randomly sampled from a model pool) each make **one combined API call** that evaluates three things:

| Evaluation | Method | Output |
|---|---|---|
| **Ambiguity Classification** | 3 boolean questions per ambiguity type (structural fit, natural phrasing, code divergence) + perturbation sketch | Score 0–3 per type |
| **Risk Assessment** | 4 boolean questions (irreversibility, external state, security, data integrity). Any true → high | `"high"` or `"low"` |
| **Feasibility Scoring** | 5 boolean dimensions (multi-entity, structural complexity, testability, natural perturbation, interpretation divergence) | Score 0–5 |

All scoring uses **structured boolean rubrics** — judges answer yes/no questions rather than subjective scales, ensuring consistency across different LLMs.

### Configuration

Three config files control the pipeline:

**`config/pipeline.yaml`** — runtime parameters:

```yaml
anchor_selection:
  judge_models:          # randomly sampled per task
    - gpt-5.4-mini
    - claude-haiku
    - gemini-3-flash
    - deepseek-v3.2
    - qwen-3.5
    - minimax-m2.5
  judges_per_task: 5     # how many judges per task
  temperature: 0.0
  max_tokens: 2048
  max_workers: 6         # concurrent tasks (question-level parallelism)
  sources:
    - humaneval          # which sources to evaluate
```

**`config/prompts.yaml`** — all system/task prompts (single source of truth). Currently has `anchor_selection.combined_evaluation` with system + task prompts including few-shot example.

**`config/models.yaml`** — maps model aliases to OpenRouter IDs. To add a new model, add one line here.

### Running

```bash
# Run on all configured sources
python scripts/run_anchor_selection.py

# Run on one source
python scripts/run_anchor_selection.py --source humaneval

# Run on first N tasks
python scripts/run_anchor_selection.py --source humaneval --limit 20

# Run on specific tasks
python scripts/run_anchor_selection.py --task-ids HumanEval/74 MBPP/11

# Enable debug mode (logs all errors with raw responses)
python scripts/run_anchor_selection.py --source humaneval --debug
```

Progress output:

```
Evaluating 164 tasks with 6 workers (5 judges per task, 1 API call each)...

[12/164] HumanEval/74           best=coreferential        risk=low   feas=4.3/5  (4.2s)  ETA 2.1min
```

### Output

**`data/intermediate/anchor_selection/anchor_results.jsonl`** — one JSON line per task:

```jsonc
{
  "task_id": "HumanEval/74",       // look up prompt/solution from data/raw/
  "source": "humaneval",
  "entry_point": "total_match",
  "library": null,

  "votes": [...],                  // raw judge votes (for auditing)

  "ambiguity_scores": {            // avg boolean score (0–3) per type
    "coreferential": 3.0,
    "syntactic": 1.5,
    "scopal": 1.5,
    "collective_distributive": 3.0,
    "elliptical": 1.5
  },
  "best_ambiguity_type": "coreferential",
  "perturbation_sketches": {       // judge-generated perturbation ideas
    "coreferential": ["Reword to '...returns it...' where 'it' could refer to..."]
  },

  "risk_level": "low",
  "risk_agreement": 1.0,

  "feasibility_score": 4.33,      // avg across judges (0–5)
  "feasibility_dimensions": {     // per-dimension agreement ratio
    "d1_multi_entity": 1.0,
    "d2_structural_complexity": 0.5,
    "d3_testability": 1.0,
    "d4_natural_perturbation": 1.0,
    "d5_interpretation_divergence": 0.5
  }
}
```

**`data/intermediate/anchor_selection/errors.log`** (debug mode only) — full error details with raw model responses.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Empty response" errors | `max_tokens` too low or model doesn't support the prompt | Increase `max_tokens` in `pipeline.yaml` (2048+ recommended) |
| Truncated JSON | Same as above | Same fix |
| Rate limit errors | Too many concurrent requests | Lower `max_workers` in `pipeline.yaml` |
| Model ID errors | Wrong OpenRouter ID | Check `config/models.yaml`, verify with OpenRouter |

### Selecting Anchors from Results

After running the pipeline, analyze `anchor_results.jsonl` to select 50 anchors:

1. **Filter** by `feasibility_score >= 3` (good candidates)
2. **Balance** across `best_ambiguity_type` (target 10 per type)
3. **Mix** `risk_level` (high and low within each type)
4. **Review** `perturbation_sketches` — LLM-suggested starting points for the manual perturbation step
5. **Cross-reference** with raw data: `store.get(task_id)` to read the full prompt

The script prints a summary with top candidates and distributions.

---

## Steps 1.2–1.5 — Benchmark Construction

After selecting 50 anchors, team members manually build perturbed benchmark items. The final output is `data/benchmark/benchmark.jsonl`.

### BenchmarkItem Schema

Each line in `benchmark.jsonl` is a JSON object with 20 fields in 3 layers:

#### Layer 1 — Anchor (from raw data)

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Benchmark ID: `"AMBI/001"` |
| `anchor_task_id` | str | Original source ID: `"HumanEval/74"` |
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

### Creating a BenchmarkItem

```python
from src.data.model import BenchmarkItem
from src.data import BenchmarkStore

store = BenchmarkStore.load_local("data/raw")
task = store.get("HumanEval/74")

# Pre-fills all Layer 1 fields from the anchor
item = BenchmarkItem.from_task(task, task_id="AMBI/001")

# Fill in Layer 2 (your perturbation work)
item.perturbed_prompt = "..."
item.ambiguity_type = "coreferential"
item.risk_level = "low"
item.interpretation_a = "..."
item.interpretation_b = "..."
item.ref_solution_a = "..."
item.ref_solution_b = "..."
item.test_a = "..."
item.test_b = "..."
```

### Running Quality Gate Stage A

Requires Docker Desktop running.

```python
from src.util.sandbox import Sandbox

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

### Example

See `data/benchmark/benchmark.jsonl` for a worked example (`AMBI/001` based on HumanEval/15, elliptical ambiguity).

---

## How Phase 2 Consumes the Benchmark

Phase 2 (inference + evaluation) reads `data/benchmark/benchmark.jsonl` and for each item:

### 1. Baseline (clean prompt)

Send `item.prompt` to each model → execute responses against `item.test_code` → compute **pass@k(Baseline)**.

### 2. Perturbed (ambiguous prompt)

Send `item.perturbed_prompt` to each model → classify response as **SA / EA / AC** → for SA/EA responses, run dual-blind execution:

```python
from src.util.sandbox import Sandbox

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

## Scaled Pipeline

The scaled pipeline (`scripts/run_scaled_pipeline.py`) automates the full benchmark construction process with 10 parallel workers.

### How It Works

Each worker handles one `(ambiguity_type, risk_level)` combination:
1. Pops the highest-feasibility task from its queue
2. Runs Stages 1-4 for that task
3. If Stage 4 passes, stores the item in the benchmark
4. If not, pops the next task
5. Stops when target reached or queue exhausted

### Running

```bash
# Dry run — show worker allocations
python scripts/run_scaled_pipeline.py --dry-run

# Full run (default: 7 low-risk + 3 high-risk per type = 50 target)
python scripts/run_scaled_pipeline.py

# Custom targets
python scripts/run_scaled_pipeline.py --low-target 7 --high-target 3

# Fewer parallel workers (reduce API load)
python scripts/run_scaled_pipeline.py --max-workers 5
```

### Output

Appends passing items to `data/benchmark/benchmark.jsonl`. Each worker writes intermediate results to `data/intermediate/scaled_pipeline/worker_NN_type_risk/`.

---

## Benchmark Distribution

The final benchmark contains **62 items**:

| Ambiguity Type | Low | High | Total |
|---|---|---|---|
| Coreferential | 8 | 3 | 11 |
| Syntactic | 10 | 4 | 14 |
| Scopal | 5 | 3 | 8 |
| Collective/Distributive | 16 | 3 | 19 |
| Elliptical | 7 | 3 | 10 |
| **Total** | **46** | **16** | **62** |

All 62 pass both Quality Gate Stage A (sandbox exclusivity) and Stage B (entropy gate).

---

## File Map

| Path | Description |
|---|---|
| `config/models.yaml` | Model alias -> OpenRouter ID mapping |
| `config/pipeline.yaml` | Pipeline runtime config (judge models, concurrency, tokens) |
| `config/prompts.yaml` | All prompts (system + task) -- single source of truth |
| `src/data/model.py` | `BenchmarkTask` + `BenchmarkItem` dataclasses |
| `src/data/loaders.py` | HuggingFace loaders for HumanEval, MBPP, DS-1000 |
| `src/data/ds1000_normalizer.py` | DS-1000 harness-to-concatenation format converter |
| `src/data/store.py` | `BenchmarkStore` -- unified load/filter/save interface |
| `src/pipeline/prompts.py` | `get_prompt()`, `render_prompt()`, `load_pipeline_config()` |
| `src/pipeline/anchor_selection.py` | Anchor scoring pipeline |
| `src/pipeline/stage1_perturbation.py` | Stage 1: perturbation generation |
| `src/pipeline/stage2_entropy_gate.py` | Stage 2: entropy gate |
| `src/pipeline/stage3_test_generation.py` | Stage 3: ref_solution_b + test_b generation |
| `src/pipeline/stage4_exclusivity_gate.py` | Stage 4: 2x2 sandbox exclusivity |
| `src/util/llm.py` | `LLMClient` -- OpenRouter wrapper |
| `src/util/sandbox.py` | `Sandbox` -- Docker-based Python execution |
| `scripts/download_data.py` | CLI: download raw benchmarks |
| `scripts/normalize_ds1000.py` | CLI: normalize DS-1000 tasks |
| `scripts/run_anchor_selection.py` | CLI: run anchor selection pipeline |
| `scripts/run_perturbation.py` | CLI: run 4-stage perturbation pipeline |
| `scripts/run_scaled_pipeline.py` | CLI: scaled pipeline (10 parallel workers) |
| `data/raw/*.jsonl` | Downloaded + normalized benchmark data (gitignored) |
| `data/intermediate/` | Pipeline intermediate outputs (gitignored) |
| `data/benchmark/benchmark.jsonl` | **Final benchmark (62 items)** |
