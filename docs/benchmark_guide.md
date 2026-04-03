# AmbiCode-Eval Benchmark Guide

## Overview

AmbiCode-Eval is a micro-benchmark measuring how LLMs handle linguistically ambiguous coding prompts. Each benchmark item contains a **clean prompt** (baseline) and a **perturbed prompt** (with injected linguistic ambiguity), along with two valid interpretations, reference solutions, and discriminative test suites.

The benchmark enables measuring the **Ambiguity Tax** — the drop in pass@k when models encounter ambiguous prompts — and classifying model behavior into **Silent Assumption (SA)**, **Explicit Assumption (EA)**, or **Active Clarification (AC)**.

## Benchmark Item Format

Each item in `data/benchmark/benchmark.jsonl` is a JSON object with these fields:

### Layer 1 — Anchor (from original benchmark)

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Benchmark ID, e.g. `"AMBI/001"` |
| `anchor_task_id` | string | Original source ID, e.g. `"MBPP/106"` |
| `source` | string | `"humaneval"`, `"mbpp"`, or `"ds1000"` |
| `prompt` | string | Clean original prompt (baseline condition) |
| `canonical_solution` | string | Original reference solution (implements interpretation A) |
| `test_code` | string | Original test suite (same as `test_a`) |
| `entry_point` | string? | Function name for HumanEval tasks, null for others |
| `library` | string? | Library name for DS-1000 tasks, null for others |

### Layer 2 — Perturbation

| Field | Type | Description |
|---|---|---|
| `perturbed_prompt` | string | Ambiguous version of the prompt (experimental condition) |
| `ambiguity_type` | string | One of: `coreferential`, `syntactic`, `scopal`, `collective_distributive`, `elliptical` |
| `risk_level` | string | `"high"` or `"low"` — risk of the underlying task |
| `interpretation_a` | string | One-sentence description of interpretation A (original meaning) |
| `interpretation_b` | string | One-sentence description of interpretation B (alternative meaning) |
| `ref_solution_a` | string | Code implementing interpretation A (= `canonical_solution`) |
| `ref_solution_b` | string | Code implementing interpretation B |
| `test_a` | string | Test suite for interpretation A (= `test_code`) |
| `test_b` | string | Test suite for interpretation B |

### Layer 3 — Quality Gates

| Field | Type | Description |
|---|---|---|
| `quality_gate_a` | bool | Passed Stage 4 sandbox exclusivity check |
| `quality_gate_b` | bool | Passed Stage 2 entropy gate |
| `quality_gate_b_votes` | dict | Judge model votes (available in stage2_results.jsonl) |

### Exclusivity Guarantee

Every item in the benchmark satisfies strict mutual exclusivity, verified by Docker sandbox:

```
ref_solution_a + test_a → PASS
ref_solution_a + test_b → FAIL
ref_solution_b + test_a → FAIL
ref_solution_b + test_b → PASS
```

## How to Use (Downstream — Phase 2+)

### Loading

```python
from src.data.model import BenchmarkItem
import json

with open("data/benchmark/benchmark.jsonl") as f:
    items = [BenchmarkItem.from_dict(json.loads(line)) for line in f]
```

### Phase 2 — Inference (Two-Condition Sampling)

For each item, send both prompts to each target model:

```python
# Baseline condition
clean_response = model.generate(item.prompt, temperature=0.8, n=10)

# Experimental condition (ambiguous)
perturbed_response = model.generate(item.perturbed_prompt, temperature=0.8, n=10)
```

### Phase 3 — Behavioral Classification

Classify each perturbed response as:
- **Silent Assumption (SA)**: model picks one interpretation without mentioning ambiguity
- **Explicit Assumption (EA)**: model states its assumption before answering
- **Active Clarification (AC)**: model asks for clarification

### Phase 4 — Dual-Blind Execution

Run each generated solution against both test suites:

```python
from src.util.sandbox import Sandbox

sandbox = Sandbox()
result_a, result_b = sandbox.run_dual_blind(
    code=model_response,
    test_a=item.test_a,
    test_b=item.test_b,
)
# result_a.passed → matches interpretation A
# result_b.passed → matches interpretation B
```

### Phase 5 — Analysis

- **Ambiguity Tax**: `pass@k(clean) - pass@k(perturbed)`
- **Behavioral Distribution**: proportion of SA / EA / AC across models
- **Conditional pass@k**: pass@k(A) and pass@k(B) given ambiguous prompt

## Ambiguity Types

| Type | Description | Example |
|---|---|---|
| **Coreferential** | Pronoun/noun phrase with ambiguous antecedent | "merge dict_a into dict_b and return **it**" |
| **Syntactic** | Modifier phrase attaching to different constituents | "replace the last element **as a whole**" |
| **Scopal** | Quantifier/operator with underdetermined scope | "sum from 0 to n, **integer-divided by 2**" |
| **Collective/Distributive** | Operation applying to set as whole vs. each member | "append list **to the tuples**" |
| **Elliptical** | Omitted verb phrase with multiple valid recoveries | "find the response **to a sinusoid**" |

## Risk Levels

- **Low**: Pure in-memory computation (sorting, math, string manipulation)
- **High**: Touches external state, file I/O, data integrity, security-sensitive operations
