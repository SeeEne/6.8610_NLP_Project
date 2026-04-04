# AmbiCode-Eval Benchmark Guide

## Overview

AmbiCode-Eval is a benchmark of **62 tasks** measuring how LLMs handle linguistically ambiguous coding prompts. Each benchmark item contains a **clean prompt** (baseline) and a **perturbed prompt** (with injected linguistic ambiguity), along with two valid interpretations, reference solutions, and discriminative test suites.

The benchmark enables measuring the **Ambiguity Tax** — the drop in pass@k when models encounter ambiguous prompts — and classifying model behavior into **Silent Assumption (SA)**, **Explicit Assumption (EA)**, or **Active Clarification (AC)**.

## Benchmark Distribution

### By Ambiguity Type

| Type | Low Risk | High Risk | Total |
|------|----------|-----------|-------|
| Coreferential | 8 | 3 | 11 |
| Syntactic | 10 | 4 | 14 |
| Scopal | 5 | 3 | 8 |
| Collective/Distributive | 16 | 3 | 19 |
| Elliptical | 7 | 3 | 10 |
| **Total** | **46** | **16** | **62** |

### By Source

| Source | Count | Libraries |
|--------|-------|-----------|
| MBPP | 26 | — |
| DS-1000 | 36 | Pandas (22), Sklearn (5), Scipy (4), Numpy (4), Pytorch (1) |

## Benchmark Item Format

Each item in `data/benchmark/benchmark.jsonl` is a JSON object with 19 fields across 3 layers.

### Layer 1 — Anchor (from original benchmark)

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Benchmark ID, e.g. `"AMBI/001"` |
| `anchor_task_id` | string | Original source ID, e.g. `"MBPP/106"` |
| `source` | string | `"mbpp"` or `"ds1000"` |
| `prompt` | string | Clean original prompt (baseline condition) |
| `canonical_solution` | string | Original reference solution (implements interpretation A) |
| `test_code` | string | Original test suite (same as `test_a`) |
| `entry_point` | string? | Function name for HumanEval tasks, null for others |
| `library` | string? | Library name for DS-1000 tasks, null for MBPP |

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
| `quality_gate_b_votes` | dict | Judge model votes |

### Exclusivity Guarantee

Every item in the benchmark satisfies strict mutual exclusivity, verified by Docker sandbox:

```
ref_solution_a + test_a -> PASS
ref_solution_a + test_b -> FAIL
ref_solution_b + test_a -> FAIL
ref_solution_b + test_b -> PASS
```

### DS-1000 Format Note

DS-1000 items use a **normalized format**. The `canonical_solution` wraps the original code fragment as a `__SOLUTION__` string variable, and `test_code` is the original DS-1000 test harness with `test_execution(__SOLUTION__)` appended. This allows uniform `exec(code + test)` execution across all sources.

For DS-1000 items, `ref_solution_b` and `test_b` are **self-contained** (include their own imports and test data), while `ref_solution_a` and `test_a` use the harness format.

## How to Use

### Loading

```python
from src.data.model import BenchmarkItem
import json

with open("data/benchmark/benchmark.jsonl") as f:
    items = [BenchmarkItem.from_dict(json.loads(line)) for line in f if line.strip()]

print(f"Loaded {len(items)} benchmark items")
```

### Source-Specific Prompting

Each benchmark source requires a different prompting strategy because of different prompt formats, function naming conventions, and output expectations.

#### MBPP (26 items)

**Prompt format**: Natural language description (e.g., "Write a function to append the given list to the given tuples.")

**Key issue**: MBPP prompts do **not** specify the expected function name, but tests call a specific name (e.g., `add_lists`). The function name must be extracted from `test_code` and appended to the prompt.

**LLM output**: Complete, self-contained function.

**Sandbox execution**: `sandbox.run(code, test_code)` — direct concatenation.

```python
import re

def extract_function_name(test_code: str) -> str:
    m = re.search(r'assert\s+(\w+)\s*\(', test_code)
    return m.group(1) if m else None

# Build prompt with function name
func_name = extract_function_name(item.test_code)  # e.g., "add_lists"
prompt = f"{item.prompt}\nThe function should be named `{func_name}`."

response = client.call(config, prompt=prompt, system=system_prompt)
code = response.choices[0]

# Execute
result = sandbox.run(code, item.test_code)
```

#### HumanEval (not in benchmark, for reference)

**Prompt format**: Function signature + docstring with `>>>` examples (e.g., `def has_close_elements(...): """..."""`).

**Key issue**: Asking the LLM to output "just the function body" causes indentation errors. Instead, send the full prompt and ask for the **complete function** implementation.

**LLM output**: Complete function (signature + body).

**Sandbox execution**: `sandbox.run(code, test_code)` — test code contains `check(function_name)`.

```python
response = client.call(config, prompt=task.prompt, system=system_prompt)
code = response.choices[0]

# Execute (test_code calls check(entry_point))
result = sandbox.run(code, task.test_code)
```

#### DS-1000 (36 items)

**Prompt format**: Data science problem description with example data and `<code>` blocks containing setup code (imports, DataFrame definitions, etc.).

**Key issue**: DS-1000 uses a normalized format in the benchmark:
- `canonical_solution` is `__SOLUTION__ = r"""<code fragment>"""`
- `test_code` is the original harness + `test_execution(__SOLUTION__)`
- `ref_solution_b` and `test_b` are self-contained

When evaluating LLM output, the generated code fragment must be wrapped as `__SOLUTION__` before testing against `test_a`, but can be tested directly against `test_b` (which is self-contained).

**LLM output**: Code fragment (not a complete function).

**Sandbox execution**: Requires `ambicode-ds1000` Docker image.

```python
from src.data.ds1000_normalizer import _wrap_solution_as_string

sandbox_ds = Sandbox(image="ambicode-ds1000")

response = client.call(config, prompt=item.prompt, system=system_prompt)
code_fragment = response.choices[0]

# Test against test_a (harness format): wrap as __SOLUTION__
wrapped = _wrap_solution_as_string(code_fragment)
result_a = sandbox_ds.run(wrapped, item.test_a, timeout_s=60)

# Test against test_b (self-contained): use directly
result_b = sandbox_ds.run(code_fragment, item.test_b, timeout_s=60)
```

### Common Considerations

**Markdown fence stripping**: LLMs often wrap output in ` ```python ... ``` ` despite instructions. Always strip fences before execution:

```python
import re

def strip_markdown_fences(code: str) -> str:
    code = re.sub(r'^```(?:python)?\s*\n', '', code.strip())
    code = re.sub(r'\n```\s*$', '', code)
    return code.strip()
```

**System prompt**: Use a consistent system prompt across all sources:

```
You are a Python code generator. Write ONLY the Python function implementation.
No explanation, no markdown fences, no extra text. Just the code.
```

### Phase 2 — Inference (Two-Condition Sampling)

For each item, send both the clean and perturbed prompts using the source-specific prompting strategy above:

```python
from src.util.llm import LLMClient, ModelConfig

client = LLMClient()
config = ModelConfig(model="gpt-5.4", temperature=0.8, max_tokens=1024)

for item in items:
    prompt_fn = build_source_prompt(item)  # apply source-specific logic

    # Baseline condition (clean prompt)
    clean_response = client.call(config, prompt=prompt_fn(item.prompt))

    # Experimental condition (perturbed prompt)
    perturbed_response = client.call(config, prompt=prompt_fn(item.perturbed_prompt))
```

### Phase 3 — Behavioral Classification

Classify each perturbed response as:
- **Silent Assumption (SA)**: model picks one interpretation without mentioning ambiguity
- **Explicit Assumption (EA)**: model states its assumption before answering
- **Active Clarification (AC)**: model asks for clarification

(Classification strategy TBD — requires careful prompt design for LLM-as-Judge.)

### Phase 4 — Dual-Blind Execution

Run each generated solution against both test suites:

```python
from src.util.sandbox import Sandbox

sandbox = Sandbox()                          # for MBPP / HumanEval
sandbox_ds = Sandbox(image="ambicode-ds1000")  # for DS-1000

result_a, result_b = sandbox.run_dual_blind(
    code=model_response,
    test_a=item.test_a,
    test_b=item.test_b,
)
# result_a.passed -> matches interpretation A
# result_b.passed -> matches interpretation B
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

- **Low** (46 items): Pure in-memory computation (sorting, math, string manipulation, data transformations)
- **High** (16 items): Touches external state, file I/O, data integrity, security-sensitive operations
