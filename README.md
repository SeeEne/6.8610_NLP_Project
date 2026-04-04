# AmbiCode-Eval

A benchmark of **62 tasks** measuring how LLMs handle linguistically ambiguous coding prompts.

Quantifies the **Ambiguity Tax** (pass@k drop from ambiguity injection) and classifies model behavior into Silent Assumption / Explicit Assumption / Active Clarification.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set up API key
cp .env.example .env   # edit with your OPENROUTER_API_KEY

# Docker (required for sandbox execution)
docker build -t ambicode-ds1000 -f docker/ds1000.Dockerfile .
```

### Load the Benchmark

```python
import json
from src.data.model import BenchmarkItem

with open("data/benchmark/benchmark.jsonl") as f:
    items = [BenchmarkItem.from_dict(json.loads(line)) for line in f if line.strip()]
```

### Run a Simple Evaluation

```python
from src.util.llm import LLMClient, ModelConfig
from src.util.sandbox import Sandbox

client = LLMClient()
sandbox = Sandbox()
config = ModelConfig(model="gpt-5.4-mini", temperature=0.8, max_tokens=1024)

item = items[0]

# Baseline
resp = client.call(config, prompt=item.prompt, system="Write ONLY Python code.")
result = sandbox.run(resp.choices[0], item.test_code)

# Perturbed (ambiguous)
resp = client.call(config, prompt=item.perturbed_prompt, system="Write ONLY Python code.")
result_a, result_b = sandbox.run_dual_blind(resp.choices[0], item.test_a, item.test_b)
```

See `notebooks/evaluation_demo.ipynb` for a complete walkthrough.

## Benchmark Overview

Each benchmark item contains:

| Layer | Fields | Purpose |
|-------|--------|---------|
| **Anchor** | `prompt`, `canonical_solution`, `test_code` | Baseline condition |
| **Perturbation** | `perturbed_prompt`, `interpretation_a/b`, `ref_solution_a/b`, `test_a/b` | Experimental condition |
| **Quality Gates** | `quality_gate_a` (sandbox exclusivity), `quality_gate_b` (entropy gate) | Validation |

### Distribution

| Ambiguity Type | Low Risk | High Risk | Total |
|----------------|----------|-----------|-------|
| Coreferential | 8 | 3 | 11 |
| Syntactic | 10 | 4 | 14 |
| Scopal | 5 | 3 | 8 |
| Collective/Distributive | 16 | 3 | 19 |
| Elliptical | 7 | 3 | 10 |
| **Total** | **46** | **16** | **62** |

Sources: MBPP (26), DS-1000 (36).

## Source-Specific Prompting

Each benchmark source requires dedicated prompt engineering for fair evaluation. **Failing to apply these strategies will cause false negatives** (correct code that fails tests due to naming/format mismatches).

### MBPP Items

MBPP prompts are natural language descriptions that do **not** specify the expected function name.
Tests call a specific function name (e.g., `add_lists`).

**Required**: Extract the function name from `test_code` and append it to the prompt.

```python
import re

def extract_function_name(test_code):
    m = re.search(r'assert\s+(\w+)\s*\(', test_code)
    return m.group(1) if m else None

func_name = extract_function_name(item.test_code)
prompt = f"{item.prompt}\nThe function should be named `{func_name}`."
```

**Output format**: Complete, self-contained Python function.

**Sandbox**: `Sandbox()` (default `python:3.11-slim` image).

### DS-1000 Items

DS-1000 prompts are data science problems with embedded setup code and example data.
The benchmark uses a **normalized format**:

- `canonical_solution`: code fragment wrapped as `__SOLUTION__ = r"""..."""`
- `test_code`: original DS-1000 harness + `test_execution(__SOLUTION__)`
- `ref_solution_b` / `test_b`: self-contained (include their own imports and test data)

**Required**: Wrap LLM output as `__SOLUTION__` when testing against `test_a`.

```python
from src.data.ds1000_normalizer import _wrap_solution_as_string

# Testing against test_a (harness format)
wrapped = _wrap_solution_as_string(llm_output)
result_a = sandbox_ds.run(wrapped, item.test_a, timeout_s=60)

# Testing against test_b (self-contained)
result_b = sandbox_ds.run(llm_output, item.test_b, timeout_s=60)
```

**Output format**: Code fragment (not a complete function).

**Sandbox**: `Sandbox(image="ambicode-ds1000")` — requires building the Docker image.

### Common: Markdown Fence Stripping

LLMs frequently wrap output in ` ```python ... ``` ` despite system prompt instructions.
**Always strip fences** before sandbox execution.

```python
import re

def strip_markdown_fences(code):
    code = re.sub(r'^```(?:python)?\s*\n', '', code.strip())
    code = re.sub(r'\n```\s*$', '', code)
    return code.strip()
```

### Recommended System Prompt

```
You are a Python code generator. Write ONLY the Python function implementation.
No explanation, no markdown fences, no extra text. Just the code.
```

## Evaluation Metrics

- **Ambiguity Tax** = pass@k(baseline) - pass@k(perturbed)
- **Interpretation Bias** = asymmetry between pass@k(A) and pass@k(B)
- **Behavioral Distribution** = proportion of SA / EA / AC responses per model
- **Conditional pass@k** = pass@k given ambiguous prompt, stratified by interpretation

## Project Structure

```
data/
  benchmark/benchmark.jsonl     # The benchmark (62 items)
  raw/                          # Raw benchmark sources (gitignored)
  intermediate/                 # Pipeline intermediate outputs (gitignored)

src/
  data/                         # Data models, loaders, DS-1000 normalizer
  pipeline/                     # 4-stage perturbation pipeline
  util/                         # LLM client, Docker sandbox, parsing

config/
  models.yaml                   # Model alias -> OpenRouter ID registry
  pipeline.yaml                 # Pipeline parameters
  prompts.yaml                  # All LLM prompts (single source of truth)

scripts/
  download_data.py              # Download raw benchmarks
  normalize_ds1000.py           # Normalize DS-1000 format
  run_perturbation.py           # Run 4-stage pipeline
  run_scaled_pipeline.py        # Scaled pipeline (10 parallel workers)

notebooks/
  benchmark_demo.ipynb          # Explore benchmark structure
  evaluation_demo.ipynb         # Full evaluation walkthrough

docs/
  benchmark_guide.md            # Detailed benchmark format + usage
  data_guide.md                 # Pipeline construction guide
  project_status.md             # Current status + known limitations

docker/
  ds1000.Dockerfile             # Docker image for DS-1000 execution
```

## Reproducing the Benchmark

```bash
# 1. Download raw data
python scripts/download_data.py

# 2. Normalize DS-1000
python scripts/normalize_ds1000.py

# 3. Run anchor selection (scores 1,421 tasks)
python scripts/run_anchor_selection.py

# 4. Run scaled pipeline (generates benchmark items)
python scripts/run_scaled_pipeline.py --dry-run    # preview
python scripts/run_scaled_pipeline.py              # run
```

## Requirements

- Python 3.9+
- Docker Desktop (for sandbox execution)
- OpenRouter API key (for LLM calls)
- ~$5-10 in API credits for full benchmark construction

## License

MIT 6.8610 NLP course project.
