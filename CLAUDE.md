# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AmbiCode-Eval** — a micro-benchmark of 50 tasks measuring how LLMs handle linguistically ambiguous coding prompts. Quantifies the "Ambiguity Tax" (pass@k drop from ambiguity injection) and classifies model behavior into Silent Assumption / Explicit Assumption / Active Clarification.

Target models: GPT, Claude, Gemini, DeepSeek, Qwen — all called via OpenRouter.

## Commands

```bash
pip install -e ".[dev]"        # install project + dev dependencies
pytest                          # run all tests
pytest tests/ -k "test_name"   # run a single test
```

Requires a `.env` file with `OPENROUTER_API_KEY` (see `.env.example`).
Docker must be running for sandbox execution.

## Architecture

```
src/
├── data/
│   ├── model.py    # BenchmarkTask — unified schema across all sources
│   ├── loaders.py  # Per-source loaders (HumanEval, MBPP, DS-1000) via HuggingFace
│   └── store.py    # BenchmarkStore — load, filter, save/reload from JSONL
├── util/
│   ├── llm.py      # Unified LLM client — OpenRouter (OpenAI-compatible API)
│   └── sandbox.py  # Docker-based Python sandbox — no network, mem/pid limits
config/
└── models.yaml     # Model alias → OpenRouter ID registry
scripts/
└── download_data.py  # Download all benchmarks to data/raw/
docs/
└── data_guide.md     # Guide for Steps 1.2–1.5 (perturbation, tests, quality gate)
```

### Data Layer (`src/data/`)

- `BenchmarkTask` — unified dataclass: `task_id`, `prompt`, `canonical_solution`, `test_code`, etc.
- Loaders normalise HumanEval (164), MBPP-sanitized (257), DS-1000 (1000) into the same schema
- `BenchmarkStore` — in-memory store with `filter(source=, library=)`, `save()` to JSONL, `load_local()` from JSONL
- Raw data lives in `data/raw/` (gitignored, regenerated via `scripts/download_data.py`)

### LLM Client (`src/util/llm.py`)

- Uses OpenRouter as the single gateway to all model families
- Model aliases configured in `config/models.yaml`
- `LLMClient.call()` returns `LLMResponse` with `choices` list (one per `n`)
- To add a new model: add an entry to `config/models.yaml`

### Sandbox (`src/util/sandbox.py`)

- Each execution runs in a fresh Docker container (`python:3.11-slim`)
- Containers have: no network, 256MB memory limit, 64 PID limit, configurable timeout
- `run(code, test_code)` — concatenates code + tests, returns `SandboxResult`
- `run_dual_blind(code, test_a, test_b)` — runs against both interpretations
- `validate_quality_gate_a()` — checks strict exclusivity of reference solutions

## Pipeline Phases (from proposal)

1. **Data**: Anchor selection → ambiguity injection → reference solutions → test authoring → quality gate
2. **Inference**: Two-condition batch sampling (clean + perturbed) with temperature sampling
3. **Classification**: LLM-as-Judge with structured boolean rubric → SA/EA/AC labels
4. **Execution**: Dual-blind sandbox against Test-A and Test-B
5. **Analysis**: Ambiguity Tax, behavioral distributions, conditional pass@k
