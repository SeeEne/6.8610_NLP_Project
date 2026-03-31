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
src/util/
├── llm.py      # Unified LLM client — OpenRouter (OpenAI-compatible API)
│                 ModelConfig for per-call params, MODEL_REGISTRY for aliases
│                 Supports temperature sampling (n>1) for pass@k
└── sandbox.py  # Docker-based Python sandbox — no network, mem/pid limits
                  run() for code+tests, run_dual_blind() for Test-A/Test-B,
                  validate_quality_gate_a() for Stage A mechanical check
```

### LLM Client (`src/util/llm.py`)

- Uses OpenRouter as the single gateway to all model families
- `MODEL_REGISTRY` maps short aliases (e.g. `"gpt-4o"`) to OpenRouter model IDs
- `LLMClient.call()` returns `LLMResponse` with `choices` list (one per `n`)
- To add a new model: add an entry to `MODEL_REGISTRY`

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
