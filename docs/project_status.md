# AmbiCode-Eval — Project Status

*Last updated: 2026-04-04*

## Current State

**62 verified benchmark items** in `data/benchmark/benchmark.jsonl`, from MBPP (26) and DS-1000 (36), all passing strict 2x2 sandbox exclusivity and entropy gate quality checks.

## Benchmark Coverage

| Dimension | Current | Notes |
|---|---|---|
| Total items | 62 | Exceeds original 50 target |
| collective_distributive | 19 | 16 low, 3 high |
| syntactic | 14 | 10 low, 4 high |
| coreferential | 11 | 8 low, 3 high |
| elliptical | 10 | 7 low, 3 high |
| scopal | 8 | 5 low, 3 high |
| Source: MBPP | 26 | |
| Source: DS-1000 | 36 | Pandas (22), Sklearn (5), Scipy (4), Numpy (4), Pytorch (1) |
| Risk: low | 46 | 74% |
| Risk: high | 16 | 26% |

## Pipeline Architecture

```
Anchor Selection (done, 1,421 tasks scored)
    |
DS-1000 Normalization (done, 845 tasks normalized)
    |
Stage 1 -- Perturbation Generation
    3 SOTA models (gpt-5.4, claude-sonnet, gemini-3.1-pro) generate:
      perturbed_prompt + interpretation_a + interpretation_b
    |
Stage 2 -- Entropy Gate
    5 judge models vote on which interpretation the prompt conveys.
    Shannon entropy H >= 0.72 required (at least 4-1 judge split).
    |
Stage 3 -- Reference Solution B + Test Generation
    Same generator model writes ref_solution_b + test_b.
    DS-1000: generates self-contained code (with imports + test data).
    |
Stage 4 -- Exclusivity Gate (Docker sandbox, no LLM calls)
    Verifies 2x2 matrix:
      ref_a passes test_a, fails test_b
      ref_b fails test_a, passes test_b
    DS-1000: handles cross-format pairs (harness vs self-contained).
```

### Yield Rates

| Stage | Yield | Notes |
|---|---|---|
| Stage 1 (generation) | ~96% | Most tasks get valid perturbations |
| Stage 2 (entropy gate) | ~21% | Biggest bottleneck |
| Stage 3 (test gen) | ~94% | Mostly succeeds |
| Stage 4 (exclusivity) | ~50% | Half fail discriminativity |
| **Overall** | **~10%** | Tasks in -> benchmark items out |

## Completed Milestones

- **Phase 1 Step 1.1 (Raw Data)**: DONE — 1,421 tasks downloaded to `data/raw/`
- **Phase 1 Step 1.1b (Anchor Selection)**: DONE — 1,421 tasks scored
- **Phase 1 DS-1000 Normalization**: DONE — 845 non-Matplotlib tasks converted to concatenation-friendly format
- **Phase 1 Steps 1.2-1.5 (Perturbation Pipeline)**: DONE — 62 verified benchmark items
  - Scaled pipeline (`scripts/run_scaled_pipeline.py`) with 10 parallel workers
  - Each worker targets one (ambiguity_type, risk_level) combination
  - Workers pop from priority queue sorted by feasibility score

## Known Limitations

### 1. HumanEval Excluded

HumanEval prompts contain detailed docstrings with `>>>` examples that disambiguate perturbations. 0% Stage 4 pass rate in testing. Focus shifted to MBPP + DS-1000.

### 2. Matplotlib DS-1000 Tasks Excluded (155 tasks)

Matplotlib tasks compare rendered images pixel-by-pixel. Cannot be converted to assert-based testing.

### 3. Scopal Ambiguity Under-represented

Only 36 candidates total (28 low, 8 high) vs 300+ for other types. Scopal ambiguity is rare in code prompts.

### 4. Some Anchor Duplication

6 anchor tasks appear in multiple benchmark items (different perturbation wordings from different generator models). All pass quality gates independently.

## Phase 2+ Readiness

Infrastructure for downstream phases is ready:
- `LLMClient` — OpenRouter gateway to all target models
- `Sandbox` — Docker-based dual-blind execution
- `BenchmarkItem.from_dict()` — load benchmark items
- DS-1000 Docker image (`ambicode-ds1000`) with pandas, numpy, scipy, sklearn, torch, tensorflow

Next: Phase 2 (inference), Phase 3 (behavioral classification), Phase 4 (execution), Phase 5 (analysis).
