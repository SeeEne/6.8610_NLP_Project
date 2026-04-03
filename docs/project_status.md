# AmbiCode-Eval — Project Status

*Last updated: 2026-04-03*

## Current State

**14 verified benchmark items** in `data/benchmark/benchmark.jsonl`, all from MBPP, passing strict 2×2 sandbox exclusivity. Target is 50 items with balanced ambiguity types and risk levels.

## Pipeline Architecture

```
Anchor Selection (done, 1,421 tasks scored)
    ↓
Stage 1 — Perturbation Generation
    3 SOTA models (gpt-5.4, claude-sonnet, gemini-3.1-pro) generate:
      perturbed_prompt + interpretation_a + interpretation_b
    Cost: ~$0.05/task
    ↓
Stage 2 — Entropy Gate
    5 judge models vote on which interpretation the prompt conveys.
    Shannon entropy H >= 0.72 required (at least 4-1 judge split).
    Cost: ~$0.01/generation
    ↓
Stage 3 — Reference Solution B + Test Generation
    Same generator model writes ref_solution_b + test_b.
    Cost: ~$0.03/item
    ↓
Stage 4 — Exclusivity Gate (Docker sandbox, no LLM calls)
    Verifies 2×2 matrix:
      ref_a passes test_a, fails test_b
      ref_b fails test_a, passes test_b
    Cost: ~$0 (local Docker)
```

### Yield Rates (from 59-candidate MBPP run)

| Stage | Input | Output | Yield |
|---|---|---|---|
| Stage 1 (generation) | 59 tasks × 3 models = 177 | 170 successful | 96% |
| Stage 2 (entropy gate) | 170 generations | 36 passed | 21% |
| Stage 3 (test gen) | 36 items | 34 successful | 94% |
| Stage 4 (exclusivity) | 34 items | 17 passed (14 unique) | 50% |
| **Overall** | **59 tasks** | **14 unique items** | **24%** |

## Running the Pipeline

```bash
# Full pipeline (all 4 stages)
python scripts/run_perturbation.py --task-ids MBPP/106 MBPP/299 ...

# Individual stages
python scripts/run_perturbation.py --stage 1
python scripts/run_perturbation.py --stage 2
python scripts/run_perturbation.py --stage 3
python scripts/run_perturbation.py --stage 4    # requires Docker

# Selection options
python scripts/run_perturbation.py --max-tasks 20
python scripts/run_perturbation.py --min-feasibility 3.0
python scripts/run_perturbation.py --risk-level high
```

### Intermediate Outputs

```
data/intermediate/
├── anchor_selection/
│   └── anchor_results.jsonl      # 1,421 scored tasks
├── perturbation/
│   └── stage1_results.jsonl      # perturbed prompts + interpretations
├── entropy_gate/
│   └── stage2_results.jsonl      # judge votes + entropy scores
├── test_generation/
│   └── stage3_results.jsonl      # ref_solution_b + test_b
└── exclusivity_gate/
    └── stage4_results.jsonl      # 2×2 sandbox verification
```

## Benchmark Coverage

| Dimension | Current | Target |
|---|---|---|
| Total items | 14 | 50 |
| collective_distributive | 9 | ~10 |
| syntactic | 4 | ~10 |
| coreferential | 1 | ~10 |
| scopal | 0 | ~10 |
| elliptical | 0 | ~10 |
| Source: MBPP | 14 | ~17 |
| Source: HumanEval | 0 | ~17 |
| Source: DS1000 | 0 | ~16 |
| Risk: low | 13 | ~35 (70%) |
| Risk: high | 1 | ~15 (30%) |

## Known Issues

### 1. DS1000 Sandbox Adapter (blocks scaling)

**Problem**: DS1000 tasks use a different execution model than HumanEval/MBPP:
- Setup code (`df`, `X`, imports) is embedded in the prompt's `<code>` blocks, not in `test_code`
- Tests use an `exec_context` template with `[insert]` placeholder, not simple `assert` statements
- The test harness uses `generate_test_case()` / `exec_test()` functions

**What's done**: Docker image `ambicode-ds1000` built with pandas/numpy/scipy/matplotlib/sklearn. Stage 4 selects the right image per source.

**What's needed**: Parse DS1000 prompt to extract setup code, build adapter that wraps solutions into DS1000's `exec_context` format, adapt Stage 4 to call their test harness.

**Impact**: Without DS1000, we cannot get high-risk items (177/181 are DS1000), scopal/elliptical ambiguity types, or 3:7 risk balance.

**Estimated effort**: 2-3 hours.

### 2. HumanEval Ambiguity Injection (0% Stage 4 pass rate)

**Problem**: HumanEval prompts contain detailed docstrings with `>>>` examples and type hints.

- **Stage 2**: Examples disambiguate — judges always pick A (87% filtered at H=0)
- **Stage 4**: Even when ambiguity passes entropy gate, `ref_solution_b` still passes `test_a` — original tests are too thorough for ref_b to genuinely diverge

**Attempted fixes**:
- Stage 1 prompt: instructions to remove/rewrite disambiguating `>>>` examples
- Stage 3 prompt: instructions to choose inputs where ref_a and ref_b diverge
- Neither solved the problem (0/7 HumanEval items pass Stage 4)

**Possible approaches** (not yet attempted):
- **Test-first generation**: Generate diverging test cases first, then write ref_b to match
- **Strip to MBPP-style**: Convert HumanEval docstrings to plain English before Stage 1
- **Accept limitation**: Focus on MBPP + DS1000

**Estimated effort**: 2-3 hours, uncertain outcome.

### 3. Ambiguity Type Imbalance

Collective/distributive dominates (9/14) because it naturally fits code prompts about operations on collections. Scopal and elliptical types are mostly found in DS1000 tasks — fixing the DS1000 sandbox unlocks them.

### 4. Risk Level Imbalance

Current: 13 low, 1 high. High-risk tasks are almost exclusively in DS1000 (177/181 candidates). Fixing the DS1000 sandbox directly solves this.

## Scaling Plan to 50 Items

1. **Fix DS1000 sandbox adapter** → unlocks 733 candidates (177 high-risk, all 5 ambiguity types)
2. **Run pipeline on remaining MBPP** (feas >= 2.0) → ~270 candidates → ~65 items at 24% yield
3. **Run pipeline on DS1000 candidates** → high-risk items + scopal/elliptical types
4. **Curate final 50** from combined pool, balancing ~10 per ambiguity type, 7:3 low:high risk
