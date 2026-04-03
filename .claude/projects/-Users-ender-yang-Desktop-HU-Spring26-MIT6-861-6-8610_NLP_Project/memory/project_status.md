---
name: project-status-phase1
description: Phase 1 progress — 4-stage perturbation pipeline built, 14 verified benchmark items, DS1000 + HumanEval blockers remain
type: project
---

As of 2026-04-03:

**Completed:**
- Raw data collection: 1,421 tasks (HumanEval 164 + MBPP 257 + DS-1000 1000) in `data/raw/`
- Anchor selection pipeline: 1,421 tasks scored in `data/intermediate/anchor_selection/anchor_results.jsonl`
- 4-stage perturbation pipeline fully built and tested:
  - Stage 1 (`src/pipeline/stage1_perturbation.py`): 3 SOTA models generate perturbed_prompt + interpretations
  - Stage 2 (`src/pipeline/stage2_entropy_gate.py`): 5 judge models vote → Shannon entropy filter (H >= 0.72)
  - Stage 3 (`src/pipeline/stage3_test_generation.py`): generates ref_solution_b + test_b
  - Stage 4 (`src/pipeline/stage4_exclusivity_gate.py`): Docker sandbox 2×2 exclusivity verification
- Shared utilities: `src/util/parsing.py` (JSON extraction), `src/util/pipeline_runner.py` (concurrent runner)
- **14 verified benchmark items** in `data/benchmark/benchmark.jsonl` (all MBPP, all passing exclusivity)
- Benchmark guide: `docs/benchmark_guide.md`
- Docker image `ambicode-ds1000` built for DS1000 tasks
- Overall pipeline yield: ~24% (59 candidates → 14 unique items)

**Current benchmark coverage:**
- 14 items total (target: 50)
- Ambiguity types: collective_distributive=9, syntactic=4, coreferential=1, scopal=0, elliptical=0
- Risk levels: low=13, high=1 (target: 7:3)
- Sources: MBPP only

**Blockers to reaching 50 items:**
1. DS1000 sandbox adapter needed (2-3h) — DS1000 uses `exec_context` + `[insert]` execution model, setup code in prompt `<code>` blocks. Unlocks 177 high-risk candidates + scopal/elliptical types.
2. HumanEval 0% Stage 4 pass rate — docstring examples disambiguate, original tests too thorough for ref_b to diverge. Possible fix: test-first generation or strip to MBPP-style. (2-3h, uncertain)

**Next steps:**
1. Fix DS1000 sandbox adapter in Stage 4
2. Run pipeline on remaining MBPP candidates (feas >= 2.0, ~270 tasks)
3. Run pipeline on DS1000 candidates for high-risk + missing types
4. Curate final 50 items with balanced types and risk

**Why:** 6-week timeline, currently mid Phase 1 data construction. Downstream team can start Phase 2 inference pipeline with existing 14 items.
**How to apply:** Pipeline is run via `python scripts/run_perturbation.py`. Check intermediate outputs before re-running. Config in `config/pipeline.yaml`, prompts in `config/prompts.yaml`.
