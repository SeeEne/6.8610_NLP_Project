---
name: project-status-phase1
description: Phase 1 progress — anchor selection pipeline complete, ready for full run and manual perturbation
type: project
---

As of 2026-04-02:

**Completed:**
- Raw data collection: 1,421 tasks (HumanEval 164 + MBPP 257 + DS-1000 1000) in `data/raw/`
- Anchor selection pipeline: fully built and tested in `src/pipeline/anchor_selection.py`
  - Combined single-call evaluation (ambiguity + risk + feasibility per judge)
  - Structured boolean rubrics for cross-model consistency
  - Concurrent execution at question level with progress tracking and debug mode
  - Config: 6 flash-tier judge models, 5 judges per task, max_workers=16
- BenchmarkItem schema defined with example in `data/benchmark/benchmark.jsonl`
- Full documentation in `docs/data_guide.md`

**Next steps:**
1. Run anchor selection on full HumanEval dataset: `python scripts/run_anchor_selection.py --source humaneval`
2. Analyze `anchor_results.jsonl` — filter feasibility >= 3, balance across 5 ambiguity types
3. Select 50 anchors, then team does manual Steps 1.2–1.5 (perturbation, reference solutions, tests, quality gate)
4. After 50 items pass quality gate → Phase 2 (inference pipeline)

**Why:** 6-week timeline, currently in Week 1-2 data phase.
**How to apply:** When resuming, check `anchor_results.jsonl` for existing results before re-running. Next code work is either analyzing results or building Phase 2.
