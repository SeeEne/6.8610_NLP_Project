---
name: feedback-separate-stage-files
description: User prefers each pipeline stage in its own .py file, shared code in util/
type: feedback
---

Write each pipeline stage as a separate Python file, not combined into one module. Shared utilities (JSON parsing, concurrent runner) go in `src/util/`.

**Why:** User explicitly requested "write each stage in single py file and assembly them later" when code was initially combined.
**How to apply:** When adding new stages, create `src/pipeline/stageN_name.py`. Shared logic goes in `src/util/`. The runner script (`scripts/run_perturbation.py`) assembles stages.
