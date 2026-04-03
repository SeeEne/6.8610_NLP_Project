---
name: feedback-keep-tests-rigorous
description: User wants test suites to remain rigorous, don't sacrifice quality for token savings
type: feedback
---

When generating test_b, keep the test suite rigorous — don't shorten or simplify tests just to reduce token usage. Explanations can be brief, but tests must be thorough.

**Why:** User rejected a prompt change that shortened test instructions to save tokens, saying "Test suite should keep rigorous."
**How to apply:** In Stage 3 prompts, keep full test requirements (at least 3 assertions, discriminative, self-contained). Only shorten interpretation descriptions, not test specifications.
