---
name: user-runs-tests-themselves
description: User prefers to run test/pipeline scripts themselves rather than having Claude execute them
type: feedback
---

Don't run test or pipeline scripts automatically — provide the command and let the user run it.

**Why:** User explicitly asked to run tests themselves.
**How to apply:** When testing pipeline code or running scripts that call APIs / cost money, give the user the command instead of executing it.
