# Notebook: experiment-template

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/.agents/skills/jupyter-notebook/assets/experiment-template.ipynb

---

# Experiment: TITLE

Objective:
- State the question you want to answer.
- Define the success criteria.


```python
# Setup: imports and reproducibility
from __future__ import annotations

import random
import statistics

SEED = 7
random.seed(SEED)
SEED

```

## Plan

- Hypothesis:
- Variables to sweep:
- Metrics to record:


```python
# Define parameters and lightweight helpers
sample_size = 20
values = [random.random() for _ in range(sample_size)]
summary = {
    "count": len(values),
    "mean": statistics.fmean(values),
    "min": min(values),
    "max": max(values),
}
summary

```

## Results

- Key observations:
- Surprises or failure modes:
- Decision: continue, pivot, or stop:


```python
# Record findings in a minimal, copy-pasteable structure
result = {
    "seed": SEED,
    "mean": summary["mean"],
    "range": summary["max"] - summary["min"],
}
result

```

## Next steps

- What to try next:
- What to document elsewhere (PRD, notes, issue):
