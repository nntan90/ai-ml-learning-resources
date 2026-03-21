# Notebook: tutorial-template

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/.agents/skills/jupyter-notebook/assets/tutorial-template.ipynb

---

# Tutorial: TITLE

Audience:
- Describe who this is for.

Prerequisites:
- List required concepts or setup.

Learning goals:
- By the end, the reader can...


## Outline

1. Setup
2. A minimal working example
3. Variations and pitfalls
4. Exercises


```python
# Setup cell: keep it short and deterministic
from __future__ import annotations

import math
import random

SEED = 21
random.seed(SEED)
SEED

```

## Step 1 - Start with a tiny example

Explain what the next cell does in plain language.


```python
# Minimal working example
angles = [0, math.pi / 4, math.pi / 2]
sines = [math.sin(a) for a in angles]
list(zip(angles, sines))

```

## Exercises

- Try a different input.
- Predict the output before running the code.
- Note one common mistake and how to fix it.


```python
# Exercise answer scaffold
def describe(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "max": max(values)}

describe(sines)

```