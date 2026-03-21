# Notebook: simple

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/transformers/glu_variants/simple.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/glu_variants/simple.ipynb)                    

## Gated Linear Units and Variants

This trains a simple [transformer](https://nn.labml.ai/transformers/) model for auto-regression.
We try different variants for the [position-wise feedforward network](https://nn.labml.ai/transformers/feed_forward.html).

Annotated trainer code is at [`simple.py`](https://nn.labml.ai/transformers/glu_variants/simple.html)

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
import dataclasses

import torch
import torch.nn as nn
from labml import experiment
from labml_nn.transformers.glu_variants.simple import Configs, Trainer
```

Create an experiment

```python
experiment.create(name="glu_variants")
```

Initialize configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(dataclasses.asdict(conf))
```

Create [`Trainer`](https://nn.labml.ai/transformers/glu_variants/simple.html)

```python
trainer = Trainer(conf)
```

Set PyTorch models for loading and saving

```python
experiment.add_pytorch_models({'model': trainer.model})
```

Start the experiment and run the training loop.

```python
with experiment.start():
    trainer.train()
```