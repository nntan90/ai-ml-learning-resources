# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/activations/fta/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/activations/fta/experiment.ipynb)

## [Fuzzy Tiling Activations](https://nn.labml.ai/activations/fta/index.html)

Here we train a transformer that uses [Fuzzy Tiling Activation](https://nn.labml.ai/activations/fta/index.html) in the
[Feed-Forward Network](https://nn.labml.ai/transformers/feed_forward.html).
We use it for a language model and train it on Tiny Shakespeare dataset
for demonstration.
However, this is probably not the ideal task for FTA, and we
believe FTA is more suitable for modeling data with continuous variables.

### Install the packages

```python
!pip install labml-nn --quiet
```

### Imports

```python
import torch
import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_nn.activations.fta.experiment import Configs
```

### Create an experiment

```python
experiment.create(name="fta", writers={'screen'})
```

### Configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf, {
    'tokenizer': 'character',
    'prompt_separator': '',
    'prompt': 'It is ',
    'text': 'tiny_shakespeare',

    'seq_len': 256,
    'epochs': 32,
    'batch_size': 16,
    'inner_iterations': 10,

    'optimizer.optimizer': 'Adam',
    'optimizer.learning_rate': 3e-4,
})
```

Set PyTorch models for loading and saving

```python
experiment.add_pytorch_models({'model': conf.model})
```

### Start the experiment and run the training loop.

```python
# Start the experiment
with experiment.start():
    conf.run()
```