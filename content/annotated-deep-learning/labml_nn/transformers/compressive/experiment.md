# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/transformers/compressive/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/compressive/experiment.ipynb)                    

## Compressive Transformer

This is an experiment training Shakespeare dataset with a Compressive Transformer model.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
import torch
import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_nn.transformers.compressive.experiment import Configs
```

Create an experiment

```python
experiment.create(name="compressive_transformer")
```

Initialize configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf,
                # A dictionary of configurations to override
                {'tokenizer': 'character',
                'text': 'tiny_shakespeare',
                'optimizer.learning_rate': 2.5e-4,
                'optimizer.optimizer': 'AdamW',
                'prompt': 'It is',
                'prompt_separator': '',

                'train_loader': 'sequential_train_loader',
                'valid_loader': 'sequential_valid_loader',

                'seq_len': 8,
                'mem_len': 8,
                'epochs': 128,
                'batch_size': 32,
                'inner_iterations': 25,
                'compression_rate': 2,
                })
```

Set PyTorch models for loading and saving

```python
experiment.add_pytorch_models({'model': conf.model})
```

Start the experiment and run the training loop.

```python
# Start the experiment
with experiment.start():
    conf.run()
```