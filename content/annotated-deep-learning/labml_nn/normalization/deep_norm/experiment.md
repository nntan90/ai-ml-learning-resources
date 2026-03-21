# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/normalization/deep_norm/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/deep_norm/experiment.ipynb)

## DeepNorm

This is an experiment training Shakespeare dataset with a deep transformer using [DeepNorm](https://nn.labml.ai/normalization/deep_norm/index.html).

### Install the packages

```python
!pip install labml-nn --quiet
```

### Imports

```python
from labml import experiment
from labml_nn.normalization.deep_norm.experiment import Configs
```

### Create an experiment

```python
experiment.create(name="deep_norm", writers={'screen'})
```

### Configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf, {
    # Use character level tokenizer
    'tokenizer': 'character',
    # Prompt separator is blank
    'prompt_separator': '',
    # Starting prompt for sampling
    'prompt': 'It is ',
    # Use Tiny Shakespeare dataset
    'text': 'tiny_shakespeare',

    # Use a context size of $256$
    'seq_len': 256,
    # Train for 32 epochs
    'epochs': 32,
    # Batch size $16$
    'batch_size': 16,
    # Switch between training and validation for $10$ times per epoch
    'inner_iterations': 10,

    # Adam optimizer with no warmup
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