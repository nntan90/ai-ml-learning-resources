# Notebook: autoregressive_experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/transformers/basic/autoregressive_experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/basic/autoregressive_experiment.ipynb)

## Transformer Experiment

This trains a simple transformer with
[multi headed attention](https://nn.labml.ai/transformers/mha.html)
introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
on an NLP auto-regression task (with Tiny Shakespeare dataset).

### Install the packages

```python
!pip install labml-nn --quiet
```

### Imports

```python
from labml import experiment
from labml_nn.transformers.basic.autoregressive_experiment import Configs
```

### Create an experiment

```python
experiment.create(name="transformer", writers={'screen'})
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
    'seq_len': 512,
    # Train for 32 epochs
    'epochs': 32,
    # Batch size $32$
    'batch_size': 16,
    # Switch between training and validation for $10$ times
    # per epoch
    'inner_iterations': 10,

    # Model size
    'd_model': 256,
    'transformer.n_heads': 16,
    'transformer.ffn.d_ff': 1024,

    # Use [Noam optimizer](../../optimizers/noam.html)
    'optimizer.optimizer': 'Noam',
    'optimizer.learning_rate': 1.,
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