# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/transformers/gpt/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/gpt/experiment.ipynb)                    

## Training a model with GPT architecture

This is an experiment training Tiny Shakespeare dataset with GPT architecture model.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
from labml import experiment
from labml_nn.transformers.gpt import Configs
```

Create an experiment

```python
experiment.create(name="gpt")
```

Initialize [GPT configurations](https://nn.labml.ai/transformers/gpt/)

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

    # Use a context size of $128$
    'seq_len': 128,
    # Train for $32$ epochs
    'epochs': 32,
    # Batch size $128$
    'batch_size': 128,
    # Switch between training and validation for $10$ times
    # per epoch
    'inner_iterations': 10,

    # Transformer configurations
    'transformer.d_model': 512,
    'transformer.ffn.d_ff': 2048,
    'transformer.n_heads': 8,
    'transformer.n_layers': 6
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