# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/transformers/switch/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/experiment.ipynb)                    

## Switch Transformer

This is an experiment training Shakespeare dataset with a small Switch Transformer.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
from labml import experiment
from labml_nn.transformers.switch.experiment import Configs
```

Create an experiment

```python
experiment.create(name="switch_transformer")
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
                    'optimizer.learning_rate': 1.,
                    'optimizer.optimizer': 'Noam',
                    'prompt': 'It is',
                    'prompt_separator': '',

                    'transformer': 'switch_transformer',
                    'is_scale_prob': False,
                    'n_experts': 4,

                    'drop_tokens': True,
                    'capacity_factor': 1.2,

                    'train_loader': 'shuffled_train_loader',
                    'valid_loader': 'shuffled_valid_loader',

                    'seq_len': 64,
                    'epochs': 128,
                    'batch_size': 32,
                    'inner_iterations': 25,
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