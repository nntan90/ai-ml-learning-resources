# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/gan/cycle_gan/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/gan/cycle_gan/experiment.ipynb)

## Cycle GAN

This is an experiment training Cycle GAN model.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
from labml import experiment
from labml.utils.pytorch import get_modules
from labml_nn.gan.cycle_gan import Configs
```

Create an experiment

```python
experiment.create(name="cycle_gan")
```

Initialize configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf, {'dataset_name': 'summer2winter_yosemite'})
```

Initialize

```python
conf.initialize()
```

Set PyTorch models for loading and saving

```python
experiment.add_pytorch_models(get_modules(conf))
```

Start the experiment and run the training loop.

```python
with experiment.start():
    conf.run()
```