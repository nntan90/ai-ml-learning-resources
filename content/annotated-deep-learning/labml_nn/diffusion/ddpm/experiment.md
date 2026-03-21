# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/diffusion/ddpm/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/experiment.ipynb)

## [Denoising Diffusion Probabilistic Models (DDPM)](https://nn.labml.ai/diffusion/ddpm/index.html)

This notebook trains a DDPM based model on MNIST digits dataset.

### Install the packages

```python
!pip install labml-nn --quiet
```

### Imports

```python
from labml import experiment
from labml_nn.diffusion.ddpm.experiment import Configs
```

### Create an experiment

```python
experiment.create(name="diffuse", writers={'screen'})
```

### Configurations

```python
configs = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(configs, {
    'dataset': 'MNIST',
    'image_channels': 1,
    'epochs': 5,
})
```

Initializ

```python
configs.init()
```

Set PyTorch models for loading and saving

```python
experiment.add_pytorch_models({'eps_model': configs.eps_model})
```

### Start the experiment and run the training loop.

```python
# Start the experiment
with experiment.start():
    configs.run()
```