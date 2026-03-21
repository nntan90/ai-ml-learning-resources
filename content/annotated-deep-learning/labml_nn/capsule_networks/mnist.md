# Notebook: mnist

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/capsule_networks/mnist.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/capsule_networks/mnist.ipynb)                    

## Training a Capsule Network to classify MNIST digits

This is an experiment to train a Capsule Network to classify MNIST digits using PyTorch.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
import torch

from labml import experiment
from labml_nn.capsule_networks.mnist import Configs
```

Create an experiment

```python
experiment.create(name="capsule_networks")
```

Initialize [Capsule Network configurations](https://nn.labml.ai/capsule_networks/mnist.html)

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                         'optimizer.learning_rate': 1e-3,
                         'inner_iterations': 5})
```

Set PyTorch models for loading and saving

```python
experiment.add_pytorch_models({'model': conf.model})
```

Start the experiment and run the training loop.

```python
with experiment.start():
    conf.run()
```