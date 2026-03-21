# Notebook: mnist

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/normalization/batch_norm/mnist.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/batch_norm/mnist.ipynb)                    

## Batch Normaliztion

This trains is a simple convolutional neural network that uses
[batch normalization](https://nn.labml.ai/normalization/batch_norm/index.html) to classify MNIST digits.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
import torch
from labml import experiment
from labml_nn.normalization.batch_norm.mnist import MNISTConfigs
```

Create an experiment

```python
experiment.create(name="mnist_batch_norm")
```

Initialize configurations

```python
conf = MNISTConfigs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf, {'optimizer.optimizer': 'Adam'})
```

Start the experiment and run the training loop.

```python
with experiment.start():
    conf.run()
```