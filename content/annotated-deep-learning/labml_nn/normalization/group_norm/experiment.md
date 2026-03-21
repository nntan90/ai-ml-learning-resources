# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/normalization/group_norm/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/group_norm/experiment.ipynb)                    

## Group Norm - CIFAR 10

This is an experiment training a model with group norm to classify CIFAR-10 dataset.

Install the `labml-nn` package.

```python
!pip install labml-nn
```

Imports

```python
import torch
import torch.nn as nn

from labml import experiment
from labml_nn.normalization.group_norm.experiment import Configs
```

Create an experiment

```python
experiment.create(name="cifar10", comment="group norm")
```

Initialize configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf, {
    'optimizer.optimizer': 'Adam',
    'optimizer.learning_rate': 2.5e-4,
})
```

Start the experiment and run the training loop.

```python
with experiment.start():
    conf.run()
```