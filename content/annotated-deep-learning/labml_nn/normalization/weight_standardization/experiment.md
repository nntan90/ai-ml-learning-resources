# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/normalization/weight_standardization/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/group_norm/experiment.ipynb)                    

## Weight Standardization & Batch-Channel Normalization - CIFAR 10

This is an experiment training a model with Weight Standardization & Batch-Channel Normalization to classify CIFAR-10 dataset.

Install the `labml-nn` package.

```python
!pip install labml-nn
```

Imports

```python
from labml import experiment
from labml_nn.normalization.weight_standardization.experiment import CIFAR10Configs as Configs
```

Create an experiment

```python
experiment.create(name="cifar10", comment="WS + BCN")
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
    'train_batch_size': 64,
})
```

Start the experiment and run the training loop.

```python
with experiment.start():
    conf.run()
```