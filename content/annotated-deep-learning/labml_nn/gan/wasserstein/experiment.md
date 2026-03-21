# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/gan/wasserstein/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/gan/wasserstein/experiment.ipynb)

## DCGAN

This is an experiment training DCGAN model.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python

from labml import experiment
from labml_nn.gan.wasserstein.experiment import Configs
```

Create an experiment

```python
experiment.create(name="mnist_wgan")
```

Initialize configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf,
                   {
                       'discriminator': 'cnn',
                       'generator': 'cnn',
                       'label_smoothing': 0.01,
                       'generator_loss': 'wasserstein',
                       'discriminator_loss': 'wasserstein',
                   })
```

Start the experiment and run the training loop.

```python
with experiment.start():
    conf.run()
```