# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/rl/dqn/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/experiment.ipynb)                    

## Deep Q Networks (DQN)

This is an experiment training an agent to play Atari Breakout game using Deep Q Networks (DQN)

Install the `labml-nn` package

```python
!pip install labml-nn
```

Add Atari ROMs (Doesn't work without this in Google Colab)

```python
! wget http://www.atarimania.com/roms/Roms.rar
! mkdir /content/ROM/
! unrar e /content/Roms.rar /content/ROM/
! python -m atari_py.import_roms /content/ROM/
```

Imports

```python
from labml import experiment
from labml.configs import FloatDynamicHyperParam
from labml_nn.rl.dqn.experiment import Trainer
```

Create an experiment

```python
experiment.create(name="dqn")
```

### Configurations

`FloatDynamicHyperParam` is a dynamic hyper-parameter
that you can change while the experiment is running.

```python
configs = {
    # Number of updates
    'updates': 1_000_000,
    # Number of epochs to train the model with sampled data.
    'epochs': 8,
    # Number of worker processes
    'n_workers': 8,
    # Number of steps to run on each process for a single update
    'worker_steps': 4,
    # Mini batch size
    'mini_batch_size': 32,
    # Target model updating interval
    'update_target_model': 250,
    # Learning rate.
    'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
}
```

Set experiment configurations

```python
experiment.configs(configs)
```

Create trainer

```python
trainer = Trainer(**configs)
```

Start the experiment and run the training loop.

```python
with experiment.start():
    trainer.run_training_loop()
```