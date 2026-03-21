# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/rl/ppo/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/experiment.ipynb)                    

## Proximal Policy Optimization - PPO

This is an experiment training an agent to play Atari Breakout game using  Proximal Policy Optimization - PPO

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
from labml.configs import FloatDynamicHyperParam, IntDynamicHyperParam
from labml_nn.rl.ppo.experiment import Trainer
```

Create an experiment

```python
experiment.create(name="ppo")
```

### Configurations

`IntDynamicHyperParam` and `FloatDynamicHyperParam` are dynamic hyper parameters
that you can change while the experiment is running.

```python
configs = {
    # number of updates
    'updates': 10000,
    # number of epochs to train the model with sampled data
    'epochs': IntDynamicHyperParam(8),
    # number of worker processes
    'n_workers': 8,
    # number of steps to run on each process for a single update
    'worker_steps': 128,
    # number of mini batches
    'batches': 4,
    # Value loss coefficient
    'value_loss_coef': FloatDynamicHyperParam(0.5),
    # Entropy bonus coefficient
    'entropy_bonus_coef': FloatDynamicHyperParam(0.01),
    # Clip range
    'clip_range': FloatDynamicHyperParam(0.1),
    # Learning rate
    'learning_rate': FloatDynamicHyperParam(2.5e-4, (0, 1e-3)),
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